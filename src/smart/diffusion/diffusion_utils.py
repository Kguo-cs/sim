from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor


STATE_DIM = 8
POS = slice(0, 2)
HEADING = slice(2, 4)
SHAPE = slice(4, 6)
VEL = slice(6, 8)


def _split_state(state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split [x, y, heading_2d, shape_2d, velocity_2d]."""
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected [N, 8], got {tuple(state.shape)}")
    return state[:, POS], state[:, HEADING], state[:, SHAPE], state[:, VEL]

def bounded_log_std(
    raw_scale: Tensor,
    min_log_std: float = -3.0,
    max_log_std: float = 0.0,
) -> Tensor:
    return min_log_std + (
        max_log_std - min_log_std
    ) * torch.sigmoid(raw_scale)

def gaussian_nll_2d(
    mu: Tensor,
    log_sigma: Tensor,
    target: Tensor,
    min: float = 1e-3,
) -> Tensor:
    """Per-sample diagonal Gaussian negative log likelihood."""
    log_std = bounded_log_std(log_sigma)

    sigma = log_std.exp()
    error = (target - mu) / sigma
    return 0.5 * (
        error.square() + 2.0 * log_std + math.log(2.0 * math.pi)
    ).sum(-1)


def gm_kl_loss(
    means: Tensor,
    logweights: Tensor,
    logstds: Tensor,
    sample: Tensor,
    eps: float = 1e-4,
) -> Tensor:
    """Per-sample Gaussian-mixture NLL (legacy name kept for compatibility)."""
    if logweights.ndim == 3:
        logweights = logweights.squeeze(-1)
    if logstds.ndim == 2:
        logstds = logstds[:, None, :]

    sigma = logstds.exp().clamp_min(eps)
    error = (sample[:, None, :] - means) / sigma
    component_log_prob = -0.5 * (
        error.square() + 2.0 * sigma.log() + math.log(2.0 * math.pi)
    ).sum(-1)

    logweights = logweights.log_softmax(-1)
    return -torch.logsumexp(component_log_prob + logweights, dim=-1)


def get_scale(x0_prediction: Tensor, x0: Tensor, eps: float = 1e-5) -> Tensor:
    """Squared error normalized by detached per-sample mean absolute error."""
    if x0_prediction.shape != x0.shape:
        raise ValueError("x0_prediction and x0 must have the same shape")

    dims = tuple(range(1, x0.ndim))
    error = x0_prediction - x0
    scale = error.detach().abs().mean(dims, keepdim=True).clamp_min(eps)
    return (error.square() / scale).mean(dims)


def _component_loss(
    prediction: Tensor,
    target: Tensor,
    use_huber: bool,
    huber_beta: float,
) -> Tensor:
    if use_huber:
        # if huber_beta <= 0:
        #     raise ValueError("huber_beta must be positive")
        # loss = F.smooth_l1_loss(
        #     prediction,
        #     target,
        #     beta=huber_beta,
        #     reduction="none",
        # )
        loss=F.l1_loss(prediction, target, reduction="none")
    else:
        loss = F.mse_loss(prediction, target, reduction="none")#.square()
    return loss.mean(-1)


def _parse_prediction(fake_state: Tensor):
    """Parse deterministic, Gaussian, or Gaussian-mixture output.

    Layouts:
      [N, 8]       deterministic
      [N, 16]      8 means + 8 log standard deviations
      [N, 9*K + 8] K means + K logits + 8 shared log standard deviations
    """
    dim = fake_state.shape[-1]

    if dim == STATE_DIM:
        return "deterministic", fake_state, None, None

    if dim == 2 * STATE_DIM:
        return "gaussian", fake_state[:, :8], None, fake_state[:, 8:]

    if dim <= STATE_DIM or (dim - STATE_DIM) % (STATE_DIM + 1) != 0:
        raise ValueError(f"Unsupported prediction dimension: {dim}")

    num_components = (dim - STATE_DIM) // (STATE_DIM + 1)
    means_end = num_components * STATE_DIM
    logits_end = means_end + num_components

    means = fake_state[:, :means_end].reshape(-1, num_components, STATE_DIM)
    logits = fake_state[:, means_end:logits_end]
    logstds = fake_state[:, logits_end:]
    return "mixture", means, logits, logstds


def matching_loss(
    real_state: Tensor,
    fake_state: Tensor,
    w_pos=0.1,
    w_heading=0.5,
    w_shape=0.2,
    w_vel=0.2,
    use_huber: bool = False,
    huber_beta: float = 0.1,
    scale=None,
):
    """Return total and component losses, each with shape [N]."""

    if real_state.shape[0] != fake_state.shape[0]:
        raise ValueError("real_state and fake_state must have the same length")

    real_pos, real_heading, real_shape, real_vel = _split_state(
        real_state[:, :STATE_DIM]
    )
    mode, prediction, logits, logstds = _parse_prediction(fake_state)

    if mode == "deterministic":
        fake_pos, fake_heading, fake_shape, fake_vel = _split_state(prediction)
        pos_loss = _component_loss(fake_pos, real_pos, True, huber_beta)
        heading_loss = _component_loss(
            fake_heading, real_heading, True, huber_beta
        )
        shape_loss = _component_loss(fake_shape, real_shape, True, huber_beta)
        vel_loss = _component_loss(fake_vel, real_vel, True, huber_beta)

    elif mode == "gaussian":
        fake_pos, fake_heading, fake_shape, fake_vel = _split_state(prediction)
        pos_std, heading_std, shape_std, vel_std = _split_state(logstds)

        pos_loss = gaussian_nll_2d(fake_pos/scale[:,:2], pos_std, real_pos/scale[:,:2], 1e-1)
        heading_loss = gaussian_nll_2d(
            fake_heading/scale[:,2:4], heading_std, real_heading/scale[:,2:4], 1e-1
        )
        shape_loss = gaussian_nll_2d(fake_shape/scale[:,4:6], shape_std, real_shape/scale[:,4:6], 1e-1)
        vel_loss = gaussian_nll_2d(fake_vel/scale[:,6:8], vel_std, real_vel/scale[:,6:8], 1e-1)

        pos_loss1 = _component_loss(fake_pos, real_pos, True, huber_beta)
        heading_loss1 = _component_loss(
            fake_heading, real_heading, True, huber_beta
        )
        shape_loss1 = _component_loss(fake_shape, real_shape, use_huber, huber_beta)
        vel_loss1 = _component_loss(fake_vel, real_vel, use_huber, huber_beta)

    else:
        fake_pos = prediction[..., POS]
        fake_heading = prediction[..., HEADING]
        fake_shape = prediction[..., SHAPE]
        fake_vel = prediction[..., VEL]
        pos_std, heading_std, shape_std, vel_std = _split_state(logstds)

        pos_loss = gm_kl_loss(fake_pos, logits, pos_std, real_pos)
        heading_loss = gm_kl_loss(fake_heading, logits, heading_std, real_heading)
        shape_loss = gm_kl_loss(fake_shape, logits, shape_std, real_shape)
        vel_loss = gm_kl_loss(fake_vel, logits, vel_std, real_vel)

    total_loss = (
        w_pos * pos_loss
        + w_heading * heading_loss
        + w_shape * shape_loss
        + w_vel * vel_loss
    )
    #total_loss = F.mse_loss(real_state,fake_state, reduction="none").mean(-1)*w_pos
    #total_loss=w_pos*F.l1_loss(real_state,fake_state,reduction='none').mean(-1)
    return total_loss, pos_loss, heading_loss, shape_loss, vel_loss


def compute_vehicle_circles_torch(
    pos: Tensor,
    heading: Tensor,
    length: Tensor,
    width: Tensor,
    num_circles: int = 5,
):
    """Approximate each oriented box by circles along its centerline."""
    if num_circles < 1:
        raise ValueError("num_circles must be positive")

    length = length.clamp_min(1e-6)
    width = width.clamp_min(1e-6)
    offsets = torch.linspace(
        -0.5,
        0.5,
        num_circles,
        device=pos.device,
        dtype=pos.dtype,
    )

    centerline_length = (length - width).clamp_min(0)
    local_x = centerline_length[:, None] * offsets[None]
    direction = torch.stack([heading.cos(), heading.sin()], dim=-1)
    centers = pos[:, None] + local_x[..., None] * direction[:, None]

    radius = width / math.sqrt(3.8)
    radii = radius[:, None].expand(-1, num_circles)
    return centers, radii


def compute_penetration(
    state: Tensor,
    start_idx: Tensor,
    end_idx: Tensor,
    num_circles: int = 5,
    eps: float = 1e-8,
):
    """Maximum circle penetration for each pair; positive means collision."""
    if start_idx.numel() == 0:
        return state.new_empty(0)

    pos = state[:, :2]
    heading = torch.atan2(state[:, 3], state[:, 2])

    # Detach shape so collision loss cannot be reduced by shrinking vehicles.
    length = state[:, 4].detach()
    width = state[:, 5].detach()
    centers, radii = compute_vehicle_circles_torch(
        pos, heading, length, width, num_circles
    )

    delta = centers[start_idx, :, None] - centers[end_idx, None, :]
    distance = (delta.square().sum(-1) + eps).sqrt()
    penetration = (
        radii[start_idx, :, None] + radii[end_idx, None, :] - distance
    )
    return penetration.amax(dim=(1, 2))


def _within_batch_pairs(batch: Tensor):
    """Create each unordered pair once without constructing an N x N mask."""
    start_all, end_all = [], []

    for batch_id in batch.unique():
        members = torch.where(batch == batch_id)[0]
        if members.numel() < 2:
            continue

        start, end = torch.triu_indices(
            members.numel(),
            members.numel(),
            offset=1,
            device=batch.device,
        )
        start_all.append(members[start])
        end_all.append(members[end])

    if not start_all:
        empty = torch.empty(0, dtype=torch.long, device=batch.device)
        return empty, empty

    return torch.cat(start_all), torch.cat(end_all)


def multi_circle_collision_loss_mem_efficient(
    fake_state: Tensor,
    batch: Tensor,
):
    """Collision penalty for all unordered pairs in the same scene."""

    start_idx, end_idx = _within_batch_pairs(batch.to(fake_state.device))
    penetration = compute_penetration(fake_state, start_idx, end_idx)
    loss = torch.expm1(penetration.relu())

    # Preserve the original output order.
    return loss, end_idx, start_idx


def get_col_rate(tokenized_agent, pred_init: Tensor):
    """Return 1 for non-colliding agents and 0 for colliding agents."""
    num_agents = len(pred_init)
    batch = tokenized_agent["batch"][-num_agents:].to(pred_init.device)
    loss, end_idx, start_idx = multi_circle_collision_loss_mem_efficient(
        pred_init,  batch
    )

    colliding = torch.zeros(
        num_agents, dtype=torch.bool, device=pred_init.device
    )
    collision_edge = loss > 0
    colliding[start_idx[collision_edge]] = True
    colliding[end_idx[collision_edge]] = True
    return (~colliding).to(pred_init.dtype)


def _group_labels(tokenized_agent, num_states: int, use_all_type: bool):
    if isinstance(tokenized_agent, Mapping):
        batch = tokenized_agent["batch"][-num_states:]
    else:
        batch = tokenized_agent[-num_states:]

    batch = batch.detach().cpu().long().numpy()
    if use_all_type:
        return batch, None

    if not isinstance(tokenized_agent, Mapping):
        raise TypeError("tokenized_agent must contain 'batch' and 'type'")
    agent_type = tokenized_agent["type"][-num_states:]
    return batch, agent_type.detach().cpu().long().numpy()


@torch.no_grad()
def get_closest_sum_idx_fast(
    fake_state: Tensor,
    real_state: Tensor,
    tokenized_agent,
    all_state: bool = False,
    use_all_type: bool = False,
):
    """Hungarian matching within each scene, optionally separated by type."""
    if len(fake_state) != len(real_state):
        raise ValueError("fake_state and real_state must have the same length")

    num_states = len(fake_state)
    if num_states == 0:
        return torch.empty(0, dtype=torch.long, device=fake_state.device)

    fake = fake_state if all_state else fake_state[:, :2]
    real = real_state if all_state else real_state[:, :2]
    fake = fake.detach().float().cpu().numpy()
    real = real.detach().float().cpu().numpy()

    if not np.isfinite(fake).all() or not np.isfinite(real).all():
        raise ValueError("Matching input contains NaN or infinity")

    batch, agent_type = _group_labels(
        tokenized_agent, num_states, use_all_type
    )
    if agent_type is None:
        order = np.argsort(batch, kind="stable")
        group_key = batch[order, None]
    else:
        order = np.lexsort((agent_type, batch))
        group_key = np.stack([batch[order], agent_type[order]], axis=1)

    group_change = np.any(group_key[1:] != group_key[:-1], axis=1)
    starts = np.r_[0, np.flatnonzero(group_change) + 1]
    ends = np.r_[starts[1:], num_states]

    matched = np.arange(num_states, dtype=np.int64)
    for start, end in zip(starts, ends):
        idx = order[start:end]
        if len(idx) <= 1:
            continue

        real_group = real[idx]
        fake_group = fake[idx]
       # cost=np.abs(real_group - fake_group).sum(axis=1, keepdims=True)
        cost = (
            np.sum(real_group**2, axis=1, keepdims=True)
            + np.sum(fake_group**2, axis=1, keepdims=True).T
            - 2.0 * real_group @ fake_group.T
        )
        np.maximum(cost, 0, out=cost)
        row, col = linear_sum_assignment(cost)
        matched[idx[row]] = idx[col]

    return torch.from_numpy(matched).to(fake_state.device)


def get_closest_sum_idx(
    fake_state: Tensor,
    real_state: Tensor,
    tokenized_agent,
    all_state: bool = False,
    use_all_type: bool = False,
):
    """Compatibility wrapper around the optimized implementation."""
    return get_closest_sum_idx_fast(
        fake_state,
        real_state,
        tokenized_agent,
        all_state,
        use_all_type,
    )


def _time_weight(
    t: Tensor,
    num_states: int,
    t_eps: float,
    x_pred: bool,
    max_loss_weight: float | None,
):
    t = t if t.ndim == 1 else t.reshape(num_states, -1)[:, 0]
    valid = (t > 0) & (t < 1)

    if x_pred:
        weight = t.clamp_min(t_eps).reciprocal()#.square()
        if max_loss_weight is not None:
            weight = weight.clamp_max(max_loss_weight)
    else:
        weight = torch.ones_like(t)

    return weight * valid.to(weight.dtype)


def get_diff_loss(
    tokenized_agent,
    fake_state: Tensor,
    real_state: Tensor,
    t: Tensor,
    t_eps: float,
    scale=1,
    all_state: bool = False,
    use_col: bool = False,
    use_all_type: bool = False,
    use_match: bool = False,
    x_pred: bool = False,
    w_pos: float = 0.1/ 5,
    w_heading: float = 0.5/ 5,
    w_shape: float = 0.2/ 5,
    w_vel: float = 1 / 5,
    max_loss_weight: float | None = None,
    use_huber: bool = False,
    huber_beta: float = 0.1,
):
    """State reconstruction loss plus optional symmetric collision loss."""
    num_states = len(fake_state)
    batch = tokenized_agent["batch"][-num_states:].to(fake_state.device)
    weight = _time_weight(t, num_states, t_eps, x_pred, max_loss_weight)

    if use_match:
        fake_idx = get_closest_sum_idx_fast(
            fake_state / scale,
            real_state / scale,
            tokenized_agent,
            all_state,
            use_all_type,
        )
        fake_state = fake_state[fake_idx]

    collision_loss = fake_state.new_zeros(())
    if use_col and x_pred:
        edge_loss, end_idx, start_idx = multi_circle_collision_loss_mem_efficient(
            fake_state, batch
        )
        collision_loss = (edge_loss * weight[start_idx]).mean()

    # #w_pos=w_heading=w_shape=w_vel=1
    # # w_heading=1#0.05
    # # w_shape=1#0.05
    # # w_vel=1
    #
    #
    # w_pos=w_heading=w_shape=w_vel=1#0.1
    # # w_pos=5
    # # w_heading=0.05
    # # w_shape=0.01
    # # w_vel=1
    #
    #
    #
    # real_state=real_state/scale
    # fake_state=fake_state/scale

    losses = matching_loss(
        real_state,
        fake_state,
        w_pos=w_pos * weight,
        w_heading=w_heading * weight,
        w_shape=w_shape * weight,
        w_vel=w_vel * weight,
        use_huber=use_huber,
        huber_beta=huber_beta,
        scale=scale
    )
    return losses[0], collision_loss, *losses[1:]