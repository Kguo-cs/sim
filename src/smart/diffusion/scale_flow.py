"""Rectified Flow with timestep-adaptive SDE noise.

State convention:
    [x, y, heading_cos, heading_sin, length, width, vx, vy]

Rectified-Flow convention:
    x0 ~ data, x1 ~ noise
    x_t = (1 - t) * x0 + t * x1
    velocity = dx_t/dt = x1 - x0

Training uses t in [0, 1]. Generation starts from noise at t=1 and
integrates backward to data at t=0. The SDE/PPO transition uses the same
Rectified-Flow time directly; no complementary ``1 - t`` variable is used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.diffusion.diffusion_utils import (
    get_closest_sum_idx_fast,
    get_diff_loss,
)
from src.smart.utils import weight_init
import copy
from .denoiser import InitDenoiser


class ScaleFlow(nn.Module):
    """Initial-state flow with simple timestep-adaptive branch noise."""

    def __init__(
        self,
        args,
        token_processor,
        gail: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(args.hidden_dim)

        # Standard x0-prediction flow. No Gaussian or MeanFlow output.
        self.x_pred = True
        self.model = InitDenoiser(
            token_processor,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.input_dim,
            num_freq_bands=args.num_freq_bands,
            num_layers=args.num_denoiser_layers,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            dropout=args.dropout,
        )

        self.t_eps = 0.05
        self.token_processor=token_processor

        # --------------------------------------------------------------
        # Initial-state policy optimization
        # --------------------------------------------------------------
        self.use_sde = bool( gail and getattr(token_processor, "learn_init", False) )

        # --------------------------------------------------------------
        # Multi-branch sampling
        # --------------------------------------------------------------
        self.num_branch_steps = int(
            getattr(args, "num_branch_steps", 1)
        )
        self.fixed_branch_steps = self._parse_fixed_branch_steps(
            getattr(args, "branch_steps", None)
        )
        self.use_refiner = token_processor.use_refiner

        if self.use_refiner:
            self.use_sde=False
            self.refine_model = InitDenoiser(
                token_processor,
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.input_dim*2,
                num_freq_bands=args.num_freq_bands,
                num_layers=1,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                dropout=args.dropout,
            )

        self.apply(weight_init)

    @staticmethod
    def _parse_fixed_branch_steps(
        value,
    ) -> Optional[tuple[int, ...]]:
        if value is None:
            return None

        if torch.is_tensor(value):
            value = (
                value.detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
        ):
            raise TypeError(
                "branch_steps must be a sequence "
                "of timestep indices."
            )

        steps = tuple(
            sorted(
                {
                    int(step)
                    for step in value
                }
            )
        )

        if not steps:
            raise ValueError(
                "branch_steps cannot be empty."
            )

        return steps

    # ==================================================================
    # Standard flow helpers
    # ==================================================================
    def _sample_noise(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
    ) -> Tensor:
        #x = self.model.normalize(x)
        noise = torch.randn_like(x)
        noise=self.model.denormalize(
            noise
        )

        ego_mask = tokenized_agent[
            "ego_mask"
        ].bool()
        noise[ego_mask] = x[ego_mask]

        matched_index = get_closest_sum_idx_fast(
            noise,
            x,
            tokenized_agent,
            all_state=True,
        )

        return noise[matched_index]

        # return self.model.denormalize(
        #     noise[matched_index]
        # )
        #

    def _sample_time(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
    ) -> Tensor:
        batch = tokenized_agent["batch"].long()
        num_graphs = int(
            tokenized_agent["num_graphs"]
        )

        scene_time = torch.rand(
            (num_graphs, 1),
            device=x.device,
            dtype=x.dtype,
        )

        return scene_time[batch]

    @staticmethod
    def _fix_conditioned_agents(
        clean: Tensor,
        latent: Tensor,
        time: Tensor,
        tokenized_agent: HeteroData,
    ) -> None:
        ego_mask = tokenized_agent[ "ego_mask"]
        latent[ego_mask] = clean[ego_mask]
        time[ego_mask] = 0.0

    def _prepare_supervised_batch(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
    ) -> tuple[Tensor, Tensor, Tensor]:
        noise = self._sample_noise(
            x,
            tokenized_agent,
        )

        time = self._sample_time(
            x,
            tokenized_agent,
        )

        self._fix_conditioned_agents(
            x,
            noise,
            time,
            tokenized_agent,
        )

        # Rectified Flow: x0=data, x1=noise.
        latent = (
            (1.0 - time) * x
            + time * noise
        )

        return noise, time, latent

    def _model_velocity(
        self,
        latent: Tensor,
        time: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
        eval_mask: Optional[Tensor] = None,
        mode: int = 1,
        use_map_condition: bool = True,
    ) -> tuple[Tensor, Tensor]:
        prediction = self.model(
            latent,
            time,
            tokenized_agent,
            map_feature,
            eval_mask,
            mode=mode,
            use_map_condition=use_map_condition,
        )

        x0 = prediction[:, : latent.shape[-1]]

        velocity = (
            latent - x0
        ) / time.clamp_min(
            self.t_eps
        )

        return velocity, x0

    def get_loss(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ):
        loss=self._supervised_loss(
            x,
            tokenized_agent,
            map_feature,
        )

        if "advantages" in tokenized_agent:
            if self.use_sde:
                rl_loss = self._sde_advantage_loss(
                    tokenized_agent,
                    map_feature,
                )
            else:
                rl_loss = self._direct_advantage_loss(
                    tokenized_agent,
                    map_feature,
                )

            tokenized_agent["rl_loss"] = rl_loss

        return loss

    def _supervised_loss(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ):
        _, time, latent = (
            self._prepare_supervised_batch(
                x,
                tokenized_agent,
            )
        )

        _, x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        ego_mask = tokenized_agent[
            "ego_mask"
        ].bool()[:, None]

        # Avoid in-place modification of model output.
        x0 = torch.where(
            ego_mask,
            x.detach(),
            x0,
        )

        loss = get_diff_loss(
            tokenized_agent,
            x0,
            x,
            time,
            self.t_eps,
            scale=self.model.normal_scale,
            use_col=True,
            x_pred=True,
        )

        return loss, x0, latent, time

    def _sde_advantage_loss(
        self,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ) -> Tensor:
        (
            current,
            next_sample,
            old_log_prob,
        ) = tokenized_agent["sde_z"]

        (
            time,
            next_time,
        ) = tokenized_agent["sde_t"]

        saved_noise_level = tokenized_agent.get(
            "sde_noise_level"
        )

        if current.ndim == 2:
            current = current[:, None]
            next_sample = next_sample[:, None]
            time = time[:, None]
            next_time = next_time[:, None]

            if old_log_prob is not None:
                old_log_prob = old_log_prob[:, None]

            if saved_noise_level is not None:
                saved_noise_level = saved_noise_level[:, None]

        num_agents, num_branches, _ = current.shape

        non_ego = ~tokenized_agent[
            "ego_mask"
        ].bool()

        if not torch.any(non_ego):
            return current.new_zeros(())

        tokenized_agent=self.repeat_input_copy(tokenized_agent,num_branches)

        velocities, pred_x0 = self._model_velocity(
            current.transpose(0, 1).flatten(0, 1),
            time.transpose(0, 1).flatten(0, 1),
            tokenized_agent,
            map_feature,
        )

        velocities = velocities.reshape(
            num_branches,
            num_agents,
            -1,
        ).transpose(0, 1)  # [A, B, D]

        branch_noise = saved_noise_level.detach()

        # delta_t = time - next_time

        active = (
                non_ego[:, None]
                & (branch_noise.amax(dim=-1) > 0)
        )

        (
            _,
            selected_log_prob,
            _,
            _,
        ) = self.sde_step_with_logprob(
            time=time[active],
            next_time=next_time[active],
            model_output=velocities[active],
            sample=current[active],
            noise_level=branch_noise[active],
            prev_sample=next_sample[active],
        )

        advantages = tokenized_agent["advantages"].transpose(0, 1) #a,t

        selected_advantage = advantages[active]

        # time_for_ratio = torch.where(
        #     time >= 1.0,
        #     torch.full_like(time, 0.9),
        #     time,
        # )
        #
        # diffusion = (
        #         torch.sqrt(
        #             time.clamp_min(0.0)
        #             / (1.0 - time_for_ratio)
        #         )
        #         * branch_noise#noise_level
        # )#smaller t , smaller std
        #
        # step_std_dim = (
        #         diffusion * torch.sqrt(torch.abs(delta_t))
        # )
        #
        # step_std = step_std_dim.mean(dim=-1)
        #
        # selected_step_std = 1/step_std[active]  # [N]
        #
        # weight = selected_step_std / (
        #         selected_step_std.mean().detach() + 1e-8
        # )
        #v_norm=velocities[active].norm(dim=-1).mean()

        weight=1

        loss=-(weight*
            selected_log_prob
            * selected_advantage
        ).mean()

        return loss#+v_norm*0.01

    def _direct_advantage_loss(
            self,
            tokenized_agent: HeteroData,
            map_feature: Mapping[str, Tensor],
    ) -> Tensor:
        sampled_x0 = tokenized_agent["gen_z"].detach()

        _, time, latent = self._prepare_supervised_batch(
            sampled_x0,
            tokenized_agent,
        )

        _, pred_x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        non_ego = ~tokenized_agent["ego_mask"].bool()

        if not torch.any(non_ego):
            return sampled_x0.new_zeros(())

        advantage = (
            tokenized_agent["advantages"].transpose(0, 1)[:,0][non_ego]
            .detach()
        )

        # Assuming advantage has already been normalized.
        advantage = advantage.clamp(-2.0, 2.0)

        pred = pred_x0[non_ego]
        target_endpoint = sampled_x0[non_ego]

        alpha_pos = 0.10
        alpha_neg = 0.05

        coeff = torch.where(
            advantage >= 0,
            alpha_pos * advantage,
            alpha_neg * advantage,
        )

        coeff = coeff.clamp(
            min=-0.10,
            max=0.20,
        )

        target_x0 = (
                pred.detach()
                + coeff[..., None]
                * (
                        target_endpoint
                        - pred.detach()
                )
        )

        loss = 0.5 * (
                pred - target_x0.detach()
        ).square().mean()

        return loss
    # ==================================================================
    # Multi-branch sampling
    # ==================================================================
    def _choose_branch_steps(
        self,
        num_graphs: int,
        total_steps: int,
        device: torch.device,
        branch_steps: Optional[
            int | Sequence[int] | Tensor
        ],
    ) -> Tensor:
        if (
            branch_steps is None
            and self.fixed_branch_steps is not None
        ):
            branch_steps = self.fixed_branch_steps

        if torch.is_tensor(branch_steps):
            if branch_steps.ndim == 0:
                branch_steps = int(
                    branch_steps.item()
                )
            else:
                branch_steps = (
                    branch_steps.detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )

        if (
            isinstance(branch_steps, Sequence)
            and not isinstance(
                branch_steps,
                (str, bytes),
            )
        ):
            selected = sorted(
                {
                    int(step)
                    for step in branch_steps
                }
            )

            return torch.tensor(
                selected,
                device=device,
                dtype=torch.long,
            )[None].expand(
                num_graphs,
                -1,
            )

        count = (
            self.num_branch_steps
            if branch_steps is None
            else int(branch_steps)
        )

        count = min(
            count,
            total_steps,
        )

        # Sampling without replacement for every scene.
        random_score = torch.rand(
            num_graphs,
            total_steps-1,
            device=device,
        )

        selected = random_score.argsort(
            dim=1
        )[:, :count]

        return selected

    @staticmethod
    def _gather_steps(
        values: Tensor,
        step_index: Tensor,
    ) -> Tensor:
        """Gather [N, T, ...] at per-agent indices [N, B]."""
        if values.ndim < 2:
            raise ValueError(
                "values must have shape [N, T, ...]."
            )

        if (
            step_index.ndim != 2
            or step_index.shape[0] != values.shape[0]
        ):
            raise ValueError(
                "step_index must have shape [N, B]."
            )

        index = step_index

        for _ in range(values.ndim - 2):
            index = index.unsqueeze(-1)

        index = index.expand(
            *step_index.shape,
            *values.shape[2:],
        )

        return values.gather(
            dim=1,
            index=index,
        )

    @torch.no_grad()
    def _sample_step(
        self,
        latent: Tensor,
        time_scalar: Tensor,
        next_time_scalar: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
        branch_mask: Optional[Tensor] = None,
    ):
        num_agents = latent.shape[0]

        time = torch.full(
            (num_agents, 1),
            time_scalar,
            device=latent.device,
            dtype=latent.dtype,
        )
        next_time = torch.full_like(time, next_time_scalar)

        self._fix_conditioned_agents(
            tokenized_agent["expert_input"],
            latent,
            time,
            tokenized_agent,
        )

        ego_mask = tokenized_agent["ego_mask"]
        next_time[ego_mask] = 0.0

        velocity, x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        next_latent = latent + (next_time - time) * velocity
        log_prob = latent.new_zeros(num_agents)
        used_noise_level = latent.new_zeros(latent.shape)

        if (
            self.use_sde
            and branch_mask is not None
            and branch_mask.any()
            and self.token_processor.learn_init
           # and "gt_z_raw" not in tokenized_agent
        ):
            noise_level =0.5 #self.get_adaptive_noise_level(time, next_time)
            noise_level = noise_level * branch_mask[:, None].to(latent.dtype)

            stochastic = branch_mask.bool() & (~ego_mask)
            stochastic &= noise_level.amax(dim=-1) > 0

            if torch.any(stochastic):
                (
                    stochastic_next,
                    stochastic_log_prob,
                    _,
                    _,
                ) = self.sde_step_with_logprob(
                    time=time[stochastic],
                    next_time=next_time[stochastic],
                    model_output=velocity[stochastic],
                    sample=latent[stochastic],
                    noise_level=noise_level[stochastic],
                )
                next_latent[stochastic] = stochastic_next
                log_prob[stochastic] = stochastic_log_prob

                # Expand [N,1] scheduled noise to [N,D] for replay storage.
                used_noise_level[stochastic] = noise_level[stochastic]

        return (
            next_latent,
            x0,
            time,
            next_time,
            log_prob,
            used_noise_level,
        )

    @torch.no_grad()
    def sample(
        self,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
        steps: int = 20,
        branch_steps: Optional[
            int | Sequence[int] | Tensor
        ] = None,
    ) -> Tensor:

        agent_batch = tokenized_agent[
            "batch"
        ].long()

        num_graphs = int(
            tokenized_agent["num_graphs"]
        )

        num_agents = agent_batch.numel()

        ego_mask = tokenized_agent[
            "ego_mask"
        ].bool()

        latent = torch.randn(
            num_agents,
            self.model.m_delta_dim,
            device=agent_batch.device,
            dtype=self.model.normal_scale.dtype,
        )

        latent = self.model.denormalize(
            latent
        )
        tokenized_agent["gen_noise"]=latent.clone()

        if "expert_input" not in tokenized_agent:
            expert_input, _ = self.model.get_input(
                tokenized_agent
            )
            tokenized_agent[
                "expert_input"
            ] = expert_input

        # Generation: start from x1~noise at t=1 and integrate to x0 at t=0.
        timesteps = torch.linspace(
            1.0,
            0.0,
            steps + 1,
            device=agent_batch.device,
            dtype=latent.dtype,
        )

        if self.use_sde:
            graph_branch_steps = (
                self._choose_branch_steps(
                    num_graphs,
                    steps,
                    agent_batch.device,
                    branch_steps,
                )
            )

            agent_branch_steps = (
                graph_branch_steps[
                    agent_batch
                ]
            )

            step_branch_mask = torch.zeros(
                num_agents,
                steps,
                device=latent.device,
                dtype=torch.bool,
            )
            step_branch_mask.scatter_(
                dim=1,
                index=agent_branch_steps,
                value=True,
            )
            step_branch_mask[ego_mask] = False

            latent_history = [
                latent.clone()
            ]
            time_history = []
            next_time_history = []
            log_prob_history = []
            noise_level_history = []
            feature_history = []

        # Rollout sampling should not use dropout. This also ensures that
        # denoisers which expose noise_feat_cur only in eval mode work.
        model_was_training = self.model.training
        self.model.eval()

        try:
            for step in range(steps):
                (
                    latent,
                    _,
                    time,
                    next_time,
                    log_prob,
                    used_noise_level,
                ) = self._sample_step(
                    latent,
                    timesteps[step],
                    timesteps[step + 1],
                    tokenized_agent,
                    map_feature,
                    branch_mask=(
                        step_branch_mask[:, step]
                        if self.use_sde
                        else None
                    ),
                )

                if self.use_sde:
                    latent_history.append(
                        latent.clone()
                    )
                    time_history.append(time)
                    next_time_history.append(
                        next_time
                    )
                    log_prob_history.append(
                        log_prob
                    )
                    noise_level_history.append(
                        used_noise_level
                    )

                    feature_history.append(
                        tokenized_agent[
                            "noise_feat_cur"
                        ].clone()
                    )

                elif (
                    step == 0
                    and "noise_feat_cur"
                    in tokenized_agent
                ):
                    tokenized_agent[
                        "noise_feat"
                    ] = tokenized_agent[
                        "noise_feat_cur"
                    ][:,None]

        finally:
            self.model.train(
                model_was_training
            )

        latent[
            ego_mask
        ] = tokenized_agent[
            "expert_input"
        ][ego_mask]

        if not self.use_sde:
            tokenized_agent["gen_z"] = latent

            if self.use_refiner:
                prediction = self.refine_model(
                    latent,
                    torch.zeros_like(latent[:,:1]),
                    tokenized_agent,
                    map_feature,
                )

                prediction_mean=prediction[:,:self.model.input_dim]
                prediction_logstd=prediction[:,self.model.input_dim:]

                latent=prediction_mean+prediction_logstd.exp()*torch.randn_like(prediction_mean)

                tokenized_agent["noise_feat"]=tokenized_agent[ "noise_feat_cur" ][:,None]
                tokenized_agent["refined_z"] = latent

            return latent

        latent_stack = torch.stack(
            latent_history,
            dim=1,
        )

        time_stack = torch.stack(
            time_history,
            dim=1,
        )

        next_time_stack = torch.stack(
            next_time_history,
            dim=1,
        )

        log_prob_stack = torch.stack(
            log_prob_history,
            dim=1,
        )

        noise_level_stack = torch.stack(
            noise_level_history,
            dim=1,
        )

        feature_stack = torch.stack(
            feature_history,
            dim=1,
        )

        selected_current = self._gather_steps(
            latent_stack[:, :-1],
            agent_branch_steps,
        )

        selected_next = self._gather_steps(
            latent_stack[:, 1:],
            agent_branch_steps,
        )

        selected_time = self._gather_steps(
            time_stack,
            agent_branch_steps,
        )

        selected_next_time = self._gather_steps(
            next_time_stack,
            agent_branch_steps,
        )

        selected_log_prob = self._gather_steps(
            log_prob_stack,
            agent_branch_steps,
        )

        selected_noise_level = self._gather_steps(
            noise_level_stack,
            agent_branch_steps,
        )

        selected_features = self._gather_steps(
            feature_stack,
            agent_branch_steps,
        )

        tokenized_agent["sde_z"] = (
            selected_current,
            selected_next,
            selected_log_prob.detach(),
        )

        tokenized_agent["sde_t"] = (
            selected_time,
            selected_next_time,
        )

        tokenized_agent["gen_z"]=latent

        tokenized_agent[ "sde_noise_level"] = selected_noise_level.detach()

        tokenized_agent["noise_feat"] = selected_features

        return latent

    def sde_step_with_logprob(
            self,
            time: Tensor,
            next_time: Tensor,
            model_output: Tensor,
            sample: Tensor,
            noise_level=0.7,
            prev_sample: Optional[Tensor] = None,
    ):
        """SDE transition directly in raw data space."""

        eps = 1e-5

        scale = self.model.normal_scale.to(
            device=sample.device,
            dtype=sample.dtype,
        ).clamp_min(eps)

        mean = self.model.normal_mean.to(
            device=sample.device,
            dtype=sample.dtype,
        )

        dt = next_time - time

        time_for_ratio = torch.where(
            time >= 1.0,
            torch.full_like(time, 0.9),
            time,
        )

        diffusion = (
                torch.sqrt(
                    time.clamp_min(0.0)
                    / (1.0 - time_for_ratio).clamp_min(eps)
                )
                * noise_level#noise_level
        )
        #diffusion=0.05/torch.sqrt(-dt)

        safe_time = time.clamp_min(eps)

        # Normalized-space transition coefficients.
        A = (
                1.0
                + diffusion.square()
                / (2.0 * safe_time)
                * dt
        )

        B = (
                    1.0
                    + diffusion.square()
                    * (1.0 - time)
                    / (2.0 * safe_time)
            ) * dt

        # Equivalent mean directly in raw space.
        next_mean = (
                mean
                + A * (sample - mean)
                + B * model_output
        )

        # Raw-space transition std.
        transition_std = (
                diffusion
                * torch.sqrt((-dt).clamp_min(eps))
                * scale
        ).clamp_min(eps)

        # Sample transition.
        if prev_sample is None:
            next_sample = (
                    next_mean
                    + transition_std
                    * torch.randn_like(sample)
            )
        else:
            next_sample = prev_sample

        # Raw-space Gaussian log probability.
        residual = (
                next_sample.detach()
                - next_mean
        )

        log_prob_element = (
                -0.5
                * (
                        residual.square()
                        / transition_std.square()
                        + 2.0 * torch.log(transition_std)
                        + math.log(2.0 * math.pi)
                )
        )

        log_prob = log_prob_element.sum(
            dim=tuple(range(1, log_prob_element.ndim))
        )

        return (
            next_sample,
            log_prob,
            next_mean,
            transition_std,
        )

    def repeat_input_copy(self, tokenized_agent, n_step):
        out = copy.copy(tokenized_agent)

        num_graphs = tokenized_agent["num_graphs"]
        batch = tokenized_agent["batch"]

        out["repeat_batch"] = batch.unsqueeze(1).repeat(1, n_step)

        repeated_batch = torch.stack(
            [batch + num_graphs * k for k in range(n_step)],
            dim=1,
        ).transpose(0, 1).flatten(0, 1)

        out["batch"] = repeated_batch
        out["agent_type_embed"] = tokenized_agent["agent_type_embed"][None].repeat(
            n_step, 1,1
        ).flatten(0, 1)
        out["num_graphs"] = num_graphs * n_step

        out["ego_feat"] = tokenized_agent["ego_feat"][None].repeat(
            n_step, 1, 1
        ).flatten(0, 1)

        return out