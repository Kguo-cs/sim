import math
import copy
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn

from src.smart.layers import MLPLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.modules.edge_encoder import EdgeEncoder
from src.smart.utils import (
    transform_to_global,
    transform_to_local,
    wrap_angle,
    rotate_to_global,
    rotate_to_local,
    weight_init,
)
from .noise_schedule import LearnableGroupedPowerSchedule
import torch.nn.functional as F

class InitDenoiser(nn.Module):
    """Cleaned initial-state denoiser.

    Kept public API:
        - normalize
        - denormalize
        - get_input
        - forward
        - get_output

    Removed unsupported/dead branches from the previous implementation:
        - DiT path
        - use_all_pos path
        - bin-normalization path
        - non-RoFormer path
        - previous-heading/speed/condition branches
        - return/cfg conditioning branches
        - unused padding/SkipMLP/ExploreNoiseNet code

    The embedding path can be selected by ``init_embedding_mode``:
        - "new": AgentTokenEncoder-style Fourier/categorical fusion.
        - "original": old denoiser MLP-addition embedding.

    MeanFlow/iMF support:
        When ``mean_flow=True``, the model output is interpreted as the
        interval-average velocity u(z_t,t,r). The current time remains ``beta``;
        the interval length h=r-t is read from ``tokenized_agent["meanflow_h"]``
        and embedded additively.
    """

    def __init__(
        self,
        token_processor,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        x_pred: bool = True,
        init_embedding_mode: str = "original",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.x_pred = x_pred
        self.token_processor = token_processor
        self.init_embedding_mode = init_embedding_mode

        self.label_drop_prob = 0.0
        self.map_drop_prob=0.0

        self.num_classes = 3
        self.shape_dim = 2
        self.m_delta_dim = input_dim
        self.output_dim =output_dim

        self.register_buffer("normal_mean", torch.zeros(1, self.m_delta_dim))
        self.register_buffer("normal_scale", torch.ones(1, self.m_delta_dim))

        # Different groups can still use different schedules.
        # self.schedule = LearnableGroupedPowerSchedule(
        #     group_dims=(2, 2, 2, self.m_delta_dim - 6)
        # )
        #
        if self.init_embedding_mode == "new":
            # New path: AgentTokenEncoder-style state/noise/type/shape embedding.
            self.state_embedder = DenoiserStateEmbedder(
                token_processor=token_processor,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
                m_delta_dim=self.m_delta_dim,
                num_classes=self.num_classes,
                shape_dim=self.shape_dim,
            )
        else:
            # Original path: old denoiser embedding style.
            # It projects m_delta[:, 4:] with a Linear layer and adds
            # noise embedding + type embedding directly.
            self.type_a_emb = nn.Embedding(self.num_classes + 1, hidden_dim)
            self.noise_embedding = MLPLayer(self.m_delta_dim, hidden_dim, hidden_dim)
            self.proj_in_m_delta = nn.Linear(self.m_delta_dim - 4, hidden_dim)

        # Ego-context embedding. The input is:
        #   local ego poses relative to the generated agent + per-scene type count.
        # For the current tokenization, ego pose part is 9 and type-count part is 3.
        self.ego_dim = 9
        self.ego_embed = MLPLayer(self.ego_dim + 3, hidden_dim, hidden_dim)

        self.edge_encoder = EdgeEncoder(
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
            use_a2a=True,
            use_pl2a=True,
        )

        self.lane_embed = MLPLayer(128, hidden_dim, hidden_dim)

        self.a2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=0,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.pt2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=0,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.to_out_m_delta = MLPLayer(hidden_dim, hidden_dim, self.output_dim)

        self.apply(weight_init)

    # ---------------------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------------------
    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.normal_scale.clamp_min(1e-6)
        return (input - self.normal_mean) / scale

    def denormalize(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.normal_scale.clamp_min(1e-6)
        return input * scale + self.normal_mean

    def _maybe_init_normalizer(self, diff_output: torch.Tensor) -> None:
        if not torch.all(self.normal_mean == 0):
            # if self.normal_scale[0][0]>15:#20
            #     self.normal_scale[:, :2] = self.normal_scale[:, :2] * 0.8
            # if self.normal_scale[0][2]>1.5:
            #     self.normal_scale[:, 2:6] = self.normal_scale[:, 2:6] * 0.5
            #     # self.normal_scale[:, :2] = self.normal_scale[:, :2] * 2
            return

        with torch.no_grad():
            self.normal_mean.copy_(torch.mean(diff_output, dim=0, keepdim=True))

            scale = torch.std(
                diff_output,
                dim=0,
                keepdim=True,
                unbiased=False,
            ).clamp_min(1e-6)

            # Keep the old scaling heuristic.
            scale[:, 2:6] = scale[:, 2:6] * 2.0
            scale[:, :2] = scale[:, :2] * 0.5

            self.normal_scale.copy_(scale)

    # ---------------------------------------------------------------------
    # Tokenized-agent input construction
    # ---------------------------------------------------------------------
    def get_input(self, tokenized_agent,expert_data=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build local all-agent initial state.

        Returns:
            diff_input:
                Initial state used as source distribution input.
            diff_output:
                Target state to reconstruct / generate.

        Both have shape [N_agent, 8]:
            [local_x, local_y, cos(local_heading), sin(local_heading),
             length, width, local_vx, local_vy]

        Notes:
            The previous implementation used a mask that was immediately
            set to all True in ``InitDiffusion.forward``. This cleaned version
            therefore treats every agent as part of the
            initial-state generation set and uses explicit all-agent metadata:
                ``batch`` and ``type``.
        """
        if "expert_input" in tokenized_agent.keys():
            return tokenized_agent["expert_input"], tokenized_agent["expert_input"]

        batch_ego_pos = tokenized_agent["batch_ego_pos"]
        batch_ego_heading = tokenized_agent["batch_ego_heading"]
        shape = tokenized_agent["shape"]

        agent_pos = tokenized_agent["initial_pos"]
        agent_head = tokenized_agent["initial_heading"]
        local_vel = tokenized_agent["local_vel"]

        local_pos, local_heading = transform_to_local(
            agent_pos,
            agent_head,
            batch_ego_pos,
            batch_ego_heading,
        )

        heading_vec = torch.stack(
            [local_heading.cos(), local_heading.sin()],
            dim=-1,
        )

        m_init = torch.cat(
            [
                local_pos,
                heading_vec,
                shape[:, :2],
                local_vel[:, :2],
            ],
            dim=-1,
        )

        diff_input = m_init
        diff_output = m_init

        self._maybe_init_normalizer(diff_output)

        return diff_input, diff_output


    def _format_beta(self, beta: torch.Tensor, n_agent: int) -> torch.Tensor:
        if beta.ndim == 3:
            beta = beta[:, 0]
        elif beta.ndim == 1:
            beta = beta[:, None]

        if beta.shape[0] != n_agent:
            raise ValueError(
                f"beta first dim must match agents: beta={tuple(beta.shape)}, N={n_agent}."
            )

        if beta.shape[-1] == 1:
            beta = beta.expand(-1, self.m_delta_dim)

        return beta

    def _ego_context_embedding(
        self,
        pos_s: torch.Tensor,
        theta: torch.Tensor,
        batch: torch.Tensor,
        tokenized_agent,
    ) -> torch.Tensor:

        ego_feat = tokenized_agent["ego_feat"]

        ego_pose = ego_feat[:, :-3]
        type_count = ego_feat[:, -3:][batch]

        # [num_graphs, 3, 3] -> [N_agent, 3, 3]
        # Last dim is expected to be [x, y, heading].
        ego_pose = ego_pose.reshape(-1, 3, 3)
        all_pos = ego_pose[:, :, :2][batch]
        all_head = ego_pose[:, :, 2][batch]

        local_ego_pos, local_ego_head = transform_to_local(
            all_pos,
            all_head,
            pos_s,
            theta,
        )

        local_ego_head=wrap_angle(local_ego_head)

        ego_features = torch.cat(
            [
                local_ego_pos.flatten(1, 2),
                local_ego_head,#.cos(),
             #   local_ego_head.sin(),
                type_count,
            ],
            dim=-1,
        )

        return self.ego_embed(ego_features)

    def _original_state_embedding(
        self,
        m_delta: torch.Tensor,
        beta: torch.Tensor,
        agent_type_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Original InitDenoiser embedding style.

        This follows the old implementation:
            feat_a = Linear(m_delta[:, 4:])
            feat_a = feat_a + noise_embedding(beta) + type_embedding(type)

        It does not inject shape via ``FourierEmbedding``. Shape is part of
        the projected continuous state ``m_delta[:, 4:]``.
        """
        beta = self._format_beta(1-beta, m_delta.shape[0])

        feat_a = self.proj_in_m_delta(m_delta[:, 4:])
        feat_a = feat_a + self.noise_embedding(beta)
        feat_a = feat_a + agent_type_embed
        return feat_a

    def _embed_agents(
        self,
        m_delta: torch.Tensor,
        beta: torch.Tensor,
        agent_type: torch.Tensor,
        batch: torch.Tensor,
        tokenized_agent,
        mode: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        theta = torch.atan2(m_delta[:, 3], m_delta[:, 2])
        pos_s = m_delta[:, :2]

        ego_embedding = self._ego_context_embedding(
            pos_s=pos_s,
            theta=theta,
            batch=batch,
            tokenized_agent=tokenized_agent,
        )

        if self.label_drop_prob > 0:
            if self.training and mode == 1:
                drop = torch.rand(agent_type.shape[0], device=agent_type.device) < self.label_drop_prob
                agent_type =torch.where(drop, torch.full_like(agent_type, self.num_classes), agent_type)
                #ego_embedding=torch.where(drop[:, None], torch.full_like(ego_embedding, 0), ego_embedding)
            elif mode == 0:
                agent_type = torch.full_like(agent_type, self.num_classes)

        if "agent_type_embed" not in tokenized_agent :
            agent_type_embed=self.type_a_emb(agent_type)
            tokenized_agent["agent_type_embed"]=agent_type_embed
        else:
            agent_type_embed=tokenized_agent["agent_type_embed"]

        beta = self._format_beta(beta, m_delta.shape[0])

        if self.init_embedding_mode == "new":
            feat_a = self.state_embedder(
                m_delta=m_delta,
                beta=beta,
                agent_type=agent_type,
            )
        else:
            feat_a = self._original_state_embedding(
                m_delta=m_delta,
                beta=beta,
                agent_type_embed=agent_type_embed,
            )

        feat_a = feat_a + ego_embedding

        return feat_a, pos_s, theta

    # ---------------------------------------------------------------------
    # Graph denoising
    # ---------------------------------------------------------------------
    def _apply_graph_attention(
            self,
            feat_a: torch.Tensor,
            pos_s: torch.Tensor,
            theta: torch.Tensor,
            batch: torch.Tensor,
            tokenized_agent,
            map_feature: Mapping[str, torch.Tensor],
            num_graphs: int,
            use_map_condition: bool = True,
    ) -> torch.Tensor:
        head_vector_s = torch.stack(
            [theta.cos(), theta.sin()],
            dim=-1,
        )

        edge_index_a2a, r_a2a, *_ = self.edge_encoder.build_interaction_edge(
            pos_s=pos_s,
            head_s=theta,
            head_vector_s=head_vector_s,
            batch_s=batch,
            mask=None,
            max_radius=300,
            max_num_neighbors=30,
            agent_train_mask=None,
            layer_num=self.num_layers,
        )

        if use_map_condition:
            batch_pl = map_feature["batch"]
            pos_pl = map_feature["position"]
            orient_pl = map_feature["orientation"]
            feat_map = map_feature["pt_token"]

            if batch_pl.numel() > 0 and int(batch_pl.max().item()) != num_graphs - 1:
                if "agent_valid" not in tokenized_agent:
                    batch_for_map = tokenized_agent["repeat_batch"]
                    n_step = batch_for_map.shape[1]

                    pos_for_map = pos_s.reshape(n_step, -1, 2).transpose(0, 1)
                    theta_for_map = theta.reshape(n_step, -1).transpose(0, 1)
                    mask_for_map = torch.ones_like(batch_for_map, dtype=torch.bool)
                else:
                    valid = tokenized_agent["agent_valid"]
                    n_step = valid.shape[0]

                    pos_global, theta_global = transform_to_global(
                        pos_s,
                        theta,
                        tokenized_agent["batch_ego_pos"],
                        tokenized_agent["batch_ego_heading"],
                    )

                    pos_b = torch.zeros(
                        [valid.shape[0], valid.shape[1], 2],
                        device=pos_s.device,
                        dtype=pos_s.dtype,
                    )
                    theta_b = torch.zeros(
                        [valid.shape[0], valid.shape[1]],
                        device=theta.device,
                        dtype=theta.dtype,
                    )

                    pos_b[valid] = pos_global
                    theta_b[valid] = theta_global

                    pos_for_map = pos_b.transpose(0, 1)
                    theta_for_map = theta_b.transpose(0, 1)
                    mask_for_map = valid.transpose(0, 1)
                    batch_for_map = tokenized_agent["batch_a"].unsqueeze(1).repeat(
                        1,
                        n_step,
                    )
            else:
                pos_for_map = pos_s
                theta_for_map = theta
                mask_for_map = None
                batch_for_map = batch

            head_vector_for_map = torch.stack(
                [theta_for_map.cos(), theta_for_map.sin()],
                dim=-1,
            )

            edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
                pos_pl=pos_pl,
                orient_pl=orient_pl,
                pos_a=pos_for_map,
                head_a=theta_for_map,
                head_vector_a=head_vector_for_map,
                mask=mask_for_map,
                batch_s=batch_for_map,
                batch_pl=batch_pl,
                pl2a_radius=300,
                max_num_neighbors=30,
                agent_train_mask=None,
                layer_num=self.num_layers,
            )

        for layer_i in range(self.num_layers):
            feat_a = self.a2a_attn_layers[layer_i](
                feat_a,
                r_a2a,
                edge_index_a2a,
            )

            if use_map_condition:
                feat_a = self.pt2a_attn_layers[layer_i](
                    (feat_map, feat_a),
                    r_pl2a,
                    edge_index_pl2a,
                )

        return self.to_out_m_delta(feat_a)

    def forward(
        self,
        m_delta: torch.Tensor,
        beta: torch.Tensor,
        tokenized_agent,
        map_feature: Mapping[str, torch.Tensor],
        eval_mask: torch.Tensor=None,
        mode: int = 1,
        use_map_condition: Optional[bool] = True,
    ) -> torch.Tensor:
        m_delta = m_delta.reshape(m_delta.shape[0], -1)

        batch = tokenized_agent["batch"]
        agent_type = tokenized_agent["type"]
        num_graphs = tokenized_agent["num_graphs"]

        if eval_mask is not None:
            m_delta = m_delta[eval_mask]
            beta = beta[eval_mask]
            batch = batch[eval_mask]
            agent_type = agent_type[eval_mask]

        feat_a, pos_s, theta = self._embed_agents(
            m_delta=m_delta,
            beta=beta,
            agent_type=agent_type,
            batch=batch,
            tokenized_agent=tokenized_agent,
            mode=mode,
        )

        if use_map_condition:
            if self.training and self.map_drop_prob > 0:
                use_map_condition = (
                        torch.rand((), device=m_delta.device) >= self.map_drop_prob
                )
            else:
                use_map_condition = True

        res = self._apply_graph_attention(
            feat_a=feat_a,
            pos_s=pos_s,
            theta=theta,
            batch=batch,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            num_graphs=num_graphs,
            use_map_condition=use_map_condition
        )

        if self.x_pred :
            res_theta = torch.atan2(res[:, 3], res[:, 2])

            local_pos, local_theta = transform_to_global(
                res[:, :2],
                res_theta,
                pos_s,
                theta,
            )

            # local_v = rotate_to_local(
            #     res[:,6:],
            #     res_theta,
            # )

            res = torch.cat(
                [
                    local_pos,
                    torch.cos(local_theta)[:, None],
                    torch.sin(local_theta)[:, None],
                    res[:, 4:],
                   # local_v
                ],
                dim=-1,
            )

        ego_mask = tokenized_agent.get("ego_mask", None)
        if not self.training and len(beta)==len(ego_mask): #and torch.all(beta[~ego_mask] == 0):
            tokenized_agent["noise_feat_cur"] = feat_a

        return res

    def get_output(self, pred_init: torch.Tensor, tokenized_agent):
        """Convert generated local all-agent initial state back to global fields."""
        batch_ego_pos = tokenized_agent["batch_ego_pos"]
        batch_ego_heading = tokenized_agent["batch_ego_heading"]

        pred_trans = pred_init[..., :2]
        pred_head = pred_init[..., 2:4]
        pred_shape = pred_init[..., 4:6]
        pred_vel = pred_init[..., 6:8]

        pred_heading = torch.atan2(pred_head[..., 1], pred_head[..., 0])

        global_pos, global_heading = transform_to_global(
            pred_trans,
            pred_heading,
            batch_ego_pos,
            batch_ego_heading,
        )

        gt_initial_pos = global_pos[:, None]
        gt_initial_heading = global_heading[:, None]

        #center_token_traj = tokenized_agent["token_traj"].mean(-2)
        # gt_initial_idx = torch.linalg.norm(
        #     center_token_traj - pred_vel[:, None] * 0.5,
        #     dim=-1,
        # ).argmin(-1)

        # token_traj: [N, K, 4, 2], endpoint contour.
        token_end_contour = tokenized_agent["token_traj"]

        token_dt = self.token_processor.shift * 0.1

        token_vel_current = self.token_processor.token_velocity_in_current_frame(
            token_end_contour,
            token_dt,
        )

        gt_initial_idx = torch.linalg.vector_norm(
            token_vel_current - pred_vel[:, None],
            dim=-1,
        ).argmin(dim=-1)

        #local_vel = center_token_traj[torch.arange(len(gt_initial_idx), device=gt_initial_idx.device), gt_initial_idx]

        global_vel= rotate_to_global(pred_vel,global_heading)

        return (
            gt_initial_pos,
            gt_initial_heading,
            pred_shape,
            global_vel,
            gt_initial_idx[:, None],
        )