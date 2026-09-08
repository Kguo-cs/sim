# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_sum

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.modules.edge_encoder import EdgeEncoder
from src.smart.utils import weight_init


def _time_major(x: Tensor) -> Tensor:
    """Convert [agent, time, ...] to [time * agent, ...]."""
    return x.transpose(0, 1).flatten(0, 1)


def _last(x: Tensor, count: int) -> Tensor:
    return x[:0] if count == 0 else x[-count:]


def aggregate_interaction_reward(
    logits: Tensor,
    distances: Tensor,
    destination: Tensor,
    num_nodes: int,
    distance_decay: float,
    reward_weight: float,
) -> tuple[Tensor, Tensor]:
    """Distance-weighted sum of incoming interaction logits."""
    if distance_decay <= 0:
        raise ValueError("distance_decay must be positive.")

    edge_weight = reward_weight * torch.exp(-distances / distance_decay)
    reward = scatter_sum(
        edge_weight * logits,
        destination,
        dim=0,
        dim_size=num_nodes,
    )
    return reward, edge_weight


class InterativeDecoder(nn.Module):
    """Spatial-temporal decoder.

    Node spaces:
      dense: [T * A, ...]
      valid: dense nodes compressed by validity
      later: valid nodes at or after dis_start_step
    """

    def __init__(
        self,
        hidden_dim: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        pt2a_neighbor: int,
        a2a_neighbor: int,
        token_processor,
        dis_weight: float = 0.0,
        dist_decay: float = 0.0,
        reward_weight: float = 0.0,
        reward_decay: float = 0.0,
        discriminator: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.shift = int(token_processor.shift)
        self.token_processor = token_processor
        self.discriminator = discriminator

        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.pt2a_neighbor = pt2a_neighbor
        self.a2a_neighbor = a2a_neighbor
        self.dis_weight = float(dis_weight)
        self.dis_decay = float(dist_decay)

        # Compatibility fields used by existing configs.
        self.reward_weight = reward_weight
        self.reward_decay = reward_decay
        self.use_decompose = True
        self.use_full_feature = False
        self.use_airl = False
        self.pred_map_logit = False

        self.agent_hist = (
            1 if time_span is None
            else max(1, int(time_span) // self.shift)
        )

        if token_processor.pred_init:
            self.gail_start_step = 0
            self.dis_start_step = 0 if token_processor.learn_init else 1
        else:
            self.gail_start_step = 1
            self.dis_start_step = 2

        self.edge_encoder = EdgeEncoder(
            hidden_dim,
            num_freq_bands,
            hist_drop_prob=hist_drop_prob,
            time_span=time_span,
            shift=self.shift,
            discriminator=discriminator,
            use_bird=token_processor.use_bird,
            use_pl2a=True,
            use_a2a=True,
            use_t2t=True,
            differentiable_edge=(
                token_processor.use_gradient_penalty or not discriminator
            ),
        )

        # The original created one discriminator temporal layer but iterated
        # num_layers times. This caused IndexError when num_layers > 1.
        self.decode_layers = 1 if discriminator else num_layers

        self.t_attn_layers = nn.ModuleList([
            AttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=hist_drop_prob,
                bipartite=False,
                has_pos_emb=True,
            )
            for _ in range(self.decode_layers)
        ])

        self.pt2a_attn_layers = nn.ModuleList()
        if not token_processor.use_bird:
            self.pt2a_attn_layers.extend([
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(self.decode_layers)
            ])

        self.a2a_attn_layers = nn.ModuleList()
        if not discriminator:
            self.a2a_attn_layers.extend([
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(self.decode_layers)
            ])

        if discriminator:
            self.interact_head = MLPLayer(
                input_dim=hidden_dim * 3,
                hidden_dim=hidden_dim,
                output_dim=n_token_agent,
            )

        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=n_token_agent,
        )

        # if discriminator:
        #     self.token_predict_head1 = nn.Linear(hidden_dim, 2048)
        #     self.a2a_attn_layers1= AttentionLayer(
        #             hidden_dim=hidden_dim,
        #             num_heads=num_heads,
        #             head_dim=head_dim,
        #             dropout=dropout,
        #             bipartite=False,
        #             has_pos_emb=True,
        #         )

        self.feat_a_cache: list[Optional[Tensor]] = [
            None for _ in range(self.decode_layers)
        ]
        self.apply(weight_init)

    @staticmethod
    def _after_step(mask: Tensor, start: int, n_current: int) -> Tensor:
        result = mask.clone()
        step = torch.arange(mask.shape[1], device=mask.device) + n_current
        result[:, step < start] = False
        return result

    @staticmethod
    def _repeat_agent_mask(
        agent_mask: Optional[Tensor],
        num_steps: int,
    ) -> Optional[Tensor]:
        if agent_mask is None:
            return None
        return agent_mask[None].expand(num_steps, -1).reshape(-1)

    @staticmethod
    def _select_features(
        features: Tensor,
        selection: Optional[Tensor],
        expected: int,
    ) -> Tensor:
        """Handle full-valid or already-selected feature layouts."""
        if len(features) == expected:
            return features
        if selection is not None and len(selection) == len(features):
            selected = features[selection]
            if len(selected) == expected:
                return selected
        raise ValueError(
            f"Feature layout mismatch: got {len(features)}, expected {expected}."
        )

    def _update_cache(
        self,
        pos: Tensor,
        head: Tensor,
        head_vector: Tensor,
        valid: Tensor,
        n_current: int,
    ) -> Tensor:
        if n_current == 0:
            self.pos_cache = pos
            self.head_cache = head
            self.head_vector_cache = head_vector
            self.mask_cache = valid
            return valid.clone()

        self.pos_cache = torch.cat([self.pos_cache, pos], 1)[
            :, -self.agent_hist:
        ]
        self.head_cache = torch.cat([self.head_cache, head], 1)[
            :, -self.agent_hist:
        ]
        self.head_vector_cache = torch.cat(
            [self.head_vector_cache, head_vector], 1
        )[:, -self.agent_hist:]
        self.mask_cache = torch.cat(
            [self.mask_cache, valid], 1
        )[:, -self.agent_hist:]

        output_mask = self.mask_cache.clone()
        output_mask[:, :-1] = False
        return output_mask

    def _temporal(
        self,
        layer: int,
        features: Tensor,
        local_valid: Tensor,
        output_valid: Tensor,
        cache_agent_mask: Optional[Tensor],
        n_current: int,
        r_t: Tensor,
        edge_index_t: Tensor,
    ) -> Tensor:
        output_count = int(output_valid.sum().item())

        if not self.edge_encoder.rollout_traj and not self.discriminator:
            dense = features.new_zeros(
                local_valid.shape[1],
                local_valid.shape[0],
                self.hidden_dim,
            )
            dense[local_valid.T] = features

            if n_current == 0:
                self.feat_a_cache[layer] = dense
            else:
                previous = self.feat_a_cache[layer]

                self.feat_a_cache[layer] = torch.cat(
                    [previous, dense], 0
                )[-self.agent_hist:]

                cache_valid = (
                    self.mask_cache
                    if cache_agent_mask is None
                    else self.mask_cache[cache_agent_mask]
                )
                features = self.feat_a_cache[layer][cache_valid.T]

        features = self.t_attn_layers[layer](
            features, r_t, edge_index_t
        )
        return _last(features, output_count)

    @staticmethod
    def _select_outputs(
        features: Tensor,
        output_valid: Tensor,
        start_step: int,
        n_current: int,
        pred_mask: Optional[Tensor] = None,
    ) -> Tensor:
        eligible = InterativeDecoder._after_step(
            output_valid, start_step, n_current
        )
        if pred_mask is not None:
            eligible &= pred_mask[:, None]

        # Compress the dense selection with the same validity mask as features.
        selection = _time_major(eligible)[_time_major(output_valid)]
        return features[selection]

    def predict_agent(
        self,
        feat_a: Tensor,
        feat_map: Optional[Tensor],
        r_t: Tensor,
        edge_index_t: Tensor,
        r_pl2a: Optional[Tensor],
        edge_index_pl2a: Optional[Tensor],
        r_a2a: Tensor,
        edge_index_a2a: Tensor,
        agent_train_mask: Optional[Tensor],
        dist: Tensor,
        train_repeat_mask: Optional[Tensor],
        mask_a: Tensor,
        n_current: int,
        inference_mask: Tensor,
        pred_mask: Optional[Tensor],
        feat_a_later_mask: Optional[Tensor] = None,
    ):

        expected_selected = int(mask_a.sum().item())
        interaction_logits = interaction_dst = None
        num_interaction_nodes = 0

        if self.discriminator:
            if feat_a_later_mask is None:
                raise ValueError("feat_a_later_mask is required.")

            later_features = feat_a[feat_a_later_mask]
            num_interaction_nodes = len(later_features)
            src, interaction_dst = edge_index_a2a

            destination_features = later_features

            interaction_logits = self.interact_head(torch.cat([
                later_features[src],
                r_a2a,
                destination_features[interaction_dst],
            ], dim=-1))

            # Interaction sees all agents; scene prediction uses train agents.
            feat_a = self._select_features(
                feat_a, train_repeat_mask, expected_selected
            )

        for layer in range(self.decode_layers):
            if not self.discriminator:
                feat_a = self.a2a_attn_layers[layer](
                    feat_a, r_a2a, edge_index_a2a
                )

            if not self.token_processor.use_bird:
                feat_a = self.pt2a_attn_layers[layer](
                    (feat_map, feat_a), r_pl2a, edge_index_pl2a
                )

            if agent_train_mask is not None:
                feat_a = self._select_features(
                    feat_a, train_repeat_mask, expected_selected
                )

            feat_a = self._temporal(
                layer=layer,
                features=feat_a,
                local_valid=mask_a,
                output_valid=inference_mask,
                cache_agent_mask=agent_train_mask,
                n_current=n_current,
                r_t=r_t,
                edge_index_t=edge_index_t,
            )

        # if self.discriminator:
        #     feat_a1 = self.a2a_attn_layers1(
        #         feat_a, r_a2a, edge_index_a2a
        #     )
        #
        #     dis_action_pred=self.token_predict_head1(feat_a1)
        #
        #     dis_action_pred = self._select_outputs(
        #         dis_action_pred,
        #         inference_mask,
        #         self.dis_start_step,
        #         n_current,
        #     )

        if self.discriminator:
            feat_a = self._select_outputs(
                feat_a,
                inference_mask,
                self.dis_start_step,
                n_current,
            )
        elif self.edge_encoder.rollout_traj:
            feat_a = self._select_outputs(
                feat_a,
                inference_mask,
                self.gail_start_step,
                n_current,
                pred_mask,
            )

        token_logits = self.token_predict_head(feat_a)

        if not self.discriminator:
            return token_logits, feat_a, None, None


        scene_logit = token_logits[:, 0]
        scene_reward = scene_logit.detach()

        interaction_reward_all, edge_weight = aggregate_interaction_reward(
            logits=interaction_logits[:, 0].detach(),
            distances=dist,
            destination=interaction_dst,
            num_nodes=num_interaction_nodes,
            distance_decay=self.dis_decay,
            reward_weight=self.dis_weight,
        )

        interaction_reward = interaction_reward_all
        if train_repeat_mask is not None:
            interaction_reward = interaction_reward_all[
                train_repeat_mask[feat_a_later_mask]
            ]
            
        total_reward = scene_reward + interaction_reward
        logits = (scene_logit, interaction_logits[:, 0], None)
        rewards = (
            total_reward,
            None,
            scene_reward,
            interaction_reward_all,
        )
        return logits, feat_a, rewards, edge_weight

    def forward(
        self,
        all_features: Sequence[Tensor],
        feat_a: Tensor,
        map_feature: dict[str, Tensor],
        agent_train_mask: Optional[Tensor],
        n_current: int,
        tokenized_agent: dict,
    ):
        (
            pos_a,
            head_a,
            head_vector_a,
            mask_a,
            batch_repeat,
            _,
        ) = all_features

        num_agents, num_steps = mask_a.shape
        pred_mask = tokenized_agent.get("pred_mask")

        inference_full = self._update_cache(
            pos_a, head_a, head_vector_a, mask_a, n_current
        )
        selected_inference = (
            inference_full
            if agent_train_mask is None
            else inference_full[agent_train_mask]
        )

        edge_index_t, r_t = self.edge_encoder.build_temporal_edge(
            pos_a=self.pos_cache,
            head_a=self.head_cache,
            head_vector_a=self.head_vector_cache,
            mask=self.mask_cache,
            inference_mask=selected_inference,
            agent_train_mask=agent_train_mask,
        )

        if self.token_processor.use_bird:
            feat_map = edge_index_pl2a = r_pl2a = None
        else:
            feat_map = map_feature["pt_token"]
            if self.discriminator:
                feat_map = feat_map.detach()

            edge_index_pl2a, r_pl2a = (
                self.edge_encoder.build_map2agent_edge(
                    pos_pl=map_feature["position"],
                    orient_pl=map_feature["orientation"],
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=mask_a,
                    batch_s=batch_repeat,
                    batch_pl=map_feature["batch"],
                    pl2a_radius=self.pl2a_radius,
                    max_num_neighbors=self.pt2a_neighbor,
                    agent_train_mask=agent_train_mask,
                    layer_num=self.decode_layers,
                )
            )

        (
            pos_s,
            head_s,
            head_vector_s,
            dense_valid,
            _,
            batch_s,
        ) = [_time_major(feature) for feature in all_features]

        dense_train = self._repeat_agent_mask(
            agent_train_mask, num_steps
        )
        train_within_valid = (
            None if dense_train is None else dense_train[dense_valid]
        )

        if self.discriminator:
            later_a = self._after_step(
                mask_a, self.dis_start_step, n_current
            )
            dense_later = _time_major(later_a)
            feat_a_later_mask = dense_later[dense_valid]
            train_for_edges = (
                None if dense_train is None else dense_train[dense_later]
            )
        else:
            dense_later = dense_valid
            feat_a_later_mask = None
            train_for_edges = train_within_valid

        dis_edge_mask = None
        if self.discriminator and "dis_mask" in tokenized_agent:
            dis_mask = tokenized_agent["dis_mask"]
            step = torch.arange(
                num_steps, device=mask_a.device
            ) + n_current
            later_ta = mask_a[:, step >= self.dis_start_step].T

            if agent_train_mask is None:
                dense_dis = dis_mask.reshape_as(later_ta)
            else:
                selected_shape = later_ta[:, agent_train_mask].shape
                dense_dis = torch.zeros_like(later_ta)
                dense_dis[:, agent_train_mask] = dis_mask.reshape(
                    selected_shape
                )

            dis_edge_mask = dense_dis[later_ta]

        (
            edge_index_a2a,
            r_a2a,
            dist,
            relative_pos,
            _,
            _,
            _,
        ) = self.edge_encoder.build_interaction_edge(
            pos_s=pos_s,
            head_s=head_s,
            head_vector_s=head_vector_s,
            batch_s=batch_s,
            mask=dense_later,
            max_radius=self.a2a_radius,
            max_num_neighbors=self.a2a_neighbor,
            agent_train_mask=train_for_edges,
            layer_num=self.decode_layers,
            dis_edge_mask=dis_edge_mask,
        )

        selected_mask = (
            mask_a if agent_train_mask is None
            else mask_a[agent_train_mask]
        )
        selected_pred = (
            pred_mask
            if pred_mask is None or agent_train_mask is None
            else pred_mask[agent_train_mask]
        )

        logits, features, rewards, weight = self.predict_agent(
            feat_a=feat_a,
            feat_map=feat_map,
            r_t=r_t,
            edge_index_t=edge_index_t,
            r_pl2a=r_pl2a,
            edge_index_pl2a=edge_index_pl2a,
            r_a2a=r_a2a,
            edge_index_a2a=edge_index_a2a,
            agent_train_mask=agent_train_mask,
            dist=dist,
            train_repeat_mask=train_within_valid,
            mask_a=selected_mask,
            n_current=n_current,
            inference_mask=selected_inference,
            pred_mask=selected_pred,
            feat_a_later_mask=feat_a_later_mask,
        )

        return (
            logits,
            features,
            rewards,
            weight,
            (edge_index_a2a, r_a2a, relative_pos),
        )


InteractiveDecoder = InterativeDecoder