"""Initial-state flow wrapper for SMART."""

from __future__ import annotations

from argparse import ArgumentParser
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_sum

from src.smart.utils import transform_to_local

from .diffusion_utils import multi_circle_collision_loss_mem_efficient
from .scale_flow import ScaleFlow


class InitDiffusion(nn.Module):
    """Prepare ego/map context and delegate learning to ``ScaleFlow``."""

    NUM_TYPES = 3

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_freq_bands: int,
        token_processor,
        gail: bool,
        model_args: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if token_processor is None:
            raise ValueError("token_processor is required.")

        self.token_processor = token_processor

        # Compatibility flags used by SMART/SMART_GAIL.
        self.learn_autoencoder = False
        self.latent_diffusion = False
        self.ldm = False
        self.use_gan = False

        args = self._make_args( )
        self.G1 = ScaleFlow(args, token_processor, gail)

        self.use_rl = bool(args.use_rl)
        self.sampling_steps = int(args.sampling_steps)
        self.branch_steps = getattr(args, "branch_steps", None)

    @staticmethod
    def _make_args(
    ) -> SimpleNamespace:
        """Create deterministic ScaleFlow settings without parsing process CLI."""
        values = {
            "input_dim": 8,
            "hidden_dim": 256,
            "num_freq_bands": 64,
            "num_heads": 8,
            "head_dim": 16,
            "dropout": 0.0,
            "num_denoiser_layers": 3,
            "num_branch_steps": 1,
            "branch_steps": [0,1,2,3,4,5,6,7,8],#5,6,71,3,5,7,4,4,1,2,3,4,5,10,11,12,13,
            "sampling_steps": 10,
            "use_rl": False,
        }
        return SimpleNamespace(**values)

    # ------------------------------------------------------------------
    # Ego context
    # ------------------------------------------------------------------
    @staticmethod
    def _require(data, key: str):
        if key not in data:
            raise KeyError(f"tokenized_agent is missing {key!r}.")
        return data[key]

    def _scene_ego_pose(
        self,
        agent,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        batch = self._require(agent, "batch")
        ego_mask = self._require(agent, "ego_mask")
        pos = self._require(agent, "initial_pos")
        heading = self._require(agent, "initial_heading")
        num_graphs = int(self._require(agent, "num_graphs"))

        scene_pos = pos[ego_mask]
        scene_heading = heading[ego_mask]

        # Store scene-level values under the batch_ego_* names.
        agent["batch_ego_pos"] = scene_pos[batch]
        agent["batch_ego_heading"] = scene_heading[batch]
        return scene_pos, scene_heading, batch, num_graphs

    def _prepare_ego_context(
        self,
        agent,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        scene_pos, scene_heading, batch, num_graphs = (
            self._scene_ego_pose(agent)
        )
        if "ego_feat" not in agent:
            local_pos, local_heading = transform_to_local(
                self._require(agent, "ego_pos2"),
                self._require(agent, "ego_heading2"),
                scene_pos,
                scene_heading,
            )
            local_trajectory = torch.cat(
                [local_pos, local_heading[..., None]],
                dim=-1,
            ).flatten(1)

            agent_type = self._require(agent, "type").long()

            type_id = batch * self.NUM_TYPES + agent_type
            type_count = torch.bincount(
                type_id,
                minlength=num_graphs * self.NUM_TYPES,
            ).reshape(num_graphs, self.NUM_TYPES)
            type_count = type_count.to(local_trajectory)

            feature = torch.cat([local_trajectory, type_count], dim=-1)
            agent["ego_feat"] = feature

        return scene_pos, scene_heading, batch, num_graphs

    # ------------------------------------------------------------------
    # Initial map context
    # ------------------------------------------------------------------
    def _initial_map_feature(
        self,
        agent,
        scene_pos: Tensor,
        scene_heading: Tensor,
        num_graphs: int,
    ):
        if "initial_map_feature" in agent:
            feature=agent["initial_map_feature"]["pt_token"]
            if feature.shape[-1]!=self.G1.model.hidden_dim:
                agent["initial_map_feature"]["pt_token"] = self.G1.model.lane_embed(feature)

            return agent["initial_map_feature"]

        map_feature = self._require(agent, "map_feature")
        batch = map_feature["batch"]
        position = map_feature["position"]
        orientation = map_feature["orientation"]
        feature = map_feature["pt_token"]

        if batch.numel():
            distance = torch.linalg.vector_norm(
                position[..., :2] - scene_pos[batch],
                dim=-1,
            )
            keep = distance < float(self.token_processor.init_map_range)
            batch = batch[keep]
            position = position[keep]
            orientation = orientation[keep]
            feature = feature[keep]

            # Always transform non-empty map data. The old code skipped every
            # scene when the final scene happened to contain no map points.
            position, orientation = transform_to_local(
                position,
                orientation,
                scene_pos[batch],
                scene_heading[batch],
            )

        result = {
            "pt_token": self.G1.model.lane_embed(feature),
            "position": position,
            "orientation": orientation,
            "batch": batch,
        }
        agent["initial_map_feature"] = result
        return result


    def _collision_advantage(
        self,
        agent,
        map_feature,
        batch: Tensor,
    ) -> None:
        with torch.no_grad():
            sample = self.G1.sample(
                agent,
                map_feature,
                self.sampling_steps,
                self.branch_steps
            )
            collision, dst, src = (
                multi_circle_collision_loss_mem_efficient(
                    sample,
                    None,
                    batch,
                    None,
                )
            )
            penalty = scatter_sum(
                collision,
                dst,
                dim=0,
                dim_size=len(sample),
            )
            penalty += scatter_sum(
                collision,
                src,
                dim=0,
                dim_size=len(sample),
            )
            advantage = (penalty <= 0).to(sample.dtype)

        agent["noncol_rate"] = advantage
        agent["advantages"] = advantage

    @staticmethod
    def _mean(value: Tensor, name: str) -> Tensor:
        if not torch.is_tensor(value) or value.numel() == 0:
            raise ValueError(f"{name} must be a non-empty tensor.")
        value = value.mean()
        if not torch.isfinite(value):
            raise FloatingPointError(f"{name} is not finite.")
        return value

    def _train(
        self,
        agent,
        map_feature,
        batch: Tensor,
    ):
        diff_input, _ = self.G1.model.get_input(agent)

        if self.use_rl:
            self._collision_advantage(agent, map_feature, batch)

        loss, _, _, _ = self.G1.get_loss(
            diff_input,
            agent,
            map_feature,
        )
        if len(loss) != 6:
            raise ValueError("ScaleFlow must return six loss components.")

        names = (
            "match_loss",
            "collision_loss",
            "pos_loss",
            "heading_loss",
            "shape_loss",
            "velocity_loss",
        )
        return tuple(
            self._mean(value, name)
            for value, name in zip(loss, names)
        )

    def _infer(
        self,
        agent,
        map_feature,
    ):
        sample = self.G1.sample(
            agent,
            map_feature,
            self.sampling_steps,
            self.branch_steps
        )
        pos, heading, shape, velocity, token_index = (
            self.G1.model.get_output(sample, agent)
        )
        return pos, heading, token_index, shape, velocity

    def forward(self, tokenized_agent):
        scene_pos, scene_heading, batch, num_graphs = (
            self._prepare_ego_context(tokenized_agent)
        )
        map_feature = self._initial_map_feature(
            tokenized_agent,
            scene_pos,
            scene_heading,
            num_graphs,
        )

        if self.training:
            return self._train(tokenized_agent, map_feature, batch)
        return self._infer(tokenized_agent, map_feature)
