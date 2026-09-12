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

import math
import os
import pickle
from typing import Dict, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.utils import (
    cal_polygon_contour,
    infer_prev_pose,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    rotate_to_local
)


class TokenProcessor(torch.nn.Module):
    AGENT_NAMES = ("veh", "ped", "cyc")
    AGENT_SHAPES = ((2.0, 4.8), (1.0, 1.0), (1.0, 2.0))

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling: DictConfig,
        agent_token_sampling: DictConfig,
        pred_init: bool = False,
        learn_init: bool = False,
        learn_autoencoder: bool = False,
    ) -> None:
        super().__init__()
        self.map_token_sampling = map_token_sampling
        self.agent_token_sampling = agent_token_sampling
        self.learn_autoencoder = learn_autoencoder

        self.shift = 5
        self.pred_init = pred_init
        self.learn_init = learn_init
        self.use_bird = False
        self.use_token = True
        self.use_goal = False
        self.init_map_range = 100
        self.use_gradient_penalty = False
        self.use_refiner=True

        module_dir = os.path.dirname(__file__)
        self.init_agent_token(os.path.join(module_dir, agent_token_file))
        self.init_map_token(os.path.join(module_dir, map_token_file))
        self.n_token_agent = self.agent_token_all_veh.shape[0]
        self.n_token_map = self.map_token_traj_src.shape[0]

    @staticmethod
    def token_velocity_in_current_frame(
            token_end_contour: torch.Tensor,
            token_dt: float,
    ) -> torch.Tensor:
        """Convert token endpoint contour to current-agent-local velocity.

        Args:
            token_end_contour:
                [..., 4, 2], expressed in the previous agent frame.
            token_dt:
                Token duration in seconds.

        Returns:
            [..., 2], velocity expressed in the token endpoint/current frame.
        """
        displacement_prev = token_end_contour.mean(dim=-2)

        heading_vector = (
                token_end_contour[..., 0, :]
                - token_end_contour[..., 3, :]
        )
        delta_heading = torch.atan2(
            heading_vector[..., 1],
            heading_vector[..., 0],
        )

        velocity_prev = displacement_prev / token_dt

        return rotate_to_local(
            velocity_prev,
            delta_heading,
        )

    @torch.no_grad()
    def forward(
        self, data: HeteroData
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
      #  if self.training:
        tokenized_map, tokenized_agent = self.process_data(data)
        # else:
        #
        #     tokenized_map = self.tokenize_map(data)
        #     tokenized_agent = self.tokenize_agent(data)
        #     if self.pred_init:
        #         self.get_init(tokenized_agent)
        # if "sampled_pos" in tokenized_agent:
        #     prev_pos, prev_heading = infer_prev_pose(
        #         tokenized_agent["sampled_pos"][:, :1],
        #         tokenized_agent["sampled_heading"][:, :1],
        #         tokenized_agent["sampled_idx"][:, :1],
        #         tokenized_agent["token_traj_all"],
        #     )
        #
        #     tokenized_agent["sampled_pos"]=torch.cat([prev_pos+1e-2*torch.randn_like(prev_pos), tokenized_agent["sampled_pos"]], 1)#

        if "type" in tokenized_agent:
            tokenized_agent["type"] = tokenized_agent["type"].long()

        if "ego_mask" not in tokenized_agent:
            tokenized_agent["ego_mask"] = self._make_ego_mask(
                tokenized_agent["batch"]
            )

        if self.use_goal and self.training:
            if not hasattr(self, "compute_goal"):
                raise NotImplementedError("compute_goal is not implemented")
            self.compute_goal(tokenized_agent)
        else:
            tokenized_agent["goal_pos"] = None
            tokenized_agent["goal_mask"] = None

        return tokenized_map, tokenized_agent

    @staticmethod
    def _make_ego_mask(batch: Tensor) -> Tensor:
        """Preserve the original convention: last agent in each batch is ego."""
        mask = torch.zeros_like(batch, dtype=torch.bool)
        if batch.numel() > 0:
            mask[-1] = True
            mask[:-1] = batch[:-1] != batch[1:]
        return mask

    @staticmethod
    def _as_time_tensor(value: Tensor, ndim: int) -> Tensor:
        if value.ndim == ndim - 1:
            return value.unsqueeze(1)
        if value.ndim != ndim:
            raise ValueError(f"Expected {ndim - 1}D or {ndim}D, got {value.shape}")
        return value

    def get_init(self, agent: Dict[str, Tensor]) -> None:
        if "ego_mask" not in agent:
            agent["ego_mask"] = self._make_ego_mask(agent["batch"])

        agent["initial_pos"] = agent["sampled_pos"][:, 0]
        agent["initial_heading"] = agent["sampled_heading"][:, 0]
        agent["initial_shape"] = agent["shape"] #[:, 0]

        ego_mask = agent["ego_mask"]
        ego_idx = agent["sampled_idx"][ego_mask]
        if ego_idx.shape[0] > 0:
            prev_pos, prev_heading = infer_prev_pose(
                agent["sampled_pos"][ego_mask, :1],
                agent["sampled_heading"][ego_mask, :1],
                ego_idx[:, :1],
                agent["token_traj_all"][ego_mask],
            )
            history_len = min(2, agent["sampled_pos"].shape[1])
            agent["ego_pos2"] = torch.cat(
                [prev_pos, agent["sampled_pos"][ego_mask, :history_len]], dim=1
            )
            agent["ego_heading2"] = torch.cat(
                [prev_heading, agent["sampled_heading"][ego_mask, :history_len]],
                dim=1,
            )

        velocity_idx = agent["sampled_idx"][:, 0].clone()
        next_step = min(1, agent["sampled_idx"].shape[1] - 1)
        if "token_mask" in agent:
            agent["token_mask"] = self._as_time_tensor(
                agent["token_mask"], 2
            )
            invalid = ~agent["token_mask"][:, 0]
            velocity_idx[invalid] = agent["sampled_idx"][invalid, next_step]

        row = torch.arange(len(velocity_idx), device=velocity_idx.device)
        selected_token = agent["token_traj_all"][row, velocity_idx]
        # agent["local_vel"] = selected[:, -1].mean(-2) / 0.5
        token_end_contour = selected_token[:, -1]

        token_dt = self.shift * 0.1

        agent["local_vel"] = self.token_velocity_in_current_frame(
            token_end_contour,
            token_dt)

    def init_map_token(self, path: str, sample_len: int = 3) -> None:
        with open(path, "rb") as file:
            trajectory = pickle.load(file)["traj_src"]

        trajectory = torch.as_tensor(trajectory, dtype=torch.float32)
        index = torch.linspace(0, trajectory.shape[1] - 1, sample_len).long()
        self.register_buffer(
            "map_token_traj_src", trajectory.flatten(1, 2), persistent=False
        )
        self.register_buffer(
            "map_token_sample_pt", trajectory[:, index].unsqueeze(0), persistent=False
        )

    def init_agent_token(self, path: str) -> None:
        with open(path, "rb") as file:
            token_data = pickle.load(file)["token_all"]

        local_trajectories = []
        token_shape = None
        for name in self.AGENT_NAMES:
            token = torch.as_tensor(token_data[name], dtype=torch.float32)
            token = token[:, 1 : self.shift + 1]
            if token.shape[1] != self.shift:
                raise ValueError(f"Invalid {name} token length: {token.shape}")
            if token_shape is not None and token.shape != token_shape:
                raise ValueError("All agent token libraries must have the same shape")
            token_shape = token.shape
            self.register_buffer(
                f"agent_token_all_{name}", token, persistent=False
            )

            center = token.mean(2)
            direction = token[:, :, 0] - token[:, :, 3]
            heading = torch.atan2(direction[..., 1], direction[..., 0])
            local_trajectories.append(
                torch.cat([center, heading.unsqueeze(-1)], dim=-1)
            )

        self.register_buffer(
            "all_token_local_traj",
            torch.stack(local_trajectories),
            persistent=False,
        )
        for name in self.AGENT_NAMES:
            token = getattr(self, f"agent_token_all_{name}")
            self.register_buffer(
                f"trajectory_token_{name}",
                token[:, -1].flatten(1, 2),
                persistent=False,
            )

    def tokenize_map1(self, data: HeteroData) -> Dict[str, Tensor]:
        pos = data["map_save1"]["traj_pos"]
        heading = data["map_save1"]["traj_theta"]
        local_pos, _ = transform_to_local(
            pos_global=pos,
            head_global=None,
            pos_now=pos[:, 0],
            head_now=heading,
        )
        distance = (
            self.map_token_sample_pt - local_pos.unsqueeze(1)
        ).square().sum((-2, -1))

        result = {
            "position": pos[:, 0].contiguous(),
            "orientation": heading,
            "token_idx": distance.argmin(-1),
            "traj_pos_local": local_pos[:, 1:],
            "type": data["pt_token1"]["type"],
        }
        for key in ("batch", "light_type"):
            if key in data["pt_token1"]:
                result[key] = data["pt_token1"][key]
        return result

    def tokenize_map(self, data: HeteroData) -> Dict[str, Tensor]:
        pos = data["map_save"]["traj_pos"]
        heading = data["map_save"]["traj_theta"]
        local_pos, _ = transform_to_local(
            pos_global=pos,
            head_global=None,
            pos_now=pos[:, 0],
            head_now=heading,
        )
        distance = (
            self.map_token_sample_pt - local_pos.unsqueeze(1)
        ).square().sum((-2, -1))

        result = {
            "position": pos[:, 0].contiguous(),
            "orientation": heading,
            "token_idx": distance.argmin(-1),
            "traj_pos_local": local_pos[:, 1:],
            "type": data["pt_token"]["type"],
        }
        for key in ("batch", "light_type"):
            if key in data["pt_token"]:
                result[key] = data["pt_token"][key]
        return result

    def tokenize_agent(
        self,
        data: HeteroData,
        tokenized_map: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        del tokenized_map
        raw = data["agent"]
        agent_type = raw["type"].long()
        agent_shape, all_tokens, final_tokens = self._get_agent_tokens(agent_type)

        valid = raw["valid_mask"].clone()
        heading = self._clean_heading(valid, raw["heading"].clone())
        pos = raw["position"][..., :2].contiguous().clone()
        vel = raw["velocity"].clone()
        valid, pos, heading, vel = self._extrapolate_to_token_boundary(
            valid, pos, heading, vel
        )

        result = {
            "num_graphs": data.num_graphs,
            "type": agent_type,
            "shape": raw["shape"].clone(),
            "token_agent_shape": agent_shape,
            "batch": raw["batch"],
            "token_traj_all": all_tokens,
            "token_traj": final_tokens,
        }
        self._attach_token_libraries(result)

        if not self.training:
            result["gt_z_raw"] = raw["position"][:, 10, 2]

        result.update(
            self._match_agent_token(
                valid,
                pos,
                heading,
                agent_shape,
                final_tokens,
                shift=self.shift,
                error_dist= 0.3,
            )
        )
        if "id" in raw:
            result["id"] = raw["id"]
        return result

    def _match_agent_token(
        self,
        valid: Tensor,
        pos: Tensor,
        heading: Tensor,
        agent_shape: Tensor,
        token_traj: Tensor,
        shift: int = 5,
        error_dist: float = 0.3,
    ) -> Dict[str, Tensor]:
        num_agents, num_steps = valid.shape
        if not self.training:
            num_steps = min(num_steps, 11)

        steps = range(shift, num_steps, shift)

        row = torch.arange(num_agents, device=valid.device)
        prev_pos = pos[:, 0].clone()
        prev_heading = heading[:, 0].clone()
        output = {key: [] for key in (
            "valid_mask", "sampled_idx", "sampled_pos", "sampled_heading", "token_mask"
        )}

        for step in steps:
            segment_valid = valid[:, step - shift] & valid[:, step]
            target = cal_polygon_contour(
                pos[:, step], heading[:, step], agent_shape
            ).unsqueeze(1)
            world_tokens = transform_to_global(
                pos_local=token_traj.flatten(1, 2),
                head_local=None,
                pos_now=prev_pos,
                head_now=prev_heading,
            )[0].reshape_as(token_traj)

            distance = torch.linalg.vector_norm(world_tokens - target, dim=-1).sum(-1)
            min_distance, token_idx = distance.min(-1)
            token_valid = segment_valid & (min_distance < error_dist)

            selected = world_tokens[row, token_idx]
            selected_pos = selected.mean(1)
            direction = selected[:, 0] - selected[:, 3]
            selected_heading = torch.atan2(direction[:, 1], direction[:, 0])

            # Reset unmatched agents to GT so errors do not propagate.
            prev_pos = pos[:, step].clone()
            prev_heading = heading[:, step].clone()
            prev_pos[token_valid] = selected_pos[token_valid]
            prev_heading[token_valid] = selected_heading[token_valid]

            frame_valid = valid[:, step]
            output["valid_mask"].append(frame_valid)
            output["token_mask"].append(segment_valid)
            output["sampled_idx"].append(token_idx)
            output["sampled_pos"].append(
                prev_pos.masked_fill(~frame_valid[:, None], 0)
            )
            output["sampled_heading"].append(
                prev_heading.masked_fill(~frame_valid, 0)
            )

        return {key: torch.stack(value, 1) for key, value in output.items()}

    def _match_agent_token_reverse(
        self,
        valid: Tensor,
        pos: Tensor,
        heading: Tensor,
        agent_shape: Tensor,
        token_traj: Tensor,
        shift: int = 5,
        error_dist: float = 0.3,
    ) -> Dict[str, Tensor]:
        num_agents, num_steps = valid.shape
        if not self.training:
            num_steps = min(num_steps, 11)

        last_step = num_steps - 1
        row = torch.arange(num_agents, device=valid.device)
        current_pos = pos[:, last_step].clone()
        current_heading = heading[:, last_step].clone()
        sampled_pos = [current_pos]
        sampled_heading = [current_heading]
        sampled_idx, valid_mask, token_mask = [], [], []

        for step in range(last_step - shift, -1, -shift):
            next_step = step + shift
            segment_valid = valid[:, step] & valid[:, next_step]
            current_contour = cal_polygon_contour(
                current_pos, current_heading, agent_shape
            ).unsqueeze(1)
            world_tokens = transform_to_global(
                pos_local=token_traj.flatten(1, 2),
                head_local=None,
                pos_now=pos[:, step],
                head_now=heading[:, step],
            )[0].reshape_as(token_traj)

            distance = torch.linalg.vector_norm(
                world_tokens - current_contour, dim=-1
            ).sum(-1)
            min_distance, token_idx = distance.min(-1)
            matched = segment_valid & (min_distance < error_dist)

            local_token = token_traj[row, token_idx]
            local_pos = local_token.mean(1)
            direction = local_token[:, 0] - local_token[:, 3]
            local_heading = torch.atan2(direction[:, 1], direction[:, 0])
            recovered_heading = wrap_angle(current_heading - local_heading)

            cos_h, sin_h = recovered_heading.cos(), recovered_heading.sin()
            global_delta = torch.stack(
                [
                    cos_h * local_pos[:, 0] - sin_h * local_pos[:, 1],
                    sin_h * local_pos[:, 0] + cos_h * local_pos[:, 1],
                ],
                -1,
            )
            recovered_pos = current_pos - global_delta

            current_pos = pos[:, step].clone()
            current_heading = heading[:, step].clone()
            current_pos[matched] = recovered_pos[matched]
            current_heading[matched] = recovered_heading[matched]

            frame_valid = valid[:, step]
            sampled_idx.append(token_idx)
            valid_mask.append(frame_valid)
            token_mask.append(matched)
            sampled_pos.append(current_pos.masked_fill(~frame_valid[:, None], 0))
            sampled_heading.append(current_heading.masked_fill(~frame_valid, 0))

        result = {
            "sampled_pos": torch.stack(sampled_pos, 1).flip(1),
            "sampled_heading": torch.stack(sampled_heading, 1).flip(1),
        }
        if sampled_idx:
            result.update(
                {
                    "sampled_idx": torch.stack(sampled_idx, 1).flip(1),
                    "valid_mask": torch.stack(valid_mask, 1).flip(1),
                    "token_mask": torch.stack(token_mask, 1).flip(1),
                }
            )
        else:
            result.update(
                {
                    "sampled_idx": torch.empty(
                        num_agents, 0, device=valid.device, dtype=torch.long
                    ),
                    "valid_mask": valid.new_empty((num_agents, 0)),
                    "token_mask": valid.new_empty((num_agents, 0)),
                }
            )
        return result

    @staticmethod
    def _clean_heading(valid: Tensor, heading: Tensor) -> Tensor:
        valid_pair = valid[:, :-1] & valid[:, 1:]
        for step in range(heading.shape[1] - 1):
            difference = wrap_angle(
                heading[:, step] - heading[:, step + 1]
            ).abs()
            replace = valid_pair[:, step] & (difference > 1.5)
            heading[replace, step + 1] = heading[replace, step]
        return heading

    def _extrapolate_agent_to_first_step(
        self,
        valid: Tensor,
        pos: Tensor,
        heading: Tensor,
        vel: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        has_valid = valid.any(1)
        first_valid = valid.long().argmax(1).tolist()
        for agent, first_step in enumerate(first_valid):
            if not has_valid[agent] or first_step == 0:
                continue
            valid[agent, :first_step] = True
            heading[agent, :first_step] = heading[agent, first_step]
            vel[agent, :first_step] = vel[agent, first_step]
            for step in range(first_step - 1, -1, -1):
                pos[agent, step] = pos[agent, step + 1] - vel[agent, first_step] * 0.1
        return valid, pos, heading, vel

    def _extrapolate_to_token_boundary(
        self,
        valid: Tensor,
        pos: Tensor,
        heading: Tensor,
        vel: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        has_valid = valid.any(1)
        first_valid = valid.long().argmax(1).tolist()
        for agent, first_step in enumerate(first_valid):
            if not has_valid[agent]:
                continue

            count = first_step % self.shift
            if (
                first_step == 10
                and first_step >= self.shift
                and not valid[agent, first_step - self.shift]
            ):
                count = self.shift
            if count == 0:
                continue

            start = first_step - count
            valid[agent, start:first_step] = True
            heading[agent, start:first_step] = heading[agent, first_step]
            vel[agent, start:first_step] = vel[agent, first_step]
            for offset in range(count):
                step = first_step - offset - 1
                pos[agent, step] = pos[agent, step + 1] - vel[agent, first_step] * 0.1
        return valid, pos, heading, vel

    # Backward-compatible name.
    _extrapolate_agent_to_prev_token_step = _extrapolate_to_token_boundary

    def _get_agent_tokens(
        self, agent_type: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        agent_type = agent_type.long()
        known = (agent_type >= 0) & (agent_type < 3)
        safe_type = agent_type.clamp(0, 2)

        libraries = torch.stack(
            [getattr(self, f"agent_token_all_{name}") for name in self.AGENT_NAMES]
        )
        all_tokens = libraries[safe_type]
        shape_table = all_tokens.new_tensor(self.AGENT_SHAPES)
        shapes = shape_table[safe_type]

        all_tokens = all_tokens * known[:, None, None, None, None]
        shapes = shapes * known[:, None]
        self.token_traj_all = all_tokens.flatten(2)
        return shapes, all_tokens, all_tokens[:, :, -1].contiguous()

    # Backward-compatible name.
    _get_agent_shape_and_token_traj = _get_agent_tokens

    def process_data(
        self, data: HeteroData
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        tokenized_map = self._load_map(data)
        cached_agent = data["tokenized_agent"]

        if len(cached_agent) == 0:
            agent = self.tokenize_agent(data)
            if self.pred_init:
                self.get_init(agent)

        elif "initial_pos" in cached_agent:
            agent = self._load_cached_initial_agent(cached_agent)
        else:
            agent = self._load_cached_token_agent(cached_agent)
            if self.pred_init:
                self.get_init(agent)

        agent.setdefault("num_graphs", data.num_graphs)
        self._attach_token_libraries(agent)
        return tokenized_map, agent

    def _load_map(self, data: HeteroData) -> Dict[str, Tensor]:
        cached = data["tokenized_map"]
        if len(cached) == 0:
            return self.tokenize_map(data)

        result = {}
        if "token_idx" in cached:
            result["token_idx"] = cached["token_idx"]
        else:
            distance = (
                self.map_token_sample_pt[:, :, 1:]
                - cached["traj_pos_local"].unsqueeze(1)
            ).square().sum((-2, -1))
            result["token_idx"] = distance.argmin(-1)
            result["traj_pos_local"] = cached["traj_pos_local"]

        for key in ("position", "orientation", "batch", "type", "light_type"):
            if key in cached:
                result[key] = cached[key]
        return result

    def _load_cached_initial_agent(
        self, cached: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        result = {
            key: cached[key]
            for key in ("initial_heading", "initial_pos", "initial_shape", "batch", "type")
        }
        result["type"] = result["type"].long()
        result["shape"] = result["initial_shape"]

        if "initial_vel" in cached:
            result["initial_vel"] = cached["initial_vel"]
        else:
            result["local_vel"] = cached["local_vel"]
            ego_mask = self._make_ego_mask(result["batch"])
            for key in ("ego_pos2", "ego_heading2"):
                value = cached[key]
                result[key] = value[ego_mask] if len(value) == len(ego_mask) else value

        if "sampled_pos" not in cached:
            return result

        result["token_mask"] = (
            self._as_time_tensor(cached["token_mask"], 2)
            if "token_mask" in cached
            else torch.ones_like(result["sampled_idx"], dtype=torch.bool)
        )

        shapes, all_tokens, final_tokens = self._get_agent_tokens(result["type"])
        result["token_agent_shape"] = shapes
        result["token_traj_all"] = all_tokens
        result["token_traj"] = final_tokens
        self.get_init(result)
        return result

    def _load_cached_token_agent(
        self, cached: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        agent_type = cached["type"].long()
        shapes, all_tokens, final_tokens = self._get_agent_tokens(agent_type)
        result = {
            "type": agent_type,
            "token_agent_shape": shapes,
            "token_traj": final_tokens,
            "token_traj_all": all_tokens,
        }
        for key in ("col_mask", "pred_mask"):
            if key in cached:
                result[key] = cached[key]

        # The original checked data.keys() and passed speed as the shift value.
        if "gt_valid_raw" in cached:
            for key in ("batch", "shape"):
                result[key] = cached[key]
            result.update(
                self._match_agent_token(
                    valid=cached["gt_valid_raw"],
                    pos=cached["gt_pos_raw"],
                    heading=cached["gt_head_raw"],
                    agent_shape=shapes,
                    token_traj=final_tokens,
                    shift=self.shift,
                    error_dist=0.3,
                )
            )
            return result

        for key in (
            "sampled_pos",
            "sampled_heading",
            "batch",
            "shape",
            "valid_mask",
            "token_mask",
        ):
            result[key] = cached[key]
        result["sampled_idx"] = cached["sampled_idx"].long()

        for key in ("gt_pos_raw", "gt_head_raw", "route_map_index", "id"):
            if key in cached:
                result[key] = cached[key]
        return result

    def _attach_token_libraries(self, agent: Dict[str, Tensor]) -> None:
        for name in self.AGENT_NAMES:
            agent[f"trajectory_token_{name}"] = getattr(
                self, f"trajectory_token_{name}"
            )