"""Cleaner IQ/GAIL Lightning training module.

The public class and main method names are kept compatible with the original
implementation, while the training stages are split into small helpers.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.smart.loss.edge_gp import ZeroCenteredGradientPenalty_edge
from src.smart.loss.gp_penalty import (
    _has_elements,
    _reshape_valid_rewards,
    _select_ego_logits,
    _weighted_bce_with_logits,
    ZeroCenteredGradientPenalty
)
from src.smart.loss.rollout_buffer import (
    RunningMeanStdTorch,
    compute_advantages,
    get_train_mask,
)
import math
from src.smart.model.smart import SMART
from torch.optim.lr_scheduler import LambdaLR

TensorDict = MutableMapping[str, Any]


def _zero(reference: Tensor) -> Tensor:
    return reference.new_zeros(())

def _trainable_parameters(*modules: torch.nn.Module):
    """Return unique trainable parameters from several modules."""
    parameters = []
    seen = set()

    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))

    return parameters


def _safe_mean(value: Any, reference: Tensor) -> Tensor:
    if torch.is_tensor(value):
        return value.mean() if value.numel() else _zero(reference)
    return reference.new_tensor(float(value))


class SMART_GAIL(SMART):
    """Joint supervised, GAIL, value, and initial-state trainer.

    ``model_config`` must expose ``encoder`` and ``token_processor`` either as
    attributes or mapping entries. This explicit contract replaces the invalid
    ``LightningModule.__init__(model_config)`` call in the original code.
    """

    def __init__(self, model_config: Any) -> None:
        super().__init__(model_config)

        self.gamma = 0.99
        self.gail = bool(self.encoder.gail)
        self.alpha = float(self.encoder.alpha)

        self.use_lcf = bool(self.encoder.use_lcf)
        self.use_gradient_penalty = bool(self.token_processor.use_gradient_penalty)
        self.use_kl_penalty=False

        if self.gail:
            self.return_meanstd = RunningMeanStdTorch(shape=(1,))
            self.ego_return_meanstd = RunningMeanStdTorch(shape=(1,))
            self.global_return_meanstd = RunningMeanStdTorch(shape=(1,))

        init_uses_gan = self.token_processor.pred_init and self.encoder.init_decoder.use_gan
        #self.automatic_optimization=True
        if self.gail or init_uses_gan:
            self.automatic_optimization = False

    @staticmethod
    def _frozen_copy(module: torch.nn.Module) -> torch.nn.Module:
        frozen = copy.deepcopy(module)
        if hasattr(frozen, "target_net"):
            frozen.target_net = True
        frozen.requires_grad_(False)
        frozen.eval()
        return frozen

    def _log_train(self, name: str, value: Any) -> None:
        """Log every 50 optimizer steps and safely handle empty tensors."""

        if self.global_step % 50 != 0:
            return
        if torch.is_tensor(value) and value.numel() == 0:
            return
        self.log(name, value, on_step=True, batch_size=1)

    # ------------------------------------------------------------------
    # Supervised policy / initialization loss
    # ------------------------------------------------------------------
    def get_pred(
        self,
        tokenized_map: TensorDict,
        tokenized_agent: TensorDict,
        key: str = "expert",
    ) -> tuple[Tensor, Tensor,Tensor]:
        """Compute policy NLL and optional initial-state loss.

        Returns:
            total_loss: scalar tensor.
            selected_log_prob: one-dimensional tensor; empty when unavailable.
        """
        if key=="agent":
            tokenized_agent["train_mask"] = None

        prediction = self.encoder(tokenized_map, tokenized_agent)

        if key == "agent":
            tokenized_agent["train_mask"] = tokenized_agent["pred_mask"]

        reference = self._prediction_reference(prediction, tokenized_agent)

        policy_loss, log_prob,policy_entropy = self._policy_nll(prediction, tokenized_agent, key, reference)
        init_loss = self._initial_prediction_loss(prediction, tokenized_agent, reference)
        return policy_loss + init_loss, log_prob,policy_entropy

    @staticmethod
    def _prediction_reference(prediction: Mapping[str, Any], agent: Mapping[str, Any]) -> Tensor:
        for value in prediction.values():
            if torch.is_tensor(value):
                return value
            if isinstance(value, (tuple, list)):
                for item in value:
                    if torch.is_tensor(item):
                        return item
        for value in agent.values():
            if torch.is_tensor(value):
                return value
        raise ValueError("Could not find a tensor to determine device and dtype.")

    def _policy_nll(
        self,
        prediction: Mapping[str, Any],
        agent: TensorDict,
        key: str,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor,Tensor]:
        logits = prediction.get("next_token_logits")
        if logits is None:
            return _zero(reference), reference.new_empty((0,)),reference.new_empty((0,))

        start_step = 0 if key == "expert" else self.encoder.discriminator.interative_decoder.gail_start_step
        valid_mask = agent["valid_mask"][:, start_step:]
        actions = agent["sampled_idx"][:, start_step + 1 :].long()
        state_action_mask = get_train_mask(agent, start_step)

        explicit_agent_mask = agent.get("train_mask")
        if explicit_agent_mask is not None:
            explicit_agent_mask = explicit_agent_mask
            valid_mask = valid_mask[explicit_agent_mask]
            actions = actions[explicit_agent_mask]

        flat_sa_mask = state_action_mask.reshape(-1)
        flat_actions = actions.transpose(0, 1).reshape(-1)

        if key == "expert":
            # A transition is valid only when its source frame is valid.
            state_valid = valid_mask[:, :-1].transpose(0, 1).reshape(-1)
            logit_mask = flat_sa_mask[state_valid]
        else:
            logit_mask = flat_sa_mask

        selected_logits = logits[logit_mask] / self.alpha
        selected_actions = flat_actions[flat_sa_mask]

        log_policy = selected_logits.log_softmax(dim=-1)
        policy = log_policy.exp()
        log_prob = log_policy.gather(1, selected_actions[:, None]).squeeze(1)
        entropy = -(policy * log_policy).sum(dim=-1).mean()
        nll = -log_prob.mean()

        self._log_train(f"train/{key}_nll", nll)
        self._log_train(f"train/{key}_entropy", entropy)
        return nll, log_prob,entropy

    def _initial_prediction_loss(
        self,
        prediction: Mapping[str, Any],
        agent: TensorDict,
        reference: Tensor,
    ) -> Tensor:
        result = prediction.get("initial_logit")
        if result is None:
            return _zero(reference)

        init_decoder = self.encoder.init_decoder
        if init_decoder.use_gan:
            return init_decoder.D.gan_update(
                self.log,
                self.optimizers(),
                init_decoder.G1,
                result,
            )

        if init_decoder.learn_autoencoder:
            (
                reconstruction_loss,
                agent_loss,
                kl_loss,
                pos_loss,
                heading_loss,
                shape_loss,
                vel_loss,
                _,
            ) = result
            metrics = {
                "rec_loss": reconstruction_loss,
                "agent_loss": agent_loss,
                "kl_loss": kl_loss,
                "pos_loss": pos_loss,
                "heading_loss": heading_loss,
                "shape_loss": shape_loss,
                "vel_loss": vel_loss,
            }
            for name, value in metrics.items():
                self._log_train(f"train/{name}", _safe_mean(value, reference))
            return reconstruction_loss

        match_loss, collision_loss, pos_loss, heading_loss, shape_loss, vel_loss = result
        metrics = {
            "match_loss": match_loss,
            "pos_loss": pos_loss,
            "heading_loss": heading_loss,
            "shape_loss": shape_loss,
            "vel_loss": vel_loss,
            "col_loss": collision_loss,
        }
        for name, value in metrics.items():
            self._log_train(f"train/{name}", _safe_mean(value, reference))

        noise_std = agent.get("noise_std")
        if noise_std is not None:
            std = noise_std[:, 0].mean(0)
        else:
            std = init_decoder.G1.model.normal_scale[0]
        for name, value in zip(
            ("pos_std", "heading_std", "shape_std", "vel_std"),
            std.reshape(-1, 2).mean(-1),
        ):
            self._log_train(f"train/{name}", value)

        return match_loss + collision_loss

    # ------------------------------------------------------------------
    # Discriminator
    # ------------------------------------------------------------------
    def get_reward(
        self,
        agent: TensorDict,
        key: str,
        dis_mask: Optional[Tensor] = None,
    ):
        """Compute discriminator loss and rewards.

        Rewards returned for policy optimization are detached, preventing actor
        backward from entering the discriminator graph.
        """
        discriminator = self.encoder.discriminator
        mask_t = self._discriminator_mask(agent)

        if dis_mask is None:
            dis_mask = mask_t.reshape(-1)
            agent["dis_mask"] = dis_mask

        sampled_pos = agent["sampled_pos"]
        sampled_heading = agent["sampled_heading"]
        shape = agent["shape"][..., :2]

        use_gp = self.use_gradient_penalty and key == "expert" and discriminator.training
        if use_gp:
            sampled_pos = sampled_pos.detach().requires_grad_(True)
            sampled_heading = sampled_heading.detach().requires_grad_(True)
            shape = shape.detach().requires_grad_(True)

        disc_out = discriminator.predict_agent(
            agent["sampled_idx"],
            agent["token_mask"],
            agent["valid_mask"],
            sampled_pos,#[:,1:]
            sampled_heading,
            agent,
            agent["map_feature"],
            shape,
        )

        (ego_logits, interaction_logits, map_logits) = disc_out[0]
        ego_rewards, dis_action_pred, scene_reward, interaction_reward = disc_out[2]

        if self.encoder.discriminator.interative_decoder.dis_start_step==0:
            self._log_train(f"train/{key}_ego_score0", torch.sigmoid(ego_logits[:mask_t[0].sum()]).mean())
            self._log_train(f"train/{key}_ego_score1", torch.sigmoid(ego_logits[mask_t[0].sum():]).mean())


        target = 1.0 if key == "expert" else 0.0
        ego_logits = _select_ego_logits(ego_logits, dis_mask, mask_t) #t,a
        ego_loss = _weighted_bce_with_logits(
            logits=ego_logits,
            target=target,
            weight=torch.ones_like(ego_logits),
        )
        combined_logits = [ego_logits.reshape(-1)]

        self._log_train(f"train/{key}_ego_score", torch.sigmoid(ego_logits).mean())

        if _has_elements(map_logits):
            map_logits = _select_ego_logits(map_logits, dis_mask, mask_t)
            map_loss = _weighted_bce_with_logits(
                logits=map_logits,
                target=target,
                weight=torch.ones_like(map_logits),
            )
            ego_loss = ego_loss + map_loss
            combined_logits.append(map_logits.reshape(-1))
            self._log_train(f"train/{key}_map_score", torch.sigmoid(map_logits).mean())

        interaction_weight = None
        if _has_elements(interaction_logits):
            interaction_weight = disc_out[3].detach()
            interaction_loss = _weighted_bce_with_logits(
                logits=interaction_logits,
                target=target,
                weight=interaction_weight,
            )
            combined_logits.append(interaction_logits.reshape(-1))
            self._log_train(
                f"train/{key}_inter_score",
                torch.sigmoid(interaction_logits).mean(),
            )
        else:
            interaction_loss = _zero(ego_logits)

        if dis_action_pred is not None:
            start = (
                self.encoder.discriminator
                .interative_decoder
                .dis_start_step
            )

            # mask_t: [T - start, A_selected]
            # 已经从 dis_start_step 开始，不要再次切 start。
            #
            # state t 必须有效，并且 next state/action t+1 也必须有效。
            pair_mask = (
                    mask_t[:-1]
                    & mask_t[1:]
            )  # [T-start-1, A_selected]

            # ----------------------------------------------------------
            # target action: a_{t+1}
            # ----------------------------------------------------------
            actions = agent["sampled_idx"][
                :, start + 1:
            ].long()  # [A, T-start-1]

            # _discriminator_mask() 在非 pred_init 情况下会应用 train_mask，
            # action target 必须执行完全相同的 agent filtering。
            if not self.token_processor.pred_init:
                train_mask = agent.get("train_mask")
                if train_mask is not None:
                    actions = actions[train_mask]

            # time-major，与 discriminator feat_a 的排列一致
            action_targets = actions.transpose(0, 1)[
                pair_mask
            ]  # [N_pair]

            # ----------------------------------------------------------
            # select corresponding discriminator logits
            # ----------------------------------------------------------
            # dis_action_pred 对应 mask_t 中所有 True state：
            #
            #   mask_t.reshape(-1)[mask_t.reshape(-1)]
            #
            # 但最后一个 state 没有 next action，
            # next invalid 的 state 也不能计算 action CE。
            predict_mask = torch.zeros_like(mask_t)
            predict_mask[:-1] = pair_mask

            # 压缩到 dis_action_pred 的 index space
            selected_predict_mask = (
                predict_mask.reshape(-1)[
                    mask_t.reshape(-1)
                ]
            )

            action_logits = dis_action_pred[
                selected_predict_mask
            ]

            action_loss = torch.nn.functional.cross_entropy(
                action_logits,
                action_targets,
            )

            self._log_train(
                f"train/{key}_dis_action_loss",
                action_loss,
            )
        else:
            action_loss = _zero(ego_logits)

        discriminator_loss = ego_loss + interaction_loss+action_loss
        combined = torch.cat(combined_logits)
        probabilities = combined.sigmoid()
        self._log_train(f"train/{key}_disc_val", probabilities.mean())
        self._log_train(
            f"train/{key}_disc_val_std",
            probabilities.std(unbiased=False),
        )

        self.ego_return_meanstd.update(scene_reward.detach())
        scene_reward = self.ego_return_meanstd.normalize(scene_reward)

        self.global_return_meanstd.update(interaction_reward.detach())
        interaction_reward = self.global_return_meanstd.normalize(interaction_reward)

        ego_rewards=0.2*scene_reward+0.8*interaction_reward
        ego_reward_grid = _reshape_valid_rewards(ego_rewards, mask_t, "ego_rewards")

        # neighbour_reward_grid = None
        # if _has_elements(neighbour_rewards):
        #     neighbour_reward_grid = _reshape_valid_rewards(
        #         neighbour_rewards,
        #         mask_t,
        #         "nei_rewards",
        #     )
        # if neighbour_reward_grid is not None:
        #     self._log_train(f"train/{key}_nei_rewards", neighbour_reward_grid.mean())
        #     self._log_train(
        #         f"train/{key}_all_rewards",
        #         (ego_reward_grid + neighbour_reward_grid).mean(),
        #     )

        self._log_train(f"train/{key}_rewards", ego_reward_grid.mean())
        if _has_elements(scene_reward):
            self._log_train(f"train/{key}_scene_reward", scene_reward.mean())
        if _has_elements(interaction_reward):
            self._log_train(f"train/{key}_interact_reward", interaction_reward.mean())

        gp_loss = _zero(ego_logits)
        if use_gp:
            gp_valid_mask = agent["valid_mask"].clone()
            gp_valid_mask[:, : self.encoder.discriminator.interative_decoder.dis_start_step] = False
            #
            # (
            #     gp_loss,
            #     penalty_pos,
            #     penalty_head,
            #     penalty_shape,
            # ) = ZeroCenteredGradientPenalty(
            #     sampled_pos=sampled_pos,
            #     sampled_heading=sampled_heading,
            #     shape=shape,
            #     critic_score=combined.sum(),#[combined_logits.abs()>1]
            #     valid_mask=gp_valid_mask,
            #     gamma=0.01,
            # )
            # #
            # self._log_train(f"train/{key}_gp", gp_loss)
            # self._log_train(f"train/{key}_pos_gp", penalty_pos)
            # self._log_train(f"train/{key}_head_gp", penalty_head)
            # self._log_train(f"train/{key}_shape_gp", penalty_shape)
            edge_index = disc_out[1][0]
            destination_index = edge_index[1]

            gp_loss, scene_gp, interaction_gp, _ = ZeroCenteredGradientPenalty_edge(
                sampled_pos=sampled_pos,
                sampled_heading=sampled_heading,
                shape=shape,
                critic_score=(
                    ego_logits,
                    interaction_logits,
                    interaction_weight,
                    destination_index,
                ),
                valid_mask=gp_valid_mask,
                gamma=0.001,
                interaction_gamma=0.001,
                position_scale=1.0,
                heading_scale=1.0,
                shape_scale=1.0,
                interaction_min_mass=1,
                detach_edge_weight=True,
            )
            self._log_train(f"train/{key}_gp", gp_loss)
            self._log_train("train/scene_gp", scene_gp)
            self._log_train("train/interaction_gp", interaction_gp)

        # Rewards are targets, never differentiable critic outputs for the actor.
        return (
            discriminator_loss,
            ego_reward_grid.detach(),
            None ,#if neighbour_reward_grid is None else neighbour_reward_grid.detach()
            gp_loss,
            dis_mask,
        )

    def _discriminator_mask(self, agent: Mapping[str, Any]) -> Tensor:
        mask_t = agent["valid_mask"].transpose(0, 1)[self.encoder.discriminator.interative_decoder.dis_start_step :]
        if not self.token_processor.pred_init:
            train_mask = agent.get("train_mask")
            if train_mask is not None:
                mask_t = mask_t[:, train_mask]
        return mask_t

    def _perturb_expert_dis_mask(
            self,
            expert_dis_mask: Tensor,
            agent: Mapping[str, Any],
            perturb_prob: float = 1.0,
    ) -> Tensor:
        """Permute mask trajectories among same-batch, same-type non-ego agents.

        The time dimension is preserved. Ego agents are never changed.

        Args:
            expert_dis_mask:
                Flattened time-major discriminator mask, shape [T * A_selected].
            agent:
                Agent dictionary used by the target discriminator call.
            perturb_prob:
                Probability of perturbing each batch/type group.

        Returns:
            Perturbed flattened boolean mask.
        """
        current_valid = self._discriminator_mask(agent)  # [T, A_selected]

        if expert_dis_mask.numel() != current_valid.numel():
            raise ValueError(
                "expert_dis_mask and current discriminator mask have "
                f"different sizes: {expert_dis_mask.numel()} vs "
                f"{current_valid.numel()}."
            )

        source_mask = expert_dis_mask.bool().view_as(current_valid)
        perturbed_mask = source_mask.clone()

        batch = agent["batch"]
        agent_type = agent["type"].long()
        ego_mask = agent["ego_mask"].bool()

        # _discriminator_mask() applies train_mask on the agent dimension,
        # so metadata must use exactly the same filtering.
        if not self.token_processor.pred_init:
            train_mask = agent.get("train_mask")
            if train_mask is not None:
                batch = batch[train_mask]
                agent_type = agent_type[train_mask]
                ego_mask = ego_mask[train_mask]

        if batch.numel() != current_valid.shape[1]:
            raise ValueError(
                "Agent metadata is not aligned with discriminator columns: "
                f"{batch.numel()} vs {current_valid.shape[1]}."
            )

        group_keys = torch.stack(
            [batch.long(), agent_type],
            dim=-1,
        )

        for group_key in torch.unique(group_keys, dim=0):
            same_group = (
                    (batch == group_key[0])
                    & (agent_type == group_key[1])
                    & ~ego_mask
            )
            group_idx = same_group.nonzero(as_tuple=True)[0]

            # Cannot replace an agent when no alternative exists.
            if group_idx.numel() < 2:
                continue

            if torch.rand((), device=batch.device) >= perturb_prob:
                continue

            # A non-zero cyclic shift guarantees that every agent receives
            # another non-ego agent's temporal mask.
            shift = torch.randint(
                low=1,
                high=group_idx.numel(),
                size=(),
                device=batch.device,
            ).item()

            donor_idx = group_idx.roll(shifts=shift)

            perturbed_mask[:, group_idx] = source_mask[:, donor_idx]

        # dis_mask must be a subset of the logits available under current_valid.
        perturbed_mask &= current_valid

        # Explicitly preserve ego entries.
        perturbed_mask[:, ego_mask] = (
                source_mask[:, ego_mask] & current_valid[:, ego_mask]
        )

        return perturbed_mask.reshape(-1)

    # ------------------------------------------------------------------
    # Complete update
    # ------------------------------------------------------------------
    def update(self, tokenized_map: TensorDict, tokenized_agent: TensorDict) -> Tensor:
     #   self._initialize_reference_model_once()

        expert_agent = self._prepare_expert_agent(tokenized_agent)

        #if not self.token_processor.learn_init:

        expert_nll, _ ,_= self.get_pred(tokenized_map, tokenized_agent, key="expert")
        # else:
        #     map_feature = self.encoder._get_map_feature(
        #         tokenized_map,
        #         tokenized_agent,
        #     )
        #
        #     expert_nll=torch.zeros(1,device=self.device)

        if not self.gail:
            return expert_nll

        if self.token_processor.pred_init:
            expert_agent["pred_mask"] = None
        else:
            expert_agent["train_mask"] =  expert_agent["pred_mask"]

        #if self.token_processor.learn_init :#== False
        expert_dis_loss, _, _, expert_gp, expert_dis_mask = self.get_reward(
            expert_agent,
            "expert",
        )
        # else:
        #     expert_dis_loss=expert_gp=0
        #     expert_dis_mask = self._discriminator_mask(expert_agent).reshape(-1)

        with torch.no_grad():
            self.encoder.eval()
            rollout_agent = self.encoder.inference( expert_agent )
            self.encoder.train()

        # perturbed_dis_mask = self._perturb_expert_dis_mask(
        #     expert_dis_mask,
        #     rollout_agent,
        #     perturb_prob=1,
        # )

       # if self.token_processor.learn_init :#== False
        agent_dis_loss, agent_rewards, _, agent_gp, _ = self.get_reward(
            rollout_agent,
            "agent",
            expert_dis_mask,
        )
        # else:
        #     with torch.no_grad():
        #
        #         agent_dis_loss, agent_rewards, _, agent_gp, _ = self.get_reward(
        #             rollout_agent,
        #             "agent",
        #             expert_dis_mask,
        #         )

        critic_loss = expert_dis_loss + agent_dis_loss + expert_gp + agent_gp
        self._log_train("train/critic_loss", critic_loss)

        if not self.automatic_optimization:
            actor_optimizer, discriminator_optimizer, init_optimizer = self._optimizers()
        else:
            actor_optimizer=discriminator_optimizer=init_optimizer=None

        # if self.token_processor.learn_init :#== False
        self._optimizer_step(discriminator_optimizer, critic_loss)

        #if self.token_processor.learn_init:

        policy_loss, advantages_flat, advantages_2d = self._actor_value_loss(
            tokenized_map,
            rollout_agent,
            agent_rewards,
            expert_nll,
        )

        #if not self.token_processor.learn_init:
        self._optimizer_step(actor_optimizer, policy_loss)#policy update use no sde, initial update use sde
        # else:
        #     policy_loss=torch.zeros(1,device=critic_loss.device)

        if self.token_processor.learn_init:
            init_loss=self._initial_state_update(
                tokenized_agent,
                advantages_flat,
                advantages_2d,
                init_optimizer,
            )
        else:
            init_loss=torch.zeros(1,device=critic_loss.device)

        # Returned only for progress reporting. Both optimizers already stepped.
        return critic_loss.detach()+ policy_loss.detach()+init_loss.detach()

    def _initialize_reference_model_once(self) -> None:
        generator = self.encoder.init_decoder.G1
        if self.global_step != 0 or not generator.use_ref:
            return
        with torch.no_grad():
            for source, target in zip(
                generator.model.parameters(),
                generator.ref_model.parameters(),
                strict=True,
            ):
                target.copy_(source)
        generator.ref_model.requires_grad_(False)
        generator.ref_model.eval()

    def _prepare_expert_agent(self, tokenized_agent: TensorDict) -> TensorDict:
        if self.gail:
            for key in (
                "sampled_pos",
                "sampled_heading",
                "sampled_idx",
                "valid_mask",
                "token_mask",
            ):
                tokenized_agent[key] = tokenized_agent[key][:, :self.training_rollout_len]
        return tokenized_agent

    def _optimizers(self):
        optimizers = tuple(self.optimizers())
        if len(optimizers) == 3:
            actor, discriminator, initial = optimizers
        else:
            actor, discriminator = optimizers
            initial = None
        return actor, discriminator, initial

    def _optimizer_step(self, optimizer, loss: Tensor) -> None:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            self.manual_backward(loss)
            optimizer.step()

    def _actor_value_loss(
        self,
        tokenized_map: TensorDict,
        rollout_agent: TensorDict,
        rewards: Tensor,
        expert_nll: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        edge_encoder = self.encoder.agent_encoder.interative_decoder.edge_encoder
        old_rollout_flag = edge_encoder.rollout_traj
        edge_encoder.rollout_traj = True
        try:
            _, agent_log_prob,agent_entropy = self.get_pred(tokenized_map, rollout_agent, key="agent")
        finally:
            edge_encoder.rollout_traj = old_rollout_flag

        value = self._value_predictions(
            rollout_agent,
            num_agents=rewards.shape[1],
        )

        init_step=len(value)-len(rewards)+1

        rewards_pad= torch.cat([torch.zeros([init_step-1,value.shape[1]],device=rewards.device), rewards])

        advantages_2d, value_loss_elements = compute_advantages(
            rewards_pad,#[-len(value) :]
            value,
        )

        if "train_mask" in rollout_agent and rollout_agent["train_mask"] is not None:
            policy_transition_mask = get_train_mask(
                rollout_agent,
                self.encoder.discriminator.interative_decoder.gail_start_step,
            )

            # 对齐 advantage 的最后 policy 部分
            policy_advantages = advantages_2d[
                -policy_transition_mask.shape[0]:
            ]

            advantages_flat = policy_advantages[
                policy_transition_mask
            ]
        else:
            advantages_flat = advantages_2d.reshape(-1)

        self.return_meanstd.update(advantages_flat.detach())
        advantages_flat = self.return_meanstd.normalize(advantages_flat)

        ppo_advantages = advantages_flat[-agent_log_prob.numel() :].detach()
        init_advantages = advantages_flat[:-agent_log_prob.numel() ].detach()

        # self.return_meanstd.update(ppo_advantages.detach())
        # ppo_advantages = self.return_meanstd.normalize(ppo_advantages)
        #
        # self.ego_return_meanstd.update(init_advantages.detach())
        # init_advantages = self.ego_return_meanstd.normalize(init_advantages)

        if self.training_rollout_len > 1 and agent_log_prob.numel():
            ppo_loss = -(agent_log_prob * ppo_advantages).mean()
        else:
            ppo_loss = _zero(value)

        if self.token_processor.learn_init:
            value_loss = value_loss_elements[init_step+1:].mean()
            init_value_loss = value_loss_elements[:init_step+1].mean()
            self._log_train("train/init_value_loss", init_value_loss)
        else:
            value_loss = value_loss_elements.mean()
            init_value_loss = _zero(value)

        self._log_train("train/ppo_loss", ppo_loss)
        self._log_train("train/running_mean", self.return_meanstd.mean)
        self._log_train("train/running_var", self.return_meanstd.var)
        self._log_train("train/value_loss", value_loss)

        policy_loss = expert_nll + ppo_loss + 1e-3 * value_loss + 1e-3 * init_value_loss
        return policy_loss, init_advantages, advantages_2d

    def _value_predictions(self, rollout_agent: TensorDict  ,  num_agents: int) -> Tensor:
        initial_value = None
        if self.token_processor.learn_init:
            initial_value = self.encoder.init_value_network(
                rollout_agent["noise_feat"]
            )[...,0].transpose(0,1)

        if "feat_a" in rollout_agent:
            values = self.encoder.value_network(rollout_agent["feat_a"])[..., 0]
            values = values.reshape(-1, num_agents)
            if initial_value is not None:
                values = torch.cat((initial_value, values), dim=0)
            return values

        return initial_value

    def _initial_state_update(
        self,
        tokenized_agent: TensorDict,
        advantages_flat: Tensor,
        advantages_2d: Tensor,
        optimizer,
    ) -> None:
        #normalized = advantages_flat.view_as(advantages_2d)
        tokenized_agent["advantages"] =advantages_flat.view_as(advantages_2d[:tokenized_agent["noise_feat"].shape[1]]) #normalized[:tokenized_agent["noise_feat"].shape[1]]

        match_loss, col_loss, pos_loss, heading_loss, shape_loss, vel_loss = self.encoder.init_decoder(tokenized_agent)
        rl_loss = tokenized_agent["rl_loss"]
        reference = match_loss
        metrics = {
            "match_loss": match_loss,
            "pos_loss": pos_loss,
            "heading_loss": heading_loss,
            "shape_loss": shape_loss,
            "vel_loss": vel_loss,
            "rl_loss": rl_loss,
            "col_loss": col_loss,
        }
        for name, value in metrics.items():
            self._log_train(f"train/{name}", _safe_mean(value, reference))

        self._optimizer_step(optimizer, match_loss + rl_loss + col_loss)

        return match_loss + rl_loss + col_loss

    # ------------------------------------------------------------------
    # Lightning entry point
    # ------------------------------------------------------------------
    def training_step(self, data: Any, batch_idx: int) -> Tensor:
        # if random.random()<0.5:
        #     self.token_processor.learn_init = False
        #     self.token_processor.pred_init = True
        # else:
        #     self.token_processor.learn_init = True
        #     self.token_processor.pred_init = True
        #
        #
        # if self.token_processor.pred_init:
        #     self.encoder.discriminator.interative_decoder.gail_start_step = 0
        #     self.encoder.discriminator.interative_decoder.dis_start_step = 0 if self.token_processor.learn_init else 1
        #     self.encoder.agent_encoder.interative_decoder.gail_start_step = 0
        #     self.encoder.agent_encoder.interative_decoder.dis_start_step = 0 if self.token_processor.learn_init else 1
        #
        # else:
        #     self.encoder.agent_encoder.interative_decoder.gail_start_step = 1
        #     self.encoder.agent_encoder.interative_decoder.dis_start_step = 2
        #
        #     self.encoder.discriminator.interative_decoder.gail_start_step = 1
        #     self.encoder.discriminator.interative_decoder.dis_start_step = 2


        tokenized_map, tokenized_agent = self.token_processor(data)

       # self.token_processor.eval()

        loss = self.update(tokenized_map, tokenized_agent)
        self._log_train("train/loss", loss)

        for key in ("sampled_match_loss", "clip_ratio", "noncol_rate"):
            if key in tokenized_agent:
                self._log_train(f"train/{key}", tokenized_agent[key].mean())
        return loss


    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    def _lr_multiplier(self, step: int) -> float:
        warmup = int(self.lr_warmup_steps)
        total = int(self.lr_total_steps)
        minimum = float(self.lr_min_ratio)

        if total <= warmup:
            raise ValueError(
                "lr_total_steps must be greater than lr_warmup_steps."
            )

        if warmup > 0 and step < warmup:
            progress = step / warmup
            return minimum + (1.0 - minimum) * progress

        progress = min(1.0, (step - warmup) / (total - warmup))
        return minimum + 0.5 * (1.0 - minimum) * (
            1.0 + math.cos(math.pi * progress)
        )


    def configure_optimizers(self):
        if self.automatic_optimization:
            optimizer = torch.optim.AdamW(
                _trainable_parameters(self.encoder),
                lr=self.lr,
            )
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=self._lr_multiplier,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        if self.gail:
            discriminator_optimizer = torch.optim.AdamW(
                _trainable_parameters(self.encoder.discriminator),
                lr=self.lr,
            )

            if self.token_processor.learn_init:
                actor_optimizer = torch.optim.AdamW(
                    _trainable_parameters(
                        self.encoder.agent_encoder.agent_token_embedding,
                        self.encoder.agent_encoder.interative_decoder,
                        self.encoder.value_network,
                        self.encoder.init_value_network,
                    ),
                    lr=self.lr,
                )
                init_optimizer = torch.optim.AdamW(
                    _trainable_parameters(self.encoder.init_decoder),
                    lr=self.lr,
                )
                return [
                    actor_optimizer,
                    discriminator_optimizer,
                    init_optimizer,
                ]

            if self.token_processor.pred_init:
                actor_modules = (
                    self.encoder.agent_encoder.agent_token_embedding,
                    self.encoder.agent_encoder.interative_decoder,
                    self.encoder.value_network,
                )
            else:
                actor_modules = (
                    self.encoder.map_encoder,
                    self.encoder.agent_encoder,
                    self.encoder.value_network,
                )

            actor_optimizer = torch.optim.AdamW(
                _trainable_parameters(*actor_modules),
                lr=self.lr,
            )
            return [actor_optimizer, discriminator_optimizer]

        # Manual optimization without GAIL is used only by the initial GAN.
        generator_optimizer = torch.optim.AdamW(
            _trainable_parameters(self.encoder.init_decoder.G1),
            lr=self.lr,
        )
        discriminator_optimizer = torch.optim.AdamW(
            _trainable_parameters(self.encoder.init_decoder.D),
            lr=self.lr,
        )
        return [generator_optimizer, discriminator_optimizer]