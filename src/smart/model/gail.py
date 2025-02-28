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


from lightning import LightningModule

from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts
import torch
import torch.optim as optim
import random
from collections import deque
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch
from src.smart.model.rollout_buffer import ReplayBuffer

class GAIL(LightningModule):

    def __init__(self, model_config) -> None:
        super(GAIL, self).__init__(model_config)
        self.value_network=nn.Linear(model_config.decoder.hidden_dim,1)

        self.num_steps=15
        self.reward_type="gail"
        self.dis_update_num=1
        self.ppo_update_num=1

        self.agent_buffer = ReplayBuffer(self.num_steps)
        self.automatic_optimization = False
        self.expert_buffer = deque(maxlen=1000)

    def push_expert_sample(self,tokenized_map, tokenized_agent,step_current_2hz=2):
        hist_len=1

        for step in range(1,self.num_steps+1):
            tokenized_agent_current = {}

            tokenized_agent_current['sampled_pos'] = tokenized_agent["sampled_pos"][:,step:step+hist_len]
            tokenized_agent_current['sampled_heading'] = tokenized_agent['sampled_heading'][:, step:step+hist_len]
            tokenized_agent_current['sampled_idx'] = tokenized_agent["sampled_idx"][:, step:step+hist_len]
            tokenized_agent_current['valid_mask'] = tokenized_agent["valid_mask"][:, step:step+hist_len]
            tokenized_agent_current['trajectory_token_veh'] = tokenized_agent['trajectory_token_veh']
            tokenized_agent_current['trajectory_token_ped'] = tokenized_agent['trajectory_token_ped']
            tokenized_agent_current['trajectory_token_cyc'] = tokenized_agent['trajectory_token_cyc']
            tokenized_agent_current['type'] = tokenized_agent['type']
            tokenized_agent_current['shape'] = tokenized_agent['shape']
            tokenized_agent_current['batch'] = tokenized_agent['batch']
            tokenized_agent_current['num_graphs'] = tokenized_agent['num_graphs']

            action = tokenized_agent["sampled_idx"][:, step+hist_len]

            expert_sample = {
                "state": (tokenized_map, tokenized_agent_current),
                "action": action
            }

            self.expert_buffer.append(expert_sample)

    def rollout(self,tokenized_map,tokenized_agent):
        pred = self.encoder.inference(
            tokenized_map,
            tokenized_agent,
            sampling_scheme=self.training_rollout_sampling,
        )

        sample_list=pred["sample_list"]
        action=sample_list[0]["action"]
        self.agent_buffer.initialize(len(action),action.device)
        self.agent_buffer.map=tokenized_map

        for step,sample in enumerate(sample_list):
            sample["value"]=self.value_network(sample["feat_a_now"])[:,0]
            self.agent_buffer.insert(sample,step)

    def evaluate_actions(self, state, action):
        tokenized_map, tokenized_agent=state

        pred_dict = self.encoder(tokenized_map, tokenized_agent)
        pred_logit=pred_dict["cur_pred"]
        dist = Categorical(logits=pred_logit)
        action_log_probs=dist.log_prob(action)
        dist_entropy = dist.entropy()

        feat_a=pred_dict["feat_a"]

        value=self.value_network(feat_a)[:,0]

        return {
            'value': value,
            'log_prob': action_log_probs,
            'ent': dist_entropy,
        }

    def update_reward_func(self,dis_opt,gradient_clip=True):

        for i in range(self.dis_update_num):
            expert_batch=random.sample(self.expert_buffer,1)[0]
            agent_batch=self.agent_buffer.sample_state_action()

            expert_d = self.discriminator.compute_disc_val(expert_batch['state'], expert_batch['action'])
            agent_d = self.discriminator.compute_disc_val(agent_batch['state'] ,agent_batch['action'])

            expert_loss =  F.binary_cross_entropy(expert_d,torch.ones_like(expert_d))
            agent_loss =  F.binary_cross_entropy(agent_d,torch.zeros_like(agent_d))

            discrim_loss = expert_loss + agent_loss

            dis_opt.zero_grad()
            discrim_loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            dis_opt.step()

            self.log("train/discrim_loss", discrim_loss, on_step=True, batch_size=1)
            self.log("train/expert_loss", expert_loss, on_step=True, batch_size=1)
            self.log("train/agent_loss", agent_loss, on_step=True, batch_size=1)
            self.log("train/expert_disc_val", expert_d.mean().item(), on_step=True, batch_size=1)
            self.log("train/agent_disc_val",  agent_d.mean().item(), on_step=True, batch_size=1)
            self.log("train/agent_reward",  ((agent_d + 1e-20).log() - (1 - agent_d + 1e-20).log()).mean().item(), on_step=True, batch_size=1)

    def get_reward(self):
        self.discriminator.eval()

        map_feature=self.discriminator.map_encoder(self.agent_buffer.map)

        cum_reward=0

        for step in range(self.num_steps):
            state = (map_feature, self.agent_buffer.state_list[step])
            action = self.agent_buffer.actions[step]

            s = self.discriminator.compute_disc_val(state, action)

            eps = 1e-20
            if self.reward_type == 'airl':
                reward = (s + eps).log() - (1 - s + eps).log()
            elif self.reward_type == 'gail':
                reward = (s + eps).log()
            elif self.reward_type == 'raw':
                reward = s
            elif self.reward_type == 'airl-positive':
                reward = (s + eps).log() - (1 - s + eps).log() + 20
            elif self.reward_type == 'revise':
                d_x = (s + eps).log()
                reward = d_x + (-1 - (-d_x).log())
            else:
                raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
            self.agent_buffer.rewards[step]=reward

            cum_reward+=reward

        self.log("train/cum_reward", cum_reward.mean().item(), on_step=True, batch_size=1)

    def ppo_update(self,policy_optimizer):
        for e in range(self.ppo_update_num):
            sample=self.agent_buffer.sample(1)
            policy_optimizer.zero_grad()
            ppo_loss=self.ppo_loss(sample)
            self.manual_backward(ppo_loss)
            policy_optimizer.step()

    def ppo_loss(self,sample,
                 use_clipped_value_loss=False,
                 clip_param=0.2,
                 value_loss_coef=0.5,
                 entropy_coef=0.0001
                 ):

        ac_eval = self.evaluate_actions(sample['state'], sample['action'])

        ratio = torch.exp(ac_eval['log_prob'] - sample['prev_log_prob'])
        surr1 = ratio * sample['adv']
        surr2 = torch.clamp(ratio,
                            1.0 -clip_param,
                            1.0 + clip_param) * sample['adv']
        actor_loss = -torch.min(surr1, surr2).mean(0)

        if use_clipped_value_loss:
            value_pred_clipped = sample['value'] + (ac_eval['value'] - sample['value']).clamp(
                -clip_param,  clip_param)
            value_losses = (ac_eval['value'] - sample['return']).pow(2)
            value_losses_clipped = (
                    value_pred_clipped - sample['return']).pow(2)
            value_loss = 0.5 * torch.max(value_losses,
                                         value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (sample['return'] - ac_eval['value']).pow(2).mean()

        loss = (value_loss * value_loss_coef + actor_loss -  ac_eval['ent'].mean() * entropy_coef)

        self.log("train/value_loss", value_loss.mean().item(), on_step=True, batch_size=1)
        self.log("train/actor_loss", actor_loss.mean().item(), on_step=True, batch_size=1)
        self.log("train/dist_entropy", ac_eval['ent'].mean().item(), on_step=True, batch_size=1)


        return loss

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent)

            loss = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
                token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
                train_mask=data["agent"]["train_mask"],  # [n_agent]
                current_epoch=self.current_epoch,
            )
            self.log("train/loss", loss, on_step=True, batch_size=1)

            # Get optimizers
            policy_optimizer, discriminator_optimizer = self.optimizers()

            policy_optimizer.zero_grad()
            self.manual_backward(loss)
            policy_optimizer.step()
        else:
            with torch.no_grad():
                self.rollout(tokenized_map, tokenized_agent)
                self.push_expert_sample(tokenized_map,tokenized_agent)

            # Get optimizers
            policy_optimizer, discriminator_optimizer = self.optimizers()

            self.update_reward_func(discriminator_optimizer)

            with torch.no_grad():
                self.get_reward()

                self.agent_buffer.compute_returns()
                self.agent_buffer.compute_advantages()

            self.ppo_update(policy_optimizer)


    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            # if not self.wosac_submission.is_active:
            epoch_wosac_metrics = self.wosac_metrics.compute()
            epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
            if self.global_rank == 0:
                # epoch_wosac_metrics["epoch"] = (
                #     self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                # )
                #self.logger.log_metrics(epoch_wosac_metrics)
                for key, value in epoch_wosac_metrics.items():
                    self.log(key, value, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)

            self.wosac_metrics.reset()
            self.minADE.reset()

            # if self.global_rank == 0:
            #     if self.wosac_submission.is_active:
            #         self.wosac_submission.save_sub_file()

    def configure_optimizers(self):
        policy_optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.value_network.parameters()), lr=self.lr)
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return [policy_optimizer, discriminator_optimizer], []
