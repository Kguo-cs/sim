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
from pathlib import Path

import hydra
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import (
    CrossEntropy,
    TokenCls,
    WOSACMetrics,
    WOSACSubmission,
    minADE,
)
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils.finetune import set_model_for_finetuning
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts
import torch
import torch.optim as optim
import random
from collections import deque
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import copy
import torchvision

import cv2
import torch
import torchvision.transforms as transforms

class ReplayBuffer:
    def __init__(self, num_steps,num_processes=2):

        self.num_steps=num_steps

        self.state_list=deque(maxlen=num_steps+1)

    def initialize(self,agent_num,device):

        self.rewards = torch.zeros(self.num_steps, agent_num).to(device)
        self.value_preds = torch.zeros(self.num_steps + 1, agent_num).to(device)
        self.returns = torch.zeros(self.num_steps + 1, agent_num).to(device)
        self.action_log_probs = torch.zeros(self.num_steps, agent_num).to(device)
        self.actions = torch.zeros(self.num_steps, agent_num).to(device).to(torch.int)
        self.masks = torch.ones(self.num_steps + 1, agent_num).to(device)
        self.masks[-1]=0

    def insert(self, sample,step):
        if step==0:
            self.map=sample["state"][0]
        self.state_list.append(sample["state"][1])
        self.value_preds[step]=sample["value"]

        if step<self.num_steps:
            self.action_log_probs[step]=sample["value"]
            self.actions[step]=sample["action"]

    def sample(self, batch_size=1):
        idx=random.sample(range(self.num_steps),batch_size)[0]
        return {"state": (self.map,self.state_list[idx]),
                "action": self.actions[idx],
                "prev_log_prob":self.action_log_probs[idx],
                "adv":self.advantages[idx],
                "value":self.value_preds[idx],
                "return":self.returns[idx]
                }

    def sample_state_action(self, batch_size=1):
        idx=random.sample(range(self.num_steps),batch_size)[0]
        return {"state": (self.map,self.state_list[idx]),
                "action": self.actions[idx],
                }

    def compute_advantages(self):
        advantages = self.returns[:-1] - self.value_preds[:-1]
        # Normalize the advantages
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def compute_returns(self,gamma=0.99,gae_lambda=0.95):
        exp_rewards = self.rewards

        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = (
                exp_rewards[step]
                + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                - self.value_preds[step]
            )
            gae = (
                delta
                + gamma * gae_lambda * self.masks[step + 1] * gae
            )
            self.returns[step] = gae + self.value_preds[step]

class SMART_GAIL(LightningModule):

    def __init__(self, model_config) -> None:
        super(SMART_GAIL, self).__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        self.token_processor = TokenProcessor(**model_config.token_processor)

        self.encoder = SMARTDecoder(
            **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
        )
        set_model_for_finetuning(self.encoder, model_config.finetune)

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)
        self.wosac_metrics = WOSACMetrics("val_closed")
        self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)
        self.training_loss = CrossEntropy(**model_config.training_loss)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling

        self.discriminator=SMARTDecoder(
            **model_config.decoder, n_token_agent=self.token_processor.n_token_agent,
            discrminator=True
        )

        self.value_network=nn.Linear(model_config.decoder.hidden_dim,1)

        self.num_steps=15
        self.reward_type="gail"
        self.dis_update_num=1
        self.ppo_update_num=1

        self.agent_buffer = ReplayBuffer(self.num_steps)
        self.automatic_optimization = False
        self.expert_buffer = deque(maxlen=1000)

    def push_expert_sample(self,tokenized_map, tokenized_agent,step_current_2hz=2):
        for step in range(self.num_steps):
            tokenized_agent_current = {}

            tokenized_agent_current['sampled_pos'] = tokenized_agent["sampled_pos"][:,step:step+step_current_2hz]
            tokenized_agent_current['sampled_heading'] = tokenized_agent['sampled_heading'][:, step:step+step_current_2hz]
            tokenized_agent_current['sampled_idx'] = tokenized_agent["sampled_idx"][:, step:step+step_current_2hz]
            tokenized_agent_current['valid_mask'] = tokenized_agent["valid_mask"][:, step:step+step_current_2hz]
            tokenized_agent_current['trajectory_token_veh'] = tokenized_agent['trajectory_token_veh']
            tokenized_agent_current['trajectory_token_ped'] = tokenized_agent['trajectory_token_ped']
            tokenized_agent_current['trajectory_token_cyc'] = tokenized_agent['trajectory_token_cyc']
            tokenized_agent_current['type'] = tokenized_agent['type']
            tokenized_agent_current['shape'] = tokenized_agent['shape']
            tokenized_agent_current['batch'] = tokenized_agent['batch']
            tokenized_agent_current['num_graphs'] = tokenized_agent['num_graphs']

            action = tokenized_agent["sampled_idx"][:, step+step_current_2hz]

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

        return loss

    def update_reward_func(self,dis_opt,gradient_clip=True):
        for i in range(self.dis_update_num):
            expert_batch=random.sample(self.expert_buffer,1)[0]
            agent_batch=self.agent_buffer.sample_state_action()

            expert_d = self.discriminator.compute_disc_val(expert_batch['state'], expert_batch['action'])
            agent_d = self.discriminator.compute_disc_val(agent_batch['state'], agent_batch['action'])

            expert_loss =  F.binary_cross_entropy(expert_d,torch.ones_like(expert_d))
            agent_loss =  F.binary_cross_entropy(agent_d,torch.zeros_like(agent_d))

            total_loss = expert_loss + agent_loss

            dis_opt.zero_grad()
            total_loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            dis_opt.step()


    def get_reward(self):
        self.discriminator.eval()
        for step in range(self.num_steps):
            state = (self.agent_buffer.map, self.agent_buffer.state_list[step])
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

    def ppo_update(self,policy_optimizer):
        for e in range(self.ppo_update_num):
            sample=self.agent_buffer.sample(1)
            policy_optimizer.zero_grad()
            ppo_loss=self.ppo_loss(sample)
            self.manual_backward(ppo_loss)
            policy_optimizer.step()


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

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        # ! open-loop vlidation
        if self.val_open_loop:
            pred = self.encoder(tokenized_map, tokenized_agent)
            loss = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
                token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            )

            self.TokenCls.update(
                # action that goes from [(10->15), ..., (85->90)]
                pred=pred["next_token_logits"],  # [n_agent, 16, n_token]
                pred_valid=pred["next_token_valid"],  # [n_agent, 16]
                target=tokenized_agent["gt_idx"][:, 2:],
                target_valid=tokenized_agent["valid_mask"][:, 2:],
            )
            self.log(
                "val_open/acc",
                self.TokenCls,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)

        # ! closed-loop vlidation
        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self.n_rollout_closed_val):
                pred = self.encoder.inference(
                    tokenized_map, tokenized_agent, self.validation_rollout_sampling
                )
                pred_traj.append(pred["pred_traj_10hz"])
                pred_z.append(pred["pred_z_10hz"])
                pred_head.append(pred["pred_head_10hz"])

            pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
            pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
            pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]

            # ! WOSAC
            scenario_rollouts = None
            if self.wosac_submission.is_active:  # ! save WOSAC submission
                self.wosac_submission.update(
                    scenario_id=data["scenario_id"],
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                    global_rank=self.global_rank,
                )
                _gpu_dict_sync = self.wosac_submission.compute()
                if self.global_rank == 0:
                    for k in _gpu_dict_sync.keys():  # single gpu fix
                        if type(_gpu_dict_sync[k]) is list:
                            _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
                    scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
                    self.wosac_submission.aggregate_rollouts(scenario_rollouts)
                self.wosac_submission.reset()

            else:  # ! compute metrics, disable if save WOSAC submission
                self.minADE.update(
                    pred=pred_traj,
                    target=data["agent"]["position"][
                        :, self.num_historical_steps :, : pred_traj.shape[-1]
                    ],
                    target_valid=data["agent"]["valid_mask"][
                        :, self.num_historical_steps :
                    ],
                )

                # WOSAC metrics
                if batch_idx < self.n_batch_wosac_metric:
                    device = pred_traj.device
                    scenario_rollouts = get_scenario_rollouts(
                        scenario_id=get_scenario_id_int_tensor(
                            data["scenario_id"], device
                        ),
                        agent_id=data["agent"]["id"],
                        agent_batch=data["agent"]["batch"],
                        pred_traj=pred_traj,
                        pred_z=pred_z,
                        pred_head=pred_head,
                    )
                    self.wosac_metrics.update(data["tfrecord_path"], scenario_rollouts)

            # ! visualization
            if self.global_rank == 0 and batch_idx < self.n_vis_batch:
                if scenario_rollouts is not None:
                    for _i_sc in range(self.n_vis_scenario):
                        _vis = VisWaymo(
                            scenario_path=data["tfrecord_path"][_i_sc],
                            save_dir=self.video_dir
                            / f"batch_{batch_idx:02d}-scenario_{_i_sc:02d}",
                        )
                        _vis.save_video_scenario_rollout(
                            scenario_rollouts[_i_sc], self.n_vis_rollout
                        )
                        # for _path in _vis.video_paths:
                        #     self.logger.log_video(
                        #         "/".join(_path.split("/")[-3:]), [_path]
                        #     )

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_wosac_metrics = self.wosac_metrics.compute()
                epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
                if self.global_rank == 0:
                    epoch_wosac_metrics["epoch"] = (
                        self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    )
                    self.logger.log_metrics(epoch_wosac_metrics)

                self.wosac_metrics.reset()
                self.minADE.reset()

            if self.global_rank == 0:
                if self.wosac_submission.is_active:
                    self.wosac_submission.save_sub_file()

    def configure_optimizers(self):
        policy_optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.value_network.parameters()), lr=self.lr)
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return [policy_optimizer, discriminator_optimizer], []

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        # ! only closed-loop vlidation
        pred_traj, pred_z, pred_head = [], [], []
        for _ in range(self.n_rollout_closed_val):
            pred = self.encoder.inference(
                tokenized_map, tokenized_agent, self.validation_rollout_sampling
            )
            pred_traj.append(pred["pred_traj_10hz"])
            pred_z.append(pred["pred_z_10hz"])
            pred_head.append(pred["pred_head_10hz"])

        pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
        pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
        pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]

        # ! WOSAC submission save
        self.wosac_submission.update(
            scenario_id=data["scenario_id"],
            agent_id=data["agent"]["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            global_rank=self.global_rank,
        )
        _gpu_dict_sync = self.wosac_submission.compute()
        if self.global_rank == 0:
            for k in _gpu_dict_sync.keys():  # single gpu fix
                if type(_gpu_dict_sync[k]) is list:
                    _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
            scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
            self.wosac_submission.aggregate_rollouts(scenario_rollouts)
        self.wosac_submission.reset()

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.wosac_submission.save_sub_file()


#
# def load_video_as_tensor(video_path, max_frames=32, resize=(64, 64)):
#     """
#     Load a video file as a tensor (T, C, H, W).
#
#     Args:
#         video_path (str): Path to the video file.
#         max_frames (int): Maximum number of frames to load.
#         resize (tuple): Resize frames to (H, W).
#
#     Returns:
#         torch.Tensor: Video as a (T, C, H, W) tensor.
#     """
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(resize),
#         transforms.ToTensor()
#     ])
#
#     frame_count = 0
#     while cap.isOpened() and frame_count < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#         frame_tensor = transform(frame)  # Convert to tensor
#         frames.append(frame_tensor)
#         frame_count += 1
#
#     cap.release()
#
#     if len(frames) == 0:
#         raise ValueError(f"No frames extracted from {video_path}")
#
#     return torch.stack(frames)  # Shape: (T, C, H, W)
