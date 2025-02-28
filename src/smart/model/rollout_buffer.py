
import torch
from collections import deque

import random



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
        self.state_list.append(sample["state"])
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
