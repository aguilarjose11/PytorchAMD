import copy
from typing import Union

from .policy_networks import DeepPolicyNetwork

import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.utils import clip_grad_norm_

import numpy as np

def nan_hook(module,
             input,
             output):

    for grad in input:
        if grad.isnan().any():
            print(module)
            print("------------Input Gradient------------")
            assert False , "An input gradient is NaN!"
    for grad in output:
        if grad.isnan().any():
            print(module)
            print("------------Output Gradient------------")
            assert False, "A gradient is NaN!"


# Deprecated!
class REINFORCE(nn.Module):
    def __init__(self,
                 d_obs: int,
                 n_actions: int,
                 policy: nn.Module,
                 optimizer: type(torch.optim.Optimizer),
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 eps: float = 1e-6,
                 grad_clip: float = 1.,
                 clip_norm: Union[np.ndarray, int] = np.inf,
                 value_baseline: nn.Module = None,
                 warm_up: int = 50
                 ):
        super().__init__()
        self.d_obs = d_obs
        self.n_actions = n_actions
        self.policy = copy.deepcopy(policy)
        # We want to maximize the rewards, as this is reinforcement learning after all!
        self.optimizer = optimizer(self.policy.parameters(), learning_rate, maximize=True)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = eps
        self.grad_clip = grad_clip
        self.clip_norm = clip_norm
        self.value_baseline = copy.deepcopy(value_baseline) if value_baseline is not None else None
        self.value_optimizer = optimizer(self.value_baseline.parameters(), learning_rate, maximize=True)\
            if value_baseline is not None else None
        self.warm_up = warm_up
        self.warm_alpha = 1/warm_up
        # None, mean, rolling

        self.log_probabilities = []
        self.actions = []
        self.states = []
        # User will be in charge of appending rewards to this.
        self.rewards = []

    def forward(self,
                obs: Tensor,
                deterministic: bool = False):
        # observation -> b x d_obs
        observation = obs
        self.states.append(observation)
        # prob_dist -> b x n_actions
        prob_dist = self.policy(observation)
        # Store log probabilities for training
        self.log_probabilities.append(torch.log(prob_dist + self.eps))
        # actions -> b x 1
        if deterministic:
            actions = torch.argmax(prob_dist.detach().cpu(), dim=-1, keepdim=True)
        else:
            sampler = Categorical(prob_dist.detach().cpu())
            actions = sampler.sample().unsqueeze(-1)
        # To gain access to log probabilities, use log_probabilities[-1] for latest computed outside API.
        self.actions.append(actions.numpy())
        return self.actions[-1] # Latest action taken

    def _weigthed_rewards(self,
                          rewards) -> Tensor:
        running_g = 0
        g_t = []
        for reward in rewards[::-1]:
            running_g = reward + self.gamma * running_g
            g_t.insert(0, running_g)
        return torch.tensor(g_t)

    def update(self):
        assert len(self.rewards) == len(self.log_probabilities), f"Number of rewards and forward passes is inconsistent:\
                                                                  {len(self.rewards)} vs {len(self.log_probabilities)}"

        # Compute discounted return G_t
        deltas = self._weigthed_rewards(self.rewards)

        # Compute Delta values.
        # Compute Delta values.
        if self.value_baseline is not None:
            # Compute the compared rewards with the baseline
            device = list(self.parameters())[0].device
            states = torch.cat(self.states)
            states = states.reshape(-1, self.d_obs).to(device)

            if self.warm_alpha < 1.:
                # Compute the mean and std
                v_baseline = self.value_baseline(states).cpu().squeeze(-1)
                # Compute exponential moving average
                deltas_mean = deltas.mean()
                v_ema = deltas_mean
                v_hat = self.warm_alpha * v_baseline + (1 - self.warm_alpha) * v_ema
            else:
                v_hat = self.value_baseline(states).cpu().squeeze(-1)

            deltas = (deltas - v_hat)
            deltas = deltas / (deltas.std() + 1e-9)
        else:
            # Normalize discounted returns
            # NaNs will occur if there is not many rewards collected!
            deltas = deltas
            #deltas_std = deltas.std() if len(deltas) > 1 else 1
            #deltas = (deltas - deltas.mean()) / (deltas_std + 1e-9)

        # Compute the performance metric J. We are computing the average gradient change rather than the change
        # as we move through the environment, as described in RL book by Sutton and Barto
        J: list = []
        for log_prob, action, delta in zip(self.log_probabilities, self.actions, deltas):
            # We compute the Performance metric for the trained model
            J.append(log_prob.cpu()[action] * delta)

        # Prepare for computations
        J: Tensor = torch.cat(J).mean()

        # Backpropagate with clipped norms on __policy__
        self.optimizer.zero_grad()
        J.backward()
        self.optimizer.step()

        # Backpropagate to improve the state-value function
        if self.value_baseline is not None:
            device = list(self.parameters())[0].device
            states = torch.cat(self.states)
            states = states.reshape(-1, self.d_obs).to(device)
            v_hat = self.value_baseline(states).cpu().squeeze(-1)
            # Compute state-value
            deltas = self._weigthed_rewards(self.rewards)
            deltas = deltas - v_hat
            # We compute the Performance metric for the baseline state-value function.
            J_value = deltas * v_hat
            J_value: Tensor = J_value.mean()
            # Apply gradient
            self.value_optimizer.zero_grad()
            J_value.backward()
            self.value_optimizer.step()

        self.log_probabilities = []
        self.actions = []
        self.rewards = []
        self.states = []
        self.warm_alpha += 1 / self.warm_up