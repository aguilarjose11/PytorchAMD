"""Attention Model REINFORCE training module

The training code used for the Attention Model. REINFORCE is implemented
here.


Note update.a
A little note on the performance function.
The original algorithm presented in Sutton & Barto 2018 uses a step-based algorithm to train the policy.
This means that an update to the algorithm is applied based on the gradient of the performance measure at
each episode step. Here, we use a different approach that is similar to a "batched" version of REINFORCE.
Instead of calculating and applying each gradient individually, we compute the gradient for all steps and
calculate the mean of all the gradients of J. In this way, we manage to find the gradient that would
change the policy towards the greatest increase in reward over all actions on average. This should in
practice make no difference to the performance of the algorithm.


"""
from typing import Tuple, Union
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.nn.utils import clip_grad_norm_

Optimizer = type(torch.optim.Optimizer)
Array = np.ndarray

class REINFORCE(nn.Module):

    def __init__(self,
                 policy: nn.Module,
                 optimizer: Optimizer,
                 lr: float = 1e-2,
                 gamma: float = 0.99,
                 beta: float = 0.95,
                 gradient_clip: Tuple[float, float] = None,
                 eps: float = 1e-9,
                 ):
        """REINFORCE algorithm

        parameters
        ----------
        policy: nn.Module
            Policy to train on. It is expected to be a child of nn.Module.
        optimizer: Optimizer
            Optimizer callable to optimize policy. Uses lr.
        d_state: int
            Dimensions of input state. Used for reshaping data.
        lr: float
            Learning rate to use on optimizer
        gamma: float
            Discounting factor for discounted rewards G_t.
        gradient_clip: float
            Clipping factor for gradients. Use None if opting to not use gradient clipping. Tuple is expected:
                * clipping_factor: float specifying the value at which larger gradients will be clipped
                * clipping_norm: float specifying the norm at which the gradients will be measured and clipped.
        eps: float
            Mathematical stability factor. Helps avoid underflow.
        """
        super().__init__()
        self.policy = policy
        self.optimizer = optimizer(self.policy.parameters(),
                                   lr,
                                   maximize=True)
        # By some reason this actualy works with
        self.lr = lr
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.gradient_clip, self.gradient_norm = gradient_clip if gradient_clip is not None else (None, None)

        self.running_G = None
        self.actions = []
        self.states = []
        self.pi = []
        self.rewards = []


    def forward(self,
                state: Tensor,
                explore: bool = False):
        """Forward pass through policy"""
        # Save probability distribution, moving them to CPU.
        probability_distribution = self.policy(state).cpu()
        if explore:
            # Allow exploration, sampling from probability distribution generated.
            sampler = Categorical(probability_distribution.detach())
            # action -> 1
            action = sampler.sample()
        else:
            # action -> 1 x 1
            action = torch.argmax(probability_distribution.detach(), dim=-1, keepdim=False)
        # Store state, probability distribution, and action taken
        self.states.append(state)
        self.pi.append(probability_distribution)
        self.actions.append(action)
        return action

    def _get_rewards(self,
                     discounted: bool = False
                     ) -> Array:
        """Return rewards either weighted by gamma or not
        Will return the rewards in a 1-dimensional row tensor.

        parameters
        ----------
        discounted: bool
            Whether to return the weighted sum of rewards using gamma.

        returns
        -------
        g_t: Array
            Tensor containing the weighted rewards, not summed yet. Expected in shape episode_len x 1
        """
        # self.rewards -> R^episode_length
        if discounted:
            # Apply the weighted sum
            running_g: float = 0
            g_t = []
            for reward in self.rewards[::-1]:
                running_g = reward + self.gamma * running_g
                g_t.insert(0, np.array(running_g))
        else:
            # Return rewards as is
            g_t = self.rewards

        return np.array(g_t)


    def update(self):
        """Apply gradient ascent
        applied a step of gradient ascent using the stored actions, probability distributions, states,
        and rewards.

        Five steps are applied here:
          1. Compute the delta values using the weighted rewards using gamma, and normalize
          2. Compute log pi, the probabilities for each action selected.
          3. Compute the performance function J, which will be used for gradient ascent.
          4. Apply gradient calculated, clipping to specified norm.
          5. Clear up stored information used for gradient computation.
        """

        # Step 1: Compute deltas from weighted rewards
        G = self._get_rewards(discounted=True)
        #delta = G
        if self.running_G is None:
            self.running_G = G.mean()
        else:
            self.running_G = self.beta * self.running_G + (1 - self.beta) * G.mean()
        delta = (G - self.running_G)
        delta_std = self.eps + delta.std() if len(G) > 1 else 1
        delta = delta / delta_std
        delta = torch.from_numpy(delta)

        # Step 2: Compute log probabilities
        # pi -> episode_len x d_action
        pi = torch.stack(self.pi)
        ln_pi = torch.log(pi)
        actions = torch.stack(self.actions)
        ln_pi_a = ln_pi[np.arange(len(actions)), actions]

        # Step 3: compute performance function
        # See note update.a on the use of the mean here vs Sutton & Barto's algorithm.
        J: Tensor = (delta * ln_pi_a).mean()

        # Step 4: Calculate gradients
        self.optimizer.zero_grad()
        J.backward()
        if self.gradient_norm is not None and self.gradient_clip is not None:
            clip_grad_norm_(parameters=self.policy.parameters(),
                            max_norm=self.gradient_clip,
                            norm_type=self.gradient_norm,
                            error_if_nonfinite=True)
        self.optimizer.step()

        # Step 5: Clean up calculations
        self.actions = []
        self.states = []
        self.pi = []
        self.rewards = []



