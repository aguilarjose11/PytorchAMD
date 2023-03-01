import random

import gymnasium as gym
from DeepPolicyGradient import DeepPolicyNetwork, StateValueNetwork
from AttentionModel import REINFORCE
import combinatorial_problems

import torch

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Select Environment
nodes = 4
# GridWorld size=nodes, render_mode=None, punishment=-0.01
env = gym.make("CartPole-v1", render_mode="human")
env.metadata["render_fps"] = 200
# Used for logging progress of model.
wrapped_env = env
rewards_over_seeds = []
#torch.autograd.set_detect_anomaly(True) # For debugging purposes
total_num_episodes = int(1_500)

# # Collect problem parameters
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.n

device = "cuda"

# Begin training
for seed in [None,]:
    # Set up randomness to be pseudo-consistent
    #torch.manual_seed(seed)
    #random.seed(seed)
    #np.random.seed(seed)
    # Agent will actually be cloned in the training wrapper, so do not use this.
    agent = DeepPolicyNetwork(obs_space_dims,
                              action_space_dims,
                              [16,])
    state_value = StateValueNetwork(obs_space_dims,
                              1,
                              [16, 32, 64, 32, 16])
    r_agent = REINFORCE(policy=agent,
                        optimizer=torch.optim.AdamW,
                        lr=1e-2,
                        gamma=0.99,
                        beta=0.9,
                        gradient_clip=(1., np.inf),
                        eps=1e-9,
                        ).to(device)
    # Reward logging over episodes for current seed.
    reward_over_episodes = []
    # For each episode
    for episode in range(total_num_episodes):
        # Prepare environment
        obs, info = wrapped_env.reset(seed=seed)
        done = False
        episode_rewards = 0
        counter = 0
        while not done:
            if counter > 15_000:
                break
            # Move observation to respective device.
            obs = torch.tensor(obs, dtype=torch.float).to(device)
            # Choose action.
            action = r_agent(obs, explore=True).numpy()
            # Apply chosen action.
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            # Rewards must be stored manually, the API depends on us on doing so.
            r_agent.rewards.append(reward)
            episode_rewards += reward
            done = terminated #or truncated
        # Log rewards over episodes
        reward_over_episodes.append(episode_rewards)
        # Apply backpropagation at the end of episode.
        r_agent.update()
        # Verbosity
        if episode % 5 == 0:
            avg_reward = int(np.mean(reward_over_episodes[-5:]))
            print(f"\rEpisode: {episode} with Average Reward {avg_reward} for last 5", end="")
    rewards_over_seeds.append(reward_over_episodes)

# Code for plotting learning curve.
rewards_to_plot = [[reward for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for LunarLander-v2"
)
plt.show()

# Test of policy
# Select Environment
env = gym.make("LunarLander-v2", render_mode="human")

for episode in range(10):
    # Prepare environment
    obs, info = env.reset(seed=None)
    done = False
    timer = 0
    while not done:
        # Move observation to respective device.
        obs = torch.tensor(obs, dtype=torch.float).to(device)
        # Choose action.
        action = r_agent(obs, deterministic=False)[0]
        # Apply chosen action.
        obs, reward, terminated, truncated, info = env.step(action)
        # Rewards must be stored manually, the API depends on us on doing so.
        #r_agent.rewards.append(-0.01 if not reward else reward)
        done = terminated
    # Log rewards over episodes
    # Apply backpropagation at the end of episode.
    #r_agent.update()
    # Verbosity
