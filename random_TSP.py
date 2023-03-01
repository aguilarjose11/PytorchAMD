
import gymnasium as gym

import combinatorial_problems
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from AttentionModel import REINFORCE, AttentionModel

nodes = 20

env = gym.make("combinatorial_problems/TravelingSalesman-v0",
               nodes=nodes,)
total_episodes = 1_000

device = 'cuda'

rewards_over_seeds = []
for seed in [None]:
    # Apply seeds

    reward_over_episodes = []
    for episode in range(total_episodes):
        state, info = env.reset(seed=seed)
        start_idx = info["agent_start_idx"]
        done = False
        episode_rewards = 0
        possible_actions = list(range(nodes))
        while not done:

            action = np.random.choice(possible_actions)
            possible_actions.remove(action)

            state, reward, terminated, truncated, info = env.step(action)
            episode_rewards += reward
            done = terminated or truncated
        reward_over_episodes.append(episode_rewards)
        if episode % 5 == 0:
            avg_reward = np.mean(reward_over_episodes[-5:])
            print(f"\rEpisode: {episode} with Average Reward {avg_reward} for last 5", end="")
    rewards_over_seeds.append(reward_over_episodes)

rewards_to_plot = [[reward for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for LunarLander-v2"
)
plt.show()