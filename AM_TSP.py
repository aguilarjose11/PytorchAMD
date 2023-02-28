
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

torch.autograd.set_detect_anomaly(True)

# Attention Model Parameters
d_m = 128
d_c = d_m * 2
d_k = 128
h = 8
N = 2
d_ff = 128
n_nodes = nodes
embeder = 2
d_v = 32
c = 10.
head_split = True
dropout = 0.
use_graph_emb = True
batches = 1

rewards_over_seeds = []
for seed in [None]:
    # Apply seeds
    agent = AttentionModel(d_m=d_m,
                           d_c=d_c,
                           d_k=d_k,
                           h=h,
                           N=N,
                           d_ff=d_ff,
                           n_nodes=n_nodes,
                           embeder=embeder,
                           d_v=d_v,
                           c=c,
                           head_split=head_split,
                           dropout=dropout,
                           use_graph_emb=use_graph_emb,
                           batches=batches).to(device)
    am_REINFORCE = REINFORCE(policy=agent,
                             optimizer=torch.optim.AdamW,
                             lr=1e-4,
                             gamma=0.99,
                             beta=0.9,
                             gradient_clip=(1., torch.inf),
                             eps=1e-9).to(device)
    reward_over_episodes = []
    for episode in range(total_episodes):
        state, info = env.reset(seed=seed)
        start_idx = info["agent_start_idx"]
        done = False
        episode_rewards = 0
        while not done:
            # graph -> b x n_nodes x coords
            graph = torch.FloatTensor(info["nodes"]).reshape(1, nodes, 2).to(device)
            # The context will be the concatenation of the node embeddings for first and last nodes.
            # use am_REINFORCE.policy.encode
            # tmb_emb -> b x nodes x d_m
            tmp_emb = am_REINFORCE.policy.encoder(graph).detach()
            # start/end_node -> b x 1 x d_m
            start_node = tmp_emb[:,start_idx,:]
            end_node = tmp_emb[:,start_idx,:]
            # ctxt -> b x 1 x d_c (2 * d_m)
            ctxt = torch.cat([start_node, end_node], dim=-1).reshape(1, 1, -1)
            # For now, I will not use a mask for the embedding input.
            # mask_emb_graph -> b x 1 x nodes
            mask_emb_graph = torch.zeros(1, 1, nodes).bool().to(device) # Empty Mask!
            # mask_dex_graph -> b x 1 x nodes
            mask_dec_graph = torch.tensor(info["mask"]).reshape(1, 1, -1).to(device)
            reuse_embeding = False

            action = am_REINFORCE(graph=graph,
                                  ctxt=ctxt,
                                  mask_emb_graph=mask_emb_graph,
                                  mask_dec_graph=mask_dec_graph,
                                  reuse_embeding=reuse_embeding,
                                  explore=True).numpy()
            state, reward, terminated, truncated, info = env.step(action)
            am_REINFORCE.rewards.append(reward)
            episode_rewards += reward
            done = terminated or truncated
        reward_over_episodes.append(episode_rewards)
        am_REINFORCE.update()
        if episode % 5 == 0:
            avg_reward = np.mean(reward_over_episodes[-5:])
            print(f"\rEpisode: {episode} with Average Reward {avg_reward} for last 5", end="")
    rewards_over_seeds.append(reward_over_episodes)

# Maybe add testing, using argmax!

rewards_to_plot = [[reward for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for LunarLander-v2"
)
plt.show()