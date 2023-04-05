import sys
import warnings

import gymnasium as gym

import combinatorial_problems
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from AttentionModel import REINFORCE, AttentionModel

import tqdm

warnings.filterwarnings("ignore")

nodes = 20

total_episodes = 1_000

device = 'cuda'

torch.autograd.set_detect_anomaly(True)

# Attention Model Parameters
d_m = 128
d_c = d_m * 2
d_k = 128
h = 8
N = 3
d_ff = 128
n_nodes = nodes
embeder = 2
d_v = 128
c = 10.
head_split = True
dropout = 0.
use_graph_emb = True

samples = 1_024
batches = 1
epochs = 10
epochs *= 1_250  #
asynchronous = False

assert samples % batches == 0, f"Number of samples is not divisible by specified batches: {samples} % {batches} = {samples % batches}."
# List of environments. Use .reset({"new": False}) to reuse same environment. Useful for Training, Validation comparisons
# We reset them here already, as we want to keep the unique graphs generated here.
batched_envs = [
    gym.vector.make("combinatorial_problems/TravelingSalesman-v0",
                    num_nodes=nodes,
                    num_envs=batches,
                    new_on_reset=False,
                    asynchronous=False) for batch in range(samples // batches)
]

rewards_over_epochs = []

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
for epoch in range(epochs):
    rewards_over_batches = []
    for env in tqdm.tqdm(batched_envs, file=sys.stdout):
        # Apply seeds
        state, info = env.reset()
        start_idx = info["agent_start_idx"]
        done = False
        batch_rewards = 0
        j = 0
        while not done:
            print(j, torch.cuda.memory_allocated())
            # graph -> b x n_nodes x coords
            graph_nodes = np.stack(info["nodes"])
            graph = torch.FloatTensor(graph_nodes).reshape(batches, nodes, 2).to(device)
            # The context will be the concatenation of the node embeddings for first and last nodes.
            # use am_REINFORCE.policy.encode
            # tmb_emb -> b x nodes x d_m
            tmp_emb = am_REINFORCE.policy.encoder(graph).detach()
            # start/end_node -> b x 1 x d_m
            start_node = tmp_emb[np.arange(batches),start_idx,:].unsqueeze(1)
            end_node = tmp_emb[np.arange(batches),start_idx,:].unsqueeze(1)
            # ctxt -> b x 1 x d_c (2 * d_m)
            ctxt = torch.cat([start_node, end_node], dim=-1)
            # For now, I will not use a mask for the embedding input.
            # mask_emb_graph -> b x 1 x nodes
            mask_emb_graph = torch.zeros(batches, 1, nodes).bool().to(device) # Empty Mask!
            # mask_dex_graph -> b x 1 x nodes
            masks = np.stack(info["mask"])
            print(masks)
            mask_dec_graph = torch.tensor(masks).unsqueeze(1).to(device)
            reuse_embeding = False

            action = am_REINFORCE(graph=graph,
                                  ctxt=ctxt,
                                  mask_emb_graph=mask_emb_graph,
                                  mask_dec_graph=mask_dec_graph,
                                  reuse_embeding=reuse_embeding,
                                  explore=True).numpy()
            state, reward, terminated, truncated, info = env.step(action)
            am_REINFORCE.rewards.append(reward)
            batch_rewards += reward
            done = terminated.all() or truncated.all()
            print(j, torch.cuda.memory_allocated())
            j += 1
        rewards_over_batches.append(np.array(batch_rewards).mean())
        am_REINFORCE.update()
    rewards_over_epochs.append(np.mean(np.array(rewards_over_batches)))
    if epoch % 1 == 0:
        avg_reward = np.mean(rewards_over_epochs[-1:])
        print(f"Epoch: {epoch} with Average Reward {avg_reward} for last epoch",)

# Maybe add testing, using argmax!
# Why is the plotted graph different from what is being reported?

rewards_to_plot = [[batch_r] for batch_r in rewards_over_epochs]
df1 = pd.DataFrame(rewards_to_plot, columns=["Train"]).plot()
plt.title("Attention Model Training.")
plt.xlabel("Epochs")
plt.ylabel("Rewards")
plt.xticks(range(epochs))
plt.show()

for env in batched_envs:
    env.close()
    del env
