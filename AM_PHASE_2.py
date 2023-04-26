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
import imageio
#warnings.filterwarnings("ignore")
import os
nodes = 20
num_objectives = 7
random_objectives = True
total_episodes = 1_000
context = 1
device = 'cuda'

torch.autograd.set_detect_anomaly(True)

# Attention Model Parameters
d_m = 128
d_c = d_m * context
d_k = 128
h = 8
N = 3
d_ff = 128
n_nodes = nodes
embeder = 5
d_v = 128
c = 10.
head_split = True
dropout = 0.
use_graph_emb = True
k = 4
samples = k*1024 #1024  # 256 #1_024
batches = 1
epochs = 200  # same number of weight updates
#epochs *= 1_250  # ???

assert samples % batches == 0, f"Number of samples is not divisible by specified batches: {samples} % {batches} = {samples % batches}."
# List of environments. Use .reset({"new": False}) to reuse same environment. Useful for Training, Validation comparisons
# We reset them here already, as we want to keep the unique graphs generated here.
print("Here1")
batched_envs = [
    gym.vector.make("combinatorial_problems/Phase2Env-v0",
                    num_nodes=nodes,
                    num_envs=batches,
                    num_objectives=num_objectives,
                    new_on_reset=False,
                    random_objectives=random_objectives,
                    asynchronous=False) for batch in range(samples // batches)
]
print("Here2")
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
                       batches=None).to(device)
am_REINFORCE = REINFORCE(policy=agent,
                         optimizer=torch.optim.AdamW,
                         lr=1e-4,
                         gamma=0.99,
                         beta=0.9,
                         gradient_clip=(1., torch.inf),
                         eps=1e-9).to(device)

"""baseline_agent = AttentionModel(d_m=d_m,
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
                       batches=None).to(device)
baseline_am_REINFORCE = REINFORCE(policy=agent,
                         optimizer=torch.optim.AdamW,
                         lr=1e-4,
                         gamma=0.99,
                         beta=0.9,
                         gradient_clip=(1., torch.inf),
                         eps=1e-9).to(device)
"""
"""train = []
for i in range(0, samples):
    train.append(gym.make("combinatorial_problems/Phase1Env-v0", num_nodes=nodes, num_objectives=objectives,
                          random_objectives=random_objectives, new_on_reset=False))
train = np.asarray(train)

val = []
for i in range(0, 32):
    val.append(gym.make("combinatorial_problems/Phase1Env-v0", num_nodes=nodes, num_objectives=objectives,
                        random_objectives=random_objectives, new_on_reset=False))

grade = gym.make("combinatorial_problems/Phase1Env-v0", num_nodes=nodes, num_objectives=objectives,
                 random_objectives=random_objectives, new_on_reset=False)
"""
import copy

def eval_env(env, val=False, render=False, epoch=0, seed=None, use_masking=False, use_baseline=False):
    if seed:
        state, info = env.reset(seed=seed)
    else:
        state, info = env.reset()
    """else:
        if use_baseline:
            info = {"agent_end_idx": env.__getattr__("end_idx"),
                    "agent_curr_idx": env.__getattr__("agent_curr_idx"),
                    "nodes": env.__getattr__("nodes"),
                    "mask": env.__getattr__("mask").reshape(1, -1)}
        else:
            state, info = env.reset()
    cpy = copy.deepcopy(env)"""
    done = False
    env_reward = []
    #print(graph_nodes)
    #print(obstacle_idx)
    baseline_reward = []
    j = 0
    while not done:
        end_idx = info["agent_end_idx"]
        curr_idx = info["agent_curr_idx"]
        #print("  Num Obj: {}, Num Obst: {}, Non non-obj: {}".format(len(obj_idx), len(obstacle_idx), len(non_obj_idx)))
        if render:
            env.render()
        # graph -> b x n_nodes x coords
        graph_nodes = np.stack(info["nodes"])

        graph = torch.FloatTensor(graph_nodes).reshape(batches, nodes, embeder).to(device)
        # The context will be the concatenation of the node embeddings for first and last nodes.
        # use am_REINFORCE.policy.encode
        # tmb_emb -> b x nodes x d_m
        tmp_emb = am_REINFORCE.policy.encoder(graph).detach()
        # start/end_node -> b x 1 x d_m
        curr_node = tmp_emb[np.arange(batches), curr_idx, :].unsqueeze(1)
        end_node = tmp_emb[np.arange(batches), end_idx, :].unsqueeze(1)
        """obj_nodes = []
        for i in obj_idx[~vis_idx[obj_idx]]:
            obj_nodes.append(tmp_emb[np.arange(batches), i, :].unsqueeze(1))
        end_node = tmp_emb[np.arange(batches), end_idx, :].unsqueeze(1)
        if np.sum(~vis_idx[obj_idx])+2 != context:
            leftover = (context - (np.sum(~vis_idx[obj_idx])+2)) * [torch.zeros_like(curr_node)]
            # ctxt -> b x 1 x d_c (2 * d_m)
            ctxt = torch.cat([curr_node, end_node] + obj_nodes + leftover, dim=-1)
        else:
            # ctxt -> b x 1 x d_c (2 * d_m)
            ctxt = torch.cat([curr_node, end_node] + obj_nodes, dim=-1)"""

        ctxt = curr_node
        #ctxt = torch.cat([curr_node, end_node], dim=-1)

        # For now, I will not use a mask for the embedding input.
        # mask_emb_graph -> b x 1 x nodes
        if use_masking:
            mask_emb_graph = torch.zeros(batches, 1, nodes).bool().to(device)  # Empty Mask!
            # mask_dex_graph -> b x 1 x nodes
            masks = np.stack(info["mask"])
            mask_dec_graph = torch.tensor(masks).unsqueeze(1).to(device)
        else:
            mask_emb_graph = torch.zeros(batches, 1, nodes).bool().to(device)  # Empty Mask!
            # mask_dex_graph -> b x 1 x nodes
            mask_dec_graph = torch.tensor(
                np.logical_not(np.ones(batches * 1 * nodes)).reshape(batches, 1, nodes)).to(
                device)

        reuse_embeding = False

        action = am_REINFORCE(graph=graph,
                              ctxt=ctxt,
                              mask_emb_graph=mask_emb_graph,
                              mask_dec_graph=mask_dec_graph,
                              reuse_embeding=reuse_embeding,
                              explore=True).numpy()

        state, reward, terminated, truncated, info = env.step(action)

        """if not val:
            if not use_baseline:
                am_REINFORCE.rewards.append(reward)"""
        env_reward.append(reward)
        done = terminated or truncated  # terminated.all() or truncated.all() # terminated or truncated

        j += 1

    if render:
        env.render()
        frames = []
        for t in range(0, j + 1):
            image = imageio.v2.imread(f'tmp/render{t}.png')
            os.remove(f'tmp/render{t}.png')
            frames.append(image)
        imageio.mimsave('tmp/epoch{}.gif'.format(epoch),
                        frames,
                        fps=1,
                        loop=0)

    return env_reward, j, np.sum(env.__getattr__("greedy_rewards"))


import time
import logging
NUM_BATCHES = k*32  # batch size = num_samples // NUM_BATCHES
logging.basicConfig(filename="PHASE_2_baseline_harsh.log", level=logging.DEBUG, encoding='utf-8')
logging.debug('This will get logged')

#baseline_am_REINFORCE.load_state_dict(torch.load("PHASE_2/STATE_SAVE_EPOCH_20"))
#am_REINFORCE.load_state_dict(torch.load("PHASE_2_BASELINE/STATE_SAVE_EPOCH_15"))
for epoch in range(epochs):
    msg = "EPOCH {}/{}".format(epoch+1, epochs)
    print(msg)
    logging.info(msg)
    #np.random.shuffle(train)
    train = []
    for i in range(0, samples):
        train.append(gym.make("combinatorial_problems/Phase2Env-v0", num_nodes=nodes, num_objectives=num_objectives,
                              random_objectives=random_objectives, new_on_reset=True, obstacle_max_distance=0.2))
    train = np.asarray(train)
    """for i in range(0, 5):
        print(eval_env(train[i], render=True, epoch="phs2_test_instance{}".format(i)))"""
    rewards_over_batches = []
    #if epoch < 10:
    #    use_masking = True
    #else:
    use_masking = False
    #use_masking = True
    for i, batch in enumerate(np.array_split(train, NUM_BATCHES)):
        msg = " Train Batch {}/{}".format(i + 1, NUM_BATCHES)
        print(msg)
        logging.info(msg)
        msg = " Memory: {}".format(torch.cuda.memory_allocated())
        print(msg)
        logging.info(msg)
        batch_rewards = 0
        start = time.time()
        print(i, torch.cuda.memory_allocated())
        baseline_rewards = []
        for j, env in enumerate(batch):

            #step_reward, iters, cpy_env = eval_env(env, val=False, use_masking=use_masking, use_baseline=False)
            #baseline_step_reward, baseline_iters, _ = eval_env(cpy_env, val=False, use_masking=use_masking, use_baseline=True)
            #reward = step_reward - baseline_step_reward
            reward, iters, baseline_reward = eval_env(env, val=False, use_masking=use_masking, use_baseline=False)
            am_REINFORCE.rewards.append(reward)
            baseline_rewards.append([baseline_reward]*iters)
            batch_rewards += (np.sum(reward)-baseline_reward)
            msg = j, iters, torch.cuda.memory_allocated(), np.sum(reward), baseline_reward, np.sum(reward)-baseline_reward
            print(msg)
            logging.info(msg)
        end = time.time()
        rewards_over_batches.append(batch_rewards)
        #print(i, torch.cuda.memory_allocated())
        msg = "   Batch Reward: {}; Time: {}".format(batch_rewards, end-start)
        print(msg)
        logging.info(msg)
        am_REINFORCE.running_G = np.concatenate(baseline_rewards)
        am_REINFORCE.update()

    rewards_over_epochs.append(np.mean(rewards_over_batches))
    msg = "  Epoch Reward: {}".format(rewards_over_epochs[-1])
    print(msg)
    logging.info(msg)


    batch_rewards = 0

    print(rewards_over_epochs)
    torch.save(am_REINFORCE.state_dict(), "PHASE_2_BASELINE/STATE_SAVE_EPOCH_{}".format(epoch))


