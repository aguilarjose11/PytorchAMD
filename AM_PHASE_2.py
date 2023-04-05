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
objectives = 7
random_objectives = True
total_episodes = 1_000
context = 2
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
epochs = 60  # same number of weight updates
#epochs *= 1_250  # ???

assert samples % batches == 0, f"Number of samples is not divisible by specified batches: {samples} % {batches} = {samples % batches}."
# List of environments. Use .reset({"new": False}) to reuse same environment. Useful for Training, Validation comparisons
# We reset them here already, as we want to keep the unique graphs generated here.
print("Here1")
batched_envs = [
    gym.vector.make("combinatorial_problems/Phase2Env-v0",
                    num_nodes=nodes,
                    num_envs=batches,
                    num_objectives=objectives,
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
#am_REINFORCE.load_state_dict(torch.load("PHASE_2/STATE_SAVE_EPOCH_0"))
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


def eval_env(env, val=False, render=False, epoch=0, seed=None, use_masking=False):
    if seed:
        state, info = env.reset(seed=seed)
    else:
        state, info = env.reset()

    done = False
    env_reward = 0
    j = 0
    while not done:
        start_idx = info["agent_start_idx"]
        end_idx = info["agent_end_idx"]
        obj_idx = info["agent_obj_idx"]
        non_obj_idx = info["agent_non_obj_idx"]
        vis_idx = info["agent_visited_idx"]
        #curr_idx = np.asarray(info["agent_curr_idx"], dtype=np.int64)
        curr_idx = info["agent_curr_idx"]
        obstacle_idx = info["agent_obstacle_idx"]
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

        ctxt = torch.cat([curr_node, end_node], dim=-1)

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

        if not val:
            am_REINFORCE.rewards.append(reward)
        env_reward += reward
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

    return env_reward, j

"""import time
rewards_over_epochs = []
NUM_BATCHES = 32  # batch size = num_samples // NUM_BATCHES

rewards_std_over_epochs = []
for epoch in range(epochs):
    print("EPOCH {}/{}".format(epoch+1, epochs))
    rewards_over_batches = []
    seeds = np.arange(256*batches)

    for sub_epoch in range(0, 2):
        print(" Sub Epoch {}/{}".format(sub_epoch+1, 2))
        # Reset environment and prepare data loging
        batch_rewards = 0
        start = time.time()
        print("   Before: {}".format(torch.cuda.memory_allocated()))
        for batch_i, env in enumerate(batched_envs):
            # Do preparation of environments
            seed = seeds[(batch_i * batches):(
                        (batch_i + 1) * batches)].tolist()
            #state, info = env.reset(seed=seed)
            step_reward = eval_env(env, val=False, seed=seed)
            batch_rewards += step_reward
            print("   After: {}".format(torch.cuda.memory_allocated()))
        end = time.time()
        rewards_over_batches.append(batch_rewards)
        print("   After: {}".format(torch.cuda.memory_allocated()))
        print("   Batch Reward: {}; Time: {}".format(batch_rewards, end - start))
    torch.save(am_REINFORCE.state_dict(), "MASKS/STATE_SAVE_EPOCH_{}".format(epoch))



import sys
sys.exit()"""
import time
import logging
NUM_BATCHES = k*64  # batch size = num_samples // NUM_BATCHES
logging.basicConfig(filename="PHASE_2.log", level=logging.DEBUG, encoding='utf-8')
logging.debug('This will get logged')

for epoch in range(epochs):
    msg = "EPOCH {}/{}".format(epoch+1, epochs)
    print(msg)
    logging.info(msg)
    #np.random.shuffle(train)
    train = []
    for i in range(0, samples):
        train.append(gym.make("combinatorial_problems/Phase2Env-v0", num_nodes=nodes, num_objectives=objectives,
                              random_objectives=random_objectives, new_on_reset=True, obstacle_max_distance=0.35))
    train = np.asarray(train)
    #print(eval_env(train[0], render=True, epoch="phs2_epoch_adjusted_obstacle"))
    rewards_over_batches = []
    if epoch < 10:
        use_masking = True
    else:
        use_masking = False
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
        for j, env in enumerate(batch):

            step_reward, iters = eval_env(env, val=False, use_masking=use_masking)

            batch_rewards += step_reward
            msg = j, iters, torch.cuda.memory_allocated(), step_reward
            print(msg)
            logging.info(msg)
        end = time.time()
        rewards_over_batches.append(batch_rewards)
        #print(i, torch.cuda.memory_allocated())
        msg = "   Batch Reward: {}; Time: {}".format(batch_rewards, end-start)
        print(msg)
        logging.info(msg)
        am_REINFORCE.update()

    rewards_over_epochs.append(np.mean(rewards_over_batches))
    msg = "  Epoch Reward: {}".format(rewards_over_epochs[-1])
    print(msg)
    logging.info(msg)


    batch_rewards = 0

    """am_REINFORCE.eval()
    for j, env in enumerate(val):
        print(j, torch.cuda.memory_allocated())
        with torch.no_grad():
            step_reward = eval_env(env, val=True)
        print(j, torch.cuda.memory_allocated())
        batch_rewards += step_reward
    #am_REINFORCE.update()
    print("  Val Epoch Reward: {}".format(batch_rewards))"""

    #grade_reward = eval_env(grade, val=True, render=True, epoch=epoch)
    #print("  Grade Reward: {}".format(grade_reward))
    #am_REINFORCE.train()

    print(rewards_over_epochs)
    torch.save(am_REINFORCE.state_dict(), "PHASE_2/STATE_SAVE_EPOCH_{}".format(epoch))



import sys
sys.exit()

for epoch in range(epochs):
    for i in range(0, batches):
        env = gym.make("combinatorial_problems/Phase1Env-v0", num_nodes=nodes, num_objectives=objectives, random_objectives=random_objectives)
        state, info = env.reset()
        start_idx = info["agent_start_idx"]
        end_idx = info["agent_end_idx"]
        obj_idx = info["agent_obj_idx"][0]
        non_obj_idx = info["agent_non_obj_idx"][0]
        curr_idx = info["agent_curr_idx"]
        vis_idx = info["agent_visited_idx"][0]
        done = False
        batch_rewards = 0
        j = 0
        while not done:
            env.render()
            # graph -> b x n_nodes x coords
            graph_nodes = np.stack(info["nodes"])
            graph = torch.FloatTensor(graph_nodes).reshape(batches, nodes, embeder).to(device)
            # The context will be the concatenation of the node embeddings for first and last nodes.
            # use am_REINFORCE.policy.encode
            # tmb_emb -> b x nodes x d_m
            tmp_emb = am_REINFORCE.policy.encoder(graph).detach()
            # start/end_node -> b x 1 x d_m
            start_node = tmp_emb[np.arange(batches), start_idx, :].unsqueeze(1)
            curr_node = tmp_emb[np.arange(batches), curr_idx, :].unsqueeze(1)

            # ctxt -> b x 1 x d_c (2 * d_m)
            ctxt = torch.cat([start_node, curr_node], dim=-1)
            # For now, I will not use a mask for the embedding input.
            # mask_emb_graph -> b x 1 x nodes
            mask_emb_graph = torch.zeros(batches, 1, nodes).bool().to(device)  # Empty Mask!
            # mask_dex_graph -> b x 1 x nodes
            mask_dec_graph = torch.tensor(np.logical_not(np.ones(batches * 1 * nodes)).reshape(batches, 1, nodes)).to(
                device)
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
            done = terminated or truncated
            j += 1
        env.render()
        am_REINFORCE.update()
        frames = []
        for t in range(0, j+1):
            image = imageio.v2.imread(f'tmp/render{t}.png')
            os.remove(f'tmp/render{t}.png')
            frames.append(image)
        imageio.mimsave('tmp/example.gif',
                        frames,
                        fps=1,
                        loop=1)



for epoch in range(epochs):
    rewards_over_batches = []
    print("Here3")
    for env in tqdm.tqdm(batched_envs, file=sys.stdout):
        # Apply seeds
        state, info = env.reset()
        start_idx = info["agent_start_idx"]
        end_idx = info["agent_end_idx"]
        obj_idx = info["agent_obj_idx"][0]
        non_obj_idx = info["agent_non_obj_idx"][0]
        curr_idx = info["agent_curr_idx"][0]
        vis_idx = info["agent_visited_idx"]
        done = False
        batch_rewards = 0
        while not done:
            env.render()
            # graph -> b x n_nodes x coords
            graph_nodes = np.stack(info["nodes"])
            graph = torch.FloatTensor(graph_nodes).reshape(batches, nodes, embeder).to(device)
            # The context will be the concatenation of the node embeddings for first and last nodes.
            # use am_REINFORCE.policy.encode
            # tmb_emb -> b x nodes x d_m
            tmp_emb = am_REINFORCE.policy.encoder(graph).detach()
            # start/end_node -> b x 1 x d_m
            start_node = tmp_emb[np.arange(batches), start_idx, :].unsqueeze(1)
            curr_node = tmp_emb[np.arange(batches), curr_idx, :].unsqueeze(1)

            # ctxt -> b x 1 x d_c (2 * d_m)
            ctxt = torch.cat([start_node, curr_node], dim=-1)
            # For now, I will not use a mask for the embedding input.
            # mask_emb_graph -> b x 1 x nodes
            mask_emb_graph = torch.zeros(batches, 1, nodes).bool().to(device) # Empty Mask!
            # mask_dex_graph -> b x 1 x nodes
            mask_dec_graph = torch.tensor(np.logical_not(np.ones(batches*1*nodes)).reshape(batches, 1, nodes)).to(device)
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
        rewards_over_batches.append(np.array(batch_rewards).mean())
        am_REINFORCE.update()
    rewards_over_epochs.append(np.mean(np.array(rewards_over_batches)))
    if epoch % 1 == 0:
        avg_reward = np.mean(rewards_over_epochs[-1:])
        print(f"Epoch: {epoch} with Average Reward {avg_reward} for last epoch",)


PATH = "SAVED_AM_REINFORCE_PHASE1_ENV_RHEL0"
torch.save(am_REINFORCE.state_dict(), PATH)

for env in batched_envs:
    env.close()
    del env

# Maybe add testing, using argmax!
# Why is the plotted graph different from what is being reported?

rewards_to_plot = [[batch_r] for batch_r in rewards_over_epochs]
df1 = pd.DataFrame(rewards_to_plot, columns=["Train"]).plot()
plt.title("Attention Model Training.")
plt.xlabel("Epochs")
plt.ylabel("Rewards")
plt.xticks(range(epochs))
plt.savefig(PATH+"_PLOT")
plt.show()
