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

import time
import logging

import tqdm
import imageio
import os

"""
PHASE NAME VARIABLES
"""
PHASE = 3  # 1 or 2 or 3
logging_filename = "PHASE_{}_base_model_baseline.log".format(PHASE)
env_filename = "combinatorial_problems/Phase{}Env-v0".format(PHASE)
render_filename = "render_PHASE_{}_base_model_baseline".format(PHASE)
state_save_foldername = "PHASE_{}_base_model_baseline".format(PHASE)
if not os.path.exists(state_save_foldername):
    os.makedirs(state_save_foldername)

logging.basicConfig(filename=logging_filename, level=logging.DEBUG, encoding='utf-8')


"""
ENVIRONMENT VARIABLES
"""
nodes = 20
num_objectives = 7
num_obstacles = 2
obstacle_min_distance = 0.10  # Minimum distance bound from an obstacle before the negative reward does not count
obstacle_max_distance = 0.2  # Maximum distance bound from an obstacle before the negative reward does not count.
random_objectives = True
k = 4
samples = k * 1024
num_batches = k * 32  # batch size = num_samples // num_batches
context = 1  # Number of context embeddings

if PHASE == 1:
    # Number of features for graph (x, y, boolean end node, boolean objective)
    embeder = 4
    use_baseline = False

    args = {"num_nodes": nodes,
            "num_objectives": num_objectives,
            "random_objectives": random_objectives,
            "new_on_reset": True}
elif PHASE == 2:
    # Number of features for graph (x, y, boolean end node, boolean objective, radius of obstacle)
    embeder = 5
    use_baseline = True

    args = {"num_nodes": nodes,
            "num_objectives": num_objectives,
            "obstacle_min_distance": obstacle_min_distance,
            "num_obstacles": num_obstacles,
            "random_objectives": random_objectives,
            "new_on_reset": True,
            "obstacle_max_distance": obstacle_max_distance}

elif PHASE == 3:
    # Number of features for graph (x, y, boolean end node, boolean objective, radius of obstacle, x velo, y velo)
    embeder = 7
    use_baseline = True

    args = {"num_nodes": nodes,
            "num_objectives": num_objectives,
            "obstacle_min_distance": obstacle_min_distance,
            "num_obstacles": num_obstacles,
            "random_objectives": random_objectives,
            "new_on_reset": True,
            "obstacle_max_distance": obstacle_max_distance,
            "max_velocity": 0.075}
else:
    embeder = -1
    use_baseline = True


"""
MODEL VARIABLES
"""
# Attention Model Parameters
p = 1  # overall model parameter size hyper-parameter
d_m = p * 128  # Model dimensions for embeddings.
d_c = d_m * context  # Context dimensions.
d_k = p * 128  # Dimensions of key.
h = 8  # Number of attention heads.
N = 3  # Number of encoder layers.
d_ff = p * 128  # Number of encoder hidden layer neural network dimensions.
n_nodes = nodes
use_masking = False

d_v = p * 128  # Dimensions for value matrix.
c = 10.  # Clipping value for probability calculation of decoder.
head_split = True
dropout = 0.
use_graph_emb = True


"""
TRAINING VARIABLES
"""
batches = 1
epochs = 200
device = 'cuda'
lr = 1e-4

torch.autograd.set_detect_anomaly(True)

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
                         lr=lr,
                         gamma=0.99,
                         beta=0.9,
                         gradient_clip=(1., torch.inf),
                         eps=1e-9).to(device)


# load model from state save
# am_REINFORCE.load_state_dict(torch.load("file_path"))

def eval_env(env, render=False, name="", seed=None, use_masking=False, use_baseline=False):
    """
    Evaluates environment using attention model.
        render - Boolean - Render environment and save as gif
        name - String - Used only when rendering for saving gif as name
        seed - Int - Used to reset environment when retraining (DOES NOT CURRENTLY WORK)
        use_masking - Boolean - Mask out nodes that have already been visited
        use_baseline -Boolean - Returns baseline rewards from greedy algorithm
    """
    if seed:
        state, info = env.reset(seed=seed)
    else:
        state, info = env.reset()
    done = False
    env_reward = []
    j = 0
    while not done:
        curr_idx = info["agent_curr_idx"]
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
        ctxt = curr_node

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

        reuse_embedding = False

        action = am_REINFORCE(graph=graph,
                              ctxt=ctxt,
                              mask_emb_graph=mask_emb_graph,
                              mask_dec_graph=mask_dec_graph,
                              reuse_embeding=reuse_embedding,
                              explore=True).numpy()

        state, reward, terminated, truncated, info = env.step(action)

        env_reward.append(reward)
        done = terminated or truncated  # terminated.all() or truncated.all()

        j += 1

    if render:  # get all pngs from folder and combine
        env.render()
        frames = []
        for t in range(0, j + 1):
            image = imageio.v2.imread(f'tmp/render{t}.png')
            os.remove(f'tmp/render{t}.png')
            frames.append(image)
        imageio.mimsave('tmp/{}.gif'.format(name),
                        frames,
                        fps=1,
                        loop=0)
    if use_baseline:
        return env_reward, j, np.sum(env.__getattr__("greedy_rewards"))
    else:
        return env_reward, j

# log packages
# failures and ideas
def log_print_msg(msg):
    """
    Just's prints and logs msgs from training
    """
    print(msg)
    logging.info(msg)


total_params = sum(p.numel() for p in am_REINFORCE.parameters())
log_print_msg("Number of Parameters {}".format(total_params))

rewards_over_epochs = []
render = False
for epoch in range(epochs):
    log_print_msg("EPOCH {}/{}".format(epoch + 1, epochs))

    train = []
    for i in range(0, samples):
        train.append(gym.make(env_filename, **args))
    train = np.asarray(train)

    if render:
        # for some reason rendering still causes memory overflow, do not use it during training, only evaluation.
        for i in range(0, 5):
            print(eval_env(train[i], render=True, name="{}_epoch{}_test_instance{}".format(render_filename, epoch, i)))

    rewards_over_batches = []

    batch_rewards = 0
    for i, batch in enumerate(np.array_split(train, num_batches)):

        log_print_msg(" Train Batch {}/{}".format(i + 1, num_batches))
        log_print_msg(" Memory: {}".format(torch.cuda.memory_allocated()))

        batch_rewards = 0
        baseline_rewards = []

        start = time.time()
        log_print_msg(" Env, Steps Taken, Memory, Model, Baseline")
        for j, env in enumerate(batch):

            if use_baseline: # no baseline for Phase 1
                reward, iters, baseline_reward = eval_env(env, use_masking=use_masking, use_baseline=use_baseline)
                baseline_rewards.append([baseline_reward] * iters)
                batch_rewards += (np.sum(reward) - baseline_reward)

                log_print_msg(
                    " {},{},{},{},{},{}".format(j, iters, torch.cuda.memory_allocated(), np.sum(reward),
                                                baseline_reward,
                                                np.sum(reward) - baseline_reward))
            else:
                reward, iters = eval_env(env, use_masking=use_masking, use_baseline=use_baseline)
                batch_rewards += np.sum(reward)

                log_print_msg(
                    " {},{},{},{},{}".format(j, iters, torch.cuda.memory_allocated(), np.sum(reward),
                                                np.sum(reward)))

            am_REINFORCE.rewards.append(reward)

        end = time.time()

        rewards_over_batches.append(batch_rewards)
        log_print_msg("   Batch Reward: {}; Time: {}".format(batch_rewards, end - start))

        if use_baseline:
            am_REINFORCE.running_G = np.concatenate(baseline_rewards)
        am_REINFORCE.update()

    rewards_over_epochs.append(np.mean(rewards_over_batches))
    log_print_msg("  Epoch Reward: {}".format(rewards_over_epochs[-1]))

    log_print_msg(rewards_over_epochs)
    torch.save(am_REINFORCE.state_dict(), "{}/STATE_SAVE_EPOCH_{}_BASEL_MODEL".format(state_save_foldername, epoch))


