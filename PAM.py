import sys
import warnings

import gymnasium as gym

import combinatorial_problems
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from AttentionModel import REINFORCE, PersistentAttentionModel

import tqdm

import os
import timeit
import argparse
import pickle

def create_parser():
    """Create CLI parameters"""

    parser = argparse.ArgumentParser(description='Attention Model Experiments.', fromfile_prefix_chars='@')

    parser.add_argument('--d_m', type=int, default=128, help='Dimension used within the model for latent space.')
    parser.add_argument('--d_k', type=int, default=128, help='Dimension for key input to attention.')
    parser.add_argument('--d_v', type=int, default=None, help='Dimensions of embedding.')
    parser.add_argument('--h', type=int, default=8, help='Number of heads per attention module.')
    parser.add_argument('--N', type=int, default=3, help='Number of encoding layers.')
    parser.add_argument('--d_ff', type=int, default=128, help='Dimensions of hidden layers within neural networks.')
    parser.add_argument('--c', type=float, default=10., help='Clipping value for attention mechanisms.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability.')
    parser.add_argument('--graph_emb', type=str, default=None, help='Technique used in computing graph embedding for decoder.')
    parser.add_argument('--recompute_emb', action='store_true', default=False, help='Flag indicating whether graph embeddings must be re-computed every time.')

    # Experiment and Environment specifics
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--environments', type=int, default=512, help='Total size of data per epoch.')
    parser.add_argument('--sub_environments', type=int, default=512, help='Number of actual environments to be generated for training. The algorithm will use environments // sub_samples iterations for a single epoch.')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of sample environments per epoch. Must divide --samples without remainder.')
    parser.add_argument('--new_environments', action='store_true', default=False, help='Flag indicating whether environments should be rennewed after every epoch.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to run experiments.')
    parser.add_argument('--asynchronous', action='store_true', default=False, help='Flag indicating whether asynchronous environments will be used.')
    parser.add_argument('--problem', type=str, default="TSP", help='Problem to train on.')
    parser.add_argument('--problem_dimension', type=int, default=2, help='Node dimensions.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to train on. Either cuda or cpu.')
    parser.add_argument('--graph_size', type=int, default=20, help='Number of nodes for combinatorial problem.')
    parser.add_argument('--head_split', action='store_true', default=False, help='Flag indicating whether to split d_v/d_k over attention heads rather than having their own individual heads. Will reduce the actual dimension of these by --heads.')
    parser.add_argument('--max_edge_length', type=float, default=10, help='Maximum filtration length for rips filtration.')
    parser.add_argument('--persistence_dimension', type=int, default=2, help='Maximum dimension for persistence.')

    # Saving information
    parser.add_argument('--file', type=str, default="PAM", help='File root name.')
    parser.add_argument('--dir', type=str, default=".", help='Directory to where to save.')
    parser.add_argument('--exp_label', type=str, default="", help='Experiment number.')

    # Verbosity information
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # Loading model for further training. Both the pt and pkl files must be available!
    parser.add_argument('--continue_training', action='store_true', default=False, help='Flag indicating whether passed file and directory will be used for loading a model to continue training. Both *.pt and *.pkl files must be available!')
    """Retraining
    When using the script for continual training, it is important to note that not all of the parameters
    created for this script will be used. This note serves as a location to know what the required params
    are for continual training:
    --lr
    --environments
    --sub_environments
    --batch_size
    --new_environments
    --epochs
    --asynchronous
    --device
    --graph_size
    # Must be the same used for training
    --problem
    --problem_dimension
    """

    return parser



if __name__ == '__main__':

    # Parsing from bash
    parser = create_parser()
    args = parser.parse_args()

    # Apply assertions
    assert args.sub_environments % args.batch_size == 0, f"Selected number of sub environments ({args.sub_environments}) is not divisible by batch size ({args.batch_size})."
    assert args.environments % args.sub_environments == 0, f"Selected number of environments ({args.environments}) is not divisible by sub environments ({args.sub_environments})."
    assert args.sub_environments >= args.batch_size, f"Specified batch size {args.batch_size} is larger than the number of sub environments {args.sub_environments}!"

    if args.continue_training:
        # Load pkl file's args as args_old
        # set all of relevant model information to current args
        """Relevant parameters to copy:
        --d_m
        --d_k
        --d_v
        --h
        --N
        --d_ff
        --c
        --dropout
        --graph_emb
        --recompute_emb
        --head_split
        """
        pass

    assert args.problem.lower() in ['tsp'], f"Problem {args.problem} not in list of implemented problems."
    # Get batched environments
    # Select environment
    if args.problem.lower() == 'tsp':
        problem_env = "combinatorial_problems/TravelingSalesman-v0"
        # Special parameter for algorithm
        d_c = args.d_m * 2

    # Create file names for experiments
    experiment_params = f"heads_{args.h}_layers_{args.N}_g-embedding_{args.graph_emb}_lr_{args.lr}_envs_{args.environments}_epochs_{args.epochs}_batch-len_{args.batch_size}_problem_{args.problem}_nodes_{args.graph_size}_dim_{args.problem_dimension}"
    exp_file = os.path.join(args.dir, f"experiment_{args.exp_label}_{args.file}_{experiment_params}")
    if args.verbose:
        print(f'Experiment File Root Name: {exp_file}')

    # Create batched environments. There will be sub_environments // batch_size batches
    total_batches = args.sub_environments // args.batch_size # Batches per sub_sample
    batched_envs = [
        gym.vector.make(problem_env,
                        num_nodes=args.graph_size,
                        num_envs=args.batch_size,
                        new_on_reset=args.new_environments,
                        asynchronous=args.asynchronous) for batch in range(total_batches)
    ]
    # Note that actual batches per epoch would be environments // batch_size = (envoronments // sub_environments) * (total_batches)

    # Create training algorithm with attention model
    # Make Agent
    d_v = args.d_v if args.d_v is not None else args.d_m
    agent = PersistentAttentionModel(d_m=args.d_m,
                                     d_c=d_c,
                                     d_k=args.d_k,
                                     h=args.h,
                                     N=args.N,
                                     d_ff=args.d_ff,
                                     n_nodes=args.graph_size, # Not used
                                     embeder=args.problem_dimension,
                                     d_v=d_v,
                                     max_edge_length= args.max_edge_length,
                                     persistence_dimension=args.persistence_dimension,
                                     c=args.c,
                                     head_split=args.head_split,
                                     dropout=args.dropout,
                                     use_graph_emb=args.graph_emb,
                                     batches=args.batch_size,
                                     reuse_graph_emb=not args.recompute_emb).to(args.device)

    # Make REINFORCE
    am_REINFORCE = REINFORCE(policy=agent,
                             optimizer=torch.optim.AdamW,
                             lr=args.lr,
                             gamma=0.99,
                             beta=0.9,
                             gradient_clip=None, #(1., torch.inf),
                             eps=1e-9).to(args.device)
    if args.continue_training:
        pass
        #am_REINFORCE.load_state_dict(torch.load(PATH))

    # Training
    sub_epochs = args.environments // args.sub_environments
    start_time = timeit.default_timer()
    rewards_over_epochs = []
    rewards_std_over_epochs = []
    for epoch in range(args.epochs):
        rewards_over_batches = []
        ''' In the original paper, the authors use 1,250,000 samples per epoch. Due to computational and memory limitations
        we need to be a bit more creative over this. To accomplish this, we use a sub_sample which would be an actual
        number of environments created. Then, during every sub_epoch, an unique seed is given to each environment. 
        this unique seed is incremented every sub_epoch by the number of sub_samples used. The loop runs as many times as
        needed to train the model on the specified samples. Every epoch, the unique seeds get repeated; thus, giving the
        sense of a whole dataset generated at once.'''
        seeds = np.arange(args.sub_environments)
        tqdm_sub_epochs = tqdm.tqdm(range(sub_epochs), file=sys.stdout, disable=not args.verbose)

        for sub_epoch in tqdm_sub_epochs:
            # Reset environment and prepare data loging
            for batch_i, env in enumerate(batched_envs):
                # Do preparation of environments
                seed = seeds[(batch_i * args.batch_size):((batch_i + 1) * args.batch_size)].tolist() if not args.new_environments else None
                state, info = env.reset(seed=seed)
                start_idx = info["agent_start_idx"] # Specific to enviornment
                # Initially, the start and end_idx are the same. Later, it becomes the latest-chosen node/action
                end_idx = start_idx
                done = False
                batch_rewards = 0
                # Run through environment
                while not done:

                    graph_nodes = np.stack(info["nodes"])
                    # Convert to tensor and reshape nodes for obtaining their encoding.
                    # graph -> b x n_nodes x coords
                    graph = torch.FloatTensor(graph_nodes).reshape(args.batch_size, args.graph_size, 2).to(args.device)
                    # The context will be the concatenation of the node embeddings for first and last nodes.
                    # use am_REINFORCE.policy.encode
                    # tmb_emb -> b x nodes x d_m
                    tmp_emb = am_REINFORCE.policy.encoder(graph).detach()
                    # start or end_node -> b x 1 x d_m
                    start_node = tmp_emb[np.arange(args.batch_size), start_idx, :].unsqueeze(1)
                    end_node = tmp_emb[np.arange(args.batch_size), end_idx, :].unsqueeze(1)
                    # The rest of the context is added within the AM algorithm
                    # ctxt -> b x 1 x d_c (2 * d_m)
                    ctxt = torch.cat([start_node, end_node], dim=-1)
                    # For now, I will not use a mask for the embedding input.
                    # mask_emb_graph -> b x 1 x nodes
                    mask_emb_graph = torch.zeros(args.batch_size, 1, args.graph_size).bool().to(args.device)  # Empty Mask!
                    # As nodes are visited, the mask is updated to avoid considering these nodes.
                    # mask_dex_graph -> b x 1 x nodes
                    masks = np.stack(info["mask"])
                    mask_dec_graph = torch.tensor(masks).unsqueeze(1).to(args.device)
                    reuse_embeding = False
                    with torch.cuda.amp.autocast():
                        action = am_REINFORCE(graph=graph,
                                              ctxt=ctxt,
                                              mask_emb_graph=mask_emb_graph,
                                              mask_dec_graph=mask_dec_graph,
                                              reuse_embeding=reuse_embeding,
                                              explore=True,
                                              re_compute_embedding=False).numpy()
                    state, reward, terminated, truncated, info = env.step(action)
                    end_idx = action.squeeze()
                    am_REINFORCE.rewards.append(reward)
                    batch_rewards += reward
                    done = terminated.all() or truncated.all()
                # Maybe validate?
                rewards_over_batches.append(np.array(batch_rewards).mean())
                am_REINFORCE.update()
                # Reset saved graph embeddings
                am_REINFORCE.policy.graph_emb_vect = None
                # Save scores (including validation if done)
                if args.verbose:
                    tqdm_sub_epochs.set_description(f'Batch Score: {rewards_over_batches[-1]:2.5}')
            # Increment seeds to create pseudo-offline training.
            seeds += args.sub_environments
        # Verbosity
        rewards_over_epochs.append(np.mean(np.array(rewards_over_batches)))
        rewards_std_over_epochs.append(np.std(np.array(rewards_over_batches)))
        if args.verbose and epoch % 1 == 0:
            avg_reward = np.mean(rewards_over_epochs[-1:])
            print(f"Epoch: {epoch} with Average Reward {avg_reward} - std: {np.std(rewards_over_batches[:])} for last epoch", )
    end_time = timeit.default_timer()
    # Save collected data
    rewards_to_plot = [[batch_r] for batch_r in rewards_over_epochs]
    rewards_std_to_plot = [[batch_r] for batch_r in rewards_std_over_epochs]
    rewards = pd.DataFrame(rewards_to_plot, columns=["Train"])
    rewards_std = pd.DataFrame(rewards_std_to_plot, columns=["Train"])
    experiment_data = {
        "rewards": rewards,
        "rewards_std": rewards_std,
        "args": args,
        "runtime": start_time - end_time,
    }
    with open(f"{exp_file}.pkl", "wb") as jar:
        pickle.dump(experiment_data, jar)

    # Save model
    torch.save(am_REINFORCE, f"{exp_file}.pt")

    # Close all environments
    for env in batched_envs:
        env.close()
        del env