from typing import List, Tuple

import torch.nn as nn
from torch import Tensor


class DeepPolicyNetwork(nn.Module):
    def __init__(self,
                 d_obs: int,
                 n_actions: int,
                 d_ff: List[int],):
        """Deep Reinforcement Learning Policy Network
        parameters
        ----------
        d_obs: int
            Dimension of input observations.
        n_actions: int
            Number of actions to select from.
        d_ff: List[int]
            List of dimensions for hidden layers.
        """
        super().__init__()

        self.d_obs = d_obs
        self.n_actions = n_actions
        assert len(d_ff) >= 1, "List of hidden dimensions is empty!"
        self.d_ff = d_ff
        # Set up input layer
        linear_input = nn.Linear(d_obs, d_ff[0])
        input_relu = nn.Tanh()
        # Add hidden layers
        hidden_layers = []
        for dim_in, dim_out in zip(d_ff[:-1], d_ff[1:]):
            hidden_layers.append(
                nn.Linear(dim_in, dim_out)
            )
            hidden_layers.append(
                nn.ReLU()
            )
        # Set up output layer, using softmax for probability distribution
        linear_output = nn.Linear(d_ff[-1], n_actions)
        output_softmax = nn.Softmax(-1)
        # Assemble everything into a ModuleList. Could have used Sequential instead!
        layers = [linear_input,
                  input_relu,
                  *hidden_layers,
                  linear_output,
                  output_softmax
                  ]
        self.layers = nn.ModuleList(layers)

    def forward(self,
                obs: Tensor,
                ) -> Tensor:
        """Forward to obtain action.
        parameters
        ----------
        obs: Tensor
            Observation Tensor. Expected in shape b x d_obs

        returns
        -------
        Tensor
            Probability distribution over all actions possible. Expected in shape b x n_actions
        """
        output = obs
        for layer in self.layers:
            output = layer(output)
        return output


class StateValueNetwork(nn.Module):
    def __init__(self,
                 d_obs: int,
                 n_actions: int,
                 d_ff: List[int],):
        """Deep Reinforcement Learning Policy Network
        parameters
        ----------
        d_obs: int
            Dimension of input observations.
        n_actions: int
            Number of actions to select from.
        d_ff: List[int]
            List of dimensions for hidden layers.
        """
        super().__init__()

        self.d_obs = d_obs
        self.n_actions = n_actions
        assert len(d_ff) >= 1, "List of hidden dimensions is empty!"
        self.d_ff = d_ff
        # Set up input layer
        linear_input = nn.Linear(d_obs, d_ff[0])
        input_relu = nn.Tanh()
        # Add hidden layers
        hidden_layers = []
        for dim_in, dim_out in zip(d_ff[:-1], d_ff[1:]):
            hidden_layers.append(
                nn.Linear(dim_in, dim_out)
            )
            hidden_layers.append(
                nn.ReLU()
            )
        # Set up output layer, using softmax for probability distribution
        linear_output = nn.Linear(d_ff[-1], n_actions)
        # Assemble everything into a ModuleList. Could have used Sequential instead!
        layers = [linear_input,
                  input_relu,
                  *hidden_layers,
                  linear_output,
                  ]
        self.layers = nn.ModuleList(layers)

    def forward(self,
                obs: Tensor,
                ) -> Tensor:
        """Forward to obtain action.
        parameters
        ----------
        obs: Tensor
            Observation Tensor. Expected in shape b x d_obs

        returns
        -------
        Tensor
            Probability distribution over all actions possible. Expected in shape b x n_actions
        """
        output = obs
        for layer in self.layers:
            output = layer(output)
        return output
