
# Python Core Libraries
from typing import List, Tuple, Dict, Union
# External Libraries
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Custom data types for checking
integer = np.int64
array = np.ndarray
Observation = Dict[str, array]
Information = Dict[str, Union[array, int, List[int]]]


class TravelingSalesmanEnv(gym.Env):

    metadata = {
        "render_modes": [None],
        "render_fps": None
    }

    def __init__(self,
                 num_nodes: int,
                 max_coord: float = 1.,
                 new_on_reset: bool = True,
                 seed: int = None,
                 render_mode = None):
        """Traveling Salesman Problem
        Classical Traveling Salesman Problem, where the agent will start at a random location (nodes[0])
        and will have to find the fastest path to traverse all nodes. The agent is not allowed to travel
        back to an already-visited node. The graph is assumed to be fully connected.
        parameters
        ----------
        nodes: int
            Total number of nodes, not including agent's initial position.
        max_coord: float
            Maximum x, y coordinates for 2D problem. 0 <= x,y < max_coord
        seed: int
            Random seed
        """
        # Save constructor parameters
        # We add one to account for agent's initial position.
        self.num_nodes = num_nodes
        # Maximum X, Y values, starting at 0
        self.max_coord = max_coord
        # Whether to reset the graph generated at first.
        self.new_on_reset = new_on_reset
        self.seed = seed
        # Cartesian coordinate dimensions
        self.coordinate_dimensions = 2
        # Problem-unique variables. See reset for more information
        self.nodes: array
        self.agent_start_idx: int
        self.agent_curr_idx: int
        self.visited_nodes: List[float]
        self.mask: array
        self.distance: float
        self.trajectory: List[int]

        # Definition of observation and action spaces
        # agent: Index of node the agent currently sits on.
        # distance: Distance currently traveled
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, self.num_nodes, dtype=int),
            "distance": spaces.Box(0, np.inf, dtype=float)
        })
        # Possible actions are node indices
        self.action_space = spaces.Discrete(self.num_nodes)
        # If rendering, code would go after this comment.
        assert render_mode in self.metadata["render_modes"], \
            f"Specified render mode not available from possible modes: {self.metadata['render_modes']}"
        self.render_mode = render_mode

    def _get_observation(self) -> Observation:
        return {
            "agent": np.array([self.agent_curr_idx]),
            "distance": np.array([self.distance]),
        }

    def _get_info(self) -> Information:
        return {
            "mask": self.mask,
            "agent_start_idx": self.agent_start_idx,
            "visited_nodes": self.visited_nodes,
            "trajectory": self.trajectory,
            "nodes": self.nodes,
        }

    def reset(self,
              seed: int = None,
              options: Dict[str, all] = None
              ) -> Tuple[Observation, Information]:
        super().reset(seed=seed)
        # Create initial graph problem nodes
        if self.new_on_reset or not hasattr(self, "nodes"):
            self.nodes = self.np_random.random((self.num_nodes, self.coordinate_dimensions))
            # Convert to maximum dimensions
            self.nodes *= self.max_coord
        # Agent's and Goal's nodes.
        self.agent_start_idx = 0
        # Agent's current node location
        self.agent_curr_idx = self.agent_start_idx
        # List of visited nodes. Will keep track of the distance traveled when the ith node was visited.
        # If Nan, then, it has not been visited, else, it is the distance it took to arrive.
        self.visited_nodes = np.array([
            np.nan if i != self.agent_start_idx else 0 for i in range(self.num_nodes)
        ])
        # Mask for attention mechanisms.
        self.mask = np.logical_not(np.isnan(self.visited_nodes))
        # Agent's Traveled distance
        self.distance = 0.
        # Agent's trajectory. Initially will be the start location. Only index.
        self.trajectory = [self.agent_start_idx, ]

        # Get observations and information
        observation = self._get_observation()
        info = self._get_info()

        # Add code here for renderization.

        return observation, info


    def step(self,
             action: int,
             ) -> Tuple[Observation, float, bool, bool, Information]:
        assert action < self.num_nodes, f"Invalid node {action} given! Should have been < {self.num_nodes}."
        # May need to add an assert or something to keep agent from selecting same node.
        # Compute the distance traveled by going to the new node.
        same_node = self.trajectory[-1] == action
        prev_node = self.nodes[self.trajectory[-1]]
        new_node = self.nodes[action]
        distance = np.linalg.norm(prev_node - new_node)
        # Compute distance and mark node as visited.
        self.distance += distance
        self.visited_nodes[action] = self.distance
        self.trajectory.append(action)
        self.agent_curr_idx = action
        # Compute mask
        self.mask = np.logical_not(np.isnan(self.visited_nodes))
        # Check if agent visited all nodes: if we masked all nodes, then, we visited all nodes!
        terminated = self.mask.all()
        # Truncation never happens as actions do not allow to go off bounds.
        truncated = False
        # Calculate Rewards. Gives high punishment for staying, but no error
        reward = (-1) * distance
        # Calculate observation and information
        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info


    def render(self):
        pass

    def close(self):
        pass