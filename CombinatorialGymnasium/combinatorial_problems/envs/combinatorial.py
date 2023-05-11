
# Python Core Libraries
from typing import List, Tuple, Dict, Union
# External Libraries
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

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
        self.agent_curr_idx = action.squeeze()
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

class Phase1Env(gym.Env):

    metadata = {
        "render_modes": [None],
        "render_fps": None
    }

    def __init__(self,
                 num_nodes: int,
                 num_objectives: int,
                 max_coord: float = 1.,
                 new_on_reset: bool = True,
                 seed: int = None,
                 render_mode = None,
                 random_objectives: bool = False  # creates sample(2, num_objectives) objectives
                 ):
        """Phase 1 Problem
        Find shortest path between a set of objective nodes given a graph. The agent will start at a random location,
        where two sets of nodes will be returned (index 0 for list of objective nodes, and index 1 for non-objective
        nodes). The graph is assumed to be fully connected.
        parameters
        ----------
        num_nodes: int
            Total number of nodes, not including agent's initial position.
        max_coord: float
            Maximum x, y coordinates for 2D problem. 0 <= x,y < max_coord
        num_objectives: int
            Maximum number of objectives in the graph.
        new_on_reset: bool
            Upon reset, randomly generate new nodes
        render_mode: bool
            Render environement for plotting
        random_objectives:
            Samples number of objectives from (2, num_objectives)
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
        self.nodes_objective: array
        self.nodes_non_objective: array
        self.mask: array
        self.agent_start_idx: int
        self.agent_curr_idx: int
        self.distance: float
        self.current_reward = 0
        self.trajectory: List[int]
        self.random_objectives = random_objectives
        self.num_objectives = num_objectives
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
            "mask": self.mask.reshape(1,-1 ),
            "agent_start_idx": self.agent_start_idx,
            "trajectory": self.trajectory,
            "nodes": self.nodes,
            "agent_end_idx": self.end_idx,
            "agent_obj_idx": self.obj_idx,
            "agent_non_obj_idx": self.non_obj_idx,
            "agent_curr_idx": self.agent_curr_idx,
            "agent_visited_idx": self.visited_nodes
        }

    def _get_nearest_obj_idx(self):
        #curr = self.nodes[self.agent_curr_idx][0:2]
        #idx = self.obj_idx[self.visited_nodes[self.obj_idx] == False]
        pass
        # np.argmin([np.linalg.norm(nodes[4][0:2]-node[0:2]) for node in nodes[np.concatenate((np.arange(0, 4), np.arange(4+1, len(nodes))))]])

    def reset(self,
              seed: int = None,
              options: Dict[str, all] = None
              ) -> Tuple[Observation, Information]:
        super().reset(seed=seed)
        # Create initial graph problem nodes
        if self.new_on_reset or not hasattr(self, "nodes"):
            nodes = self.np_random.random((self.num_nodes, self.coordinate_dimensions))
            end_node = np.zeros(self.num_nodes)
            objectives = np.zeros(self.num_nodes)
            if self.random_objectives:

                self.num_objectives = self.np_random.choice(range(2, self.num_objectives))

            objectives[self.np_random.choice(range(1, self.num_nodes), self.num_objectives, replace=False)] = 1
            if self.np_random.uniform(0, 1) <= 0.005:  # start node is end node
                idx = 0
            else:  # end node is non-objective node
                idx = self.np_random.choice(np.where(objectives[1:] == 0)[0]+1)  # adjust to not include start node in random choice, but then add one after to account for start node

            end_node[idx] = 1
            self.end_idx = idx
            self.obj_idx = np.where(objectives == 1)[0]
            self.non_obj_idx = np.where(objectives != 1)[0]
            self.nodes = np.hstack((nodes, objectives.reshape(-1, 1), end_node.reshape(-1, 1)))

        # Agent's and Goal's nodes.
        self.agent_start_idx = 0
        self.iter = 0
        # Agent's current node location
        self.agent_curr_idx = self.agent_start_idx
        # List of visited nodes. Will keep track of the distance traveled when the ith node was visited.
        # Agent's Traveled distance
        self.distance = 0.
        # Agent's trajectory. Initially will be the start location. Only index.
        self.trajectory = [self.agent_start_idx, ]
        self.visited_nodes = np.asarray([False]*self.num_nodes)
        self.visited_nodes[0] = 1  # start at node 0
        self.mask = np.copy(self.visited_nodes)
        self.mask[self.end_idx] = False
        # Get observations and information
        observation = self._get_observation()
        info = self._get_info()
        self.start = True
        self.count = 0

        # Add code here for renderization.

        return observation, info

    def render(self):
        plt.close('all')
        if self.start:
            self.start = False
            self.set_up_render()
            return
        self.count += 1
        start = plt.scatter(self.nodes[self.agent_start_idx][0], self.nodes[self.agent_start_idx][1], marker="D")  # start
        end = plt.scatter(self.nodes[self.end_idx][0], self.nodes[self.end_idx][1], marker="X")
        obj = plt.scatter(self.nodes[self.obj_idx][:, 0], self.nodes[self.obj_idx][:, 1], marker="*")
        non_obj = plt.scatter(self.nodes[self.non_obj_idx][:, 0], self.nodes[self.non_obj_idx][:, 1], marker=".")
        plt.legend((start, end, obj, non_obj),
                   ('Start', 'End', 'Objective', 'Non-Objective'),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=2,
                   fontsize=8)
        prev = 0
        alphas = np.linspace(0.2, 0.6, len(self.trajectory))
        for i, nxt in enumerate(self.trajectory[1:]):
            plt.arrow(self.nodes[prev][0], self.nodes[prev][1], self.nodes[nxt][0] - self.nodes[prev][0],
                      self.nodes[nxt][1] - self.nodes[prev][1], width=0.001, alpha=alphas[i], color='black')
            prev = nxt
        plt.suptitle("Phase1 - TimeStep: {} - Action Reward: {}".format(self.count, np.around(self.current_reward, 2)))
        plt.savefig("tmp/render{}.png".format(self.count), dpi=300)


    def set_up_render(self):

        start = plt.scatter(self.nodes[self.agent_start_idx][0], self.nodes[self.agent_start_idx][1],
                            marker="D")  # start
        end = plt.scatter(self.nodes[self.end_idx][0], self.nodes[self.end_idx][1], marker="X")
        obj = plt.scatter(self.nodes[self.obj_idx][:, 0], self.nodes[self.obj_idx][:, 1], marker="*")
        non_obj = plt.scatter(self.nodes[self.non_obj_idx][:, 0], self.nodes[self.non_obj_idx][:, 1], marker=".")
        plt.legend((start, end, obj, non_obj),
                   ('Start', 'End', 'Objective', 'Non-Objective'),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=2,
                   fontsize=8)

        plt.suptitle("Phase1 - TimeStep: {} - Action Reward: {}".format(self.count, np.around(self.current_reward, 2)))
        plt.savefig("tmp/render{}.png".format(self.count), dpi=300)

    def step(self,
             action: int,
             ) -> Tuple[Observation, float, bool, bool, Information]:
        assert action < self.num_nodes, f"Invalid node {action} given! Should have been < {self.num_nodes}."
        # Avoid extra dimensions that mess with Gymnasium.
        action = action.squeeze()
        # May need to add an assert or something to keep agent from selecting same node.
        # Compute the distance traveled by going to the new node.
        prev_node = self.nodes[self.trajectory[-1]]
        new_node = self.nodes[action]
        distance = np.linalg.norm(prev_node[0:2] - new_node[0:2])
        # Compute distance and mark node as visited.
        self.distance += distance
        if self.visited_nodes[action] == 0 and (action in self.obj_idx or (np.all(self.visited_nodes[self.obj_idx]) and action == self.end_idx)):
            vis_obj_reward = 10 / (len(self.obj_idx)+1)
            self.obj_idx = np.setdiff1d(self.obj_idx[self.obj_idx != action], action)  # get rid of obj_idx
            t = np.copy(self.nodes[action])
            t[2] = 0
            self.nodes[action] = np.copy(t)
            self.non_obj_idx = np.concatenate((self.non_obj_idx, [action]))

        else:
            vis_obj_reward = 0

        self.visited_nodes[action] = 1
        self.trajectory.append(action)
        self.agent_curr_idx = action
        # Compute mask
        self.mask = np.copy(self.visited_nodes)
        self.mask[self.end_idx] = False

        # Check if agent visited all objective nodes
        terminated = np.all(self.visited_nodes[self.obj_idx]) and action == self.end_idx
        if terminated or self.iter == 49:
            non_vis_obj_penalty = - 10 * np.sum(np.invert(self.visited_nodes[self.obj_idx]))
        else:
            non_vis_obj_penalty = 0
        self.iter += 1
        # Truncation never happens as actions do not allow to go off bounds.
        truncated = False
        # Calculate Rewards. Gives high punishment for staying, but no error
        time_penalty = - 0.1
        reward = (-1) * distance + time_penalty + vis_obj_reward + non_vis_obj_penalty
        self.current_reward = reward

        # Calculate observation and information
        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def close(self):
        pass



class Phase2Env(gym.Env):

    metadata = {
        "render_modes": [None],
        "render_fps": None
    }

    def __init__(self,
                 num_nodes: int,
                 num_objectives: int,
                 max_coord: float = 1.,
                 new_on_reset: bool = True,
                 seed: int = None,
                 render_mode = None,
                 random_objectives: bool = False,  # creates sample(2, num_objectives) objectives
                 num_obstacles: int = 2,
                 random_obstacles: bool = False,
                 dynamic_obstacles: bool = False,
                 dynamic_probability_1: float = 0.005,
                 dynamic_probability_2: float = 0.02,
                 obstacle_min_distance: float = 0.10,
                 obstacle_max_distance: float = 0.15,
                 time_penalty: float = -1.0,
                 visited_reward: float = 50.0
                 ):
        """Phase 2 Problem
        Find shortest path between a set of objective nodes given a graph. The agent will start at a random location,
        where two sets of nodes will be returned (index 0 for list of objective nodes, and index 1 for non-objective
        nodes). The graph is assumed to be fully connected. Phase 2 differs from Phase 1 in that some nodes
        now are obstacles with negative reward radius (1/x), along with dynamic obstacles. Agent also contains
        greedy baseline that selects the nearest objective at each timestep.
        parameters
        ----------
        num_nodes: int
            Total number of nodes, not including agent's initial position.
        max_coord: float
            Maximum x, y coordinates for 2D problem. 0 <= x,y < max_coord
        num_objectives: int
            Maximum number of objectives in the graph.
        new_on_reset: bool
            Upon reset, randomly generate new nodes
        render_mode: bool
            Render environement for plotting
        random_objectives:
            Samples number of objectives from (2, num_objectives)
        num_obstacles:
            Maximum number of obstacles in the graph.
        random_obstacles:
            Samples number of obstacles from (2, num_obstacles)
        dynamic_obstacles:
            Allows for obstacles to become non-objective nodes, and non-objective nodes to become obstacles
        dynamic_probability_1:
            Probability that a non-obstacle becomes an obstacle
        dynamic_probability_2:
            Probability that an obstacle becomes non-obstacle
        obstacle_min_distance:
            Minimum distance bound from an obstacle before the negative reward does not count
        obstacle_max_distance:
            Maximum distance bound from an obstacle before the negative reward does not count.
        seed: int
            Random seed
        time_penalty: float
            The penalty added at each timestep.
        visited_reward: float
            The reward for visiting objective nodes (only the first time).
        """
        self.num_obstacles = num_obstacles
        self.random_obstacles = random_obstacles
        self.obstacle_max_distance = obstacle_max_distance
        self.dynamic_probability_1 = dynamic_probability_1
        self.dynamic_probability_2 = dynamic_probability_2
        self.dynamic_obstacles = dynamic_obstacles
        self.obstacle_min_distance = obstacle_min_distance
        self.time_penalty = time_penalty
        self.visited_reward = visited_reward
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
        self.nodes_objective: array
        self.nodes_non_objective: array
        self.mask: array
        self.agent_start_idx: int
        self.agent_curr_idx: int
        self.distance: float
        self.current_reward = 0
        self.trajectory: List[int]
        self.random_objectives = random_objectives
        self.num_objectives = num_objectives

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
            "mask": self.mask.reshape(1,-1 ),
            "agent_start_idx": self.agent_start_idx,
            "trajectory": self.trajectory,
            "nodes": self.nodes,
            "agent_end_idx": self.end_idx,
            "agent_obj_idx": self.obj_idx,
            "agent_non_obj_idx": self.non_obj_idx,
            "agent_curr_idx": self.agent_curr_idx,
            "agent_visited_idx": self.visited_nodes,
            "agent_obstacle_idx": self.obstacle_idx,
        }

    def reset(self,
              seed: int = None,
              options: Dict[str, all] = None
              ) -> Tuple[Observation, Information]:
        super().reset(seed=seed)
        # Create initial graph problem nodes
        if self.new_on_reset or not hasattr(self, "nodes"):
            nodes = self.np_random.random((self.num_nodes, self.coordinate_dimensions))
            end_node = np.zeros(self.num_nodes)  # boolean encoding
            objectives = np.zeros(self.num_nodes)  # boolean encoding
            obstacles = np.zeros(self.num_nodes)  # boolean encoding

            if self.random_objectives:
                self.num_objectives = self.np_random.choice(range(2, self.num_objectives))

            if self.random_obstacles:
                self.num_obstacles = self.np_random.choice(range(2, self.num_obstacles))

            objectives[self.np_random.choice(range(1, self.num_nodes), self.num_objectives, replace=False)] = 1
            if self.np_random.uniform(0, 1) <= 0.5:  # start node is end node
                idx = 0
            else:  # end node is non-objective node
                # adjust to not include start node in random choice, but then add one after to account for start node
                idx = self.np_random.choice(np.where(objectives[1:] == 0)[0]+1)
            end_node[idx] = 1
            self.end_idx = idx

            self.obj_idx = np.where(objectives == 1)[0]
            if self.end_idx == 0:
                self.non_obj_idx = np.where(objectives != 1)[0]
                idx = self.np_random.choice(self.non_obj_idx[np.where(self.non_obj_idx != self.end_idx)[0]],
                                            self.num_obstacles)
            else:
                self.non_obj_idx = np.where(objectives != 1)[0][1:]

                idx = self.np_random.choice(self.non_obj_idx[np.where(self.non_obj_idx != self.end_idx)[0]],
                                            self.num_obstacles)
            self.non_obj_idx = np.setdiff1d(self.non_obj_idx, idx)

            obstacles[idx] = self.np_random.uniform(self.obstacle_min_distance, self.obstacle_max_distance, len(idx))
            self.obstacle_idx = idx

            self.nodes = np.hstack((nodes, objectives.reshape(-1, 1), end_node.reshape(-1, 1), obstacles.reshape(-1, 1)))


        # Agent's and Goal's nodes.
        self.agent_start_idx = 0
        self.iter = 0
        # Agent's current node location
        self.agent_curr_idx = self.agent_start_idx
        # List of visited nodes. Will keep track of the distance traveled when the ith node was visited.
        # Agent's Traveled distance
        self.distance = 0.
        # Agent's trajectory. Initially will be the start location. Only index.
        self.trajectory = [self.agent_start_idx, ]
        self.visited_nodes = np.asarray([False]*self.num_nodes)
        self.visited_nodes[0] = 1  # start at node 0
        self.mask = np.copy(self.visited_nodes)
        self.mask[self.end_idx] = False
        # Get observations and information
        observation = self._get_observation()
        info = self._get_info()
        self.start = True
        self.count = 0

        self.greedy_unvisited_obj = np.copy(self.obj_idx)
        self.greedy_curr_node = self.agent_start_idx
        self.greedy_rewards = []
        return observation, info

    def render(self):
        """
        Plots current only the current timestep of the environment.
        """
        plt.close('all')
        if self.start:  # for first render setup and initialize everything
            self.start = False
            self.set_up_render()
            return
        self.count += 1
        plt.suptitle("Phase2 - TimeStep: {} - Action Reward: {}".format(self.count, np.around(self.current_reward, 2)))

        start = plt.scatter(self.nodes[self.agent_start_idx][0], self.nodes[self.agent_start_idx][1],
                            marker="D")
        end = plt.scatter(self.nodes[self.end_idx][0], self.nodes[self.end_idx][1], marker="X")
        obj = plt.scatter(self.nodes[self.obj_idx][:, 0], self.nodes[self.obj_idx][:, 1], marker="*")
        non_obj = plt.scatter(self.nodes[self.non_obj_idx][:, 0], self.nodes[self.non_obj_idx][:, 1], marker=".")
        obs = plt.scatter(self.nodes[self.obstacle_idx][:, 0], self.nodes[self.obstacle_idx][:, 1], marker="^")

        plt.legend((start, end, obj, non_obj, obs),
                   ('Start', 'End', 'Objective', 'Non-Objective', "Turrets"),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=2,
                   fontsize=8)

        for obstacle in self.obstacle_idx:
            position = self.nodes[obstacle][:2]
            c = plt.Circle(position, self.nodes[obstacle][4], color='r', fill=False)
            plt.gca().add_artist(c)

        prev = 0
        alphas = np.linspace(0.2, 0.6, len(self.trajectory))
        for i, nxt in enumerate(self.trajectory[1:]):
            plt.arrow(self.nodes[prev][0], self.nodes[prev][1], self.nodes[nxt][0] - self.nodes[prev][0],
                      self.nodes[nxt][1] - self.nodes[prev][1], width=0.001, alpha=alphas[i], color='black')
            prev = nxt
        plt.savefig("tmp/render{}.png".format(self.count), dpi=300)


    def set_up_render(self):
        """
        Initializes everything for rendering each plot.
        Scatter plots each type of node along with legend, and turrets with their penalty circle.
        """
        start = plt.scatter(self.nodes[self.agent_start_idx][0], self.nodes[self.agent_start_idx][1],
                            marker="D")  # start
        end = plt.scatter(self.nodes[self.end_idx][0], self.nodes[self.end_idx][1], marker="X")
        obj = plt.scatter(self.nodes[self.obj_idx][:, 0], self.nodes[self.obj_idx][:, 1], marker="*")
        non_obj = plt.scatter(self.nodes[self.non_obj_idx][:, 0], self.nodes[self.non_obj_idx][:, 1], marker=".")
        obs = plt.scatter(self.nodes[self.obstacle_idx][:, 0], self.nodes[self.obstacle_idx][:, 1], marker="^")
        plt.legend((start, end, obj, non_obj, obs),
                   ('Start', 'End', 'Objective', 'Non-Objective', "Turrets"),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=2,
                   fontsize=8)

        for obstacle in self.obstacle_idx:
            position = self.nodes[obstacle][:2]
            c = plt.Circle(position, self.nodes[obstacle][4], color='r', fill=False)
            plt.gca().add_artist(c)
        plt.suptitle("Phase2 - TimeStep: {} - Action Reward: {}".format(self.count, np.around(self.current_reward, 2)))
        plt.savefig("tmp/render{}.png".format(self.count), dpi=300)

    @staticmethod
    def _get_points(prev_node, next_node):
        """
        Creates a line of points between prev_node and next_node
        """
        if np.all(prev_node == next_node):
            points = None
        else:
            points = np.stack((next_node, prev_node))[:, 0:2].T
            coefficients = np.polyfit(points[0, :], points[1, :], 1)
            fill = np.linspace(np.min(points[0, :]), np.max(points[0, :]))
            line = np.poly1d(coefficients)(fill)
            points = np.vstack((fill, line)).T
        return points

    def _get_obstacle_penalties(self, points, next_node):
        """
        Calculates distances between all the points and obstacles. Calculates penalty if below obstacle
        radius.

        points : Array
            Line of points
        next_node :
            Used if points is None
        """
        distances = []
        obstacles = self.nodes[self.obstacle_idx]
        if len(obstacles) == 0:
            obstacle_penalty = 0
        else:
            if points is not None:
                for obstacle in obstacles:
                    distances.append(np.vstack([np.linalg.norm(points - np.tile(obstacle[0:2], (50, 1)), axis=-1),
                                                np.tile(obstacle[4], (50,))]).T)
                distances = np.vstack(distances)
            else:
                for obstacle in obstacles:
                    distances.append([np.linalg.norm(next_node[0:2] - obstacle[0:2]), obstacle[4]])
                distances = np.asarray(distances)
            distances = distances[distances[:, 0] < distances[:, 1]][:, 0] + 1e-3  # add 1e-3 for 0 (do not want infty)
            obstacle_penalty = - np.sum(10.0 / distances)  # my heuristic
        return obstacle_penalty

    def _step_greedy(self):
        """
        Performs step in environment using greedy baseline.
        Greedy first calculates distances between current pos and all objectives, selects minimum as next position.
        Calculate distances between all obstacles and line points between next and current position for turrent
        penalties.
        """
        if len(self.greedy_unvisited_obj) == 0 and self.greedy_curr_node == self.end_idx:
            greedy_reward = 0
        else:
            if len(self.greedy_unvisited_obj) == 0:  # all objective points are visited, just go to end node:
                min_distance = np.linalg.norm(self.nodes[self.greedy_curr_node][0:2] - self.nodes[self.end_idx][0:2])
                greedy_next_node = self.end_idx
                points = self._get_points(self.nodes[self.greedy_curr_node], self.nodes[greedy_next_node])
                self.greedy_curr_node = self.end_idx
            else:
                # find distances between all non visited greedy objectives
                distances = []
                for node in self.greedy_unvisited_obj:
                    tmp_node = self.nodes[node]
                    distances.append(np.linalg.norm(self.nodes[self.greedy_curr_node][0:2] - tmp_node[0:2]))
                idx = np.argmin(distances)

                min_distance = np.min(distances)
                greedy_next_node = self.nodes[self.greedy_unvisited_obj[idx]]
                points = self._get_points(self.nodes[self.greedy_curr_node], greedy_next_node)
                self.greedy_curr_node = self.greedy_unvisited_obj[idx]
                self.greedy_unvisited_obj = np.concatenate(
                    (self.greedy_unvisited_obj[0:idx], self.greedy_unvisited_obj[idx + 1:]))

            vis_obj_reward = self.visited_reward / (self.num_objectives + 1)  # divide reward to normalize

            # calculate distances between all obstacles and points
            obstacle_penalty = self._get_obstacle_penalties(points, greedy_next_node)

            greedy_reward = (-1) * min_distance + self.time_penalty + vis_obj_reward + obstacle_penalty

        self.greedy_rewards.append(greedy_reward)

    def _step_model(self, prev_node, next_node, action):
        """
        Performs step in environment for the model.
        Greedy first calculates distances between current pos and all objectives, selects minimum as next position.
        Calculate distances between all obstacles and line points between next and current position for turrent
        penalties.
        """
        points = self._get_points(prev_node, next_node)
        distance = np.linalg.norm(prev_node[0:2] - next_node[0:2])
        self.distance += distance
        # if node has not been visited and is an objective or all visited nodes have been reached and action is end node
        if self.visited_nodes[action] == 0 and (
                action in self.obj_idx or (np.all(self.visited_nodes[self.obj_idx]) and action == self.end_idx)):

            self.obj_idx = np.setdiff1d(self.obj_idx[self.obj_idx != action], action)  # get rid of obj_idx
            t = np.copy(self.nodes[action])
            t[2] = 0
            self.nodes[action] = np.copy(t)
            self.non_obj_idx = np.concatenate((self.non_obj_idx, [action]))

            vis_obj_reward = self.visited_reward / (self.num_objectives + 1)  # divide reward to normalize
        else:
            vis_obj_reward = 0

        self.visited_nodes[action] = 1
        self.trajectory.append(action)
        self.agent_curr_idx = action

        self.mask = np.copy(self.visited_nodes)  # Compute mask
        self.mask[self.end_idx] = False

        obstacle_penalty = self._get_obstacle_penalties(points, next_node)

        # Check if agent visited all objective nodes
        terminated = np.all(self.visited_nodes[self.obj_idx]) and action == self.end_idx
        if terminated or self.iter == 49:  # 49  is maximum number of iterations in evironment
            # heavily penalize model for not visiting all nodes in time
            non_vis_obj_penalty = - 1000 * np.sum(np.invert(self.visited_nodes[self.obj_idx]))
        else:
            non_vis_obj_penalty = 0

        return distance, vis_obj_reward, obstacle_penalty, non_vis_obj_penalty, terminated

    def _update_turrets_non_objectives(self):
        """
        Performs calculations for randomly switching non-objective nodes to objective and visa versa.
        """
        if self.end_idx in self.non_obj_idx:
            idx2 = self.non_obj_idx[np.where(self.non_obj_idx != self.end_idx)[0]]
            non_obj_switch_idx = idx2[self.np_random.uniform(0, 1, len(idx2)) < self.dynamic_probability_1]
        else:
            non_obj_switch_idx = self.non_obj_idx[self.np_random.uniform(0, 1, len(self.non_obj_idx)) < self.dynamic_probability_1]

        if len(non_obj_switch_idx) != 0:
            t = np.copy(self.nodes[non_obj_switch_idx])
            t[:, -1] = self.np_random.uniform(self.obstacle_min_distance, self.obstacle_max_distance, len(t))
            self.nodes[non_obj_switch_idx] = np.copy(t)
            self.obstacle_idx = np.concatenate((self.obstacle_idx, non_obj_switch_idx))
            self.non_obj_idx = np.setdiff1d(self.non_obj_idx, non_obj_switch_idx)

        obst_switch_idx = self.obstacle_idx[self.np_random.uniform(0, 1, len(self.obstacle_idx)) < self.dynamic_probability_2]
        if len(obst_switch_idx) != 0:
            t = np.copy(self.nodes[obst_switch_idx])
            t[:, -1] = 0
            self.nodes[obst_switch_idx] = np.copy(t)
            self.obstacle_idx = np.setdiff1d(self.obstacle_idx, obst_switch_idx)
            self.non_obj_idx = np.concatenate((self.non_obj_idx, obst_switch_idx))

    def step(self,
             action: int,
             ) -> Tuple[Observation, float, bool, bool, Information]:
        assert action < self.num_nodes, f"Invalid node {action} given! Should have been < {self.num_nodes}."
        # take care of greedy baseline
        self._step_greedy()
        prev_node = self.nodes[self.trajectory[-1]]
        action = action.squeeze()
        next_node = self.nodes[action]
        self._step_model(prev_node, next_node, action)

        distance, vis_obj_reward, obstacle_penalty, non_vis_obj_penalty, terminated = self._step_model(prev_node,
                                                                                                       next_node,
                                                                                                       action)

        reward = (-1) * distance + self.time_penalty + vis_obj_reward + non_vis_obj_penalty + obstacle_penalty
        self.current_reward = reward

        self._update_turrets_non_objectives()

        # Truncation never happens as actions do not allow to go off bounds.
        truncated = False
        self.iter += 1

        # Calculate observation and information
        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def close(self):
        pass





class Phase3Env(gym.Env):

    metadata = {
        "render_modes": [None],
        "render_fps": None
    }

    def __init__(self,
                 num_nodes: int,
                 num_objectives: int,
                 max_coord: float = 1.,
                 new_on_reset: bool = True,
                 seed: int = None,
                 render_mode = None,
                 random_objectives: bool = False,  # creates sample(2, num_objectives) objectives
                 num_obstacles: int = 2,
                 random_obstacles: bool = False,
                 dynamic_obstacles: bool = False,
                 dynamic_probability_1: float = 0.005,
                 dynamic_probability_2: float = 0.02,
                 obstacle_min_distance: float = 0.10,
                 obstacle_max_distance: float = 0.15,
                 velo_max: float = 0.075
                 ):
        """Phase 2 Problem
        Find shortest path between a set of objective nodes given a graph. The agent will start at a random location,
        where two sets of nodes will be returned (index 0 for list of objective nodes, and index 1 for non-objective
        nodes). The graph is assumed to be fully connected. Phase 2 differs from Phase 1 in that some nodes
        now are obstacles with negative reward radius (1/x), along with dynamic obstacles. In addition, obstacles
        are now moving with some given velocity.
        parameters
        ----------
        num_nodes: int
            Total number of nodes, not including agent's initial position.
        max_coord: float
            Maximum x, y coordinates for 2D problem. 0 <= x,y < max_coord
        num_objectives: int
            Maximum number of objectives in the graph.
        new_on_reset: bool
            Upon reset, randomly generate new nodes
        render_mode: bool
            Render environement for plotting
        random_objectives:
            Samples number of objectives from (2, num_objectives)
        num_obstacles:
            Maximum number of obstacles in the graph.
        random_obstacles:
            Samples number of obstacles from (2, num_obstacles)
        dynamic_obstacles:
            Allows for obstacles to become non-objective nodes, and non-objective nodes to become obstacles
        dynamic_probability_1:
            Probability that a non-obstacle becomes an obstacle
        dynamic_probability_2:
            Probability that an obstacle becomes non-obstacle
        obstacle_min_distance:
            Minimum distance bound from an obstacle before the negative reward does not count
        obstacle_max_distance:
            Maximum distance bound from an obstacle before the negative reward does not count.
        seed: int
            Random seed
        """
        self.num_obstacles = num_obstacles
        self.random_obstacles = random_obstacles
        self.obstacle_max_distance = obstacle_max_distance
        self.dynamic_probability_1 = dynamic_probability_1
        self.dynamic_probability_2 = dynamic_probability_2
        self.dynamic_obstacles = dynamic_obstacles
        self.obstacle_min_distance = obstacle_min_distance
        self.velo_max = velo_max
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
        self.nodes_objective: array
        self.nodes_non_objective: array
        self.mask: array
        self.agent_start_idx: int
        self.agent_curr_idx: int
        self.distance: float
        self.current_reward = 0
        self.trajectory: List[int]
        self.random_objectives = random_objectives
        self.num_objectives = num_objectives

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
            "mask": self.mask.reshape(1,-1 ),
            "agent_start_idx": self.agent_start_idx,
            "trajectory": self.trajectory,
            "nodes": self.nodes,
            "agent_end_idx": self.end_idx,
            "agent_obj_idx": self.obj_idx,
            "agent_non_obj_idx": self.non_obj_idx,
            "agent_curr_idx": self.agent_curr_idx,
            "agent_visited_idx": self.visited_nodes,
            "agent_obstacle_idx": self.obstacle_idx,
        }

    def _get_nearest_obj_idx(self):
        #curr = self.nodes[self.agent_curr_idx][0:2]
        #idx = self.obj_idx[self.visited_nodes[self.obj_idx] == False]
        pass
        # np.argmin([np.linalg.norm(nodes[4][0:2]-node[0:2]) for node in nodes[np.concatenate((np.arange(0, 4), np.arange(4+1, len(nodes))))]])

    def reset(self,
              seed: int = None,
              options: Dict[str, all] = None
              ) -> Tuple[Observation, Information]:
        super().reset(seed=seed)
        # Create initial graph problem nodes
        if self.new_on_reset or not hasattr(self, "nodes"):
            nodes = self.np_random.random((self.num_nodes, self.coordinate_dimensions))
            end_node = np.zeros(self.num_nodes)
            objectives = np.zeros(self.num_nodes)
            obstacles = np.zeros(self.num_nodes)
            velocities = np.zeros((self.num_nodes, 2))
            if self.random_objectives:

                self.num_objectives = self.np_random.choice(range(2, self.num_objectives))

            if self.random_obstacles:
                self.num_obstacles = self.np_random.choice(range(2, self.num_obstacles))

            objectives[self.np_random.choice(range(1, self.num_nodes), self.num_objectives, replace=False)] = 1
            if self.np_random.uniform(0, 1) <= 0.5:  # start node is end node
                idx = 0
            else:  # end node is non-objective node
                idx = self.np_random.choice(np.where(objectives[1:] == 0)[0]+1)  # adjust to not include start node in random choice, but then add one after to account for start node

            end_node[idx] = 1
            self.end_idx = idx
            self.obj_idx = np.where(objectives == 1)[0]
            if self.end_idx == 0:
                self.non_obj_idx = np.where(objectives != 1)[0]
                idx = self.np_random.choice(self.non_obj_idx[np.where(self.non_obj_idx != self.end_idx)[0]],
                                            self.num_obstacles)
            else:
                self.non_obj_idx = np.where(objectives != 1)[0][1:]

                idx = self.np_random.choice(self.non_obj_idx[np.where(self.non_obj_idx != self.end_idx)[0]],
                                            self.num_obstacles)


            obstacles[idx] = self.np_random.uniform(self.obstacle_min_distance, self.obstacle_max_distance, len(idx))
            for i in idx:
                velo = self.np_random.uniform(-self.velo_max, self.velo_max,  2)
                while np.any(nodes[i] + velo > 1) or np.any(nodes[i] + velo < 0):
                    velo = self.np_random.uniform(-self.velo_max, self.velo_max, 2)
                velocities[i] = velo
            self.obstacle_idx = idx
            self.non_obj_idx = np.setdiff1d(self.non_obj_idx, idx)

            self.nodes = np.hstack((nodes, objectives.reshape(-1, 1), end_node.reshape(-1, 1), obstacles.reshape(-1, 1), velocities))

        self.greedy_unvisited_obj = np.copy(self.obj_idx)
        self.greedy_rewards = []
        # Agent's and Goal's nodes.
        self.agent_start_idx = 0
        self.iter = 0
        # Agent's current node location
        self.agent_curr_idx = self.agent_start_idx
        # List of visited nodes. Will keep track of the distance traveled when the ith node was visited.
        # Agent's Traveled distance
        self.distance = 0.
        # Agent's trajectory. Initially will be the start location. Only index.
        self.trajectory = [self.agent_start_idx, ]
        self.visited_nodes = np.asarray([False]*self.num_nodes)
        self.visited_nodes[0] = 1  # start at node 0
        self.mask = np.copy(self.visited_nodes)
        self.mask[self.end_idx] = False
        # Get observations and information
        observation = self._get_observation()
        info = self._get_info()
        self.start = True
        self.count = 0

        # Add code here for renderization.

        return observation, info

    def render(self):
        plt.close('all')
        if self.start:
            self.start = False
            self.set_up_render()
            return
        self.count += 1
        plt.suptitle("Phase2 - TimeStep: {} - Action Reward: {}".format(self.count, np.around(self.current_reward, 2)))
        start = plt.scatter(self.nodes[self.agent_start_idx][0], self.nodes[self.agent_start_idx][1],
                            marker="D")  # start
        end = plt.scatter(self.nodes[self.end_idx][0], self.nodes[self.end_idx][1], marker="X")
        obj = plt.scatter(self.nodes[self.obj_idx][:, 0], self.nodes[self.obj_idx][:, 1], marker="*")
        non_obj = plt.scatter(self.nodes[self.non_obj_idx][:, 0], self.nodes[self.non_obj_idx][:, 1], marker=".")
        obs = plt.scatter(self.nodes[self.obstacle_idx][:, 0], self.nodes[self.obstacle_idx][:, 1], marker="^")
        plt.legend((start, end, obj, non_obj, obs),
                   ('Start', 'End', 'Objective', 'Non-Objective', "Turrets"),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=2,
                   fontsize=8)

        for obstacle in self.obstacle_idx:
            position = self.nodes[obstacle][:2]
            c = plt.Circle(position, self.nodes[obstacle][4], color='r', fill=False)
            plt.gca().add_artist(c)

        prev = 0
        alphas = np.linspace(0.2, 0.6, len(self.trajectory))
        for i, nxt in enumerate(self.trajectory[1:]):
            plt.arrow(self.nodes[prev][0], self.nodes[prev][1], self.nodes[nxt][0] - self.nodes[prev][0],
                      self.nodes[nxt][1] - self.nodes[prev][1], width=0.001, alpha=alphas[i], color='black')
            prev = nxt
        plt.savefig("tmp/render{}.png".format(self.count), dpi=300)


    def set_up_render(self):

        start = plt.scatter(self.nodes[self.agent_start_idx][0], self.nodes[self.agent_start_idx][1],
                            marker="D")  # start
        end = plt.scatter(self.nodes[self.end_idx][0], self.nodes[self.end_idx][1], marker="X")
        obj = plt.scatter(self.nodes[self.obj_idx][:, 0], self.nodes[self.obj_idx][:, 1], marker="*")
        non_obj = plt.scatter(self.nodes[self.non_obj_idx][:, 0], self.nodes[self.non_obj_idx][:, 1], marker=".")
        obs = plt.scatter(self.nodes[self.obstacle_idx][:, 0], self.nodes[self.obstacle_idx][:, 1], marker="^")
        plt.legend((start, end, obj, non_obj, obs),
                   ('Start', 'End', 'Objective', 'Non-Objective', "Turrets"),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=2,
                   fontsize=8)

        for obstacle in self.obstacle_idx:
            position = self.nodes[obstacle][:2]
            c = plt.Circle(position, self.nodes[obstacle][4], color='r', fill=False)
            plt.gca().add_artist(c)
        plt.suptitle("Phase2 - TimeStep: {} - Action Reward: {}".format(self.count, np.around(self.current_reward, 2)))
        plt.savefig("tmp/render{}.png".format(self.count), dpi=300)

    def step(self,
             action: int,
             ) -> Tuple[Observation, float, bool, bool, Information]:
        assert action < self.num_nodes, f"Invalid node {action} given! Should have been < {self.num_nodes}."
        # Calculate Rewards. Gives high punishment for staying, but no error
        time_penalty = - 1.0

        # MOVE OBSTACLES
        for i in self.obstacle_idx:
            self.nodes[i][0:2] = self.nodes[i][0:2] + self.nodes[i][5:7]

        # take care of greedy baseline
        prev_node = self.nodes[self.trajectory[-1]]
        action = action.squeeze()
        new_node = self.nodes[action]

        if np.all(prev_node == new_node):
            points = None
        else:
            points = np.stack((new_node, prev_node))[:, 0:2].T
            coefficients = np.polyfit(points[0, :], points[1, :], 1)
            fill = np.linspace(np.min(points[0, :]), np.max(points[0, :]))
            line = np.poly1d(coefficients)(fill)
            points = np.vstack((fill, line)).T


        if len(self.greedy_unvisited_obj) == 0:
            greedy_reward = 0 # time_penalty + (50 / (self.num_objectives + 1))
        else:
            ds = []
            for node in self.greedy_unvisited_obj:
                tmp_node = self.nodes[node]
                ds.append(np.linalg.norm(prev_node[0:2] - tmp_node[0:2]))
            idx = np.argmin(ds)
            distance = np.min(ds)
            greedy_next_node = self.nodes[self.greedy_unvisited_obj[idx]]
            self.greedy_unvisited_obj = np.concatenate((self.greedy_unvisited_obj[0:idx], self.greedy_unvisited_obj[idx+1:]))
            vis_obj_reward = 50 / (self.num_objectives + 1)
            obstacles = self.nodes[self.obstacle_idx]
            distances = []
            if len(obstacles) == 0:
                obstacle_penalty = 0
            else:
                if points is not None:
                    for obstacle in obstacles:
                        distances.append(np.vstack([np.linalg.norm(points - np.tile(obstacle[0:2], (50, 1)), axis=-1),
                                                    np.tile(obstacle[4], (50,))]).T)
                    distances = np.vstack(distances)
                else:
                    for obstacle in obstacles:
                        distances.append([np.linalg.norm(greedy_next_node[0:2] - obstacle[0:2]), obstacle[4]])
                    distances = np.asarray(distances)
                distances = distances[distances[:, 0] < distances[:, 1]][:,
                            0] + 1e-3  # add 1e-3 for 0 (do not want infty)
                obstacle_penalty = - np.sum(10.0 / distances)  # my heuristic
            greedy_reward = (-1) * distance + time_penalty + vis_obj_reward + obstacle_penalty
        self.greedy_rewards.append(greedy_reward)

        # Avoid extra dimensions that mess with Gymnasium.


        distance = np.linalg.norm(prev_node[0:2] - new_node[0:2])
        # Compute distance and mark node as visited.
        self.distance += distance
        if self.visited_nodes[action] == 0 and (
                action in self.obj_idx or (np.all(self.visited_nodes[self.obj_idx]) and action == self.end_idx)):
            vis_obj_reward = 50 / (self.num_objectives + 1)
            self.obj_idx = np.setdiff1d(self.obj_idx[self.obj_idx != action], action)  # get rid of obj_idx
            t = np.copy(self.nodes[action])
            t[2] = 0
            self.nodes[action] = np.copy(t)
            self.non_obj_idx = np.concatenate((self.non_obj_idx, [action]))

        else:
            vis_obj_reward = 0

        self.visited_nodes[action] = 1
        self.trajectory.append(action)
        self.agent_curr_idx = action
        # Compute mask
        self.mask = np.copy(self.visited_nodes)
        self.mask[self.end_idx] = False

        # Check if agent visited all objective nodes
        terminated = np.all(self.visited_nodes[self.obj_idx]) and action == self.end_idx
        if terminated or self.iter == 49:
            non_vis_obj_penalty = - 1000 * np.sum(np.invert(self.visited_nodes[self.obj_idx]))
        else:
            non_vis_obj_penalty = 0
        self.iter += 1
        # Truncation never happens as actions do not allow to go off bounds.
        truncated = False


        obstacles = self.nodes[self.obstacle_idx]
        distances = []
        if len(obstacles) == 0:
            obstacle_penalty = 0
        else:
            if points is not None:
                for obstacle in obstacles:
                    distances.append(np.vstack([np.linalg.norm(points - np.tile(obstacle[0:2],(50,1)), axis=-1), np.tile(obstacle[4], (50, ))]).T)
                distances = np.vstack(distances)
            else:
                for obstacle in obstacles:
                    distances.append([np.linalg.norm(new_node[0:2] - obstacle[0:2]), obstacle[4]])
                distances = np.asarray(distances)
            distances = distances[distances[:, 0] < distances[:, 1]][:, 0] + 1e-3  # add 1e-3 for 0 (do not want infty)
            obstacle_penalty = - np.sum(10.0 / distances)  # my heuristic

        reward = (-1) * distance + time_penalty + vis_obj_reward + non_vis_obj_penalty + obstacle_penalty
        self.current_reward = reward
        if self.end_idx in self.non_obj_idx:
            idx2 = self.non_obj_idx[np.where(self.non_obj_idx != self.end_idx)[0]]
            non_obj_switch_idx = idx2[self.np_random.uniform(0, 1, len(idx2)) < self.dynamic_probability_1]
        else:
            non_obj_switch_idx = self.non_obj_idx[self.np_random.uniform(0, 1, len(self.non_obj_idx)) < self.dynamic_probability_1]

        if len(non_obj_switch_idx) != 0:
            t = np.copy(self.nodes[non_obj_switch_idx])
            t[:, 4] = self.np_random.uniform(self.obstacle_min_distance, self.obstacle_max_distance, len(t))
            self.nodes[non_obj_switch_idx] = np.copy(t)
            self.obstacle_idx = np.concatenate((self.obstacle_idx, non_obj_switch_idx))
            self.non_obj_idx = np.setdiff1d(self.non_obj_idx, non_obj_switch_idx)


        obst_switch_idx = self.obstacle_idx[self.np_random.uniform(0, 1, len(self.obstacle_idx)) < self.dynamic_probability_2]
        if len(obst_switch_idx) != 0:
            t = np.copy(self.nodes[obst_switch_idx])
            t[:, 4] = 0
            self.nodes[obst_switch_idx] = np.copy(t)
            self.obstacle_idx = np.setdiff1d(self.obstacle_idx, obst_switch_idx)
            self.non_obj_idx = np.concatenate((self.non_obj_idx, obst_switch_idx))

        # CALCULATE NEW OBSTACLE VELO
        for i in self.obstacle_idx:
            velo = self.np_random.uniform(-self.velo_max, self.velo_max, 2)
            while np.any(self.nodes[i][0:2] + velo > 1) or np.any(self.nodes[i][0:2]  + velo < 0):
                velo = self.np_random.uniform(-self.velo_max, self.velo_max, 2)
            self.nodes[i][5:7] = velo

        # Calculate observation and information
        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def close(self):
        pass