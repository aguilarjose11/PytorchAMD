from typing import Tuple, Dict

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


integer = np.int64


class GridWorldEnv(gym.Env):
    """GridWorld Grid Reinforcement Learning Environment."""

    # Metadata required for rendering. See Gymnasium's API for more.
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self,
                 render_mode: str = None,
                 size: int = 5,
                 punishment: float = 0.):
        """GridWorld Environment.
        parameters
        ----------
        render_mode: str
            Render Mode. Only human and rgb_array are defined.
        size: int
            GridWorld dimensions. Will be size x size.
        """
        self.punishment = punishment
        # Set up the renderization values
        self.size = size
        self.window_size = 512
        # Define observation space as containing two pieces of data:
        # agent, which is a 2d array containing x,y location of the agent between 0 and size - 1
        # target, which is a 2d array containing the x,y location of the target between 0 and size - 1
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=integer),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=integer)
            }
        )
        # Define the action space, in this case containing only 4 actions.
        self.action_space = spaces.Discrete(4)
        # We define the possible actions as 0, 1, 2, and 3.
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        # Make sure a valid render mode was selected.
        assert render_mode is None or render_mode in self.metadata["render_modes"], f"Invalid render mode {render_mode}"
        self.render_mode = render_mode
        # To be set when reset is called. Empty for now.
        self.window = None
        self.clock = None
        # Attributes to be used as part of environment state
        self._agent_location: np.array = None
        self._target_location: np.aray = None

    def _get_obs(self
                 ) -> Dict[str, np.array]:
        """Return current observations.
        return
        ------
        obs: Dict[str, np.array]
            A dictionary containing two observations:

            agent
                * Current agent location.
            target
                * Target's location.
        """
        return {
            "agent": self._agent_location,
            "target": self._target_location
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }

    def reset(self,
              seed: int = None,
              options: Dict[str, all] = None
              ) -> Tuple[Dict[str, np.array], Dict[str, np.array]]:
        """Reset environment.
        Resets the environment, generating new locations for the agent and target (not on the same place). This function
        must be called before conducting a new episode, and must be called before using the environment to permit the
        creation of renderization requirements.

        parameters
        ----------
        seed: int
            A random seed to set the random number generator. Leave as None if seeking to create a new environment every
            time.
        options: Dict[str, all]
            Options directory. Not used here.

        returns
        ------
        Tuple[Dict[str, np.array], Dict[str, np.array]]
            A Tuple containing two dictionaries: one for the observation, another for the extra information on state.
        """

        super().reset(seed=seed)
        # Set up the initial agent location using Gymnasium's provided number generator.
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        # Set up an initial target location and change if same as agent's.
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        # Collect observations and information to return.
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            # Set up renderization if needed.
            self._render_frame()
        return observation, info

    def step(self,
             action: int,
             ) -> Tuple[Dict[str, np.array], int, bool, bool, Dict[str, np.array]]:
        """Apply selected action to environment.

        parameters
        ----------
        action: int
            Integer action for the agent to take on the environment.

        returns
        -------
        state: Tuple[Dict[str, np.array], int, bool, bool, Dict[str, np.array]]
            Information on the state of the environment. See function for more.
        observation: Dict[str, np.array]
            Dictionary with environment observations.
        reward: int
            Reward obtained after applying the action to environment.
        terminated: bool
            Signal indicating whether the episode has ended.
        truncated: bool
            Signal indicating whether the action caused a change in the environment that had to be truncated.
        info: Dict[str, np.array]
            A dictionary containing additional information on the environment state.
        """
        # Obtain vector form of action.
        direction = self._action_to_direction[action]
        # Apply the action to the state and clip to make sure we remain within bounds of the GridWorld.
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        # Check if we reached the target location, ending the episode.
        terminated = np.array_equal(self._agent_location, self._target_location)
        # Compute sparse reward. 1 if we arrive at the target, 0 otherwise.
        # Note: A sparse reward is a reward signal that does not offer much useful information!
        reward = 1 if terminated else self.punishment
        # Obtain observation and information
        observation = self._get_obs()
        info = self._get_info()
        # Render the frame with new changes.
        if self.render_mode == "human":
            self._render_frame()
        # Note that the truncated value is always False despite being technically True at times.
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create drawable canvas to make changes
        canvas = pygame.Surface((self.window_size, self.window_size),)
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.size)
        # Draw the target and agent
        pygame.draw.rect(canvas,
                         (255, 0, 0),
                         pygame.Rect(pix_square_size * self._target_location,
                                     (pix_square_size, pix_square_size)
                                     ),
                         )
        pygame.draw.circle(canvas,
                           (0, 0, 255),
                           (self._agent_location + 0.5) * pix_square_size,
                           pix_square_size / 3,
                           )
        # Draw vertical and horizontal lines for the grid
        for x in range(self.size + 1):
            pygame.draw.line(canvas,
                             0,
                             (0, pix_square_size * x),
                             (self.window_size, pix_square_size * x),
                             width=3
                             )
            pygame.draw.line(canvas,
                             0,
                             (pix_square_size * x, 0),
                             (pix_square_size * x, self.window_size),
                             width=3
                             )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)),
                                         axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()