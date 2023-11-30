import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_pcgrl.envs.helper import get_int_prob, get_string_map
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from nmmo import Terrain

"""
The PCGRL GYM Environment
"""


def get_action_size(action_space):
    if isinstance(action_space, spaces.MultiDiscrete):
        return np.prod(action_space.nvec)
    else:
        return action_space.n


def rl_to_nmmo(a, conf):
    terrain_idx_mapping = {0: Terrain.GRASS, 1: Terrain.FOREST, 2: Terrain.STONE, 3: Terrain.WATER,
                           4: Terrain.GRASS}  # 4 make player to grass
    a = a.copy()
    r = np.where(a == 4)
    conf.PLAYER_POSITIONS = [r[1], r[0]]
    masks = [a == k for k in terrain_idx_mapping.keys()]
    for m, v in zip(masks, terrain_idx_mapping.values()):
        a[m] = v

    # add stone wall
    a = np.pad(a, 1, "constant", constant_values=[Terrain.STONE])
    a = np.pad(a, 6)
    conf.MAP_NP = a


class PcgrlEnv(gym.Env):
    """
    The type of supported rendering
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """

    def __init__(self, prob="binary", rep="narrow", **kwargs):
        self._prob = PROBLEMS[prob](**kwargs)
        self._rep = REPRESENTATIONS[rep](**kwargs)
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self._prob._width * self._prob._height), 8)
        self.render_mode = "rgb_array"
        self._max_changes = 8  # 15

        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._max_iterations = 100  # 30

        self._heatmap = np.zeros((self._prob._height, self._prob._width))

        self.viewer = None

        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())

        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height,
                                                                 self.get_num_tiles())

    def get_rep(self):
        return self._rep

    def get_prob(self):
        return self._prob

    def seed(self, seed=None):
        seed = self._rep.seed(seed)
        self._prob.seed(seed)
        return [seed]

    def reset(self, options=None, seed=None):
        self._changes = 0
        self._iteration = 0
        self._rep.reset(self._prob._width, self._prob._height,
                        get_int_prob(self._prob._prob, self._prob.get_tile_types()))
        self._rep_stats = self._prob.get_stats(self._rep._map)  # without string map

        self._prob.reset(self._rep_stats)

        observation = self._rep.get_observation()
        return observation, self.get_rep_stats()

    """
    Get the border tile that can be used for padding

    Returns:
        int: the tile number that can be used for padding
    """

    def get_border_tile(self):
        return self._prob.get_tile_types().index(self._prob._border_tile)

    """
    Get the number of different type of tiles that are allowed in the observation

    Returns:
        int: the number of different tiles
    """

    def get_num_tiles(self):
        return len(self._prob.get_tile_types())

    """
    Adjust the used parameters by the problem or representation

    Parameters:
        change_percentage (float): a value between 0 and 1 that determine the
        percentage of tiles the algorithm is allowed to modify. Having small
        values encourage the agent to learn to react to the input screen.
        **kwargs (dict(string,any)): the defined parameters depend on the used
        representation and the used problem
    """

    def adjust_param(self, **kwargs):
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage')))
            self._max_changes = max(int(percentage * self._prob._width * self._prob._height), 1)

        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)
        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height,
                                                                 self.get_num_tiles())

        if "max_iterations" in kwargs:
            self._max_iterations = kwargs["max_iterations"]
        if "max_changes" in kwargs:
            self._max_changes = kwargs["max_changes"]

    """
    Advance the environment using a specific action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        observation: the current observation after applying the action
        float: the reward that happened because of applying that action
        boolean: if the problem eneded (episode is over)
        dictionary: debug information that might be useful to understand what's happening
    """

    def step(self, action):

        # if initialy is balanced skip
        if "balancing" in self._rep_stats:
            if self._rep_stats["balancing"] == self._prob.balancing:
                info = self._prob.get_debug_info(self._rep_stats, self._rep_stats)
                info["iterations"] = self._iteration
                info["changes"] = self._changes
                info["max_iterations"] = self._max_iterations
                info["max_changes"] = self._max_changes
                return self._rep.get_observation(), 0, True, info

        self._iteration += 1
        # save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        change, x, y = self._rep.update(action)
        if change > 0:
            self._changes += change
            self._rep_stats = self._prob.get_stats(self._rep._map)

        # calculate the values
        observation = self._rep.get_observation()
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        done = self._prob.get_episode_over(self._rep_stats,
                                           old_stats) or self._changes >= self._max_changes or self._iteration >= self._max_iterations

        info = self._prob.get_debug_info(self._rep_stats, old_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        return observation, reward, done, False, info

    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """

    def render(self, mode='human'):
        if mode == "graph":
            return self._prob.render(self.get_map())

        tile_size = 16
        img = self._prob.render(get_string_map(self._rep._map, self._prob.get_tile_types()))
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        if self.render_mode == 'rgb_array':
            return img
        elif self.render_mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            if not hasattr(img, 'shape'):
                img = np.array(img)
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_map(self):
        return self._rep._map

    def set_map(self, m):
        self._rep._map = m
        self._rep_stats = self._prob.get_stats(self._rep._map)  # without string map

    def get_rep_stats(self):
        return self._rep_stats
