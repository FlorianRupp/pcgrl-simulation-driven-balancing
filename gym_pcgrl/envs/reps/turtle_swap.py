from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gymnasium import spaces
import numpy as np
from collections import OrderedDict

"""
The turtle representation where the agent is trying to modify the position of the
turtle or the tile value of its current location similar to turtle graphics.
The difference with narrow representation is the agent now controls the next tile to be modified.
"""
class TurtleSwapRepresentation(Representation):
    def __init__(self, init_random_map=True, num_players=2, **kwargs):
        super().__init__()
        self._dirs = [(-1,0), (1,0), (0,-1), (0,1)]
        self._warp = False
        self.init_random_map = init_random_map
        self.num_players = num_players
        self.width = kwargs["width"]
        self.height = kwargs["height"]
        self._warp = kwargs["warp"]
        self.save = {"maps": [], "pos1": [], "pos2": []}

    def reset(self, width, height, prob):
        self.save = {"maps": [], "pos1": [], "pos2": []}
        self.x1 = self._random.randint(width)
        self.y1 = self._random.randint(height)
        self.x2 = self._random.randint(width)
        self.y2 = self._random.randint(height)
        
        if self._random_start or self._old_map is None:
            if self.init_random_map is False:
                self._map = gen_preset_random_map(self._random, width, height, self.num_players)
            elif callable(self.init_random_map):
                self._map = self.init_random_map(self._random, width, height, self.num_players, prob)
            else:
                self._map = gen_random_map(self._random, width, height, prob)

            self._old_map = self._map.copy().astype(np.uint8)
        else:
            self._map = self._old_map.copy().astype(np.uint8)

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self._warp = kwargs.get('warp', self._warp)

    def get_action_space(self, width, height, num_tiles):
        return spaces.Discrete(len(self._dirs) * len(self._dirs) * 2)

    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            "pos": spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([width-1, height-1, width-1, height-1]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width))
        })

    def get_observation(self):
        return OrderedDict({
            "pos": np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.uint8),
            "map": self._map.copy()
        })

    
    def move(self, action, x, y):
        x += self._dirs[action][0]
        if x < 0:
            if self._warp:
                x += self._map.shape[1]
            else:
                x = 0
        if x >= self._map.shape[1]:
            if self._warp:
                x -= self._map.shape[1]
            else:
                x = self._map.shape[1] - 1

        y += self._dirs[action][1]
        if y < 0:
            if self._warp:
                y += self._map.shape[0]
            else:
                y = 0
        if y >= self._map.shape[0]:
            if self._warp:
                y -= self._map.shape[0]
            else:
                y = self._map.shape[0] - 1
        return x, y
    
    
    def update(self, action):
        change = False
        # action: [move1, move2, change]
        action = np.unravel_index(action, (4, 4, 2))
        move1 = action[0]
        move2 = action[1]
        
        if action[-1] == 0:
            # move, dont swap
            self.x1, self.y1 = self.move(move1, self.x1, self.y1)
            self.x2, self.y2 = self.move(move2, self.x2, self.y2)
        elif self._map[self.y1][self.x1] == self._map[self.y2][self.x2]:
            change = False
        else:
            # save
            self.save["maps"].append(self._map.copy())
            self.save["pos1"].append([self.x1, self.y1])
            self.save["pos2"].append([self.x2, self.y2])
            
            # swap
            tmp = self._map[self.y1][self.x1]
            self._map[self.y1][self.x1] = self._map[self.y2][self.x2]
            self._map[self.y2][self.x2] = tmp
            change = True
        return change, self.x1, self.y1
