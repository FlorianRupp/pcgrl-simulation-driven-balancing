from gym_pcgrl.envs.reps.representation import Representation
from gymnasium import spaces
import numpy as np
from gym_pcgrl.envs.helper import gen_preset_random_map, gen_random_map
from nmmo import Terrain
from collections import OrderedDict
from PIL import Image
from numpy.random import randint



class NarrowSwapRepresentation(Representation):
    def __init__(self, init_random_map=True, num_players=2, **kwargs):
        super().__init__()
        self.init_random_map = init_random_map
        self.num_players = num_players
        self.width = kwargs["width"]
        self.height = kwargs["height"]
        self.save = {"maps": [], "pos1": [], "pos2": [], "action": []}

    def get_action_space(self, width, height, num_tiles):
        return spaces.Discrete(2)

    def get_observation(self):
        return OrderedDict({
            "pos": np.array([self._x1, self._y1, self._x2, self._y2], dtype=np.uint8),
            "map": self._map.copy()
        })

    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            # 2 2d position = 4
            "pos": spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([width-1, height-1, width-1, height-1]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width))
        })

    def reset(self, width, height, prob):
        self.save = {"maps": [], "pos1": [], "pos2": [], "action": []}
        self._x1 = randint(width)
        self._y1 = randint(height)
        self._x2 = randint(width)
        self._y2 = randint(height)
        
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

    def update(self, action):
        x1, y1, x2, y2 = self._x1, self._y1, self._x2, self._y2
        
        # dont swap if the same
        if self._map[y1][x1] == self._map[y2][x2]:
            change = False

        # swap (is only 1 integer)
        elif action == 1:
            # saving stats, only change=True
            #self.save["maps"].append(self._map.copy())
            #self.save["pos1"].append([x1, y1])
            #self.save["pos2"].append([x2, y2])
            
            tmp = self._map[y1][x1]
            self._map[y1][x1] = self._map[y2][x2]
            self._map[y2][x2] = tmp
            change = True
        else:
            change = False
            
        # save all actions, also change=False
        self.save["maps"].append(self._map.copy())
        self.save["pos1"].append([x1, y1])
        self.save["pos2"].append([x2, y2])
        self.save["action"].append(change)
        
        # new random positions
        self._x1 = randint(self.width) #self._random.randint(self.width)
        self._y1 = randint(self.height)
        self._x2 = randint(self.width)
        self._y2 = randint(self.height)
        return change, x1, y1
