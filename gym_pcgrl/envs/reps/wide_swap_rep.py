# this is swap wide

from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gymnasium import spaces
import numpy as np
from gym_pcgrl.envs.helper import gen_preset_random_map, gen_random_map
from nmmo import Terrain


class SwapRepresentation(Representation):
    # aka swap wide
    
    def __init__(self, init_random_map, num_players=2, **kwargs):
        super().__init__()
        self.init_random_map = init_random_map
        self.num_players = num_players
        self.save = {"maps": [], "pos1": [], "pos2": [], "action": []}

    def get_action_space(self, width, height, num_tiles):
        return spaces.MultiDiscrete([width, height, width, height, 1])

    def get_observation(self):
        return {
            "map": self._map.copy().astype(np.uint8)
        }

    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width))
        })

    def reset(self, width, height, prob):
        self.save = {"maps": [], "pos1": [], "pos2": [], "action": []}
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
        pos1x, pos1y = action[0], action[1]
        pos2x, pos2y = action[2], action[3]
                
        # saving stats        
        self.save["maps"].append(self._map.copy())
        self.save["pos1"].append([pos1x, pos1y])
        self.save["pos2"].append([pos2x, pos2y])

        # dont swap if the same
        if self._map[pos1y][pos1x] == self._map[pos2y][pos2x]:
            self.save["action"].append(False)       
            return False, pos1x, pos1y
            
        if action[-1] == 1:                      
            tmp = self._map[pos1y][pos1x]
            self._map[pos1y][pos1x] = self._map[pos2y][pos2x]
            self._map[pos2y][pos2x] = tmp
            self.save["action"].append(True)       
            return True, pos1x, pos1y
        else:
            self.save["action"].append(False)       
            return False, pos1x, pos1y
        
class SwapWideLiteRepresentation(SwapRepresentation):
    # same as swap, but here swap always also if tile types are not the same
    
    def get_action_space(self, width, height, num_tiles):
        return spaces.MultiDiscrete([width, height, width, height])
    
    def update(self, action):
        pos1x, pos1y = action[0], action[1]
        pos2x, pos2y = action[2], action[3]
                
        # saving stats        
        self.save["maps"].append(self._map.copy())
        self.save["pos1"].append([pos1x, pos1y])
        self.save["pos2"].append([pos2x, pos2y])

        # dont swap if the same
        if self._map[pos1y][pos1x] == self._map[pos2y][pos2x]:
            self.save["action"].append(False)       
            return False, pos1x, pos1y
                    
        tmp = self._map[pos1y][pos1x]
        self._map[pos1y][pos1x] = self._map[pos2y][pos2x]
        self._map[pos2y][pos2x] = tmp
        self.save["action"].append(True)       
        return True, pos1x, pos1y
