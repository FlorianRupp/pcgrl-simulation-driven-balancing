import os
from PIL import Image
import numpy as np
from gym_pcgrl.envs.probs import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, calc_num_regions
from scipy.spatial.distance import cityblock
import nmmo
from nmmo import Terrain
from gym_pcgrl.scripted import baselines
from random import random
from gym_pcgrl.envs.helper import run_dikjstra

class NMMOGen(Problem):
    def __init__(self, width=6, height=6, num_players=2, init_random_map=False, gflow_net=False):
        super().__init__()

        self._width = width
        self._height = height
        self._prob = {"grass": 0.5, "forest": 0.15, "stone": 0.2, "water": 0.1, "player": 0.05}
        self._border_size = (0, 0)
        self._border_tile = "empty"

        self.num_players = num_players
        self.init_random_map = init_random_map
        self.gflow_net = gflow_net


    def get_tile_types(self):
        return ["grass", "forest", "stone", "water", "player"]

    def players_connected(self, m):
        players = np.where(np.array(m) == 4) # "player"
        if len(players[0]) == 2:
            player1 = (players[1][0], players[0][0]) #(x,y)
            player2 = (players[1][1], players[0][1]) #(x,y)
            dikjstra = run_dikjstra(x=player1[0], y=player1[1], map=m, passable_values=["stone", "forest", "player"])[0] #0,1,4
            if dikjstra[player2[1]][player2[0]] == -1:
                return -1
            return 1
        return 0

    def get_stats(self, map):
        map_stats = {
            "num-players": (np.array(map) == 4).sum(), # "player"
            "players-connected": self.players_connected(map)
        }
        return map_stats

    def get_reward(self, new_stats, old_stats):
        rewards = {
            "num-players": get_range_reward(new_stats["num-players"], old_stats["num-players"], 2, 2),
            "players-connected": get_range_reward(new_stats["players-connected"], old_stats["players-connected"], 1, 1)
        }
        return rewards["num-players"] * 3 + rewards["players-connected"] * 2

    def get_episode_over(self, new_stats, old_stats):
        # print(new_stats)
        return new_stats["num-players"] == 2 and new_stats["players-connected"] == 1

    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "grass": Image.open(os.path.dirname(__file__) + "/nmmo/grass16.png").convert('RGBA'),
                "forest": Image.open(os.path.dirname(__file__) + "/nmmo/forest16.png").convert('RGBA'),
                "stone": Image.open(os.path.dirname(__file__) + "/nmmo/stone16.png").convert('RGBA'),
                "water": Image.open(os.path.dirname(__file__) + "/nmmo/water16.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/nmmo/player16.png").convert('RGBA')
            }
        return super().render(map)

    def get_debug_info(self, new_stats, old_stats):
        return new_stats
    
    
class NMMOGenPlayers(NMMOGen):
    def get_episode_over(self, new_stats, old_stats):
        # print(new_stats)
        #if new_stats["num-players"] == 2 and new_stats["players-connected"] == 1:
            
        return new_stats["num-players"] == 2 and new_stats["players-connected"] == 1
    
    
    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "grass": Image.open(os.path.dirname(__file__) + "/nmmo/grass16.png").convert('RGBA'),
                "forest": Image.open(os.path.dirname(__file__) + "/nmmo/forest16.png").convert('RGBA'),
                "stone": Image.open(os.path.dirname(__file__) + "/nmmo/stone16.png").convert('RGBA'),
                "water": Image.open(os.path.dirname(__file__) + "/nmmo/water16.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/nmmo/player16.png").convert('RGBA'),
                "player2": Image.open(os.path.dirname(__file__) + "/nmmo/player2_16.png").convert('RGBA')
            }
        return super().render(map)


class NMMOGenNPlayers(NMMOGen):
    def __init__(self, width=6, height=6, num_players=2, init_random_map=False, *args, **kwargs):
        super().__init__(width, height, num_players, init_random_map)
        self._prob = {"grass": 0.45, "forest": 0.25, "stone": 0.075, "water": 0.15, "player": 0.075}
        
    def connected(self, player_a, player_b, m):
        dikjstra = run_dikjstra(x=player_a[0], y=player_a[1], map=m, passable_values=[0, 1, 4])[0] #0,1,4
        if dikjstra[player_b[1]][player_b[0]] == -1:
            return 0
        return 1
    
    def players_connected(self, m):
        players_y, players_x = np.where(np.array(m) == 4) # "player"
        connections = []
        if len(players_y) == self.num_players:
            connections.append(self.connected((players_x[0], players_y[0]), (players_x[1], players_y[1]), m))
            connections.append(self.connected((players_x[0], players_y[0]), (players_x[2], players_y[2]), m))
            connections.append(self.connected((players_x[2], players_y[2]), (players_x[1], players_y[1]), m))
            #if sum(connections) == self.n_players
            #        return 1
            #return -1
            return sum(connections)
        return 0
    
    
    def get_reward(self, new_stats, old_stats):
        rewards = {
            "num-players": get_range_reward(new_stats["num-players"], old_stats["num-players"], self.num_players, self.num_players),
            "players-connected": get_range_reward(new_stats["players-connected"], old_stats["players-connected"], self.num_players, self.num_players)
        }
        return rewards["num-players"] * 3 + rewards["players-connected"] * 2
    
    def get_episode_over(self, new_stats, old_stats):
        # print(new_stats)
        return new_stats["num-players"] == self.num_players and new_stats["players-connected"] == self.num_players