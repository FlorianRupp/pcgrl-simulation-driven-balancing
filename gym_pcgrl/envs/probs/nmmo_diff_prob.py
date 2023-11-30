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
import pandas as pd
import multiprocessing



nmmo.Env()

def spawn_pcgrl(config, *args):
    pos_y = config.PLAYER_POSITIONS[0]
    pos_x = config.PLAYER_POSITIONS[1]
    return [(y + config.MAP_BORDER + 1, x + config.MAP_BORDER + 1) for x, y in zip(pos_x, pos_y)]


class ConfigSim(nmmo.config.Small, nmmo.config.AllGameSystems):
    SPECIALIZE = True
    COMBAT_SYSTEM_ENABLED = True
    PLAYERS = [baselines.ForageOnly, baselines.ForageOnly] #baselines.ForageHandicap

    PLAYER_N = 2
    PLAYER_DEATH_FOG = None
    PATH_MAPS = 'maps'
    RENDER = False
    MAP_FORCE_GENERATION = False

    NPC_N = 0
    MAP_GENERATE_PREVIEWS = False
    MAP_CENTER = 8
    MAP_BORDER = 6

    LOG_VERBOSE = False
    LOG_EVENTS = False
    PROGRESSION_SYSTEM_ENABLED = False
    PLAYER_SPAWN_FUNCTION = spawn_pcgrl
    
    MAP_SAVE = False
    

def ate_food(env, food_state):
    for player_id in env.agents:
        f = env.realm.players.entities[player_id].packet()["resource"]["food"]["val"]
        if f > food_state[player_id]["v"]:
            #print(f"Player {player_id} ate")
            food_state[player_id]["eaten"] += 1
        food_state[player_id]["v"] = f
    return food_state



# use this for different players encoding and n players
class NMMODiff(Problem):
    def __init__(self, width=6, height=6, balancing=0.5, num_players=2, init_random_map=False, b_method=1, sim_runs=10, **kwargs):
        self.num_players = num_players
        super().__init__()
        
        self._width = width
        self._height = height
        #self._prob = {"grass": 0.45, "forest": 0.15, "stone": 0.2, "water": 0.15, "player1": 0.025, "player2": 0.025}
            
        self._border_size = (0, 0)
        self._border_tile = "empty"
        self.sim_runs = sim_runs

        self.balancing = balancing
        self.init_random_map = init_random_map
        self._border_tile = "stone"
        
        b_methods = [self.calc_balancing1, self.calc_balancing2]
        self.b_method = b_methods[b_method]
        
        if self.balancing == 0:
            self.reward_function = self.get_reward_Nplayers
        else:
            self.reward_function = self.get_reward_2players

        
        self.init_prob()
        
        self.balanced = False
        self.conf = ConfigSim()
        self.conf.PLAYERS = [baselines.ForageOnly] * self.num_players
        
        if self.num_players == 2:
            self.rl_to_nmmo(np.array([[0, 2, 0, 3, 2, 0],
                                       [0, 0, 2, 0, 2, 0],
                                       [2, 0, 1, 2, 0, 1],
                                       [0, 0, 4, 5, 1, 1],
                                       [0, 0, 3, 0, 2, 1],
                                       [0, 2, 0, 0, 0, 3]]))
        else:
            self.rl_to_nmmo(np.array([[0, 2, 0, 3, 2, 0],
                                       [0, 0, 2, 0, 2, 0],
                                       [2, 0, 1, 2, 0, 1],
                                       [0, 6, 4, 5, 1, 1],
                                       [0, 0, 3, 0, 2, 1],
                                       [0, 2, 0, 0, 0, 3]]))
        self.nmmo_env = nmmo.Env(self.conf)
        self.nmmo_env.reset()
        self.players = list(range(1, self.num_players+1))
            
    def init_prob(self):
        # needed for mocking prob values (not used) but needed for pcgrl
        self._prob = {"grass": 0.1, "forest": 0.1, "stone": 0.1, "water": 0.1}
        p = 0.6 / self.num_players
        for i in range(self.num_players):
            self._prob["player"+str(i+1)] = p


    def get_tile_types(self):
        tiles = ["grass", "forest", "stone", "water"]
        for i in range(self.num_players):
            tiles.append(f"player{i+1}")
        return tiles
    
    
    def simulate_winner(self, idx):
        winners = []
        steps = []
        #env = nmmo.Env(self.conf)
        #_ = env.reset()
        _ = self.nmmo_env.reset()
        if self.nmmo_env.num_agents != self.num_players:
            print("num_a", self.nmmo_env.agents)
            print(self.conf.PLAYER_POSITIONS)
            print(self.nmmo_env.realm.map.get_map())
            print(spawn_pcgrl(self.conf))
            raise ValueError(f"Not all players spawned, number agents does not equal {self.num_players}, is", self.nmmo_env.agents)
        
        # init game stats
        food = {i+1: {"v": 100, "eaten": 0} for i in range(self.num_players)}
        running = True

        while running:
            # simulate one step in env
            _ = self.nmmo_env.step({})
            # stopping to starvation
            if self.nmmo_env.num_agents <= 1:
                if self.nmmo_env.num_agents == 1:
                    winners.append(self.nmmo_env.agents[0])
                else:
                    winners.extend([1, 2])
                running = False

            food = ate_food(self.nmmo_env, food)
            #stopping if agents ate 10 food
            for player_id in food:
                if food[player_id]["eaten"] >= 5:
                    winners.append(player_id)
                    running = False

        #env.close()
        #del env
        del food
        return winners
    
    def calc_balancing1(self, winners):
        return round(sum(winners) / len(winners), 1) - 1

    def calc_balancing2(self, winners):
        w = []
        for idx in self.players:
            w.append(len(list(filter(lambda x: x==idx, winners))))
        wr = [i/sum(w) for i in w]
        b = round(sum([abs(i-1/len(wr)) for i in wr]), 1)
        return b

    def get_stats2(self, map, n_runs=10):
        # convert pcgrl map to nmmo map, save it in conf object in MAP_NP
        self.rl_to_nmmo(map)

        pool = multiprocessing.Pool(n_runs)
        results = pool.map(self.simulate_winner, range(self.sim_runs))
        winners = []
        for l in results:
            winners.extend(l)

        # balancing = round(sum(winners) / len(winners), 1) - 1
        balancing = self.b_method(winners)
        del pool
        return {"balancing": balancing, "winners": winners}

    def get_stats(self, map):
        self.rl_to_nmmo(map)
        self.nmmo_env.realm.update_map(self.conf)
        #_ = self.nmmo_env.reset()
        #return {"balancing": random(), "winners": [1,1,1,1,1]}
        #env = nmmo.Env(self.conf)

        winners = []
        for i in range(self.sim_runs):
            winners.extend(self.simulate_winner(i)) #env
            #winners.extend([1]) #env

        #env.close()
        # balancing = round(sum(winners) / len(winners), 1) - 1
        # balancing = self.calc_balancing(winners)
        balancing = self.b_method(winners)
        #steps = sum(steps) / len(steps)

        #del env
        return {"balancing": balancing, "winners": winners}  # , "end-reason": end_reason
    
    def get_reward_2players(self, new_stats, old_stats):
        # b_method1
        old_balancing = abs(int(old_stats["balancing"] * 10) - (self.balancing*10)) #-5
        new_balancing = abs(int(new_stats["balancing"] * 10) - (self.balancing*10))
        # print("old", old_balancing, "new", new_balancing)
        balancing_reward = 0
        if new_stats["balancing"] == self.balancing:
            balancing_reward = 6
        
        return get_range_reward(new_balancing, old_balancing, 0, 0) + balancing_reward
    
    def get_reward_Nplayers(self, new_stats, old_stats):
        # b_method2
        balancing_reward = 0
        if new_stats["balancing"] == self.balancing:
            balancing_reward = 10
        return (round(old_stats["balancing"] - new_stats["balancing"], 2)*10) + balancing_reward
        #return new_stats["steps"]# - old_stats["steps"]

    def get_reward(self, new_stats, old_stats):
        return self.reward_function(new_stats, old_stats)
    
    def get_episode_over(self, new_stats, old_stats):
        #if self.b_method == NMMODiff.calc_balancing1:
        return round(new_stats["balancing"], 1) == self.balancing
    
        #return round(new_stats["balancing"], 1) == 0.5 #USE THIS NORMALLY!!
        #return round(new_stats["balancing"], 2) == 0

    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "grass": Image.open(os.path.dirname(__file__) + "/nmmo/grass16.png").convert('RGBA'),
                "forest": Image.open(os.path.dirname(__file__) + "/nmmo/forest16.png").convert('RGBA'),
                "stone": Image.open(os.path.dirname(__file__) + "/nmmo/stone16.png").convert('RGBA'),
                "water": Image.open(os.path.dirname(__file__) + "/nmmo/water16.png").convert('RGBA'),
                "player1": Image.open(os.path.dirname(__file__) + "/nmmo/player16.png").convert('RGBA'),
                "player2": Image.open(os.path.dirname(__file__) + "/nmmo/player2_16.png").convert('RGBA'),
                "player3": Image.open(os.path.dirname(__file__) + "/nmmo/player3_16.png").convert('RGBA'),
                "player4": Image.open(os.path.dirname(__file__) + "/nmmo/player4_16.png").convert('RGBA'),
                "player5": Image.open(os.path.dirname(__file__) + "/nmmo/player5_16.png").convert('RGBA')
            }
        return super().render(map)

    def get_debug_info(self, new_stats, old_stats):
        return new_stats
    
    def reset(self, start_stats):
        super().reset(start_stats)
        self.conf = ConfigSim()
        self.conf.PLAYERS = [baselines.Forage] * self.num_players
        
    def rl_to_nmmo(self, a):
        #terrain_idx_mapping = {"grass": Terrain.GRASS, "forest": Terrain.FOREST, "stone": Terrain.STONE,
        #                       "water": Terrain.WATER, "player": Terrain.GRASS, "player2": Terrain.GRASS}  # 4,5 make player to grass

        terrain_idx_mapping = {0: Terrain.GRASS, 1: Terrain.FOREST, 2: Terrain.STONE,
                               3: Terrain.WATER, 4: Terrain.GRASS, 5: Terrain.GRASS}  # 4,5 make player to grass
        
        a = np.array(a)
        #player1 = np.where(a == 4)
        #player2 = np.where(a == 5)
        #self.conf.PLAYER_POSITIONS = [[player1[0][0], player2[0][0]], [player1[1][0], player2[1][0]]]
        
        self.conf.PLAYER_POSITIONS = [[], []]
        for i in range(self.num_players):
            y, x = np.where(a == (4+i))
            try:
                self.conf.PLAYER_POSITIONS[0].append(y[0])
                self.conf.PLAYER_POSITIONS[1].append(x[0])
            except IndexError as e:
                print(e)
                print(a)
                print(y, x)
            

        masks = [a == k for k in terrain_idx_mapping.keys()]
        for m, v in zip(masks, terrain_idx_mapping.values()):
            a[m] = v

        # add stone wall
        a = np.pad(a, 1, "constant", constant_values=[Terrain.STONE])
        a = np.pad(a, 6)
        #print(repr(a))
        self.conf.MAP_NP = a
