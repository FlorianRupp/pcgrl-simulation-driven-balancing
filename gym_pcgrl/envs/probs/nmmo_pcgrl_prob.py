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


#nmmo.Env()

def spawn_pcgrl(config, *args):
    pos = config.PLAYER_POSITIONS
    # TODO ugly x and y is swapped here bc was saved wrong
    return [(x+config.MAP_BORDER+1, y+config.MAP_BORDER+1) for x, y in zip(pos["x"], pos["y"])]


class ConfigSim(nmmo.config.Small, nmmo.config.AllGameSystems):
    SPECIALIZE = True
    COMBAT_SYSTEM_ENABLED = True
    PLAYERS = [baselines.Forage, baselines.Forage]

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




class NMMO_PCGRL(Problem):
    def __init__(self, width=6, height=6, balancing=1, num_players=2, init_random_map=False, b_method=0, **kwargs):
        super().__init__()

        self._width = width
        self._height = height
        self._prob = {"grass": 0.45, "forest": 0.15, "stone": 0.2, "water": 0.15, "player": 0.05}
        self._border_size = (0, 0)
        self._border_tile = "empty"

        self.balancing_factor = balancing
        self.num_players = num_players
        self.init_random_map = init_random_map
        self._border_tile = "stone"
        b_methods = [NMMO_PCGRL.calc_balancing1, NMMO_PCGRL.calc_balancing2]
        self.b_method = b_methods[b_method]
        
        self.balanced = False
        self.conf = ConfigSim()
        nmmo.Env(self.conf)

    def get_tile_types(self):
        return ["grass", "forest", "stone", "water", "player"] # tree
    
    
    def simulate_winner(self, idx):
        winners = []
        steps = []
        #end_reason = []
        
        env = nmmo.Env(self.conf)
        env.reset()
        #step = 1
        food = {1: {"v": 100, "eaten": 0}, 2: {"v": 100, "eaten": 0}}
        running = True

        while running:
            # simulate one step in env
            _ = env.step({})
            # stopping to starvation
            if env.num_agents <= 1:
                if env.num_agents == 1:
                    winners.append(env.agents[0])
                    #end_reason.append(f"Starvation {env.agents[0]}")
                else:
                    winners.extend([1, 2])
                    #end_reason.append("Starvation both")
                #steps.append(step)
                running = False

            food = ate_food(env, food)
            # stopping if agents ate 10 food
            for player_id in food:
                if food[player_id]["eaten"] >= 5:
                    winners.append(player_id)
                    #end_reason.append(f"Fed {player_id}")
                    running = False
                    #steps.append(step)
                    #print("eaten")
            #step += 1
        
        env.close()
        del env, food
        return winners
    
    @staticmethod
    def calc_balancing1(winners):
        return round(sum(winners) / len(winners), 1) - 1
    
    @staticmethod
    def calc_balancing2(winners):
        wins1 = len(list(filter(lambda x: x==1, winners)))
        wins2 = len(list(filter(lambda x: x==2, winners)))
        w = [wins1, wins2]
        wr = [i/sum(w) for i in w]
        b = round(sum([abs(i-1/len(wr)) for i in wr]), 1)
        return b

    def get_stats(self, map, n_runs=10):
        nmap = np.array(map)
        players = np.where(nmap == "player")
        
        # simulate only if game playable and exactly 2 players exist
        if len(players[0]) != 2:
            return {
                "num-players": len(players[0]),
                "balancing": -1
            }
        
        self.conf.MAP_NP = self.rl_to_nmmo(map)
        
        pool = multiprocessing.Pool(n_runs)
        results = pool.map(self.simulate_winner, range(n_runs))
        winners = []
        for l in results:
            winners.extend(l)
            
        #balancing = round(sum(winners) / len(winners), 1) - 1
        balancing = self.b_method(winners)
        del pool
        return {"balancing": balancing, "num-players": 2}
    
    def get_reward(self, new_stats, old_stats):
        num_player_reward = get_range_reward(new_stats["num-players"], old_stats["num-players"], 2, 2)
        
        old_balancing = abs(int(old_stats["balancing"] * 10) -5)
        new_balancing = abs(int(new_stats["balancing"] * 10) -5)
        
        balancing_reward = get_range_reward(new_balancing, old_balancing, 0, 0)
        if new_stats["balancing"] == 0.5:
            balancing_reward += 6
        return num_player_reward * 2 + balancing_reward

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["balancing"] == 0.5

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
    
    def reset(self, start_stats):
        super().reset(start_stats)
        self.conf = ConfigSim()
        
    def rl_to_nmmo(self, a, path=None):
        # GRASS 2, STONE 5, FOREST 4, WATER 1
        terrain_idx_mapping = {'grass': 2, 'forest': 4, 'stone': 5, 'water': 1, 'player': 2}
        terrain_idx_mapping = {"grass": Terrain.GRASS, "forest": Terrain.FOREST, "stone": Terrain.STONE, "water": Terrain.WATER, "player": Terrain.GRASS} # 4 make player to grass       
        
        a = np.array(a)
        players = np.where(a=="player")
        self.conf.PLAYER_POSITIONS = pd.DataFrame({"y": players[1], "x": players[0]})

        masks = [a==k for k in terrain_idx_mapping.keys()]
        for m, v in zip(masks, terrain_idx_mapping.values()):
            a[m] = v

        # add stone wall
        a = np.pad(a, 1, "constant", constant_values=[Terrain.STONE])
        a = np.pad(a, 6)
        self.conf.MAP_NP = a
        a = a.astype(int)
        if path is not None:
            np.save(path, a)
            self.conf.PLAYER_POSITIONS.to_csv(path.split(".")[0]+".csv")
        return a
