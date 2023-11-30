import os
from PIL import Image
import numpy as np
from gym_pcgrl.envs.probs import Problem
from gym_pcgrl.envs.helper import get_range_reward
import nmmo
from nmmo import Terrain
from gym_pcgrl.scripted import baselines
import multiprocessing


def spawn_pcgrl(config, *args):
    pos_y = config.PLAYER_POSITIONS[0]
    pos_x = config.PLAYER_POSITIONS[1]
    return [(x + config.MAP_BORDER + 1, y + config.MAP_BORDER + 1) for x, y in zip(pos_x, pos_y)]


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
            # print(f"Player {player_id} ate")
            food_state[player_id]["eaten"] += 1
        food_state[player_id]["v"] = f
    return food_state


class NMMO(Problem):
    def __init__(self, width=6, height=6, balancing=1, num_players=2, init_random_map=False, b_method=1, **kwargs):
        super().__init__()

        self._width = width
        self._height = height
        self._prob = {"grass": 0.45, "forest": 0.15, "stone": 0.2, "water": 0.15, "player": 0.05}
        self._border_size = (0, 0)
        self._border_tile = "empty"

        self.balancing_factor = balancing
        self.num_players = num_players
        self.init_random_map = init_random_map
        self.sim_runs = kwargs["sim_runs"]
        self._border_tile = "stone"
        b_methods = [NMMO.calc_balancing1, NMMO.calc_balancing2]
        self.b_method = b_methods[b_method]

        self.balanced = False
        self.conf = ConfigSim()
        nmmo.Env(self.conf)

    def get_tile_types(self):
        return ["grass", "forest", "stone", "water", "player"]  # tree

    def simulate_winner(self, idx, env):
        winners = []
        steps = []
        _ = env.reset()

        food = {1: {"v": 100, "eaten": 0}, 2: {"v": 100, "eaten": 0}}
        running = True

        while running:
            # simulate one step in env
            _ = env.step({})
            # stopping to starvation
            if env.num_agents <= 1:
                if env.num_agents == 1:
                    winners.append(env.agents[0])
                else:
                    winners.extend([1, 2])
                running = False

            food = ate_food(env, food)
            # stopping if agents ate 10 food
            for player_id in food:
                if food[player_id]["eaten"] >= 5:
                    winners.append(player_id)
                    running = False

        env.close()
        del env, food
        return winners

    @staticmethod
    def calc_balancing1(winners):
        return round(sum(winners) / len(winners), 1) - 1

    @staticmethod
    def calc_balancing2(winners):
        wins1 = len(list(filter(lambda x: x == 1, winners)))
        wins2 = len(list(filter(lambda x: x == 2, winners)))
        w = [wins1, wins2]
        wr = [i / sum(w) for i in w]
        b = round(sum([abs(i - 1 / len(wr)) for i in wr]), 1)
        return b

    def get_stats2(self, map, n_runs=14):
        # convert pcgrl map to nmmo map, save it in conf object in MAP_NP
        self.rl_to_nmmo(map)

        pool = multiprocessing.Pool(n_runs)
        results = pool.map(self.simulate_winner, range(self.sim_runs))
        winners = []
        for l in results:
            winners.extend(l)

        balancing = self.b_method(winners)
        del pool
        return {"balancing": balancing, "winners": winners}

    def get_stats(self, map, n_runs=14):
        self.rl_to_nmmo(map)
        env = nmmo.Env(self.conf)

        winners = []
        for i in range(self.sim_runs):
            winners.extend(self.simulate_winner(i, env))
        balancing = self.b_method(winners)
        del env
        return {"balancing": balancing, "winners": winners}  # , "end-reason": end_reason

    def get_reward(self, new_stats, old_stats):
        old_balancing = abs(int(old_stats["balancing"] * 10) - 5)
        new_balancing = abs(int(new_stats["balancing"] * 10) - 5)
        balancing_reward = 0
        if new_stats["balancing"] == 0.5:
            balancing_reward = 6

        return get_range_reward(new_balancing, old_balancing, 0, 0) + balancing_reward

    def get_reward2(self, new_stats, old_stats):
        balancing_reward = 0
        if new_stats["balancing"] == 0:
            balancing_reward = 2
        return round(old_stats["balancing"] - new_stats["balancing"], 2) + balancing_reward

    def get_episode_over(self, new_stats, old_stats):
        if self.b_method == NMMO.calc_balancing1:
            return new_stats["balancing"] == 0.5
        else:
            return new_stats["balancing"] == 0

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
        terrain_idx_mapping = {0: Terrain.GRASS, 1: Terrain.FOREST, 2: Terrain.STONE,
                               3: Terrain.WATER, 4: Terrain.GRASS, 5: Terrain.GRASS}  # 4,5 make player to grass

        a = np.array(a)
        players = np.where(a == 4)

        self.conf.PLAYER_POSITIONS = [players[1], players[0]]
        masks = [a == k for k in terrain_idx_mapping.keys()]
        for m, v in zip(masks, terrain_idx_mapping.values()):
            a[m] = v

        # add stone wall
        a = np.pad(a, 1, "constant", constant_values=[Terrain.STONE])
        a = np.pad(a, 6)
        self.conf.MAP_NP = a
