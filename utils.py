import nmmo
from nmmo import Terrain
import numpy as np
from gym_pcgrl.scripted import baselines
from gym_pcgrl.wrappers import Cropped, CroppedImagePCGRLWrapper, ActionMapImagePCGRLWrapper, ToImage
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
import pickle
import numpy as np
from random import choice
import pandas as pd
import time


def rl_to_nmmo(a, conf):
    terrain_idx_mapping = {0: Terrain.GRASS, 1: Terrain.FOREST, 2: Terrain.STONE, 3: Terrain.WATER,
                           4: Terrain.GRASS}  # 4 make player to grass

    r = np.where(a == 4)
    conf.PLAYER_POSITIONS = pd.DataFrame({"y": r[0], "x": r[1]})

    masks = [a == k for k in terrain_idx_mapping.keys()]
    for m, v in zip(masks, terrain_idx_mapping.values()):
        a[m] = v

    # add stone wall
    a = np.pad(a, 1, "constant", constant_values=[Terrain.STONE])
    a = np.pad(a, 6)
    conf.MAP_NP = a
    return a


def nmmo_to_rl(conf):
    terrain_idx_mapping = {Terrain.GRASS: 0, Terrain.FOREST: 1, Terrain.STONE: 2, Terrain.WATER: 3}
    a = conf.MAP_NP

    b = conf.MAP_BORDER + 1
    s = b + conf.MAP_CENTER - 2
    a = a[b:s, b:s] - 1

    a = a + 1
    masks = [a == k for k in terrain_idx_mapping.keys()]
    for m, v in zip(masks, terrain_idx_mapping.values()):
        a[m] = v

    # insert player
    pos = conf.PLAYER_POSITIONS
    pos = [(x, y) for x, y in zip(pos["x"], pos["y"])]
    for p in pos:
        a[p[1]][p[0]] = 4
    return a


def generate_pcgrl(*args, **kwargs):
    # print("generating new map")
    conf = ConfigPCGRL()
    nmmo.Env(conf)
    return nmmo_to_rl(conf)


# faster map initialization
class PcgrlNmmoMapGenerator():
    def __init__(self):
        self.env_config = {
            "num_players": 2,
            "init_random_map": True,
            "width": 6,
            "height": 6
        }
        self.env_name = '{}-{}-v0'.format("nmmo_gen", "wide")
        self.env = ActionMapImagePCGRLWrapper(self.env_name, **self.env_config)
        self.model = PPO(MlpPolicy, self.env, verbose=0).load("models/generation/wide/mapgen6x6.zip")
        self.model.set_env(self.env)

    def generate(self, *args, **kwargs):
        two_players = False
        while two_players is False:
            obs, _ = self.env.reset()
            done = False
            while True:
                action, _ = self.model.predict(obs)
                obs, reward, done, truncated, info = self.env.step(action)
                if done:
                    break
            if info["num-players"] == 2:
                two_players = True

        return self.env.pcgrl_env.env.get_map()

    def to_nmmo(self, m, config):
        return rl_to_nmmo(m, config)


# wrap generator method at module top level to work with multiprocessing
gen = PcgrlNmmoMapGenerator()


def generate(*args, **kwargs):
    return gen.generate(**kwargs)


class PcgrlNmmoDiffGenerator(PcgrlNmmoMapGenerator):
    def add_2nd_player(self, m):
        pos = np.where(m == 4)
        n_players = len(pos[0])
        idx = np.random.randint(n_players)
        m[pos[0][idx]][pos[1][idx]] = 5
        return m

    def add_nth_player(self, m, n_players):
        for p in range(n_players - 2):
            y, x = np.where(m == 0)
            idx = np.random.randint(len(y))
            m[y[idx]][x[idx]] = 6 + p
        return m

    def generate(self, *args, **kwargs):
        m = super().generate()
        m = self.add_2nd_player(m)

        n_players = args[3]
        if n_players >= 3:
            m = self.add_nth_player(m, n_players)
        return m

    def __call__(self, *args, **kwargs):
        return self.generate()


gen_diff = PcgrlNmmoDiffGenerator()


def generate_diff(*args, **kwargs):
    return gen_diff.generate(*args, **kwargs)


class PresetGenerator():
    def __init__(self):
        with open("output/maps-test-1k.pkl", "rb") as f:
            self.maps = pickle.load(f)
        self.idx = 0

    def generate(self):
        m = self.maps[self.idx]
        self.idx += 1
        if self.idx >= len(self.maps):
            self.reset()
        return m

    def reset(self, idx=0):
        self.idx = idx


gen_pre = PresetGenerator()


def generate_preset(*args, **kwargs):
    return gen_pre.generate()


def add_2nd_player(m):
    pos = np.where(m == 4)
    n_players = len(pos[0])
    idx = np.random.randint(n_players)
    m[pos[0][idx]][pos[1][idx]] = 5
    return m


def generate_preset_diff(*args, **kwargs):
    return add_2nd_player(gen_pre.generate())


# Callbacks


class MyCallback(BaseCallback):
    def __init__(self, wandb, steps=0, verbose=0):
        super(MyCallback, self).__init__(verbose)
        self.wandb = wandb
        # self.data = []
        self.steps = steps
        self.time = time.time()

    def _on_step(self):
        self.steps += 1
        if self.steps % 100 == 0:
            self.wandb.log({"steps": self.steps})
            # log compute time per 100 steps in seconds
            now = time.time()
            self.wandb.log({"time_per_step": round(now - self.time, 2)})
            self.time = now
        return True

    def _on_rollout_end(self):
        # self.data.append(self.locals)
        try:
            self.log_rollout()
        except Exception as e:
            # print(e)
            # print("Logging -1")
            self.wandb.log({"rollout/mean_ep_len": -1}, step=self.steps)
            self.wandb.log({"rollout/mean_ep_reward": -1}, step=self.steps)
        return True

    def log_rollout(self):
        rollout_elen = []
        rollout_rewards = []
        ep_starts = self.locals["rollout_buffer"].episode_starts
        rewards = self.locals["rollout_buffer"].rewards

        # iterate envs
        for env in range(ep_starts.shape[1]):
            episode = ep_starts[:, env]
            idx_new_ep = np.where(episode == 1)[0]
            ep_len = np.diff(idx_new_ep)
            # ep_len = np.insert(ep_len, 0, idx_new_ep[0])
            rollout_elen.extend(list(ep_len))

            # mean ep reward
            rewards_ = np.split(rewards[:, env], idx_new_ep)
            # only use full episodes
            rewards = rewards[1:-1]
            rollout_rewards.extend([r.sum() for r in rewards_])

        mean_ep_len = np.mean(rollout_elen)
        mean_ep_re = np.mean(rollout_rewards)

        self.wandb.log({"rollout/mean_ep_len": mean_ep_len}, step=self.steps)
        self.wandb.log({"rollout/mean_ep_reward": mean_ep_re}, step=self.steps)


#### 3 PLAYERS

class PcgrlNmmoMapGenerator3Players():
    def __init__(self):
        self.num_players = 3
        self.env_config = {
            "num_players": self.num_players,
            "init_random_map": True,
            "width": 6,
            "height": 6
        }
        self.env_name = '{}-{}-v0'.format("nmmo_gen_nplayers", "wide")
        self.env = ActionMapImagePCGRLWrapper(self.env_name, **self.env_config)
        self.model = PPO(MlpPolicy, self.env, verbose=0).load("logs/model/mapgen6x6_3players_7_10.zip")
        self.model.set_env(self.env)

    def generate(self, *args, **kwargs):
        two_players = False
        while two_players is False:
            obs, _ = self.env.reset()
            done = False
            while True:
                action, _ = self.model.predict(obs)
                obs, reward, done, truncated, info = self.env.step(action)
                if done:
                    break
            if info["num-players"] == 3 and info["players-connected"] == 3:
                two_players = True

        return self.env.unwrapped.get_map()

    def to_nmmo(self, m, config):
        return rl_to_nmmo(m, config)


def change_player_idx(m):
    # for easier generation players are generated all the same as number 4
    # for better learning for balancing we encode each player with an own number e.g., 4,5,6
    posy, posx = np.where(m == 4)
    m[posy[0]][posx[0]] = 5
    m[posy[1]][posx[1]] = 6
    return m


gen_diff_3players = PcgrlNmmoMapGenerator3Players()


def generate_3players(*args, **kwargs):
    return change_player_idx(gen_diff_3players.generate(*args, **kwargs))
