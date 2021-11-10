import argparse
import glob
import os
import sys
import random
import traceback
import numpy as np 
import shutil
from pathlib import Path 
from functools import partial  # pip install functools
import logging 
from logging import DEBUG, INFO
from numpy.lib.function_base import extract
from stable_baselines3.common import callbacks
from gym import spaces
import pathlib
import pickle
import tempfile
import torch
import json 
import yaml
import wandb  
import ast 
import pandas as pd 
from tqdm import tqdm
from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn, get_device
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common import policies
from wandb.integration.sb3 import WandbCallback
from imitation.algorithms.bc import BC, ConstantLRSchedule
from imitation.algorithms.adversarial import airl, gail
from torch.utils.data import Dataset, DataLoader
from agent_policy import AgentPolicy, LuxNet
from imitation.util import logger, util

sys.path.append("../../LuxPythonEnvGym")
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default, LuxMatchConfigs_Replay
from luxai2021.game.game import Game


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def create_dataset_from_json(episode_dir, data_dir, team_name='Toad Brigade', only_win=False): 
    print(f"Team: {team_name}")
    labels = []
    obs_ids = []
    unit_ids = []

    os.makedirs(data_dir, exist_ok=True)
    non_actions_count = 0
    episodes = [path for path in Path(episode_dir).glob('*.json') if 'output' not in path.name]

    submission_id_list = []
    latest_lb_list = []
    for filepath in episodes: 
        with open(filepath) as f:
            json_load = json.load(f)
            submission_id_list.append(json_load['other']['SubmissionId'])            
            latest_lb_list.append(json_load['other']['LatestLB'])            
    sub_df = pd.DataFrame([submission_id_list, latest_lb_list], index=['SubmissionId', 'LatestLB']).T
    target_sub_id = sub_df["SubmissionId"].value_counts().index[0]

    for filepath in tqdm(episodes, total=len(episodes)): 
        with open(filepath) as f:
            json_load = json.load(f)

        if json_load['other']['SubmissionId'] != target_sub_id:
            continue

        ep_id = json_load['info']['EpisodeId']

        # if os.path.exists(f"{data_dir}/{ep_id}_*.pickle"):
        #     continue

        win_index = np.argmax([r or 0 for r in json_load['rewards']])  # win or tie
        if only_win:  # 指定したチームが勝ったepisodeのみ取得
            if json_load['info']['TeamNames'][win_index] != team_name:
                continue
            own_index = win_index
        else:  # 指定したチームの勝敗関わらずepisodeを取得
            if team_name not in json_load['info']['TeamNames']: 
                continue
            own_index = json_load['info']['TeamNames'].index(team_name)  # 指定チームのindex

        # reward = get_reward(json_load, own_index)
        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][own_index]['status'] == 'ACTIVE':
                # 現在のstep=iのobsを見て選択されたactionが正解データになるのでi+1のactionを取得する
                actions = json_load['steps'][i+1][own_index]['action']

                # 空のactionsもある actions=[]
                # その場合skip
                if actions == None:
                    non_actions_count += 1
                    continue
                obs = json_load['steps'][i][0]['observation']
                
                # if depleted_resources(obs):
                #     break
                
                obs['player'] = own_index
                obs = dict([
                    (k,v) for k,v in obs.items() 
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])

                # episode_idとstep数からobs_idを振る
                obs_id = f'{ep_id}_{i}'
                with open(data_dir + f'{obs_id}.pickle', mode="wb") as f:
                    pickle.dump(obs, f)
                for action in actions:
                    # moveとbuild cityのaction labelのみが取得される?
                    label, unit_id = to_label(action)
                    if label is not None:
                        labels.append(label)
                        obs_ids.append(obs_id)
                        unit_ids.append(unit_id)

    df = pd.DataFrame()
    df['action'] = labels
    df['obs_id'] = obs_ids
    df['unit_id'] = unit_ids
    df.to_csv(data_dir + 'data.csv', index=False)
    # logger.info(f"空のactionsの数: {non_actions_count}")
    return df


# Input for Neural Network
def make_input(obs, unit_id, n_obs_channel):
    width, height = obs['width'], obs['height']

    # mapのサイズを調整するためにshiftするマス数
    # width=20の場合は6 width=21の場合5
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    # (c, w, h)
    # mapの最大サイズが(32,32)なのでそれに合わせている
    b = np.zeros((n_obs_channel, 32, 32), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if unit_id == strs[3]:
                # Position and Cargo
                b[:2, x, y] = (
                    1,
                    (wood + coal + uranium) / 2000
                )
            else:
                # Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 2 + (team - obs['player']) % 2 * 3
                b[idx:idx + 3, x, y] += (
                    1,
                    cooldown / 6,
                    (wood + coal + uranium) / 2000
                )
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            cooldown = int(strs[5])
            idx = 8 + (team - obs['player']) % 2 * 3
            b[idx:idx + 3, x, y] = (
                1,
                cities[city_id],
                cooldown / 10
            )
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 14, 'coal': 15, 'uranium': 16}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[17 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
        elif input_identifier == "ccd":
            x = int(strs[1]) + x_shift
            y = int(strs[2]) + y_shift
            road_level = float(strs[3])
            b[19, x, y] =  road_level / 6
    
    # Day/Night Cycle
    b[20, :] = obs['step'] % 40 / 40
    # Turns
    b[21, :] = obs['step'] / 360
    # Map Size
    b[22, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return b


class LuxDataset(Dataset):
    def __init__(self, df, data_dir, n_obs_channel):
        self.actions = df['action'].to_numpy()
        self.obs_ids = df['obs_id'].to_numpy()
        self.unit_ids = df['unit_id'].to_numpy()
        self.data_dir = data_dir 
        self.n_obs_channel = n_obs_channel
        
    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action = self.actions[idx]
        obs_id = self.obs_ids[idx]
        unit_id = self.unit_ids[idx]
        with open(self.data_dir + f"{obs_id}.pickle", mode="rb") as f:    
            obs = pickle.load(f)
        state = make_input(obs, unit_id, self.n_obs_channel)  
        return {'obs':state, 'acts':action}


def to_label(action):
    """action記号をラベルに変換する関数
    扱っているのはunit系のactionのみでcity系のactionはNone扱い？
    unit系のactionにはtransferやpillageも含まれるがそれらのラベルはNone扱い？

    Args:
        action (list): [description]

    Returns:
        [type]: [description]
    
    ex)
    input: action=['m u_1 w']
    strs = ['m', 'u_1', 'w']
    strs[0] - action
    strs[1] - unit_id
    strs[2] - direction(if action is 'm')
    """
    label = None 
    tile_pos = None 
    unit_id = None 
    strs = action.split(' ')
    if strs[0] in ["m", "t", "bcity"]:
        unit_id = strs[1]
        if strs[0] == 'm':
            label_dict = {'c': 0, 'n': 1, 'w': 2, 's': 3, 'e': 4}
            label = label_dict[strs[2]]
        elif strs[0] == 't':
            label = 5
        elif strs[0] == 'bcity':
            label = 6
    # elif strs[0] in ["r", "bw"]:
    #     x = int(strs[1])
    #     y = int(strs[2])
    #     tile_pos = Position(x,y)
    #     if strs[0] == "bw":
    #         label = 0
    #     elif strs[0] == "r":
    #         label = 1

    return label, unit_id

def main():

    ############
    #  config
    ############
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    seed = config["basic"]["seed"]
    run_id = config['basic']['run_id']
    debug = config["basic"]["debug"]
    data_dir = config["basic"]["data_dir"]
    episode_dir = config["basic"]["episode_dir"]
    batch_size = config["basic"]["batch_size"]
    method = config["basic"]["method"]
    bc_trainer_params = config['BC']['trainer_params']
    ckpt_params = config["callbacks"]["checkpoints"]
    # model_params = config["model"]["params"]
    
    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]

    mode = None
    if debug:
        mode = "disabled"

    if run_id == "None":
        run_id = wandb.util.generate_id()

    run = wandb.init(
        project='lux-ai', 
        entity='kuto5046', 
        config=config, 
        group=EXP_NAME, 
        id = run_id,
        mode=mode,
        sync_tensorboard=True,# auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )


    df = create_dataset_from_json(episode_dir, data_dir, only_win=False)
    print(f"obses:{df['obs_id'].nunique()} samples:{len(df)}")

    actions = df['action'].to_numpy()
    action_names = ['center', 'north', 'west', 'south', 'east', 'transfer', 'bcity']
    for value, count in zip(*np.unique(actions, return_counts=True)):
        print(f'{action_names[value]:^5}: {count:>3}')
    
    action_space = spaces.Discrete(7)
    n_obs_channel = 23
    observation_space = spaces.Box(low=0, high=1, shape=(n_obs_channel, 32, 32), dtype=np.float16)

    # jsonデータからtrajectoryを作成
    data_loader = DataLoader(
        LuxDataset(df, data_dir, n_obs_channel), 
        batch_size=batch_size,
        shuffle=True, 
        drop_last=True, 
        num_workers=1
    )

    if method == "BC":
        bc_logger = logger.configure("./logs/")
        policy = policies.ActorCriticCnnPolicy(
            observation_space=observation_space, 
            action_space=action_space, 
            lr_schedule=ConstantLRSchedule(torch.finfo(torch.float32).max),
            features_extractor_class=LuxNet,
            features_extractor_kwargs=dict(features_dim=128)
            )

        bc_trainer = BC(
            observation_space=observation_space,
            action_space=action_space,
            policy=policy,
            batch_size=batch_size,
            demonstrations=data_loader,
            custom_logger=bc_logger
        )
        
        bc_trainer.train(**bc_trainer_params)
        bc_trainer.save_policy('bc_policy')

    # elif method == "GAIL":
    #     policy_kwargs = dict(
    #         features_extractor_class=LuxNet,
    #         features_extractor_kwargs=dict(features_dim=action_space.n),
    #     )
    #     # gail_logger = logger.configure("./")
    #     gail_trainer = gail.GAIL(
    #         venv=env,
    #         demonstrations=data_loader,
    #         demo_batch_size=batch_size,
    #         gen_algo=PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, **model_params),
    #         allow_variable_horizon=True,
    #     )
    #     gail_trainer.train(
    #         total_timesteps=step_count)
    #     gail_trainer.gen_algo.save("gail_policy")
        # elif method == "AIRL":
    #     pass
    
    run.finish()
    shutil.rmtree(data_dir)


if __name__ == "__main__":
    main()

