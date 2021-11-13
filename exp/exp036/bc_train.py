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
from sklearn.model_selection import train_test_split
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
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn, get_device
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common import policies
from imitation.algorithms.bc import BC, ConstantLRSchedule
from imitation.algorithms.adversarial import airl, gail
from torch.utils.data import Dataset, DataLoader
from agent_policy import AgentPolicy, LuxNet
from sklearn.metrics import confusion_matrix
from imitation.util import logger, util
from torch.optim.lr_scheduler import CosineAnnealingLR
sys.path.append("../../")
from agents.imitation.agent_policy import ImitationAgent
sys.path.append("../../LuxPythonEnvGym")
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import (LuxEnvironment, SaveReplayAndModelCallback,
                                   CustomEnvWrapper)
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


# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=SubprocVecEnv#multiprocessing-unleashing-the-power-of-vectorized-environments
def make_env(local_env, rank, seed=0):
    """
    Utility function for multi-processed env.

    :param local_env: (LuxEnvironment) the environment 
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        local_env.seed(seed + rank)
        return CustomEnvWrapper(local_env)

    set_random_seed(seed)
    return _init

def visualize_lbscore_and_num_episodes_by_sub(episodes):
    submission_id_list = []
    latest_lb_list = []
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)
            submission_id_list.append(json_load['other']['SubmissionId'])            
            latest_lb_list.append(json_load['other']['LatestLB'])            
    sub_df = pd.DataFrame([submission_id_list, latest_lb_list], index=['SubmissionId', 'LatestLB']).T
    print(sub_df.groupby(['SubmissionId'])['LatestLB'].mean())
    print(sub_df.groupby(['SubmissionId'])['LatestLB'].count())

def filter(episodes, target_sub_id_list, team_name, only_win):
    filtering_episodes = []
    for filepath in episodes: 
        with open(filepath) as f:
            json_load = json.load(f)

        assert len(target_sub_id_list) > 0, "There is not any target submission id in list"
        if json_load['other']['SubmissionId'] not in target_sub_id_list:
            continue
        win_index = np.argmax([r or 0 for r in json_load['rewards']])  # win or tie
        if only_win:  # 指定したチームが勝ったepisodeのみ取得
            if json_load['info']['TeamNames'][win_index] != team_name:
                continue
        else:  # 指定したチームの勝敗関わらずepisodeを取得
            if team_name not in json_load['info']['TeamNames']: 
                continue
        filtering_episodes.append(filepath)
    print(f"Number of using episodes: {len(filtering_episodes)}")
    return filtering_episodes

def get_most_large_sub_id(episodes):

    submission_id_list = []
    latest_lb_list = []
    for filepath in episodes: 
        with open(filepath) as f:
            json_load = json.load(f)
            submission_id_list.append(json_load['other']['SubmissionId'])            
            latest_lb_list.append(json_load['other']['LatestLB'])            
    sub_df = pd.DataFrame([submission_id_list, latest_lb_list], index=['SubmissionId', 'LatestLB']).T
    target_sub_id = sub_df["SubmissionId"].value_counts().index[0]
    print("target(most large) submission id:", target_sub_id)
    return target_sub_id

def create_dataset_from_json(episode_dir, data_dir, target_sub_id_list=[], team_name='Toad Brigade', only_win=False): 
    print(f"Team: {team_name}")
    labels = []
    obs_ids = []
    unit_ids = []
    os.makedirs(data_dir, exist_ok=True)
    non_actions_count = 0
    episodes = [path for path in Path(episode_dir).glob('*.json') if 'output' not in path.name]
    # target_sub_id = get_most_large_sub_id(episodes)
    episodes = filter(episodes, target_sub_id_list, team_name, only_win)
    for filepath in tqdm(episodes, total=len(episodes)): 
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        team = json_load['info']['TeamNames'].index(team_name)  # 指定チームのindex
        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][team]['status'] != 'ACTIVE':
                continue 

            # 現在のstep=iのobsを見て選択されたactionが正解データになるのでi+1のactionを取得する
            actions = json_load['steps'][i+1][team]['action']

            # 空のactionsもある actions=[]
            # その場合skip
            if actions == None:
                non_actions_count += 1
                continue

            obs = json_load['steps'][i][0]['observation']
            obs['player'] = team
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

# Input for Neural Network
def make_last_input(obs):
    width, height = obs['width'], obs['height']

    # mapのサイズを調整するためにshiftするマス数
    # width=20の場合は6 width=21の場合5
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    # (c, w, h)
    # mapの最大サイズが(32,32)なのでそれに合わせている
    b = np.zeros((8, 32, 32), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            # Units
            team = int(strs[2])
            cooldown = float(strs[6])
            idx = 0 + (team - obs['player']) % 2 * 2
            b[idx:idx + 2, x, y] += (
                1,
                (wood + coal + uranium) / 2000
            )
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            cooldown = int(strs[5])
            idx = 4 + (team - obs['player']) % 2 * 2
            b[idx:idx + 2, x, y] = (
                1,
                cities[city_id],
            )

        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
    return b

def vertical_flip(state, action):
    """
    swap north(=1) and south(=3)
    """
    # flip up/down
    state = state.transpose(2,1,0)  #(c,x,y) -> (y,x,c)
    state = np.flipud(state).copy()
    if action == 1:
        action = 3
    elif action == 3:
        action = 1
    state = state.transpose(2,1,0)  # (w,h,c) -> (c,w,h)
    return state, action

def horizontal_flip(state, action):
    """
    swap west(=2) and east(=4)
    """
    # flip left/right
    state = state.transpose(2,1,0)  #(x,y,c) -> (y,x,c)
    state = np.fliplr(state).copy()
    if action == 4:
        action = 2
    elif action == 2:
        action = 4
    state = state.transpose(2,1,0)  # (w,h,c) -> (c,w,h)
    return state, action

class LuxDataset(Dataset):
    def __init__(self, df, data_dir, n_obs_channel=23, n_stack=1, phase="train"):
        self.actions = df['action'].to_numpy()
        self.obs_ids = df['obs_id'].to_numpy()
        self.unit_ids = df['unit_id'].to_numpy()
        self.data_dir = data_dir 
        self.n_obs_channel = n_obs_channel
        self.n_stack = n_stack
        self.phase = phase
        
    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action = self.actions[idx]
        obs_id = self.obs_ids[idx]
        unit_id = self.unit_ids[idx]

        ep_id = obs_id.split("_")[0]
        step = int(obs_id.split("_")[1])
        with open(self.data_dir + f"{obs_id}.pickle", mode="rb") as f:    
            obs = pickle.load(f)
        state = make_input(obs, unit_id, self.n_obs_channel)

        for i in range(1, self.n_stack):
            if os.path.exists(self.data_dir + f"{ep_id}_{step-i}.pickle"):
                with open(self.data_dir + f"{ep_id}_{step-i}.pickle", mode="rb") as f:    
                    last_obs = pickle.load(f)
                last_state = make_last_input(last_obs)
            else:
                last_state = np.zeros((8, 32, 32), dtype=np.float32)
            state = np.concatenate([state, last_state], axis=0)
        assert state.shape[0] == self.n_obs_channel + 8*(self.n_stack-1)

        if self.phase == 'train':
            if random.random() < 0.3:
                state, action = horizontal_flip(state, action)
            if random.random() < 0.3:
                state, action = vertical_flip(state, action)  

        return {'obs':state, 'acts':action}

def to_label(action):
    label = None 
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
    return label, unit_id

def valid_model(model, val_loader, label):
    epoch_acc = 0
    all_preds = []
    all_targets = []
    for item in tqdm(val_loader, leave=False):
        states = item["obs"].cpu().float()
        actions = item["acts"].cpu().long()
        with torch.no_grad():
            preds, _ = model.policy.predict(states)
        
        epoch_acc += np.sum(preds == actions.numpy())
        all_preds.append(preds)
        all_targets.append(actions.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    print(label)
    print(confusion_matrix(y_true, y_pred))
    data_size = len(val_loader.dataset)
    epoch_acc = epoch_acc/ data_size
    print({'acc': epoch_acc})

def main():
    ############
    #  config
    ############
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    seed = config["basic"]["seed"]
    run_id = config['basic']['run_id']
    debug = config["basic"]["debug"]
    batch_size = config["basic"]["batch_size"]
    method = config["basic"]["method"]
    n_stack = config["basic"]["n_stack"]
    n_envs = config["basic"]["n_envs"]
    data_dir = config["trajectory"]["data_dir"]
    episode_dir = config["trajectory"]["episode_dir"]
    only_win = config["trajectory"]["only_win"]
    only_one_sub = config["trajectory"]["only_one_sub"]
    bc_trainer_params = config['BC']['trainer_params']
    
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
    target_sub_id_list = [23281649, 23297953] 
    df = create_dataset_from_json(episode_dir, data_dir, target_sub_id_list, only_win=False)
    print(f"obses:{df['obs_id'].nunique()} samples:{len(df)}")

    unit_action_names = ['center', 'north', 'west', 'south', 'east', 'transfer', 'bcity']
    for action, _df in df.groupby(['action']):
        print(f"{unit_action_names[action]}:{len(_df)}")

    action_space = spaces.Discrete(7)
    _n_obs_channel = 23
    n_obs_channel = _n_obs_channel + 8*(n_stack-1)
    observation_space = spaces.Box(low=0, high=1, shape=(n_obs_channel, 32, 32), dtype=np.float16)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=seed, stratify=df["action"])
    train_loader = DataLoader(
        LuxDataset(train_df, data_dir, _n_obs_channel, n_stack, phase="train"), 
        batch_size=batch_size,
        shuffle=True, 
        drop_last=True, 
        num_workers=24
    )
    val_loader = DataLoader(
        LuxDataset(val_df, data_dir, _n_obs_channel, n_stack, phase='val'), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=24
    )
    if method == "BC":
        bc_logger = logger.configure(folder="./logs/", format_strs=["stdout", "tensorboard"])
        policy = policies.ActorCriticCnnPolicy(
            observation_space=observation_space, 
            action_space=action_space, 
            lr_schedule=ConstantLRSchedule(lr=1e-3),
            net_arch = [dict(pi=[64], vf=[64])],
            optimizer_class=torch.optim.AdamW,
            features_extractor_class=LuxNet,
            features_extractor_kwargs=dict(features_dim=64)
            )

        bc_trainer = BC(
            observation_space=observation_space,
            action_space=action_space,
            policy=policy,
            batch_size=batch_size,
            demonstrations=train_loader,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr":2e-3},
            custom_logger=bc_logger
        )

        # env = SubprocVecEnv([make_env(LuxEnvironment(configs=LuxMatchConfigs_Default,
        #                                                 learning_agent=AgentPolicy(mode="train", n_stack=n_stack),
        #                                                 opponent_agents={"imitation": ImitationAgent},
        #                                                 initial_opponent_policy="imitation"), rank=i) for i in range(n_envs)])
        os.makedirs('./models/', exist_ok=True)
        bc_trainer.train(
            # log_rollouts_venv=env,
            **bc_trainer_params)
        bc_trainer.save_policy(f'./models/bc_policy_{run_id}')
        valid_model(bc_trainer, val_loader, unit_action_names)

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

