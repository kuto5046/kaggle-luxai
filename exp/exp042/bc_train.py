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
from torch._C import _cuda_resetAccumulatedMemoryStats 
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
from agent_policy import AgentPolicy, CustomFeatureExtractor, CustomActorCriticCnnPolicy
from sklearn.metrics import confusion_matrix
from imitation.util import logger, util
from torch.optim.lr_scheduler import CosineAnnealingLR
sys.path.append("../../")
from agents.imitation.agent_policy import ImitationAgent
sys.path.append("../../LuxPythonEnvGym")
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
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


def filter(episodes, target_sub_id_list, team_name, only_win):
    filtering_episodes = []
    for filepath in episodes: 
        with open(filepath) as f:
            json_load = json.load(f)

        assert len(target_sub_id_list) > 0, "There is not any target submission id in list"
        if json_load['other']['SubmissionId'] not in target_sub_id_list:
            continue
        win_index = np.argmax([r or 0 for r in json_load['rewards']])  # win or tie
        if only_win:  # ?????????????????????????????????episode????????????
            if json_load['info']['TeamNames'][win_index] != team_name:
                continue
        else:  # ??????????????????????????????????????????episode?????????
            if team_name not in json_load['info']['TeamNames']: 
                continue
        filtering_episodes.append(filepath)
    print(f"Number of using episodes: {len(filtering_episodes)}")
    return filtering_episodes

def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True


def create_dataset_from_json(episode_dir, data_dir, team_name='Toad Brigade', only_win=False, target_sub_id_list=[]): 
    labels = []
    obs_ids = []
    unit_ids = []

    os.makedirs(data_dir, exist_ok=True)
    non_actions_count = 0
    episodes = [path for path in Path(episode_dir).glob('*.json') if 'output' not in path.name]

    submission_id_list = []
    latest_lb_list = []
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)
            submission_id_list.append(json_load['other']['SubmissionId'])            
            latest_lb_list.append(json_load['other']['LatestLB'])            
    sub_df = pd.DataFrame([submission_id_list, latest_lb_list], index=['SubmissionId', 'LatestLB']).T
    # target_sub_id = sub_df["SubmissionId"].value_counts().index[0]
    # target_sub_id_list.append(target_sub_id)
    print(sub_df.groupby(['SubmissionId'])['LatestLB'].mean())
    print(sub_df.groupby(['SubmissionId'])['LatestLB'].count())
    print('target sub id:', target_sub_id_list)
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)

        if (len(target_sub_id_list)>0)&(json_load['other']['SubmissionId'] not in target_sub_id_list):
            continue

        ep_id = json_load['info']['EpisodeId']

        if os.path.exists(f"{data_dir}/{ep_id}_*.pickle"):
            continue

        win_index = np.argmax([r or 0 for r in json_load['rewards']])  # win or tie
        if only_win:  # ?????????????????????????????????episode????????????
            if json_load['info']['TeamNames'][win_index] != team_name:
                continue
            own_index = win_index
        else:  # ??????????????????????????????????????????episode?????????
            if team_name not in json_load['info']['TeamNames']: 
                continue
            own_index = json_load['info']['TeamNames'].index(team_name)  # ??????????????????index

        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][own_index]['status'] == 'ACTIVE':
                # ?????????step=i???obs????????????????????????action?????????????????????????????????i+1???action???????????????
                actions = json_load['steps'][i+1][own_index]['action']

                # ??????actions????????? actions=[]
                # ????????????skip
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

                # episode_id???step?????????obs_id?????????
                obs_id = f'{ep_id}_{i}'
                with open(data_dir + f'{obs_id}.pickle', mode="wb") as f:
                    pickle.dump(obs, f)
                for action in actions:
                    # move???build city???action label?????????????????????????
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
    def __init__(self, df, data_dir, _n_obs_channel=23, n_stack=1, phase="train"):
        self.actions = df['action'].to_numpy()
        self.obs_ids = df['obs_id'].to_numpy()
        self.unit_ids = df['unit_id'].to_numpy()
        self.data_dir = data_dir 
        self._n_obs_channel = _n_obs_channel
        self.n_stack = n_stack
        self.phase = phase
        self.agent = AgentPolicy(_n_obs_channel=_n_obs_channel)
 

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
        # state = make_input(obs, unit_id, self._n_obs_channel)
    
        configs = LuxMatchConfigs_Replay
        configs["width"] = obs["width"]
        configs["height"] = obs["height"]
        game = Game(configs)
        game.reset(obs["updates"])
        team = obs["player"]
        unit = game.get_unit(team, unit_id)
        base_state = self.agent.get_base_observation(game, team, last_unit_obs=None)
        state = self.agent.get_observation(game, unit, None, team, False, base_state)

        for i in range(1, self.n_stack):
            if os.path.exists(self.data_dir + f"{ep_id}_{step-i}.pickle"):
                with open(self.data_dir + f"{ep_id}_{step-i}.pickle", mode="rb") as f:    
                    last_obs = pickle.load(f)
                last_state = self.agent.get_last_observation(last_obs)
            else:
                last_state = np.zeros((8, 32, 32), dtype=np.float32)
            state = np.concatenate([state, last_state], axis=0)
    
        assert state.shape[0] == self._n_obs_channel + 8*(self.n_stack-1)

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
            # label_dict = {'c': None, 'n': 0, 'w': 1, 's': 2, 'e': 3}
            
            label = label_dict[strs[2]]
        # elif strs[0] == 't':
        #     label = 5
        elif strs[0] == 'bcity':
            label = 5
    return label, unit_id

def valid_model(model, val_loader, label):
    epoch_acc = 0
    all_preds = []
    all_targets = []
    model.policy.to('cpu')
    for item in tqdm(val_loader, leave=False):
        states = item["obs"].cpu().float()
        actions = item["acts"].cpu().long()
        with torch.no_grad():
            preds, _, _ = model.policy(states)
        
        epoch_acc += np.sum(preds.numpy() == actions.numpy())
        all_preds.append(preds.numpy())
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
    features_dim = config["basic"]["features_dim"]
    data_dir = config["trajectory"]["data_dir"]
    episode_dir = config["trajectory"]["episode_dir"]
    only_win = config["trajectory"]["only_win"]
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

    target_sub_id_list = [23032370]  # [23281649, 23297953]  # 23032370
    df = create_dataset_from_json(episode_dir, data_dir, only_win=only_win, target_sub_id_list=target_sub_id_list)
    print(f"obses:{df['obs_id'].nunique()} samples:{len(df)}")

    unit_action_names = ['center', 'north', 'west', 'south', 'east', 'bcity']
    for action, _df in df.groupby(['action']):
        print(f"{unit_action_names[action]}:{len(_df)}")

    action_space = spaces.Discrete(6)
    _n_obs_channel = 28
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
        policy = CustomActorCriticCnnPolicy(
            observation_space=observation_space, 
            action_space=action_space, 
            lr_schedule=ConstantLRSchedule(lr=1e-3),
            net_arch = [dict(pi=[features_dim], vf=[features_dim])],
            optimizer_class=torch.optim.AdamW,
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim)
            )

        bc_trainer = BC(
            observation_space=observation_space,
            action_space=action_space,
            policy=policy,
            batch_size=batch_size,
            demonstrations=train_loader,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr":1e-3},
            custom_logger=bc_logger
        )

        os.makedirs('./models/', exist_ok=True)
        bc_trainer.train(
            # log_rollouts_venv=env,
            **bc_trainer_params)
        valid_model(bc_trainer, val_loader, unit_action_names)
        bc_trainer.save_policy(f'./models/bc_policy_{run_id}')

    run.finish()
    shutil.rmtree(data_dir)


if __name__ == "__main__":
    main()

