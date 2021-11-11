import numpy as np
import json
from pathlib import Path
import pickle
import os
import shutil
from numpy.core.overrides import array_function_from_dispatcher
import wandb
import random
import logging
from gym import spaces
import pandas as pd 
from logging import INFO, DEBUG
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agent_policy import make_input, LuxNet
import sys
sys.path.append("../../LuxPythonEnvGym/")
from luxai2021.game.city import City, CityTile

def get_logger(level=INFO, out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger

logger = get_logger(level=INFO, out_file='results.log')

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def _to_label(action):
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
    strs = action.split(' ')
    unit_id = strs[1]
    if strs[0] == 'm':
        label_dict = {'c': 0, 'n': 1, 'w': 2, 's': 3, 'e': 4}
        label = label_dict[strs[2]]
    elif strs[0] == 't':
        label = 5
    elif strs[0] == 'bcity':
        label = 6
    else:
        label = None
    return unit_id, label

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
    is_unit = None 

    strs = action.split(' ')
    if strs[0] in ["m", "t", "bcity"]:
        is_unit = 1
        unit_id = strs[1]
        if strs[0] == 'm':
            label_dict = {'c': 0, 'n': 1, 'w': 2, 's': 3, 'e': 4}
            label = label_dict[strs[2]]
        elif strs[0] == 't':
            label = 5
        elif strs[0] == 'bcity':
            label = 6
    
    elif strs[0] in ["r", "bw"]:
        is_unit = 0
        x = int(strs[1])
        y = int(strs[2])
        tile_pos = (x,y)
        if strs[0] == "bw":
            label = 0
        elif strs[0] == "r":
            label = 1

    return label, unit_id, tile_pos, is_unit
 
def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True

def get_reward(json_load, index):
    """
    WIN: 1
    TIE: 0
    LOSE: -1
    """
    rewards_list = np.array(json_load['rewards'])
    rewards_list[np.where(rewards_list==None)] = 0

    enemy_index = 1 - index
    if rewards_list[index] > rewards_list[enemy_index]:
        reward = 1
    elif rewards_list[index] == rewards_list[enemy_index]:
        reward = 0
    elif rewards_list[index] < rewards_list[enemy_index]:
        reward = -1
    else:
        NotImplementedError
    return reward

def create_dataset_from_json(episode_dir, data_dir, team_name='Toad Brigade', only_win=False): 
    logger.info(f"Team: {team_name}")
    labels = []
    obs_ids = []
    unit_ids = []
    tile_poses = []
    is_units = []

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
    target_sub_id = sub_df["SubmissionId"].value_counts().index[0]

    for filepath in tqdm(episodes): 
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

        reward = get_reward(json_load, own_index)
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
                
                if depleted_resources(obs):
                    break
                
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
                    label, unit_id, tile_pos, is_unit = to_label(action)
                    if label is not None:
                        labels.append(label)
                        obs_ids.append(obs_id)
                        unit_ids.append(unit_id)
                        tile_poses.append(tile_pos)
                        is_units.append(is_unit)

    df = pd.DataFrame()
    df['label'] = labels
    df['obs_id'] = obs_ids
    df['unit_id'] = unit_ids
    df['tile_pos'] = tile_poses
    df['is_unit'] = is_units 
    df.to_csv(data_dir + 'data.csv', index=False)
    logger.info(f"空のactionsの数: {non_actions_count}")
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
    def __init__(self, df, data_dir, n_obs_channel, phase):
        self.labels = df['label'].to_numpy()
        self.obs_ids = df['obs_id'].to_numpy()
        self.unit_ids = df['unit_id'].to_numpy()
        self.tile_poses = df['tile_pos'].to_numpy()
        self.is_units = df["is_unit"].to_numpy()
        self.data_dir = data_dir 
        self.n_obs_channel = n_obs_channel
        self.phase = phase
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        action = self.labels[idx]
        obs_id = self.obs_ids[idx]
        unit_id = self.unit_ids[idx]
        tile_pos = self.tile_poses[idx]
        is_unit = self.is_units[idx]
        with open(self.data_dir + f"{obs_id}.pickle", mode="rb") as f:    
            obs = pickle.load(f)
        state = make_input(obs, unit_id, tile_pos, self.n_obs_channel)  

        if self.phase == 'train':
            if random.random() > 0.5:
                state, action = horizontal_flip(state, action)

            if random.random() > 0.5:
                state, action = vertical_flip(state, action)  

        return state, action, is_unit


def train_model(model, dataloaders_dict, p_criterion, v_criterion, optimizer, n_obs_channel, scheduler=None, num_epochs=2):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.cuda()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_ploss = 0.0
            epoch_vloss = 0.0
            unit_epoch_acc = 0
            city_epoch_acc = 0
            epoch_acc = 0
            unit_data_size = 0
            city_data_size = 0
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states = item[0].cuda().float()
                actions = item[1].cuda().long()
                is_units = item[2]
                # rewards = item[2].cuda().float()

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # policy, value = model(states)
                    policy = model(states)
                    policy_loss = p_criterion(policy, actions)
                    # value_loss = v_criterion(value, rewards)
                    loss = policy_loss  #  + value_loss
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    batch_size = len(policy)
                    epoch_ploss += policy_loss.item() * batch_size
                    # epoch_vloss += value_loss.item() * batch_size
                    unit_mask = (is_units == 1).float()  # unitのみ取り出す
                    city_mask = (is_units == 0).float()  # cityのみ取り出す
                    num_correct = (preds.cpu() == actions.data.cpu())
                    unit_epoch_acc += torch.sum(unit_mask * num_correct)
                    city_epoch_acc += torch.sum(city_mask * num_correct)
                    epoch_acc += torch.sum(num_correct)

                    unit_data_size += torch.sum(unit_mask)
                    city_data_size += torch.sum(city_mask)

            data_size = len(dataloader.dataset)
            assert data_size == unit_data_size + city_data_size, f"all:{data_size}, unit:{unit_data_size}, city:{city_data_size}"
            epoch_ploss = epoch_ploss / data_size
            # epoch_vloss = epoch_vloss / data_size
            unit_epoch_acc = unit_epoch_acc.double() / unit_data_size
            city_epoch_acc = city_epoch_acc.double() / city_data_size
            epoch_acc = epoch_acc.double() / data_size
            if phase == 'train':
                wandb.log({'Loss/train': epoch_ploss, 'ACC/train-unit': unit_epoch_acc, 'ACC/train-city': city_epoch_acc, 'ACC/train': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Acc: {epoch_acc:.4f} (unit:{unit_epoch_acc:.4f}/city{city_epoch_acc:.4f})')
            elif phase=='val':
                wandb.log({'Loss/val': epoch_ploss, 'ACC/val-unit': unit_epoch_acc, 'ACC/val-city': city_epoch_acc, 'ACC/val': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, n_obs_channel, 32, 32))
            traced.save('jit_best.pth')
            torch.save(model.cpu().state_dict(), 'best.pth')
            best_acc = epoch_acc

        if scheduler is not None:
            scheduler.step()
     
def main():
    seed = 42
    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]
    wandb.init(project='lux-ai', entity='kuto5046', group=EXP_NAME) 
    episode_dir = '../../input/lux_ai_toad1800_episodes_1108/'
    data_dir = "./tmp_data/"
    df = create_dataset_from_json(episode_dir, data_dir, only_win=False)
    logger.info(f"obses:{df['obs_id'].nunique()} samples:{len(df)}")

    labels = df['label'].to_numpy()
    actions = ['center', 'north', 'west', 'south', 'east', 'transfer', 'bcity']
    # for value, count in zip(*np.unique(labels, return_counts=True)):
    #     logger.info(f'{actions[value]:^5}: {count:>3}')
    
    n_obs_channel = 25
    observation_space = spaces.Box(low=0, high=1, shape=(n_obs_channel, 32, 32), dtype=np.float16)
    model = LuxNet(observation_space=observation_space, features_dim=len(actions))
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=seed, stratify=labels)

    batch_size = 64  # 2048
    train_loader = DataLoader(
        LuxDataset(train_df, data_dir, n_obs_channel, phase='train'), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=24
    )
    val_loader = DataLoader(
        LuxDataset(val_df, data_dir, n_obs_channel, phase='val'), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=24
    )
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    p_criterion = nn.CrossEntropyLoss()
    v_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10)

    train_model(model, dataloaders_dict, p_criterion, v_criterion, optimizer, n_obs_channel, num_epochs=2)
    wandb.finish()
    shutil.rmtree(data_dir)


if __name__ =='__main__':
    main()