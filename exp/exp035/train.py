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


# def to_label(action):
#     unit_id = None
#     pos = None
#     label = None 
#     strs = action.split(' ')
#     if strs[0] in ['m', 't', 'bcity']:
#         unit_id = strs[1]
#         if strs[0] == 'm':
#             label_dict = {'c': 0, 'n': 1, 'w': 2, 's': 3, 'e': 4}
#             label = label_dict[strs[2]]
#         elif strs[0] == 't':
#             label = 5
#         elif strs[0] == 'bcity':
#             label = 6

#     elif strs[0] in ['r', 'bw']:
#         pos = (strs[1], strs[2])
#         if strs[0] == 'r':
#             label = 0
#         elif strs[1] == 'bw':
#             label = 1
#     return unit_id, label, pos
 
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
                    unit_id, label = to_label(action)
                    if label is not None:
                        labels.append(label)
                        obs_ids.append(obs_id)
                        unit_ids.append(unit_id)

    df = pd.DataFrame()
    df['label'] = labels
    df['obs_id'] = obs_ids
    df['unit_id'] = unit_ids
    df.to_csv(data_dir + 'data.csv', index=False)
    logger.info(f"空のactionsの数: {non_actions_count}")
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
                b[idx:idx + 3, x, y] = (
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
        self.labels = df['label'].to_numpy()
        self.obs_ids = df['obs_id'].to_numpy()
        self.unit_ids = df['unit_id'].to_numpy()
        self.data_dir = data_dir 
        self.n_obs_channel = n_obs_channel
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        action = self.labels[idx]
        obs_id = self.obs_ids[idx]
        unit_id = self.unit_ids[idx]
        with open(self.data_dir + f"{obs_id}.pickle", mode="rb") as f:    
            obs = pickle.load(f)
        state = make_input(obs, unit_id, self.n_obs_channel)  

        
        # state = state.transpose(1,2,0) # (c,w,h) -> (w,h,c)
        # # vertical flip
        # if torch.rand(1) > 0.5:
        #     state = np.fliplr(state).copy()
        #     if action == 2:
        #         action = 3
        #     elif action == 3:
        #         action = 2

        # # horizontal flip
        # if torch.rand(1) > 0.5:
        #     state = np.flipud(state).copy()
        #     if action == 0:
        #         action = 1
        #     elif action == 1:
        #         action = 0
        # state = state.transpose(2,0,1)  # (w,h,c) -> (c,w,h)

        return state, action


# Neural Network for Lux AI
class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class LuxNet(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(LuxNet, self).__init__(observation_space, features_dim)
        layers, filters = 12, 32
        self.n_obs_channel = observation_space.shape[0]
        self.conv0 = BasicConv2d(self.n_obs_channel, filters, (3, 3), False)
        self.num_actions = features_dim
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, self.num_actions, bias=False)
        # self.head_v = nn.Linear(filters*2, 1, bias=False)  # for value(reward)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        # h = (batch, c, w, h) -> (batch, c) 
        # h:(64,32,32,32)
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)  # h=head: (64, 32)
        # h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)  # h_avg: (64, 32)
        p = self.head_p(h_head)  # policy
        # v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], dim=1)))  # value
        return p  #  , v.squeeze(dim=1)

def train_model(model, dataloaders_dict, p_criterion, v_criterion, optimizer, scheduler, n_obs_channel, num_epochs=2):
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
            epoch_acc = 0
            
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states = item[0].cuda().float()
                actions = item[1].cuda().long()
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
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_ploss = epoch_ploss / data_size
            # epoch_vloss = epoch_vloss / data_size
            epoch_acc = epoch_acc.double() / data_size
        
            if phase=='val':
                # wandb.log({'Loss/policy': epoch_ploss, 'Loss/value':epoch_vloss, 'ACC/val': epoch_acc})
                # logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Loss(value): {epoch_vloss:.4f} | Acc: {epoch_acc:.4f}')
                wandb.log({'Loss/policy': epoch_ploss, 'ACC/val': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, n_obs_channel, 32, 32))
            traced.save('jit_best.pth')
            torch.save(model.cpu().state_dict(), 'best.pth')
            best_acc = epoch_acc

        # scheduler.step()
     
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
    for value, count in zip(*np.unique(labels, return_counts=True)):
        logger.info(f'{actions[value]:^5}: {count:>3}')
    
    n_obs_channel = 23
    observation_space = spaces.Box(low=0, high=1, shape=(n_obs_channel, 32, 32), dtype=np.float16)
    model = LuxNet(observation_space=observation_space, features_dim=len(actions))
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=seed, stratify=labels)

    batch_size = 64  # 2048
    train_loader = DataLoader(
        LuxDataset(train_df, data_dir, n_obs_channel), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=24
    )
    val_loader = DataLoader(
        LuxDataset(val_df, data_dir, n_obs_channel), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=24
    )
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    p_criterion = nn.CrossEntropyLoss()
    v_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    train_model(model, dataloaders_dict, p_criterion, v_criterion, optimizer, scheduler, n_obs_channel, num_epochs=20)
    wandb.finish()
    shutil.rmtree(data_dir)


if __name__ =='__main__':
    main()