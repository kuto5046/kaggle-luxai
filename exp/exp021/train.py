import numpy as np
import json
from pathlib import Path
import os
from numpy.core.overrides import array_function_from_dispatcher
import wandb
import random
import logging
import pandas as pd 
from logging import INFO, DEBUG
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split


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

def to_label(action, target):
    """action記号をラベルに変換する関数
    扱っているのはunit系のactionのみでcity系のactionはNone扱い？
    unit系のactionにはtransferやpillageも含まれるがそれらのラベルはNone扱い？

    Args:
        action (list): [description]
        target (str): "unit" or "city"

    Returns:
        [type]: [description]
    
    ex)
    input: action=['m u_1 w']
    strs = ['m', 'u_1', 'w']
    strs[0] - action
    strs[1] - x
    strs[2] - y
    """
    strs = action.split(' ')
    if target == "unit":  # unit action
        unit_id = strs[1]
        if strs[0] == 'm':
            label_dict = {'c': None, 'n': 0, 's': 1, 'w': 2, 'e': 3}
            label = label_dict[strs[2]]
        elif strs[0] == 'bcity':
            label = 4
        # elif strs[0] == 'p':  # pillage
        #     label = 5
        elif strs[0] == 't':
            label = 5
        else:
            label = None
        return unit_id, label
    else:  # city action
        if strs[0] == 'r':  # research
            label = 0
            tile_pos = (int(strs[1]), int(strs[2]))
        elif strs[0] == 'bw':  # build worker
            label = 1
            tile_pos = (int(strs[1]), int(strs[2]))
        # elif strs[0] == 'bc':  # build cargo
        #     label = 2
        #     tile_pos = (int(strs[1]), int(strs[2]))
        else:
            label = None
            tile_pos = None
        return tile_pos, label


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

def create_dataset_from_json(episode_dir, target, team_name='Toad Brigade', only_win=False): 
    logger.info(f"Team: {team_name}")
    obses = {}
    samples = []
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
                obses[obs_id] = obs
                                
                for action in actions:
                    # moveとbuild cityのaction labelのみが取得される?
                    target_id, label = to_label(action, target)
                    if label is not None:
                        samples.append((obs_id, target_id, label, reward))
    logger.info(f"空のactionsの数: {non_actions_count}")
    return obses, samples


# Input for Neural Network
def make_input(obs, target_id, n_obs_channel, target):
    """obs情報をnnが学習しやすい形式に変換する関数
    全て0~1に正規化されている
    1 ch: actionの対象のunit(city)の位置  
    2 ch: actionの対象のunitが持つresourceの合計量(/100で正規化？)(3つまとめて良い？)  

    3 ch: 自チームのworker-unitの位置 
    4 ch: cooldownの状態(/6で正規化)
    5 ch: resourceの合計量(/100で正規化？)(3つまとめて良い？)
    
    6 ch: 敵チームのworker-unitの位置 
    7 ch: cooldownの状態(/6で正規化) (workerはmax=2/cargoはmax=3という認識)
    8 ch: resourceの合計量(/100で正規化？)(3つまとめて良い？)
    
    9 ch: 自チームのcitytileの位置
    10ch: 自チームのcitytileの夜間生存期間
    11ch: cooldown(/10)
    
    12ch: 敵チームのcitytileの位置
    13ch: 敵チームのcitytileの夜間生存期間
    14ch: cooldown(/10)

    15ch: wood量
    16ch: coal量
    17ch: uranium量
    
    18ch: 自チームのresearch point(位置情報はなし)
    19ch: 敵チームのresearch point(位置情報はなし)
    
    20ch: road level

    21ch: 何cycle目かを表す
    22ch: 何step目かを表す
    23ch: map
    """
    width, height = obs['width'], obs['height']

    # mapのサイズを調整するためにshiftするマス数
    # width=20の場合は6 width=21の場合5
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    # (c, w, h)
    # mapの最大サイズが(32,32)なのでそれに合わせている
    b = np.zeros((n_obs_channel, 32, 32), dtype=np.float32)

    if target == "city":
        # x:target_id[0], y:target_id[1]
        b[0, target_id[0], target_id[1]] = 1 #(1,1)  # 0chは何も情報がない

    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])

            if target_id == strs[3]:
                # Position and Cargo
                b[:2, x, y] = (
                    1,
                    (wood + coal + uranium) / 100
                )
            else:
                # Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 2 + (team - obs['player']) % 2 * 3
                b[idx:idx + 3, x, y] = (
                    1,
                    cooldown / 6,
                    (wood + coal + uranium) / 100
                )
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            cooldown = int(strs[5])
            idx = 8 + (team - obs['player']) % 2 * 2
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
    def __init__(self, obses, samples, n_obs_channel, target):
        self.obses = obses
        self.samples = samples
        self.n_obs_channel = n_obs_channel
        self.target = target
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, target_id, action, reward = self.samples[idx]
        obs = self.obses[obs_id]
        state = make_input(obs, target_id, self.n_obs_channel, self.target)
        
        return state, action, reward


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


class LuxNet(nn.Module):
    def __init__(self, num_actions, n_obs_channel):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = BasicConv2d(n_obs_channel, filters, (3, 3), False)
        self.num_actions = num_actions
        # self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), False)] * layers)
        # TODO 12層でチャネル数が変わらないけどこれでいいのか？
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])

        self.head_p = nn.Linear(filters, self.num_actions, bias=False)
        self.head_v = nn.Linear(filters*2, 1, bias=False)  # for value(reward)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        # h = (batch, c, w, h) -> (batch, c) 
        # h:(64,32,32,32)
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)  # h=head: (64, 32)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)  # h_avg: (64, 32)
        p = self.head_p(h_head)  # policy
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], dim=1)))  # value
        return p, v.squeeze(dim=1)

def train_model(model, dataloaders_dict, p_criterion, v_criterion, optimizer, n_obs_channel, target, num_epochs=2):
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
                rewards = item[2].cuda().float()

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    policy, value = model(states)
                    policy_loss = p_criterion(policy, actions)
                    value_loss = v_criterion(value, rewards)
                    loss = policy_loss + value_loss
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    batch_size = len(policy)
                    epoch_ploss += policy_loss.item() * batch_size
                    epoch_vloss += value_loss.item() * batch_size
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_ploss = epoch_ploss / data_size
            epoch_vloss = epoch_vloss / data_size
            epoch_acc = epoch_acc.double() / data_size
        
            if phase=='val':
                wandb.log({'Loss/policy': epoch_ploss, 'Loss/value':epoch_vloss, 'ACC/val': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Loss(value): {epoch_vloss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, n_obs_channel, 32, 32))
            traced.save(f'{target}_best.pth')
            best_acc = epoch_acc

     
def main():
    # param
    target_list = ["city", "unit"]
    num_epochs = 10
    n_obs_channel = 23
    batch_size = 64
    seed = 42

    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]
    wandb.init(project='lux-ai', entity='kuto5046', group=EXP_NAME) 
    episode_dir = '../../input/lux_ai_toad_episodes_1007/'
    actions_dict = {
        'unit':['north', 'south', 'west', 'east', 'bcity', 'transfer'], 
        'city':['research', 'bworker']
        }
    for target in target_list:
        logger.info("="*20)
        logger.info(f"Training {target}'s action")
        logger.info("="*20)
        obses, samples = create_dataset_from_json(episode_dir, target, only_win=True)
        logger.info(f'obses:{len(obses)} samples:{len(samples)}')
        labels = [sample[2] for sample in samples]
        actions = actions_dict[target]
        for value, count in zip(*np.unique(labels, return_counts=True)):
            logger.info(f'{actions[value]:^5}: {count:>3}')
         
        model = LuxNet(len(actions), n_obs_channel)
        train, val = train_test_split(samples, test_size=0.1, random_state=seed, stratify=labels)
        train_loader = DataLoader(
            LuxDataset(obses, train, n_obs_channel, target), 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2
        )
        val_loader = DataLoader(
            LuxDataset(obses, val, n_obs_channel, target), 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2
        )
        dataloaders_dict = {"train": train_loader, "val": val_loader}
        p_criterion = nn.CrossEntropyLoss()
        v_criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_model(model, dataloaders_dict, p_criterion, v_criterion, optimizer, n_obs_channel, target, num_epochs)
    wandb.finish()

if __name__ =='__main__':
    main()