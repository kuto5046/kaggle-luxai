import numpy as np
import json
from pathlib import Path
import pickle
import os
import shutil
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
import sys

# sys.path.append("../../LuxPythonEnvGym/")
# from luxai2021.game.city import City, CityTile

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


def extract_unit_actions(actions):
    actioned_unit_ids = []
    unit_actions = []
    for action in actions:
        strs = action.split(' ')
        unit_id = strs[1]
        if strs[0] in ['m','bcity']:
            unit_actions.append(action)
            actioned_unit_ids.append(unit_id)
    return unit_actions, actioned_unit_ids


def extract_center_actions(obs, actioned_unit_ids):
    center_actions = []
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        if input_identifier == 'u':
            unit_id = strs[3]            
            team = int(strs[2])
            cooldown = float(strs[6])
            if (team==obs['player'])&(cooldown==0)&(unit_id not in actioned_unit_ids):
                center_actions.append(f'm {unit_id} c')
    return center_actions


def create_dataset_from_json(episode_dir, data_dir, team_name='Toad Brigade', only_win=False): 
    logger.info(f"Team: {team_name}")
    labels = []
    obs_ids = []
    unit_ids = []
    sub_ids = []
    ep_ids = []
    lb_scores = []
    os.makedirs(data_dir, exist_ok=True)
    episodes = [path for path in Path(episode_dir).glob('*.json') if 'output' not in path.name]
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)
#         if json_load['other']['SubmissionId'] not in target_sub_id_list:
#             continue
        sub_id = json_load['other']['SubmissionId']
        lb_score = json_load['other']['LatestLB']
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

        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][own_index]['status'] == 'ACTIVE':
                # episode_idとstep数からobs_idを振る
                obs_id = f'{ep_id}_{i}'
                # 現在のstep=iのobsを見て選択されたactionが正解データになるのでi+1のactionを取得する
                actions = json_load['steps'][i+1][own_index]['action']
                if actions == None:
                    continue
                    
                obs = json_load['steps'][i][0]['observation']
                
                if depleted_resources(obs):
                    break
                
                obs['player'] = own_index
                obs = dict([
                    (k,v) for k,v in obs.items() 
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])


                
                unit_actions, actioned_unit_ids = extract_unit_actions(actions)
                center_actions = extract_center_actions(obs, actioned_unit_ids)
                unit_actions += center_actions
                num_sample = min(4, len(unit_actions))
                sampling_actions = random.sample(unit_actions, k=num_sample)
                for action in sampling_actions:
                    # moveとbuild cityのaction labelのみが取得される?
                    label, unit_id = to_label(action)
            
                    if label is not None:
                        labels.append(label)
                        obs_ids.append(obs_id)
                        unit_ids.append(unit_id)
                        sub_ids.append(sub_id)
                        ep_ids.append(ep_id)
                        lb_scores.append(lb_score)


                with open(data_dir + f'{obs_id}.pickle', mode="wb") as f:
                    pickle.dump((obs, sampling_actions), f)

    df = pd.DataFrame()
    df['ep_id'] = ep_ids
    df['sub_id'] = sub_ids
    df['lb_score'] = lb_scores
    df['label'] = labels
    df['obs_id'] = obs_ids
    df['unit_id'] = unit_ids
    df.to_csv(data_dir + 'data.csv', index=False)
    return df

def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True


def to_label(action):
    strs = action.split(' ')
    unit_id = strs[1]
    if strs[0] == 'm':
        label_dict = {'c': 0, 'n': 1, 'w': 2, 's': 3, 'e': 4}
        label = label_dict[strs[2]]
    elif strs[0] == 'bcity':
        label = 5
    else:
        label = None
    return label, unit_id 


# Input for Neural Network
def make_input(obs, acts, target_unit_id):
    """
    obs(23-4)=19
    
    global features(10)
    own research point
    opponent research point
    cycle
    turn
    own unit count
    opponent unit count
    own city count
    opponent city count
    # own fuel
    # opponent fuel
    """
    own_actionable_units = {}
    width, height = obs['width'], obs['height']

    # mapのサイズを調整するためにshiftするマス数
    # width=20の場合は6 width=21の場合5
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    # (c, w, h)
    own_unit_counts = 0
    opponent_unit_counts = 0
    own_citytile_counts = 0
    opponent_citytile_counts = 0
    # mapの最大サイズが(32,32)なのでそれに合わせている
    b = np.zeros((17, 32, 32), dtype=np.float32)
    global_b = np.zeros((8, 4, 4), dtype=np.float32)
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            unit_id = strs[3]
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            
            # Units
            team = int(strs[2])
            cooldown = float(strs[6])
            idx = 0 + (team - obs['player']) % 2 * 3
            b[idx:idx + 3, x, y] += (
                1,
                cooldown / 6,
                (wood + coal + uranium) / 2000
            )
            if team == obs['player']:
                own_unit_counts += 1
            else:
                opponent_unit_counts += 1
            
            # if (obs['player'] == team)&(cooldown == 0):
            own_actionable_units[unit_id] = (x,y)
        
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            cooldown = int(strs[5])
            idx = 6 + (team - obs['player']) % 2 * 3
            b[idx:idx + 3, x, y] = (
                1,
                cities[city_id],
                cooldown / 10
            )
            if team == obs['player']:
                own_citytile_counts += 1
            else:
                opponent_citytile_counts += 1
            
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 12, 'coal': 13, 'uranium': 14}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            global_b[0 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
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
            b[15, x, y] =  road_level / 6
    
    # Day/Night Cycle
    global_b[2, :] = obs['step'] % 40 / 40
    # Turns
    global_b[3, :] = obs['step'] / 360
    global_b[4, :] = own_unit_counts / 100
    global_b[5, :] = opponent_unit_counts / 100
    global_b[6, :] = own_citytile_counts / 100
    global_b[7, :] = opponent_citytile_counts / 100

    # Map Size
    b[16, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1
    
    actions_map = np.zeros((32, 32), dtype=np.float32)  # 4d + bcity
    actions_mask = np.zeros((32, 32), dtype=np.float32)  # 4d + bcity
    for a in acts:
        label, unit_id = to_label(a)
        if (label is not None)&(unit_id == target_unit_id):
            (x,y) = own_actionable_units[unit_id]
            actions_map[x,y] = label
            actions_mask[x,y] = 1
    return b, global_b, actions_map, actions_mask


class LuxDataset(Dataset):
    def __init__(self, df, data_dir,phase='train'):
        self.obs_ids = df['obs_id'].to_numpy()
        self.unit_ids = df['unit_id'].to_numpy()
        self.data_dir = data_dir 
        self.phase = phase
        
    def __len__(self):
        return len(self.obs_ids)

    def __getitem__(self, idx):
        obs_id = self.obs_ids[idx]
        with open(self.data_dir + f"{obs_id}.pickle", mode="rb") as f:    
            (obs, acts) = pickle.load(f)
        unit_id = self.unit_ids[idx]
        north_state, global_feats, actions, actions_mask = make_input(obs, acts, unit_id)
        
        west_state = np.rot90(north_state,1, axes=(1,2))
        south_state = np.rot90(north_state,2, axes=(1,2))
        east_state = np.rot90(north_state,3, axes=(1,2))
        
        state = np.stack([north_state, west_state, south_state, east_state])
        global_feats = np.stack([global_feats,global_feats, global_feats,global_feats])
        return state, global_feats, actions, actions_mask

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LuxUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(LuxUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)  # h,wは変化しなくない？
        factor = 2 if bilinear else 1
        self.up1 = Up(256*2+8, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, obs, global_feats):
        x1 = self.inc(obs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)  # (8,8)
        x4 = torch.cat([self.down3(x3), global_feats], dim=1)  # (batch,256,4,4) + (batch,4,4,4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

def train_model(model, dataloaders_dict, p_criterion,optimizer, scheduler=None, num_epochs=2):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.cuda()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_ploss = 0.0
            epoch_num_correct = 0
            epoch_data_size = 0
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):        
                
                states = item[0].cuda().float()
                global_feats = item[1].cuda().float()
                actions = item[2].cuda().long()
                actions_mask = item[3].cuda().long()
                
                batch_size = item[0].shape[0]
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # 4directionを1つにまとめる
                    batch_states = states.view((-1, 17,32,32))  # (batch*4,23,32,32)
                    batch_global_feats = global_feats.view((-1,8,4,4))
                    _policy = model(batch_states, batch_global_feats)  # (batch*4,3)
                    _policy = _policy.view((batch_size, 4, 3, 32, 32))
                    _policy[:, 1] = torch.rot90(_policy[:, 1], k=-1, dims=(2,3)) # -90回転
                    _policy[:, 2] = torch.rot90(_policy[:, 2], k=-2, dims=(2,3)) # -180回転
                    _policy[:, 3] = torch.rot90(_policy[:, 3], k=-3, dims=(2,3)) # -270回転
                    center_policy = _policy[:,:, 0].mean(dim=1).view((-1,1,32,32))  # (64,1,32,32)
                    move_policy = _policy[:,:, 1]  # (64, 4, 32,32)
                    bcity_policy = _policy[:,:, 2].mean(dim=1).view((-1,1,32,32))  # (64,1,32,32)
                    policy = torch.cat([center_policy, move_policy, bcity_policy], dim=1)
    
                    targets = actions[actions_mask > 0]
                    predicts = policy[torch.stack([actions_mask]*6, dim=1) > 0].view(-1, 6)
                    policy_loss = p_criterion(predicts, targets)
                    loss = policy_loss
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                    epoch_num_correct += torch.sum(torch.max(predicts, 1)[1] == targets).cpu()
                    data_size = actions_mask.count_nonzero().cpu()
                    epoch_data_size += data_size 
                    epoch_ploss += policy_loss.item() * data_size

            epoch_ploss = epoch_ploss / epoch_data_size
            epoch_acc = epoch_num_correct.double() / epoch_data_size

            if phase == 'train':
                wandb.log({'Loss/train': epoch_ploss, 'ACC/train': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Acc: {epoch_acc:.4f}')
            elif phase=='val':
                wandb.log({'Loss/val': epoch_ploss, 'ACC/val': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            dummy_obs = torch.rand(4, 17, 32, 32)
            dummy_global_obs = torch.rand(4,8,4,4)
            traced = torch.jit.trace(model.cpu(), (dummy_obs, dummy_global_obs))
            traced.save('best_jit.pth')
            torch.save(model.cpu().state_dict(), 'best.pth')
            best_acc = epoch_acc

        if scheduler is not None:
            scheduler.step()

def main():
    seed = 42
    seed_everything(seed)
    team_names = ['Toad Brigade', 'RL is all you need', 'ironbar', 'A.Saito', 'Team Durrett']
    target_team_name = team_names[0]
    print("target team:", target_team_name)
    EXP_NAME = str(Path().resolve()).split('/')[-1]
    run_id = f'UNet_IL_{target_team_name}_v1'
    wandb.init(project='lux-ai', entity='kuto5046', group=EXP_NAME, id=run_id, mode='disabled') 

    episode_dir = "../../input/lux_ai_top_team_episodes_1124/"
    data_dir = "./tmp_data/"
    df = create_dataset_from_json(episode_dir, data_dir, team_name=target_team_name, only_win=True)
    target_sub_ids_dict = {
        'Toad Brigade': [23281649, 23297953, 23032370],
        'RL is all you need':[23770016, 23770123, 23769678], 
        'ironbar':[23642602, 23674317, 23674326],
        'A.Saito':[23760238, 23542431], 
        'Team Durrett': [23608696, 23889524] 
    }
    target_sub_ids = target_sub_ids_dict[target_team_name]
    _df = df[df['sub_id'].isin(target_sub_ids)]
    print(_df['label'].value_counts())

    # under sampling center action
    num_sample = int(_df.loc[_df['label'] > 0, 'label'].value_counts().mean())
    center_df = _df[_df['label'] == 0].sample(num_sample)
    other_df = _df[_df['label'] > 0]
    _df = pd.concat([center_df, other_df]).reset_index(drop=True)
    print(_df['label'].value_counts())

    logger.info(f"obses:{df['obs_id'].nunique()} samples:{len(df)}")
    # labels = _df['label'].to_numpy()
    # actions = ['center','north', 'west', 'south', 'east', 'bcity']
    # for value, count in zip(*np.unique(labels, return_counts=True)):
    #     logger.info(f'{actions[value]:^5}: {count:>3}')

    train_df, val_df = train_test_split(_df, test_size=0.1, random_state=seed, stratify=_df['ep_id'])

    batch_size = 64  # 2048
    train_loader = DataLoader(
        LuxDataset(train_df, data_dir, phase='train'), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=12
    )
    val_loader = DataLoader(
        LuxDataset(val_df, data_dir, phase='val'), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=12
    )
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    p_criterion = nn.CrossEntropyLoss()
    model = LuxUNet(n_channels=17, n_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_model(model, dataloaders_dict, p_criterion, optimizer, num_epochs=10)
    wandb.finish()

if __name__ == '__main__':
    main()