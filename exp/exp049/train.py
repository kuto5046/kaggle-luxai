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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
                    pickle.dump(obs, f)

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
def make_input(obs, unit_id, n_obs_channel):
    width, height = obs['width'], obs['height']

    # mapのサイズを調整するためにshiftするマス数
    # width=20の場合は6 width=21の場合5
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    own_actionable_city_tiles = []
    unit_count = 0
    city_tile_count = 0
    incremental_rp = 0
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
            team = int(strs[2])
            cooldown = float(strs[6])
            if unit_id == strs[3]:
                # Position and Cargo
                b[:2, x, y] = (
                    1,
                    (wood + coal + uranium) / 2000
                )
            else:
                # Units
                idx = 2 + (team - obs['player']) % 2 * 3
                b[idx:idx + 3, x, y] += (
                    1,
                    cooldown / 6,
                    (wood + coal + uranium) / 2000
                )            
            if team == obs["player"]:
                unit_count += 1

        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            cooldown = int(strs[5])
            if team == obs["player"]:
                city_tile_count += 1
                if cooldown==0:
                    own_actionable_city_tiles.append((x,y, city_id))
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
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, features_dim, bias=False)
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
        return p  #


class LuxDataset(Dataset):
    def __init__(self, df, data_dir,phase='train'):
        self.obs_ids = df['obs_id'].to_numpy()
        self.unit_ids = df['unit_id'].to_numpy()
        self.actions = df["label"].to_numpy()
        self.data_dir = data_dir 
        self.phase = phase
        
    def __len__(self):
        return len(self.obs_ids)

    def __getitem__(self, idx):
        obs_id = self.obs_ids[idx]
        with open(self.data_dir + f"{obs_id}.pickle", mode="rb") as f:    
            obs = pickle.load(f)
        unit_id = self.unit_ids[idx]
        north_state = make_input(obs, unit_id, 23)
        west_state = np.rot90(north_state,1, axes=(1,2))
        south_state = np.rot90(north_state,2, axes=(1,2))
        east_state = np.rot90(north_state,3, axes=(1,2))
        state = np.stack([north_state, west_state, south_state, east_state])

        action = self.actions[idx]
        return state, action


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
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):        
                
                states = item[0].cuda().float()
                # global_feats = item[1].cuda().float()
                actions = item[1].cuda().long()
                # actions_mask = item[3].cuda().long()
                
                batch_size = item[0].shape[0]
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # 4directionを1つにまとめる
                    batch_states = states.view((-1, 23,32,32))  # (batch*4,23,32,32)
                    _policy = model(batch_states)  # (batch*4,3)
                    _policy = _policy.view((batch_size, 4, 3))
                    center_policy = _policy[:,:, 0].mean(dim=1).unsqueeze(1)  # (64,1,32,32)
                    move_policy = _policy[:,:, 1]  # (64, 4, 32,32)
                    bcity_policy = _policy[:,:, 2].mean(dim=1).unsqueeze(1)
                    policy = torch.cat([center_policy, move_policy, bcity_policy], dim=1)
    
                    policy_loss = p_criterion(policy, actions)
                    loss = policy_loss
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    preds = torch.max(policy, 1)[1]
                    batch_size = len(policy)
                    epoch_ploss += policy_loss.item() * batch_size
                    num_correct = (preds.cpu() == actions.data.cpu())
                    epoch_num_correct += torch.sum(num_correct)

            data_size = len(dataloader.dataset)
            epoch_ploss = epoch_ploss / data_size
            epoch_acc = epoch_num_correct.double() / data_size

            if phase == 'train':
                wandb.log({'Loss/train': epoch_ploss, 'ACC/train': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Acc: {epoch_acc:.4f}')
            elif phase=='val':
                wandb.log({'Loss/val': epoch_ploss, 'ACC/val': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Acc: {epoch_acc:.4f}')
        
        if (epoch_acc > best_acc)&(phase=="val"):
            dummy_obs = torch.rand(4, 23, 32, 32)
            traced = torch.jit.trace(model.cpu(), dummy_obs)
            traced.save('best_jit.pth')
            # torch.save(model.cpu().state_dict(), 'best.pth')
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
    run_id = f'simple_IL_3action_{target_team_name}_v1'
    wandb.init(project='lux-ai', entity='kuto5046', group=EXP_NAME, id=run_id) 

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

    n_obs_channel = 23
    observation_space = spaces.Box(low=0, high=1, shape=(n_obs_channel, 32, 32), dtype=np.float16)
    model = LuxNet(observation_space=observation_space, features_dim=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_model(model, dataloaders_dict, p_criterion, optimizer, num_epochs=10)
    wandb.finish()
    shutil.rmtree(data_dir)


if __name__ == '__main__':
    main()