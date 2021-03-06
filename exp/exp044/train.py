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

def to_label(action):
    strs = action.split(' ')
    unit_id = strs[1]
    if strs[0] == 'm':
        label_dict = {'c': None, 'n': 0, 'w': 1, 's': 2, 'e': 3}
        label = label_dict[strs[2]]
    elif strs[0] == 'bcity':
        label = 4
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

def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True

def vertical_flip(state, action):
    """
    swap north(=0) and south(=2)
    """
    # flip up/down
    state = state.transpose(2,1,0)  #(c,x,y) -> (y,x,c)
    state = np.flipud(state).copy()
    if action == 0:
        action = 2
    elif action == 2:
        action = 0
    state = state.transpose(2,1,0)  # (w,h,c) -> (c,w,h)
    return state, action

def horizontal_flip(state, action):
    """
    swap west(=1) and east(=3)
    """
    # flip left/right
    state = state.transpose(2,1,0) #(x,y,c) -> (y,x,c)
    state = np.fliplr(state).copy()
    if action == 1:
        action = 3
    elif action == 3:
        action = 1
    state = state.transpose(2,1,0)  # (w,h,c) -> (c,w,h)
    return state, action

class LuxDataset(Dataset):
    def __init__(self, df, data_dir, n_obs_channel=23, n_stack=1, phase='train'):
        self.labels = df['label'].to_numpy()
        self.obs_ids = df['obs_id'].to_numpy()
        self.unit_ids = df['unit_id'].to_numpy()
        self.data_dir = data_dir 
        self.n_obs_channel = n_obs_channel
        self.n_stack = n_stack
        self.phase = phase
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        action = self.labels[idx]
        obs_id = self.obs_ids[idx]
        unit_id = self.unit_ids[idx]

        with open(self.data_dir + f"{obs_id}.pickle", mode="rb") as f:    
            obs = pickle.load(f)
        state = make_input(obs, unit_id, self.n_obs_channel)  

        ep_id = obs_id.split("_")[0]
        step = int(obs_id.split("_")[1])
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

        return state, action

    
class LuxNet(nn.Module):
    def __init__(self, observation_space, features_dim):
        super().__init__()
        layers, filters = 12, 32
        self.n_obs_channel = observation_space.shape[0]
        self.conv0 = BasicConv2d(self.n_obs_channel, filters, (3, 3), False)
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, features_dim, bias=False)
       
    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)  # h=head: (64, 32)
        p = self.head_p(h_head)  # policy
        return p  


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
                # 現在のstep=iのobsを見て選択されたactionが正解データになるのでi+1のactionを取得する
                actions = json_load['steps'][i+1][own_index]['action']

                # 空のactionsもある actions=[]
                # その場合skip
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
                        sub_ids.append(sub_id)
                        ep_ids.append(ep_id)
                        lb_scores.append(lb_score)

    df = pd.DataFrame()
    df['ep_id'] = ep_ids
    df['sub_id'] = sub_ids
    df['lb_score'] = lb_scores
    df['label'] = labels
    df['obs_id'] = obs_ids
    df['unit_id'] = unit_ids
    df.to_csv(data_dir + 'data.csv', index=False)
    return df

def train_model(model, dataloaders_dict, p_criterion,optimizer, n_obs_channel, scheduler=None, num_epochs=2):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.cuda()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_ploss = 0.0
            epoch_acc = 0
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):        
                
                states = item[0].cuda().float()
                actions = item[1].cuda().long()

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # policy, value = model(states)
                    policy = model(states)
                    policy_loss = p_criterion(policy, actions)
                    loss = policy_loss
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    batch_size = len(policy)
                    epoch_ploss += policy_loss.item() * batch_size
                    num_correct = (preds.cpu() == actions.data.cpu())
                    epoch_acc += torch.sum(num_correct)


            data_size = len(dataloader.dataset)
            epoch_ploss = epoch_ploss / data_size
            epoch_acc = epoch_acc.double() / data_size
            if phase == 'train':
                wandb.log({'Loss/train': epoch_ploss, 'ACC/train': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Acc: {epoch_acc:.4f}')
            elif phase=='val':
                wandb.log({'Loss/val': epoch_ploss, 'ACC/val': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            # traced = torch.jit.trace(model.cpu(), torch.rand(1, n_obs_channel, 32, 32))
            # traced.save('jit_best.pth')
            # torch.save(model.cpu().state_dict(), 'best.pth')
            dummy_input = torch.rand(1, n_obs_channel, 32, 32) 
            torch.onnx.export(model.cpu(), dummy_input, "best.onnx")
            best_acc = epoch_acc

        if scheduler is not None:
            scheduler.step()


def main():
    seed = 42
    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]
    team_names = ['Toad Brigade', 'RL is all you need', 'ironbar', 'A.Saito', 'Team Durrett']
    target_team_name = team_names[0]
    run_id = f"local_{target_team_name}_v2"
    print("target team:", target_team_name) 

    wandb.init(project='lux-ai', entity='kuto5046', group=EXP_NAME, id=run_id)
    episode_dir = '../../input/lux_ai_top_team_episodes_1124/'
    data_dir = "./tmp_data/"
    target_sub_ids_dict = {
        'Toad Brigade': [23032370], # [23281649, 23297953, 23032370],  # or 23032370
        'RL is all you need':[23770016, 23770123, 23769678], 
        'ironbar':[23642602, 23674317, 23674326],
        'A.Saito':[23760238, 23542431], 
        'Team Durrett': [23608696, 23889524] 
    }
    target_sub_ids = target_sub_ids_dict[target_team_name]

    df = create_dataset_from_json(episode_dir, data_dir, team_name=target_team_name, only_win=True)
    df = df[df['sub_id'].isin(target_sub_ids)]
    logger.info(f"obses:{df['obs_id'].nunique()} samples:{len(df)}")
    n_stack = 1

    labels = df['label'].to_numpy()
    actions = ['north', 'west', 'south', 'east', 'bcity']
    for value, count in zip(*np.unique(labels, return_counts=True)):
        logger.info(f'{actions[value]:^5}: {count:>3}')
    _n_obs_channel = 23
    n_obs_channel = _n_obs_channel + 8*(n_stack-1)
    observation_space = spaces.Box(low=0, high=1, shape=(n_obs_channel, 32, 32), dtype=np.float16)
    model = LuxNet(observation_space=observation_space, features_dim=len(actions))
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=seed, stratify=labels)

    batch_size = 64  # 2048
    train_loader = DataLoader(
        LuxDataset(train_df, data_dir, _n_obs_channel, n_stack=n_stack, phase='train'), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=24
    )
    val_loader = DataLoader(
        LuxDataset(val_df, data_dir, _n_obs_channel, n_stack=n_stack, phase='val'), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=24
    )
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    p_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10)

    train_model(model, dataloaders_dict, p_criterion, optimizer, n_obs_channel, num_epochs=10)
    wandb.finish()
    shutil.rmtree(data_dir)


if __name__ =='__main__':
    main()