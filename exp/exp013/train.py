import numpy as np
import json
from pathlib import Path
import os
from numpy.core.fromnumeric import squeeze
import wandb
import random
from scipy.stats import rankdata
import logging
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

def to_label(action: list):
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
    if strs[0] == 'm':  # 移動行動の場合
        label_dict = {'c': None, 'n': 0, 's': 1, 'w': 2, 'e': 3}
        label = label_dict[strs[2]]
    elif strs[0] == 'bcity':  # city build actionの場合
        label = 4
    elif strs[0] == 'p':  # pillage
        label = 5
    elif strs[0] == 't':
        label = 6
    else:
        label = None
    return unit_id, label

def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True

def create_dataset_from_json(episode_dir, team_name='Toad Brigade'): 
    logger.info(f"Team: {team_name}")
    obses = {}
    samples = []
    rewards = [-1, 1]
    non_actions_count = 0
    episodes = [path for path in Path(episode_dir).glob('*.json') if 'output' not in path.name]
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        if team_name not in json_load['info']['TeamNames']: 
            continue
        
        index = json_load['info']['TeamNames'].index(team_name)  # 指定チームのindex: 0,1
        rewards_list = np.array(json_load['rewards'])
        rewards_list[np.where(rewards_list==None)] = 0
        rank = rankdata(rewards_list, 'dense')[index] - 1  # win=1/lose=0
        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                # 現在のstep=iのobsを見て選択されたactionが正解データになるのでi+1のactionを取得する
                actions = json_load['steps'][i+1][index]['action']               
                
                # 空のactionsもある actions=[]
                # その場合skip
                if actions == None:
                    non_actions_count += 1
                    continue
                obs = json_load['steps'][i][0]['observation']
                
                if depleted_resources(obs):
                    break
                
                obs['player'] = index  # 新しくplayerという情報を追加
                # 必要な情報のみ取得
                obs = dict([
                    (k,v) for k,v in obs.items() 
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])

                # episode_idとstep数からobs_idを振る
                obs_id = f'{ep_id}_{i}'
                obses[obs_id] = obs
                                
                for action in actions:
                    # moveとbuild cityのaction labelのみが取得される?
                    unit_id, label = to_label(action)
                    if label is not None:
                        samples.append((obs_id, unit_id, label, rewards[rank]))
    logger.info(f"空のactionsの数: {non_actions_count}")
    return obses, samples


# Input for Neural Network
def make_input(obs, unit_id):
    """obs情報をnnが学習しやすい形式に変換する関数
    全て0~1に正規化されている
    1 ch: cargo-unitの位置  
    2 ch: cargo-unitが持つresourceの合計量(/100で正規化？)(3つまとめて良い？)  

    3 ch: 自チームのworker-unitの位置 
    4 ch: cooldownの状態(/6で正規化)
    5 ch: resourceの合計量(/100で正規化？)(3つまとめて良い？)
    
    6 ch: 敵チームのworker-unitの位置 
    7 ch: cooldownの状態(/6で正規化)
    8 ch: resourceの合計量(/100で正規化？)(3つまとめて良い？)
    
    9 ch: 自チームのcitytileの位置
    10ch: cities[city_id]
    
    11ch: 敵チームのcitytileの位置
    12ch: cities[city_id]
    
    13ch: wood量
    14ch: coal量
    15ch: uranium量
    
    16ch: 自チームのresearch point(位置情報はなし)(正規化)
    17ch: 敵チームのresearch point(位置情報はなし)(正規化)
    
    18ch: 何cycle目かを表す(正規化)
    19ch: 何step目かを表す(正規化)
    20ch: map
    """
    width, height = obs['width'], obs['height']

    # mapのサイズを調整するためにshiftするマス数
    # width=20の場合は6 width=21の場合5
    x_shift = (32 - width) // 2  
    y_shift = (32 - height) // 2
    cities = {}
    
    # (c, w, h)
    # mapの最大サイズが(32,32)なのでそれに合わせている
    b = np.zeros((20, 32, 32), dtype=np.float32)
    
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
            idx = 8 + (team - obs['player']) % 2 * 2
            b[idx:idx + 2, x, y] = (
                1,
                cities[city_id]
            )
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
            b[15 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
    
    # Day/Night Cycle
    b[17, :] = obs['step'] % 40 / 40
    # Turns
    b[18, :] = obs['step'] / 360
    # Map Size(使用しているマスのみ1で埋める)
    b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return b


class LuxDataset(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, unit_id, action, reward = self.samples[idx]
        obs = self.obses[obs_id]
        state = make_input(obs, unit_id)

        """
        state = state.transpose(1,2,0) # (c,w,h) -> (w,h,c)
        # vertical flip
        if torch.rand(1) > 0.5:
            state = np.fliplr(state).copy()
            if action == 2:
                action = 3
            elif action == 3:
                action = 2

        # horizontal flip
        if torch.rand(1) > 0.5:
            state = np.flipud(state).copy()
            if action == 0:
                action = 1
            elif action == 1:
                action = 0
        state = state.transpose(2,0,1)  # (w,h,c) -> (c,w,h)
        """
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
    def __init__(self, num_actions):
        super().__init__()
        layers = 12
        filters = 32
        self.num_actions = num_actions
        self.input_layer = nn.Sequential(
            BasicConv2d(20, filters, (3, 3), bn=False),
            nn.ReLU()
        )

        # self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), False)] * layers)
        # TODO 12層でチャネル数が変わらないけどこれでいいのか？
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), bn=True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, self.num_actions, bias=False)  # for policy(action)
        self.head_v = nn.Linear(filters*2, 1, bias=False)  # for value(reward)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = F.relu_(h + block(h))
        # h = (batch, c, w, h) -> (batch, c) 
        # h:(64,32,32,32)
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)  # h_head: (64, 32)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)  # h_avg: (64, 32)
        p = self.head_p(h_head)  # policy
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], dim=1)))  # value
        return p, v.squeeze(dim=1)

def train_model(model, dataloaders_dict, p_criterion, v_criterion, optimizer, num_epochs):
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
                    # loss = policy_loss + value_loss
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        policy_loss.backward()
                        optimizer.step()

                    epoch_ploss += policy_loss.item() * len(policy)
                    epoch_vloss += value_loss.item() * len(policy)
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_ploss = epoch_ploss / data_size
            epoch_vloss = epoch_vloss / data_size
            epoch_acc = epoch_acc.double() / data_size
        
            if phase=='val':
                wandb.log({'Loss/policy': epoch_ploss, 'Loss/value':epoch_vloss, 'ACC/val': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Loss(value): {epoch_vloss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, 20, 32, 32))
            traced.save(f'best.pth')
            best_acc = epoch_acc

     
def main():
    seed = 42
    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]
    wandb.init(project='lux-ai', entity='kuto5046', group=EXP_NAME) 
    episode_dir = '../../input/lux_ai_toad_episodes_1004/'
    obses, samples = create_dataset_from_json(episode_dir)
    logger.info(f'obses:{len(obses)} samples:{len(samples)}')

    labels = [sample[2] for sample in samples]
    actions = ['north', 'south', 'west', 'east', 'bcity', 'pillage', 'transfer']
    for value, count in zip(*np.unique(labels, return_counts=True)):
        logger.info(f'{actions[value]:^5}: {count:>3}')
    
    model = LuxNet(num_actions=len(actions))
    train, val = train_test_split(samples, test_size=0.1, random_state=seed, stratify=labels)
    batch_size = 1024

    train_loader = DataLoader(
        LuxDataset(obses, train), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        LuxDataset(obses, val), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    p_criterion = nn.CrossEntropyLoss()
    v_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_model(model, dataloaders_dict, p_criterion, v_criterion, optimizer, num_epochs=10)
    wandb.finish()

if __name__ =='__main__':
    main()