import numpy as np
import json
from pathlib import Path
import os
import wandb
import random
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

def to_label(action):
    strs = action.split(' ')
    unit_id = strs[1]
    if strs[0] == 'm':
        label = {'c': None, 'n': 0, 's': 1, 'w': 2, 'e': 3}[strs[2]]
    elif strs[0] == 'bcity':
        label = 4
    else:
        label = None
    return unit_id, label


def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True


def create_dataset_from_json(episode_dir, team_name='Toad Brigade'): 
    obses = {}
    samples = []
    append = samples.append
    
    episodes = [path for path in Path(episode_dir).glob('*.json') if 'output' not in path.name]
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        index = np.argmax([r or 0 for r in json_load['rewards']])
        if json_load['info']['TeamNames'][index] != team_name:
            continue

        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                actions = json_load['steps'][i+1][index]['action']
                obs = json_load['steps'][i][0]['observation']
                
                if depleted_resources(obs):
                    break
                
                obs['player'] = index
                obs = dict([
                    (k,v) for k,v in obs.items() 
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])
                obs_id = f'{ep_id}_{i}'
                obses[obs_id] = obs
                                
                for action in actions:
                    unit_id, label = to_label(action)
                    if label is not None:
                        append((obs_id, unit_id, label))

    return obses, samples


# Input for Neural Network
def make_input(obs, unit_id):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
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
    # Map Size
    b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return b


class LuxDataset(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, unit_id, action = self.samples[idx]
        obs = self.obses[obs_id]
        state = make_input(obs, unit_id)
        
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


class LuxNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = BasicConv2d(20, filters, (3, 3), False)
        # self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), False)] * layers)
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 5, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
            
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        p = self.head_p(h_head)
        return p

def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.cuda()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_loss = 0.0
            epoch_acc = 0
            
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states = item[0].cuda().float()
                actions = item[1].cuda().long()

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    policy = model(states)
                    loss = criterion(policy, actions)
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(policy)
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size
        
            if phase=='val':
                wandb.log({'Loss/val': epoch_loss, 'ACC/val': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, 20, 32, 32))
            traced.save('model.pth')
            best_acc = epoch_acc

     
def main():
    seed = 42
    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]
    wandb.init(project='lux-ai', entity='kuto5046', group=EXP_NAME) 
    episode_dir = '../../input/lux_ai_tigga_episodes_0922/'
    obses, samples = create_dataset_from_json(episode_dir, team_name="Tigga")
    logger.info('obses:', len(obses), 'samples:', len(samples))

    labels = [sample[-1] for sample in samples]
    actions = ['north', 'south', 'west', 'east', 'bcity']
    for value, count in zip(*np.unique(labels, return_counts=True)):
        logger.info(f'{actions[value]:^5}: {count:>3}')
    
    model = LuxNet()
    train, val = train_test_split(samples, test_size=0.1, random_state=seed, stratify=labels)
    batch_size = 64

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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=30)
    wandb.finish()

if __name__ =='__main__':
    main()