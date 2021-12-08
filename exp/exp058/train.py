import numpy as np
import json
from pathlib import Path
import pickle
import os
import shutil
import copy 
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
from agent_policy import CustomActorCriticCnnPolicy, CustomFeatureExtractor
from imitation.algorithms.bc import BC, ConstantLRSchedule
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
sys.path.append("../../LuxPythonEnvGym/")
from luxai2021.game.constants import LuxMatchConfigs_Default, LuxMatchConfigs_Replay
from luxai2021.game.game import Game
from agent_policy import ImitationAgent

class RLPolicy(torch.nn.Module):
    def __init__(self, feature_extractor, mlp_extractor, action_net, value_net):
        super(RLPolicy, self).__init__()
        self.feature_extractor = feature_extractor
        self.mlp_extractor = mlp_extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, observation):
        features = self.feature_extractor(observation)
        policy_latent, value_latent = self.mlp_extractor(features)
        return self.action_net(policy_latent), self.value_net(value_latent)

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
    # episodes = episodes[:50]
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)

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


class LuxDataset(Dataset):
    def __init__(self, df, data_dir,phase='train'):
        self.obs_ids = df['obs_id'].to_numpy()
        self.unit_ids = df['unit_id'].to_numpy()
        self.actions = df['label'].to_numpy()
        self.data_dir = data_dir
        self.phase = phase
        self.agent = ImitationAgent()
        
    def __len__(self):
        return len(self.obs_ids)

    def __getitem__(self, idx):
        obs_id = self.obs_ids[idx]
        with open(self.data_dir + f"{obs_id}.pickle", mode="rb") as f:    
            (obs, acts) = pickle.load(f)
        unit_id = self.unit_ids[idx]
        action = self.actions[idx]

        configs = LuxMatchConfigs_Replay
        configs["width"] = obs["width"]
        configs["height"] = obs["height"]
        game = Game(configs)
        game.reset(obs["updates"])
        game.state["turn"] = obs["step"]
        team = obs["player"]
        unit = game.get_unit(team, unit_id)
        base_obs = self.agent.get_base_observation(game, team)
        obs = self.agent.get_observation(game, unit, base_obs)
        return {"obs": obs, "action": action}


""" Parts of the U-Net model """
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
                
                obses = item["obs"].cuda().float()
                actions = item["action"].cuda().long()
                batch_size = item["obs"].shape[0]
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    feats = model.features_extractor(obses)  # (batch*4,3)
                    policy_latent, value_latent = model.mlp_extractor(feats)
                    policy = model.action_net(policy_latent)
                    value = model.value_net(value_latent)

                    policy_loss = p_criterion(policy, actions)
                    loss = policy_loss
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    preds = torch.max(policy, 1)[1]
                    epoch_ploss += policy_loss.item() * batch_size
                    num_correct = (preds.cpu() == actions.data.cpu())
                    epoch_num_correct += torch.sum(num_correct)
        
            epoch_data_size = len(dataloader.dataset)
            epoch_ploss = epoch_ploss / epoch_data_size
            epoch_acc = epoch_num_correct.double() / epoch_data_size

            if phase == 'train':
                wandb.log({'Loss/train': epoch_ploss, 'ACC/train': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Acc: {epoch_acc:.4f}')
            elif phase=='val':
                wandb.log({'Loss/val': epoch_ploss, 'ACC/val': epoch_acc})
                logger.info(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss(policy): {epoch_ploss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            torch.save(model.cpu().state_dict(), './models/best.pth')
            dummy_obs = torch.rand(1, 26, 32, 32)

            rl_model = RLPolicy(
                model.features_extractor, 
                model.mlp_extractor,
                model.action_net,
                model.value_net
            )
            copy_model = copy.deepcopy(rl_model)  # c
            torch.onnx.export(copy_model.cpu(), dummy_obs, f"./models/simple_policy.onnx", opset_version=11)
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
    run_id = f'simple_model_{target_team_name}_v2'
    wandb.init(project='lux-ai', entity='kuto5046', group=EXP_NAME, id=run_id)  #, mode="disabled") 

    episode_dir = "../../input/lux_ai_top_team_episodes_1129/"
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
        num_workers=8
    )
    val_loader = DataLoader(
        LuxDataset(val_df, data_dir, phase='val'), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8
    )
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    p_criterion = nn.CrossEntropyLoss()

    n_obs_channel = 26
    action_space = spaces.Discrete(6)
    observation_space = spaces.Box(low=0, high=1, shape=(n_obs_channel, 32, 32), dtype=np.float32)

    model = CustomActorCriticCnnPolicy(
        observation_space=observation_space, 
        action_space=action_space, 
        lr_schedule=ConstantLRSchedule(lr=1e-5),
        net_arch = [],
        optimizer_class=torch.optim.AdamW,
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128)
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_model(model, dataloaders_dict, p_criterion, optimizer, num_epochs=10)
    wandb.finish()

if __name__ == '__main__':
    main()