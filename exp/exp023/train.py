import logging
import os
import random
import sys
import time
import copy 
from pathlib import Path 
from functools import partial  # pip install functools
from logging import DEBUG, INFO

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from gym import spaces
from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback 
from stable_baselines3.common.utils import set_random_seed
import wandb
from wandb.integration.sb3 import WandbCallback

from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.actions import *
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.game.game import Game
from kaggle_environments import make

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



class MyCustomAgent(AgentWithModel):
    def __init__(self, mode="train", model=None) -> None:
        """
        Implements an agent opponent
        """
        super().__init__(mode, model)
        
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            SpawnCityAction,
        ]
        self.actions_cities = [
            SpawnWorkerAction,
            # SpawnCartAction,
            ResearchAction,
        ]
        self.model = model 
        # self.action_space = spaces.Discrete(max(len(self.actions_units), len(self.actions_cities)))
        self.action_space = spaces.Discrete(len(self.actions_units) + len(self.actions_cities))
        self.num_actions = len(self.actions_units)+len(self.actions_cities)
        self.observation_space = spaces.Box(low=0, high=1, shape=(24, 32, 32), dtype=np.float16)
        

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        pass 

    def get_observation(self, game, unit, city_tile, own_team, is_new_turn):

        """obs情報をnnが学習しやすい形式に変換する関数
        Implements getting a observation from the current game for this unit or city
    
        全て0~1に正規化されている
        1 ch: actionの対象のcitytileの位置(unitが対象の場合zero)
        2 ch: actionの対象のunitの位置(citytileが対象の場合zero)
        3 ch: actionの対象のunitが持つresourceの合計量(/100で正規化？)(citytileが対象の場合zero)

        4 ch: 自チームのworker-unitの位置 
        5 ch: cooldownの状態(/6で正規化)
        6 ch: resourceの合計量(/100で正規化？)(3つまとめて良い？)
        
        7 ch: 敵チームのworker-unitの位置 
        8 ch: cooldownの状態(/6で正規化) (workerはmax=2/cargoはmax=3という認識)
        9 ch: resourceの合計量(/100で正規化？)(3つまとめて良い？)
        
        10 ch: 自チームのcitytileの位置
        11ch: 自チームのcitytileの夜間生存期間
        12ch: cooldown(/10)
        
        13ch: 敵チームのcitytileの位置
        14ch: 敵チームのcitytileの夜間生存期間
        15ch: cooldown(/10)

        16ch: wood量
        17ch: coal量
        18ch: uranium量
        
        19ch: 自チームのresearch point(位置情報はなし)
        20ch: 敵チームのresearch point(位置情報はなし)
        
        21ch: road level

        22ch: 何cycle目かを表す
        23ch: 何step目かを表す
        24ch: map
        """
        opponent_team = abs(own_team - 1)
        width, height = game.map.width, game.map.height

        # mapのサイズを調整するためにshiftするマス数
        # width=20の場合は6 width=21の場合5
        x_shift = (32 - width) // 2
        y_shift = (32 - height) // 2

        # (c, w, h)
        # mapの最大サイズが(32,32)なのでそれに合わせている
        b = np.zeros(self.observation_space.shape, dtype=np.float32)

        # target citytile
        if city_tile is not None:
            b[0, city_tile.pos.x, city_tile.pos.y] = 1 

        # own unit 
        for unit_id, u in game.state["teamStates"][own_team]["units"].items():

            x = u.pos.x + x_shift
            y = u.pos.y + y_shift

            wood = u.cargo["wood"]
            coal = u.cargo["coal"]
            uranium = u.cargo["uranium"]
            cooldown = u.cooldown 

            if unit is not None:
                if unit.id == unit_id:
                    # Position and Cargo
                    b[1:3, x, y] = (
                        1,
                        (wood + coal + uranium) / 100
                    )

            # Worker(own) workerのみならなぜcooldownを6で割る？
            b[3:6, x, y] = (
                1,
                cooldown / 6,
                (wood + coal + uranium) / 100
            )

        # opponent unit
        for unit_id, u in game.state["teamStates"][opponent_team]["units"].items():
            x = u.pos.x + x_shift
            y = u.pos.y + y_shift

            wood = u.cargo["wood"]
            coal = u.cargo["coal"]
            uranium = u.cargo["uranium"]
            cooldown = u.cooldown 

            # Worker
            b[6:9, x, y] = (
                1,
                cooldown / 6,
                (wood + coal + uranium) / 100
            )

        # CityTiles
        lightupkeep = game.configs['parameters']['LIGHT_UPKEEP']['CITY']
        for city_id, city in game.cities.items():
            fuel = city.fuel
            if city.team == own_team:
                for citytile in city.city_cells:
                    x = citytile.pos.x + x_shift
                    y = citytile.pos.y + y_shift
                    cooldown = citytile.city_tile.cooldown
                    b[9:12, x, y] = (
                    1,
                    min(fuel / lightupkeep, 10) / 10,
                    cooldown/10
                )
            else:
                for citytile in city.city_cells:
                    x = citytile.pos.x + x_shift
                    y = citytile.pos.y + y_shift
                    cooldown = citytile.city_tile.cooldown
                    b[12:15, x, y] = (
                    1,
                    min(fuel / lightupkeep, 10) / 10,
                    cooldown/10
                )

        # Resources
        for r in game.map.resources:
            x = r.pos.x + x_shift
            y = r.pos.y + y_shift
            r_type = r.resource.type
            amt = r.resource.amount
            r_dict = {'wood': 15, 'coal': 16, 'uranium': 17}
            b[r_dict[r_type], x, y] = amt / 800

        # Research Points
        own_rp = game.state["teamStates"][own_team]["researchPoints"]
        b[18 + own_team, :] = min(own_rp, 200) / 200
        opp_rp = game.state["teamStates"][opponent_team]["researchPoints"]
        b[18 + opponent_team, :] = min(opp_rp, 200) / 200


        # road cells_with_road; type:set
        if len(game.cells_with_roads) > 0:
            for cell in list(game.cells_with_road):
                x = cell.pos.x + x_shift
                y = cell.pos.y + y_shift
                road_level = cell.road
                b[20, x, y] =  road_level / 6

        # Day/Night Cycle
        b[21, :] = game.state["turn"] % 40 / 40
        # Turns
        b[22, :] = game.state["turn"] / 360
        # Map Size
        b[23, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

        assert (b > 1).sum() == 0  # 正規化がうまくいっていない要素が1つでもあればstop
        return b

    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        # Map action_code index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y
            
            if city_tile != None:
                action =  self.actions_cities[action_code%len(self.actions_cities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            else:
                action =  self.actions_units[action_code%len(self.actions_units)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            
            return action
        except Exception as e:
            # Not a valid action
            logger.info(e)
            return None
    
    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)
    
    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        if is_game_finished:
            if game.get_winning_team() == self.team:
                return 1 # Win!
            else:
                return -1 # Loss

        return 0


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
        n_obs_channel = observation_space.shape[0]
        self.conv0 = BasicConv2d(n_obs_channel, filters, (3, 3), False)
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
        return p # , v.squeeze(dim=1)



def make_env(player, opponent, seed, rank):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = LuxEnvironment(
            configs=LuxMatchConfigs_Default,
            learning_agent=player,
            opponent_agent=opponent
        )
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
    seed = config["basic"]["seed"]
    total_timesteps = config['basic']['total_timesteps']
    num_env = config["basic"]["num_env"]
    ckpt_params = config['callbacks']['checkpoints']
    eval_params = config['callbacks']['eval']
    model_params = config["model"]
    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]
    run = wandb.init(
        project='lux-ai', 
        entity='kuto5046', 
        config=config, 
        group=EXP_NAME, 
        sync_tensorboard=True,# auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
        )
        
    opponent=MyCustomAgent()
    player=MyCustomAgent(mode="train") 
    env = SubprocVecEnv(
        [
            make_env(
            player=copy.deepcopy(player), 
            opponent=copy.deepcopy(opponent),
            seed=seed,
            rank=i) for i in range(num_env)
        ]
    )
    env = VecMonitor(env)
    # env = VecVideoRecorder(
    #     env, 
    #     f"videos/{run.id}", 
    #     record_video_trigger=lambda x: x % 2000 == 0, 
    #     video_length=200
    #     )
        
    # change custom network
    policy_kwargs = dict(
        features_extractor_class=LuxNet,
        features_extractor_kwargs=dict(features_dim=player.num_actions),
    )

    # Attach a ML model from stable_baselines3 and train a RL model
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        **model_params 
        )

    callbacks = []
    callbacks.append(CheckpointCallback(**ckpt_params))
    # callbacks.append(EvalCallback(eval_env, **eval_params))
    callbacks.append(WandbCallback())
    callback = CallbackList(callbacks)
    
    logger.info("Training model ...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )
    model.save('./model/finished_model.zip')
    logger.info("Saved model.")
    run.finish()

if __name__ == "__main__":
    main()
