import sys
import time
from functools import partial  # pip install functools
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from gym import spaces
sys.path.append("../../LuxPythonEnvGym/")
from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
import torch 
import torch.nn as nn
import torch.nn.functional as F
import gym 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy

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
  
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        n_obs_channel = observation_space["obs"].shape[0]
        n_global_obs_channel = observation_space["global_obs"].shape[0]
        bilinear = True 
        self.inc = DoubleConv(n_obs_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)  # h,wは変化しなくない？
        factor = 2 if bilinear else 1
        self.up1 = Up(512+n_global_obs_channel, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 6)

    def forward(self, input):
        x1 = self.inc(input["obs"])
        x2 = self.down1(x1)
        x3 = self.down2(x2)  # (8,8)
        x4 = torch.cat([self.down3(x3), input["global_obs"]], dim=1)  # (batch,256,4,4) + (batch,8,4,4)　
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        policy_logits = self.outc(x)  # (batch,6,32,32)        
        value_logits = x4.view(x4.shape[0], x4.shape[1], -1).mean(-1)  # (batch, 256+8) 
        return (policy_logits, value_logits, input["mask"])


class CustomMlpExtractor(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        features_dim: int,
        last_layer_dim_pi: int = 6,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomMlpExtractor, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network(3次元を1次元にする)
        # self.policy_net = nn.Sequential(

        # )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, last_layer_dim_vf),
            nn.BatchNorm1d(last_layer_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features):
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        batch_size = features[0].shape[0]
        return features[0][features[2] > 0].view(batch_size, 6), self.value_net(features[1])

    def forward_actor(self, features):
        batch_size = features[0].shape[0]
        return features[0][features[2] > 0].view(batch_size, 6)

    def forward_critic(self, features):
        return self.value_net(features[1])

class CustomActorCriticCnnPolicy(ActorCriticCnnPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMlpExtractor(self.features_dim)

class AgentPolicy(AgentWithModel):
    def __init__(self, mode="train", model=None) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
        """
        super().__init__(mode, model)

        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            SpawnCityAction,
        ]
        self.action_space = spaces.Discrete(len(self.actions_units))
        self.observation_space = spaces.Dict(
                {"obs":spaces.Box(low=0, high=1, shape=(17, 32, 32), dtype=np.float32), 
                "global_obs":spaces.Box(low=0, high=1, shape=(8, 4, 4), dtype=np.float32),
                "mask":spaces.Box(low=0, high=1, shape=(len(self.actions_units), 32, 32), dtype=np.long),
                })
        self.model = model

    def torch_predict(self, obs, global_obs, mask):
        with torch.no_grad():
            policy = self.model({
                "obs": torch.from_numpy(obs), 
                "global_obs": torch.from_numpy(global_obs), 
                "mask": torch.from_numpy(self.observation_space["mask"].sample),  # not use
                })
        return policy

    def set_model(self, model_path):
        self.model = torch.jit.load(model_path)
        print(f"[Self-Play Agent] set model by {model_path}")

    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT

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
            
            if unit is not None:
                action = self.actions_units[action_code%len(self.actions_units)](
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
            return None 
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)

    def process_city_turn(self, game, team):
        actions = []
        unit_count = len(game.get_teams_units(team))
        # city
        city_tile_count = 0
        for city in game.cities.values():
            for cell in city.city_cells:
                if city.team == team:
                    city_tile_count += 1

        # Inference the model per-city
        cities = game.cities.values()
        for city in cities:
            if city.team == team:
                for cell in city.city_cells:
                    city_tile = cell.city_tile
                    if city_tile.can_act():
                        x = city_tile.pos.x
                        y = city_tile.pos.y
                        # 保有unit数(worker)よりもcity tileの数が多いならworkerを追加
                        if unit_count < city_tile_count:
                            action = SpawnWorkerAction(team, None, x, y)
                            actions.append(action)
                            unit_count += 1
                        # # ウランの研究に必要な数のresearch pointを満たしていなければ研究をしてresearch pointを増やす
                        elif (game.state["teamStates"][team]["researchPoints"] < 200):
                            action = ResearchAction(team, x, y, None)
                            actions.append(action)
                            game.state["teamStates"][team]["researchPoints"] += 1

        return actions 

    def process_unit_turn(self, game, team):
        actions = []
        x_shift = (32 - game.map.width) // 2
        y_shift = (32 - game.map.height) // 2
        obs, global_obs = self.get_observation(game, team)
        policy_map = self.torch_predict(obs, global_obs)
        units = game.get_teams_units(team)
        for unit in units.values():
            if unit.can_act():
                x = unit.pos.x + x_shift
                y = unit.pos.y + y_shift
                policy = policy_map[:,x,y]
                for action_code in np.argsort(policy)[::-1]:
                    action = self.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=unit.team)
                    if action.is_valid(game, actions):
                        actions.append(action)
                        break      
        return actions 

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference. Generally
        don't modify this part of the code.
        Returns: Array of actions to perform.
        """
        start_time = time.time()
        unit_actions = self.process_unit_turn(game, team)
        city_actions = self.process_city_turn(game, team)
        actions = unit_actions + city_actions

        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print("[RL Agent]WARNING: Inference took %.3f seconds for computing actions. Limit is 3 second." % time_taken,
                  file=sys.stderr)
        return actions

    def get_observation(self, game, unit=None, city_tile=None, team=None):
        """
        Implements getting a observation from the current game for this unit or city
        """

        height = game.map.height
        width = game.map.width

        x_shift = (32 - width) // 2
        y_shift = (32 - height) // 2

        b = np.zeros(self.observation_space["obs"].shape, dtype=np.float32)
        global_b = np.zeros(self.observation_space["global_obs"].shape, dtype=np.float32)
        action_mask = np.zeros(self.observation_space["mask"].shape, dtype=np.float32)
        opponent_team = 1 - team
        # unit
        for _unit in game.state["teamStates"][team]["units"].values():
            x = _unit.pos.x + x_shift
            y = _unit.pos.y + y_shift
            cooldown = _unit.cooldown / 6
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / cap
            b[0:3, x,y] += (1, cooldown, resource)

            if unit is not None:
                if unit.id == _unit.id:
                    action_mask[:,x,y] = 1

        for _unit in game.state["teamStates"][opponent_team]["units"].values():
            x = _unit.pos.x + x_shift
            y = _unit.pos.y + y_shift
            cooldown = _unit.cooldown / 6
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"] 
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / cap
            b[3:6, x,y] += (1, cooldown, resource)

        # city tile
        own_unit_count = len(game.state["teamStates"][team]["units"].values())
        opponent_unit_count = len(game.state["teamStates"][opponent_team]["units"].values())
        own_citytile_count = 0
        opponent_citytile_count = 0
        for city in game.cities.values():
            fuel = city.fuel
            lightupkeep = city.get_light_upkeep()
            max_cooldown = GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]
            fuel_ratio = min(fuel / lightupkeep, max_cooldown) / max_cooldown
            for cell in city.city_cells:
                x = cell.pos.x + x_shift
                y = cell.pos.y + y_shift
                cooldown = cell.city_tile.cooldown / max_cooldown
                if city.team == team:
                    b[6:9, x, y] = (1, fuel_ratio, cooldown)
                    own_citytile_count += 1
                else:
                    b[9:12, x, y] = (1, fuel_ratio, cooldown)
                    opponent_citytile_count += 1

        # resource
        resource_dict = {'wood': 12, 'coal': 13, 'uranium': 14}
        for cell in game.map.resources:
            x = cell.pos.x + x_shift
            y = cell.pos.y + y_shift
            r_type = cell.resource.type
            amount = cell.resource.amount / 800
            idx = resource_dict[r_type]
            b[idx, x, y] = amount
        
        # road
        for row in game.map.map:
            for cell in row:
                if cell.road > 0:
                    x = cell.pos.x + x_shift
                    y = cell.pos.y + y_shift
                    b[15, x,y] = cell.road / 6

        # map
        b[16, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1
        
        # research points
        max_rp = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]
        global_b[0, :] = min(game.state["teamStates"][team]["researchPoints"], max_rp) / max_rp
        global_b[1, :] = min(game.state["teamStates"][opponent_team]["researchPoints"], max_rp) / max_rp
        
        # cycle
        cycle = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        global_b[2, :] = game.state["turn"] % cycle / cycle
        global_b[3, :] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        global_b[4, :] = own_unit_count / 100
        global_b[5, :] = opponent_unit_count / 100
        global_b[6, :] = own_citytile_count / 100
        global_b[7, :] = opponent_citytile_count / 100
        assert np.sum(action_mask) > 0
        return {"obs":b, "global_obs": global_b, "mask": action_mask}

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        self.rewards = {
            # "rew/r_city_tiles": 0,
            # "rew/r_research_points_coal_flag": 0,
            # "rew/r_research_points_uranium_flag": 0,
            # "rew/r_fuel_collected": 0,
            # "rew/r_city_tiles_end": 0,
            "rew/r_game_win": 0
            }
            
        turn = game.state["turn"]
        turn_decay = (360-turn)/360

        if is_game_error:
            # Game environment step failed, assign a game lost reward to not incentivise this
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn or at game end
            return 0

        # Get some basic stats
        # unit_count = len(game.state["teamStates"][self.team]["units"])
        # city_count = 0
        # city_count_opponent = 0
        # city_tile_count = 0
        # city_tile_count_opponent = 0
        # for city in game.cities.values():
        #     if city.team == self.team:
        #         city_count += 1
        #     else:
        #         city_count_opponent += 1

        #     for cell in city.city_cells:
        #         if city.team == self.team:
        #             city_tile_count += 1
        #         else:
        #             city_tile_count_opponent += 1
                
        # Give a reward for unit creation/death. 0.05 reward per unit.
        # self.rewards["rew/r_units"] = (unit_count - self.units_last) * 0.005
        # self.units_last = unit_count

        # Give a reward for city creation/death. 0.1 reward per city.
        # self.rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1

        # number of citytile after night
        # if (turn > 0)&(turn % 40 == 0):
        #     self.rewards["rew/r_city_tiles"] += (city_tile_count - city_tile_count_opponent) * 0.01
        #     # self.rewards["rew/r_city_tiles"] += (city_tile_count - self.city_tiles_last) * 0.01
        #     self.city_tiles_last = city_tile_count

        # Reward collecting fuel
        # fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        # self.rewards["rew/r_fuel_collected"] += ( (fuel_collected - self.fuel_collected_last) / 20000 )
        # self.fuel_collected_last = fuel_collected

        # Reward for Research Points
        # research_points = game.state["teamStates"][self.team]["researchPoints"]
        # self.rewards["rew/r_research_points"] = (research_points - self.research_points_last) / 200  # 0.005
        # if (research_points == 50)&(self.research_points_last < 50):
        #     self.rewards["rew/r_research_points_coal_flag"] += 0.25 * turn_decay
        # elif (research_points == 200)&(self.research_points_last < 200):
        #     self.rewards["rew/r_research_points_uranium_flag"] += 1 * turn_decay
        # self.research_points_last = research_points

        # Give a reward of 1.0 per city tile alive at the end of the game
        if is_game_finished:
            # self.is_last_turn = True
            # clip = 10
            # self.rewards["rew/r_city_tiles_end"] += np.clip(city_tile_count - city_tile_count_opponent, -clip, clip)
            
            if game.get_winning_team() == self.team:
                self.rewards["rew/r_game_win"] += 1 # Win
            else:
                self.rewards["rew/r_game_win"] -= 1 # Loss

        reward = 0
        for name, value in self.rewards.items():
            reward += value

        return reward

    def turn_heurstics(self, game, is_first_turn):
        """
        This is called pre-observation actions to allow for hardcoded heuristics
        to control a subset of units. Any unit or city that gets an action from this
        callback, will not create an observation+action.
        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        actions = self.process_city_turn(game, self.team)
        self.match_controller.take_actions(actions)
