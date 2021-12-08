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
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


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


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        layers, filters = 12, 32
        self.n_obs_channel = observation_space.shape[0]
        self.conv0 = BasicConv2d(self.n_obs_channel, filters, (3, 3), False)
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 3, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)  # (filter)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = torch.softmax(self.head_p(h_head), 1)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], dim=1)))
        return (p,v)


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
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomMlpExtractor, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            # nn.Linear(features_dim, last_layer_dim_pi), 
            # nn.BatchNorm1d(last_layer_dim_pi),
            # nn.ReLU(),
        )
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
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class CustomActorCriticCnnPolicy(ActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # # Disable orthogonal initialization
        # self.ortho_init = False
        
        # don't anything in action net
        self.action_net = nn.Sequential()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMlpExtractor(self.features_dim)


class ImitationAgent(Agent):
    def __init__(self, model=None) -> None:
        super().__init__()
        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            SpawnCityAction,
        ]
        self.action_space = spaces.Discrete(len(self.actions_units))
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 23, 32, 32), dtype=np.float32)
        self.model = model

    def onnx_predict(self, input):
        policy, value  = self.model.run(None, {"input.1": input})
        return policy[0], value[0]
    
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
                action = self.actions_units[action_code](
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
                        elif game.state["teamStates"][team]["researchPoints"] < 200:
                            action = ResearchAction(team, x, y, None)
                            actions.append(action)
                            game.state["teamStates"][team]["researchPoints"] += 1

        return actions 

    def process_unit_turn(self, game, team):
        actions = []
        x_shift = (32 - game.map.width) // 2
        y_shift = (32 - game.map.height) // 2
        obs = self.get_observation(game, None, None, team)
        # policy_map = self.torch_predict(obs)
        policy_map, value = self.onnx_predict(obs)
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
        if time_taken > 1.0:  # Warn if larger than 0.5 seconds.
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

        b = np.zeros(self.observation_space["obs"].shape[1:], dtype=np.float32)
        global_b = np.zeros(self.observation_space["global_obs"].shape[1:], dtype=np.float32)
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

        north_obs = b.copy()
        west_obs = np.rot90(north_obs, 1, axes=(1,2)).copy()
        south_obs = np.rot90(north_obs, 2, axes=(1,2)).copy()
        east_obs = np.rot90(north_obs, 3, axes=(1,2)).copy()
        obses = np.stack([north_obs, west_obs, south_obs, east_obs])
        global_obses = np.stack([global_b]*4)
        return {"obs":obses, "global_obs": global_obses, "mask": action_mask}