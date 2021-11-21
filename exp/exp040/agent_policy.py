import sys
import time
from functools import partial  # pip install functools
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import random 
from gym import spaces
sys.path.append("../../LuxPythonEnvGym")
from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy
import torch 
import torch.nn as nn
import torch.nn.functional as F
import gym 
from scipy.special import softmax

def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    """
    Smart-transfers from the specified unit to a nearby neighbor. Prioritizes any
    nearby carts first, then any worker. Transfers the resource type which the unit
    has most of. Picks which cart/worker based on choosing a target that is most-full
    but able to take the most amount of resources.

    Args:
        team ([type]): [description]
        unit_id ([type]): [description]

    Returns:
        Action: Returns a TransferAction object, even if the request is an invalid
                transfer. Use TransferAction.is_valid() to check validity.
    """

    # Calculate how much resources could at-most be transferred
    resource_type = None
    resource_amount = 0
    target_unit = None

    if unit != None:
        for type, amount in unit.cargo.items():
            if amount > resource_amount:
                resource_type = type
                resource_amount = amount

        # Find the best nearby unit to transfer to
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        adjacent_cells = game.map.get_adjacent_cells(unit_cell)

        
        for c in adjacent_cells:
            for id, u in c.units.items():
                # Apply the unit type target restriction
                if target_type_restriction == None or u.type == target_type_restriction:
                    if u.team == team:
                        # This unit belongs to our team, set it as the winning transfer target
                        # if it's the best match.
                        if target_unit is None:
                            target_unit = u
                        else:
                            # Compare this unit to the existing target
                            if target_unit.type == u.type:
                                # Transfer to the target with the least capacity, but can accept
                                # all of our resources
                                if( u.get_cargo_space_left() >= resource_amount and 
                                    target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if u.get_cargo_space_left() < target_unit.get_cargo_space_left():
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u
                                    
                                elif( target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass
                                    
                                elif( u.get_cargo_space_left() > target_unit.get_cargo_space_left() ):
                                    # Change targets, because neither target can accept all our resources and 
                                    # this target can take more resources.
                                    target_unit = u
                            elif u.type == Constants.UNIT_TYPES.CART:
                                # Transfer to this cart instead of the current worker target
                                target_unit = u
    
    # Build the transfer action request
    target_unit_id = None
    if target_unit is not None:
        target_unit_id = target_unit.id

        # Update the transfer amount based on the room of the target
        if target_unit.get_cargo_space_left() < resource_amount:
            resource_amount = target_unit.get_cargo_space_left()
    
    return TransferAction(team, unit_id, target_unit_id, resource_type, resource_amount)

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
        self.head = nn.Sequential(
            nn.Linear(filters*2, features_dim, bias=False),
            # nn.Linear(filters, features_dim, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)  # (filter)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        output = self.head(torch.cat([h_head, h_avg], dim=1))  # filter -> features_dim
        return output

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
            nn.Linear(features_dim, last_layer_dim_pi), 
            nn.BatchNorm1d(last_layer_dim_pi),
            nn.ReLU(),
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
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMlpExtractor(self.features_dim)


class AgentPolicy(AgentWithModel):
    def __init__(self, mode="train", model=None, _n_obs_channel=23, n_stack=1) -> None:
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
            # partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART), # Transfer to nearby cart
            # partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER), # Transfer to nearby worker
            SpawnCityAction,
            # PillageAction,
        ]
        self.action_space = spaces.Discrete(len(self.actions_units))
        self.n_stack = n_stack
        self._n_obs_channel = _n_obs_channel  #  28  # base obs
        self.n_obs_channel = self._n_obs_channel + (8 * (self.n_stack-1))  # base obs + last obs
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_obs_channel, 32, 32), dtype=np.float32)
        self.object_nodes = {}
        # self.tta = TTA()

    def onnx_predict(self, input):
        policy, value = self.model.run(None, {"input.1": np.expand_dims(input.astype(np.float32), 0)})
        return policy[0], value[0]

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

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        self.units_last = 0
        self.city_tiles_last = 0
        self.fuel_collected_last = 0
        self.research_points_last = 0
        self.rewards = {}
        self.last_unit_obs = [np.zeros((8, 32, 32)) for i in range(self.n_stack-1)]

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

    def process_unit_turn(self, game, team, base_obs):
        actions = []
        units = game.get_teams_units(team)
        for unit in units.values():
            if unit.can_act():
                obs = self.get_observation(game, unit, None, unit.team, False, base_obs)
                
                # print(tensor_obs.shape)
                # policy = self.model.policy.get_distribution(tensor_obs)
                # is_onnx = False
                # obs1 = self.tta.vertical_flip(obs)
                # obs2 = self.tta.horizontal_flip(obs)
                # obs3 = self.tta.all_flip(obs)
                # if is_onnx:
                #     stack_obs = np.stack([obs, obs1, obs2, obs3], axis=0)
                #     self.onnx_predict(stack_obs)
                # else:
                # tensor_obs = torch.from_numpy(np.stack([obs, obs1, obs2, obs3], axis=0))
                # tensor_obs = torch.from_numpy(np.stack([obs, obs], axis=0))
                # tensor_obs = torch.from_numpy(np.stack([obs, obs], axis=0))
                # features = self.model.policy.extract_features(tensor_obs)
                # latent_pi, latent_vi = self.model.policy.mlp_extractor(features)
                # policy = self.model.policy.action_net(latent_pi)[0].detach().numpy()
                action_code = self.model.predict(obs)[0]
                action = self.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=unit.team)
                if action is not None:
                    actions.append(action)
                # policies[1] = self.tta.vertical_convert_action(policies[1]) 
                # policies[2] = self.tta.horizontal_convert_action(policies[2]) 
                # policies[3] = self.tta.all_convert_action(policies[3])
                # policy = np.mean(policies, axis=0)
                # for action_code in np.argsort(policy)[::-1]:
                #     # 夜でcity上にいない場合はbuild cityはしない
                #     # if (action_code == 6)&(game.is_night())&(not game.game_map_by_pos(unit.pos).is_city_tile):
                #     #     continue 
                #     action = self.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=unit.team)
                #     if action.is_valid(game, actions):
                #         actions.append(action)
                #         break      
        return actions 

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference. Generally
        don't modify this part of the code.
        Returns: Array of actions to perform.
        """
        start_time = time.time()
        base_obs = self.get_base_observation(game, team, self.last_unit_obs)
        unit_actions = self.process_unit_turn(game, team, base_obs)
        city_actions = self.process_city_turn(game, team)
        actions = unit_actions + city_actions

        if self.n_stack > 1:
            self.get_last_observation(base_obs)
        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print("[RL Agent]WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second." % time_taken,
                  file=sys.stderr)
        return actions

    def get_last_observation(self, obs):
        current_unit_obs = obs[[2,4,5,7,8,9,11,12]]  # own_unit/opponent_unit/own_citytile/opponent_citytile
        # assert np.sum(current_unit_obs > 1) == 0
        self.last_unit_obs.append(current_unit_obs)
        if len(self.last_unit_obs)>=self.n_stack:  # 過去情報をn_stack分に保つ
            self.last_unit_obs.pop(0)
        assert len(self.last_unit_obs) == self.n_stack - 1

    def prob_unit_destroy_next_turn(self, game, _unit):
        """

        """
        # resource < 4 & is not on city tile & is not adjacent resource
        current_cell = game.map.get_cell(_unit.pos.x, _unit.pos.y)
        current_adjacent_cells = game.map.get_adjacent_cells(current_cell)
        current_resource_count = np.sum([cell.resource is not None for cell in current_adjacent_cells]*1)
        if (np.sum(list(_unit.cargo.values())) < 4)& \
            (not current_cell.is_city_tile) & \
            (current_resource_count == 0):

            adjacent_city_tile_count = 0
            for cell in current_adjacent_cells:
                if cell.is_city_tile:
                    adjacent_city_tile_count += 1

            next_resource_count = 0
            for d in ['c', 'n', 'w', 's', 'e']:
                next_adjacent_cells = game.map.get_adjacent_cells(_unit.pos.translate(d, 1))
                for cell in next_adjacent_cells:
                    if cell.resource is not None:
                        next_resource_count += 1

            # 必ず消滅する
            # 隣接cellにもcitytileがない&resourceに隣接していない場合
            if (adjacent_city_tile_count == 0)|(next_resource_count == 0):
                return 1
            else:  # 消失する可能性がある
                return 0.5
        else:  # 消失しない
            return 0

    def get_convert_fuel_loss(self, _unit):
        # convert loss from resource to fuel in night turn      
        coal_loss = 0
        uranium_loss = 0 
        consume_resource_in_night = {'wood': 0,  'coal': 0, 'uranium': 0}
        for r_name, amt in _unit.cargo.items():
            for i in range(amt):
                if np.sum(list(consume_resource_in_night.values())) < 4:
                    consume_resource_in_night[r_name] += 1
                else:
                    break
        # wood's fuel rate is 1, so wood_loss is always zero.
        coal_loss = (GAME_CONSTANTS["PARAMETERS"]["RESOURCE_TO_FUEL_RATE"]["COAL"] - 1)*consume_resource_in_night["coal"]
        uranium_loss = (GAME_CONSTANTS["PARAMETERS"]["RESOURCE_TO_FUEL_RATE"]["URANIUM"] - 1)*consume_resource_in_night["uranium"]
        loss = coal_loss + uranium_loss
        return loss 
   
    def get_base_observation(self, game, team, last_unit_obs):
        """
        Implements getting a observation from the current game for this unit or city
        """

        height = game.map.height
        width = game.map.width

        x_shift = (32 - width) // 2
        y_shift = (32 - height) // 2

        b = np.zeros((self._n_obs_channel, 32, 32), dtype=np.float32)
        opponent_team = 1 - team
        
        # unit
        for _unit in game.state["teamStates"][team]["units"].values():
            x = _unit.pos.x + x_shift
            y = _unit.pos.y + y_shift
            cooldown = _unit.cooldown / 6
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / cap
            b[2:5, x,y] += (1, cooldown, resource)

            # in night
            # if game.is_night():
            #     loss = self.get_convert_fuel_loss(_unit)
            #     b[23, x,y] = loss / 156  # max is 4*(40-1)=156
        
            # b[26,x,y] = self.prob_unit_destroy_next_turn(game, _unit)

        for _unit in game.state["teamStates"][opponent_team]["units"].values():
            x = _unit.pos.x + x_shift
            y = _unit.pos.y + y_shift
            cooldown = _unit.cooldown / 6
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"] 
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / cap
            b[5:8, x,y] += (1, cooldown, resource)
            
            # b[27,x,y] = self.prob_unit_destroy_next_turn(game, _unit)

        own_city_tile_count = 0
        for city in game.cities.values():
            for cell in city.city_cells:
                if city.team == team:
                    own_city_tile_count += 1

        # city tile
        own_unit_count = len(game.state["teamStates"][team]["units"].values())
        own_incremental_rp = 0
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
                    b[8:11, x, y] = (1, fuel_ratio, cooldown)
                    # 現ターンのcity行動により増えるunit
                    # if own_unit_count < own_city_tile_count:
                    #     b[24, x,y] = 1
                    #     own_unit_count += 1    
                    # elif game.state["teamStates"][team]["researchPoints"] < 200:
                    #     own_incremental_rp += 1
                else:
                    b[11:14, x, y] = (1, fuel_ratio, cooldown)
   
        # 現ターンのcity行動により増えるrp
        # b[25, :] = min(game.state["teamStates"][team]["researchPoints"]+own_incremental_rp, 200) / 200

        # resource
        resource_dict = {'wood': 14, 'coal': 15, 'uranium': 16}
        for cell in game.map.resources:
            x = cell.pos.x + x_shift
            y = cell.pos.y + y_shift
            r_type = cell.resource.type
            amount = cell.resource.amount / 800
            idx = resource_dict[r_type]
            b[idx, x, y] = amount
        
        # research points
        max_rp = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]
        b[17, :] = min(game.state["teamStates"][team]["researchPoints"], max_rp) / max_rp
        b[18, :] = min(game.state["teamStates"][opponent_team]["researchPoints"], max_rp) / max_rp
        
        # road
        for row in game.map.map:
            for cell in row:
                if cell.road > 0:
                    x = cell.pos.x + x_shift
                    y = cell.pos.y + y_shift
                    b[19, x,y] = cell.road / 6

        # cycle
        cycle = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        b[20, :] = game.state["turn"] % cycle / cycle
        b[21, :] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        
        # map
        b[22, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

        if self.n_stack > 1:
            additional_obs = np.concatenate(last_unit_obs, axis=0)
            b = np.concatenate([b, additional_obs], axis=0)

        assert b.shape == self.observation_space.shape
        return b 

    def get_observation(self, game, unit, citytile, team, new_turn, base_state):
        """
        Implements getting a observation from the current game for this unit or city
        """

        height = game.map.height
        width = game.map.width

        x_shift = (32 - width) // 2
        y_shift = (32 - height) // 2

        b = base_state.copy()
        
        # target unit
        if unit is not None:
            x = unit.pos.x + x_shift
            y = unit.pos.y + y_shift
            max_cooldown = GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            cooldown = np.array(unit.cooldown / max_cooldown, dtype=np.float32)
            resource = np.array((unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]) / cap, dtype=np.float32)
            b[:2, x, y] = (1, resource)
            b[2:5, x, y] -= (1, cooldown, resource)

        return b 

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


class TTA:            
    def vertical_flip(self, state):
        """
        swap north(=0) and south(=2)
        """
        # flip up/down
        state = state.transpose(2,1,0)  #(c,x,y) -> (y,x,c)
        state = np.flipud(state).copy()
        state = state.transpose(2,1,0)  # (w,h,c) -> (c,w,h)
        return state

    def horizontal_flip(self, state):
        """
        swap west(=1) and east(=3)
        """
        # flip left/right
        state = state.transpose(2,1,0) #(x,y,c) -> (y,x,c)
        state = np.fliplr(state).copy()
        state = state.transpose(2,1,0)  # (w,h,c) -> (c,w,h)
        return state
    
    def all_flip(self, state):
        state = self.vertical_flip(state)
        state = self.horizontal_flip(state)
        return state

    def random_roll(self, state):
        n = random.randint(-5, 5)
        m = random.randint(-5, 5)
        return np.roll(state, (n,m), axis=(1,2))

    def vertical_convert_action(self, action):
        # order = [2,1,0,3,4]
        order = [0,3,2,1,4,5,6]
        return action[order]

    def horizontal_convert_action(self, action):
        # order = [0,3,2,1,4]
        order = [0,1,4,3,2,5,6]
        return action[order]
    
    def all_convert_action(self, action):
        # order = [2,3,0,1,4]
        order = [0,3,4,1,2,5,6]
        return action[order]