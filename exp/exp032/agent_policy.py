import sys
import time
from functools import partial  # pip install functools
import copy
import random

import numpy as np
from gym import spaces
sys.path.append("../../LuxPythonEnvGym")
from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch.nn.functional as F

# https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)
def furthest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmax(dist_2)

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
        layers, filters = 6, 32
        self.n_obs_channel = observation_space.shape[0]
        self.conv0 = BasicConv2d(self.n_obs_channel, filters, (3, 3), False)
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

########################################################################################################################
# This is the Agent that you need to design for the competition
########################################################################################################################
class AgentPolicy(AgentWithModel):
    def __init__(self, mode="train", model=None, n_stack=4) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
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
            # partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART), # Transfer to nearby cart
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER), # Transfer to nearby worker
            SpawnCityAction,
            # PillageAction,
        ]
        self.actions_cities = [
            SpawnWorkerAction,
            # SpawnCartAction,
            ResearchAction,
        ]
        self.action_space = spaces.Discrete(max(len(self.actions_units), len(self.actions_cities)))
        self.n_stack = n_stack
        self._n_obs_channel = 25  # base obs
        self.n_obs_channel = self._n_obs_channel + (2 * (self.n_stack-1))  # base obs + last obs
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_obs_channel, 32, 32), dtype=np.float16)
        
        self.object_nodes = {}

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
        self.last_unit_obs = [np.zeros((2, 32, 32)) for i in range(self.n_stack-1)]

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference. Generally
        don't modify this part of the code.
        Returns: Array of actions to perform.
        """
        start_time = time.time()
        actions = []
        new_turn = True
        # Inference the model per-unit
        # obs = np.zeros(self.observation_space.shape)
        base_obs = self.get_base_observation(game, team, self.last_unit_obs)
        units = game.state["teamStates"][team]["units"].values()
        for unit in units:
            if unit.can_act():
                obs = self.get_observation(game, unit, None, unit.team, new_turn, base_obs)
                action_code, _states = self.model.predict(obs, deterministic=False)
                if action_code is not None:
                    actions.append(
                        self.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=unit.team))
                new_turn = False

        # Inference the model per-city
        cities = game.cities.values()
        for city in cities:
            if city.team == team:
                for cell in city.city_cells:
                    city_tile = cell.city_tile
                    if city_tile.can_act():
                        obs = self.get_observation(game, None, city_tile, city.team, new_turn, base_obs)
                        action_code, _states = self.model.predict(obs, deterministic=False)
                        if action_code is not None:
                            actions.append(
                                self.action_code_to_action(action_code, game=game, unit=None, city_tile=city_tile,
                                                           team=city.team))
                        new_turn = False
        if self.n_stack > 1:
            self.get_last_observation(base_obs)
        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print("WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second." % time_taken,
                  file=sys.stderr)

        return actions

    def get_last_observation(self, obs):
        current_unit_obs = np.array([obs[2], obs[5]])  # own unit pos, opponent unit pos
        # assert np.sum(current_unit_obs > 1) == 0
        self.last_unit_obs.append(current_unit_obs)
        if len(self.last_unit_obs)>=self.n_stack:  # 過去情報をn_stack分に保つ
            self.last_unit_obs.pop(0)
        assert len(self.last_unit_obs) == self.n_stack - 1

    def get_base_observation(self, game, team, last_unit_obs):
        """
         Implements getting a observation from the current game for this unit or city
         0ch: target unit pos
         1ch: target unit resource

         2ch: own unit pos
         3ch: own unit cooldown
         4ch: own unit resource
         5ch: opponent unit pos
         6ch: opponent unit cooldown
         7ch: opponent unit resource

         8ch: target citytile pos
         9ch: target citytile fuel_ratio

        10ch: own citytile pos
        11ch: own citytile fuel_ratio
        12ch: own citytile cooldown
        13ch: opponent citytile pos
        14ch: opponent citytile fuel_ratio
        15ch: opponent citytile cooldown

        16ch: wood
        17ch: coal
        18ch: uranium

        19ch: own research points
        20ch: opponent research points
        21ch: road level
        22ch: cycle
        23ch: turn 
        24ch: map

        25ch: own unit pos before 1step
        26ch: opoonent unit pos before 1step
        27ch: own unit pos before 2step
        28ch: opoonent unit pos before 2step
        29ch: own unit pos before 3step
        30ch: opoonent unit pos before 3step
        """

        height = game.map.height
        width = game.map.width

        x_shift = (32 - width) // 2
        y_shift = (32 - height) // 2

        b = np.zeros((self._n_obs_channel, 32, 32), dtype=np.float32)
        opponent_team = 1 - team
        # target unit
        # if unit is not None:
        #     x = unit.pos.x + x_shift
        #     y = unit.pos.y + y_shift
        #     cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
        #     resource = (unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]) / cap
        #     b[:2, x,y] = (1, resource)
    
        # unit
        for _unit in game.state["teamStates"][team]["units"].values():
            # if unit is not None:
                # if _unit.id == unit.id:
                #     continue
            x = _unit.pos.x + x_shift
            y = _unit.pos.y + y_shift
            cooldown = _unit.cooldown / 6
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / cap
            b[2:5, x,y] = (1, cooldown, resource)
    
        for _unit in game.state["teamStates"][opponent_team]["units"].values():
            x = _unit.pos.x + x_shift
            y = _unit.pos.y + y_shift
            cooldown = _unit.cooldown / 6
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"] 
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / cap
            b[5:8, x,y] = (1, cooldown, resource)

        # city tile
        for city in game.cities.values():
            fuel = city.fuel
            lightupkeep = city.get_light_upkeep()
            max_cooldown = GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]
            fuel_ratio = min(fuel / lightupkeep, max_cooldown) / max_cooldown
            for cell in city.city_cells:
                x = cell.pos.x + x_shift
                y = cell.pos.y + y_shift
                cooldown = cell.city_tile.cooldown / max_cooldown

                # target city_tile
                # if city_tile is not None:
                #     if (cell.city_tile.pos.x == city_tile.pos.x)&(cell.city_tile.pos.y == city_tile.pos.y):
                #         b[8:10, x, y] = (1, fuel_ratio)
                        # continue 
                
                if city.team == team:
                    b[10:13, x, y] = (1, fuel_ratio, cooldown)
                else:
                    b[13:16, x, y] = (1, fuel_ratio, cooldown)

        # resource
        resource_dict = {'wood': 16, 'coal': 17, 'uranium': 18}
        for cell in game.map.resources:
            x = cell.pos.x + x_shift
            y = cell.pos.y + y_shift
            r_type = cell.resource.type
            amount = cell.resource.amount / 800
            idx = resource_dict[r_type]
            b[idx, x, y] = amount
        
        # research points
        max_rp = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]
        b[19, :] = min(game.state["teamStates"][team]["researchPoints"], max_rp) / max_rp
        b[20, :] = min(game.state["teamStates"][opponent_team]["researchPoints"], max_rp) / max_rp
        
        # road
        for row in game.map.map:
            for cell in row:
                if cell.road > 0:
                    x = cell.pos.x + x_shift
                    y = cell.pos.y + y_shift
                    b[21, x,y] = cell.road / 6

        # cycle
        cycle = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        b[22, :] = game.state["turn"] % cycle / cycle
        b[23, :] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        
        # map
        b[24, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

        if self.n_stack > 1:
            additional_obs = np.concatenate(last_unit_obs, axis=0)
            b = np.concatenate([b, additional_obs], axis=0)

        assert np.sum(b > 1) == 0
        assert b.shape == self.observation_space.shape
        return b


    def get_observation(self, game, unit, city_tile, team, is_new_turn, base_obs):
        """
         Implements getting a observation from the current game for this unit or city
         0ch: target unit pos
         1ch: target unit resource

         2ch: own unit pos
         3ch: own unit cooldown
         4ch: own unit resource
         5ch: opponent unit pos
         6ch: opponent unit cooldown
         7ch: opponent unit resource

         8ch: target citytile pos
         9ch: target citytile fuel_ratio

        10ch: own citytile pos
        11ch: own citytile fuel_ratio
        12ch: own citytile cooldown
        13ch: opponent citytile pos
        14ch: opponent citytile fuel_ratio
        15ch: opponent citytile cooldown

        16ch: wood
        17ch: coal
        18ch: uranium

        19ch: own research points
        20ch: opponent research points
        21ch: road level
        22ch: cycle
        23ch: turn 
        24ch: map

        25ch: own unit pos before 1step
        26ch: opoonent unit pos before 1step
        27ch: own unit pos before 2step
        28ch: opoonent unit pos before 2step
        29ch: own unit pos before 3step
        30ch: opoonent unit pos before 3step
        """
        b = base_obs.copy()
        height = game.map.height
        width = game.map.width

        x_shift = (32 - width) // 2
        y_shift = (32 - height) // 2

        opponent_team = 1 - team
        # target unit
        if unit is not None:
            x = unit.pos.x + x_shift
            y = unit.pos.y + y_shift
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            resource = (unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]) / cap
            b[:2, x,y] = (1, resource)


        # target city_tile
        if city_tile is not None:
            city = game.cities[city_tile.city_id]
            fuel = city.fuel
            lightupkeep = city.get_light_upkeep()
            max_cooldown = GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]
            fuel_ratio = min(fuel / lightupkeep, max_cooldown) / max_cooldown
            x = city_tile.pos.x
            y = city_tile.pos.y
            b[8:10, x, y] = (1, fuel_ratio)

        assert np.sum(b > 1) == 0
        assert b.shape == self.observation_space.shape
        return b


    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        self.rewards = {
            "rew/r_city_tiles": 0,
            "rew/r_research_points_coal_flag": 0,
            "rew/r_research_points_uranium_flag": 0,
            "rew/r_game_win": 0
            }
            
        step = game.state["turn"]
        step_decay = (360-step)/360
        if is_game_error:
            # Game environment step failed, assign a game lost reward to not incentivise this
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn or at game end
            return 0

        # Get some basic stats
        unit_count = len(game.state["teamStates"][self.team]["units"])

        city_count = 0
        city_count_opponent = 0
        city_tile_count = 0
        city_tile_count_opponent = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opponent += 1

            for cell in city.city_cells:
                if city.team == self.team:
                    city_tile_count += 1
                else:
                    city_tile_count_opponent += 1
                
        # Give a reward for unit creation/death. 0.05 reward per unit.
        # rewards["rew/r_units"] = (unit_count - self.units_last) * 0.005
        # self.units_last = unit_count

        # Give a reward for city creation/death. 0.1 reward per city.
        # rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1

        # number of citytile after night
        if (step > 0)&(step % 40 == 0):
            self.rewards["rew/r_city_tiles"] += (city_tile_count - self.city_tiles_last) * 0.01
            self.city_tiles_last = city_tile_count

        # Reward collecting fuel
        # fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        # rewards["rew/r_fuel_collected"] = ( (fuel_collected - self.fuel_collected_last) / 20000 )
        # self.fuel_collected_last = fuel_collected

        # Reward for Research Points
        research_points = game.state["teamStates"][self.team]["researchPoints"]
        # rewards["rew/r_research_points"] = (research_points - self.research_points_last) / 200  # 0.005
        if (research_points == 50)&(self.research_points_last < 50):
            self.rewards["rew/r_research_points_coal_flag"] += 0.25 * step_decay
        elif (research_points == 200)&(self.research_points_last < 200):
            self.rewards["rew/r_research_points_uranium_flag"] += 1 * step_decay
        self.research_points_last = research_points

        # Give a reward of 1.0 per city tile alive at the end of the game
        # rewards["rew/r_city_tiles_end"] = 0
        # if is_game_finished:
        #     self.is_last_turn = True
        #     rewards["rew/r_city_tiles_end"] = city_tile_count - city_tile_count_opponent
        if is_game_finished:
            if game.get_winning_team() == self.team:
                self.rewards["rew/r_game_win"] += 1 # Win
            else:
                self.rewards["rew/r_game_win"] -= 1 # Loss


        reward = 0
        for name, value in self.rewards.items():
            reward += value

        return reward

    
