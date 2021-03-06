import os 
import sys
import time
from functools import partial  # pip install functools
import copy
import random
import onnxruntime as ort
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from gym import spaces
import glob 
sys.path.append("../../LuxPythonEnvGym/")
from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



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
        layers, filters = 12, 32
        self.n_obs_channel = observation_space.shape[0]
        self.conv0 = BasicConv2d(self.n_obs_channel, filters, (3, 3), False)
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, features_dim, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        # h = (batch, c, w, h) -> (batch, c) 
        # h:(64,32,32,32)
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)  # h=head: (64, 32)
        p = self.head_p(h_head)  # policy
        return p

class ImitationAgent_(Agent):
    def __init__(self) -> None:
        """
        Implements an agent opponent
        """
        # super(ImitationAgent, self).__init__(model_path)
        self.team = None
        self.match_controller = None
        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            SpawnCityAction,
  
        ]
        self.n_obs_channel = 23
        observation_space = spaces.Box(low=0, high=1, shape=(self.n_obs_channel, 32, 32), dtype=np.float16)
        self.model_path = "/work/agents/imitation_/models/toad1800_1115.onnx"
        self.model = None 
        # self.model = LuxNet(observation_space, features_dim=len(self.actions_units))
        # self.set_model()
        # model_path = "/work/agents/imitation/models/toad1800_1115.onnx"
        # self.model = ort.InferenceSession(model_path)

    def onnx_predict(self, input):
        output = self.model.run(None, {"input_1": np.expand_dims(input, 0)})[0][0]
        return output 
    
    def set_model(self):
        model_paths = glob.glob("/work/agents/imitation/models/*.pth")
        model_path = random.choice(model_paths)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

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
                        # ??????unit???(worker)?????????city tile?????????????????????worker?????????
                        if unit_count < city_tile_count:
                            action = SpawnWorkerAction(team, None, x, y)
                            actions.append(action)
                            unit_count += 1
                        # # ????????????????????????????????????research point?????????????????????????????????????????????research point????????????
                        elif game.state["teamStates"][team]["researchPoints"] < 200:
                            action = ResearchAction(team, x, y, None)
                            actions.append(action)
                            game.state["teamStates"][team]["researchPoints"] += 1

        return actions 

    def process_unit_turn(self, game, team, base_obs):
        actions = []
        units = game.get_teams_units(team)
        for unit in units.values():  
            if unit.can_act() and (game.state["turn"] % 40 < 30 or not in_city(unit.pos, game, team)):
                state = self.get_observation(game, unit, team, base_obs)
                # with torch.no_grad():
                #     p = self.model(torch.from_numpy(state).unsqueeze(0))
                # policy = p.squeeze(0).numpy()
                policy = self.onnx_predict(state)
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
        if self.model is None:
            print("[Imitation Agent]load onnx model")
            self.model = ort.InferenceSession(self.model_path)

        base_obs = self.get_base_observation(game, team)
        unit_actions = self.process_unit_turn(game, team, base_obs)
        city_actions = self.process_city_turn(game, team)
        actions = unit_actions + city_actions

        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print(f"[Imitation Agent]WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second." % time_taken,
                  file=sys.stderr)
        return actions

    def get_base_observation(self, game, team):
        """
        Implements getting a observation from the current game for this unit or city
        """

        height = game.map.height
        width = game.map.width

        x_shift = (32 - width) // 2
        y_shift = (32 - height) // 2

        b = np.zeros((self.n_obs_channel, 32, 32), dtype=np.float32)
        opponent_team = 1 - team
        
        # unit
        for _unit in game.state["teamStates"][team]["units"].values():
            x = _unit.pos.x + x_shift
            y = _unit.pos.y + y_shift
            cooldown = _unit.cooldown / 6
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / 2000
            b[2:5, x,y] += (1, cooldown, resource)
        
        for _unit in game.state["teamStates"][opponent_team]["units"].values():
            x = _unit.pos.x + x_shift
            y = _unit.pos.y + y_shift
            cooldown = _unit.cooldown / 6
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / 2000
            b[5:8, x,y] += (1, cooldown, resource)
        
        # city tile
        for city in game.cities.values():
            fuel = city.fuel
            lightupkeep = city.get_light_upkeep()
            fuel_ratio = min(fuel / lightupkeep, 10) / 10
            for cell in city.city_cells:
                x = cell.pos.x + x_shift
                y = cell.pos.y + y_shift
                cooldown = cell.city_tile.cooldown / 10
                if city.team == team:
                    b[8:11, x, y] = (1, fuel_ratio, cooldown)
                else:
                    b[11:14, x, y] = (1, fuel_ratio, cooldown)
        
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
        b[17, :] = min(game.state["teamStates"][team]["researchPoints"], 200) / 200
        b[18, :] = min(game.state["teamStates"][opponent_team]["researchPoints"], 200) / 200
        
        # road
        for row in game.map.map:
            for cell in row:
                if cell.road > 0:
                    x = cell.pos.x + x_shift
                    y = cell.pos.y + y_shift
                    b[19, x,y] = cell.road / 6

        # cycle
        b[20, :] = game.state["turn"] % 40 / 40
        b[21, :] = game.state["turn"] / 360
        
        # map
        b[22, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

        return b 

    def get_observation(self, game, unit, team, base_state):
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
            cooldown = np.array(unit.cooldown / 6, dtype=np.float32)
            resource = np.array((unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]) / 2000, dtype=np.float32)
            b[:2, x, y] = (1, resource)
            b[2:5, x, y] -= (1, cooldown, resource)
        
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

def in_city(pos, game, team):    
    try:
        citytile = game.map.get_cell(pos.x, pos.y).city_tile
        return citytile is not None and citytile.team == team
    except:
        return False