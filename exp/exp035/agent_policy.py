import os 
import sys
import time
from functools import partial  # pip install functools
import copy
import random
import onnxruntime as ort
from PIL.features import features
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from gym import spaces

sys.path.append("../../LuxPythonEnvGym/")
from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'


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
        return p  #  , v.squeeze(dim=1)

class ImitationAgent(Agent):
    def __init__(self, model_path=None) -> None:
        """
        Implements an agent opponent
        """
        # super(ImitationAgent, self).__init__(model_path)
        self.team = None
        self.match_controller = None
        self.actions_units = [
            # partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            # partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER), # Transfer to nearby worker
            SpawnCityAction,
  
        ]
        self.n_obs_channel = 23
        observation_space = spaces.Box(low=0, high=1, shape=(self.n_obs_channel, 32, 32), dtype=np.float16)
        self.model = LuxNet(observation_space, features_dim=len(self.actions_units))
        
        if model_path is not None:
            if ".pth" in model_path:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
            elif ".onnx" in model_path:
                self.model = ort.InferenceSession(model_path)

    def onnx_predict(self, state):
        state = np.expand_dims(state, 0)
        output = self.model.run(None, {"input_1": state})[0][0]
        return output 

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

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference. Generally
        don't modify this part of the code.
        Returns: Array of actions to perform.
        """
        tta = TTA()
        start_time = time.time()
        actions = []
        # new_turn = True

        # Inference the model per-unit
        dest = []
        units = game.state["teamStates"][team]["units"].values()
        unit_count = len(units)
        base_state = self.get_base_observation(game, team)
        for unit in units:
            if unit.can_act() and (game.state["turn"] % 40 < 30 or not in_city(unit.pos, game, team)):
                state = self.get_observation(game, unit, team, base_state)
                # _state = make_input(observation, unit.id, 23)
                # for c in range(_state.shape[0]):
                #     count = ((_state[c] - state[c]) != 0).sum()
                    # assert count == 0
                    # if count > 0:
                    #     print(f"??????????????????:{observation['step']}step-{c}channel", count)
                
                policy1 = tta.vertical_convert_action(self.onnx_predict(tta.vertical_flip(state)))
                policy2 = tta.horizontal_convert_action(self.onnx_predict(tta.horizontal_flip(state)))
                policy3 = tta.all_convert_action(self.onnx_predict(tta.all_flip(state)))
                policy4 = self.onnx_predict(state)
                # policy5 = self.onnx_predict(tta.random_roll(state))
                # policy6 = self.onnx_predict(tta.random_roll(state))
                # policy7 = self.onnx_predict(tta.random_roll(state))
                # policy = np.mean([policy1, policy5, policy6, policy7], axis=0)
                policy = np.mean([policy1, policy2, policy3, policy4], axis=0)
                # policy = np.mean([policy1, policy2, policy3, policy4, policy5, policy6, policy7], axis=0)

                # with torch.no_grad():
                #     p = self.model(torch.from_numpy(state).unsqueeze(0))
                # policy = p.squeeze(0).numpy()
                action, pos = self.action_code_to_action(policy, game, unit=unit, dest=dest, team=team)
                
                if action is not None:
                    actions.append(action)
                if pos is not None:
                    dest.append(pos)
                # new_turn = False

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

                        # ????????????????????????????????????research point?????????????????????????????????????????????research point????????????
                        elif (unit_count>3)&(game.state["teamStates"][team]["researchPoints"] < 200):
                            action = ResearchAction(team, x, y, None)
                            actions.append(action)
                            game.state["teamStates"][team]["researchPoints"] += 1

                        # new_turn = False
        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print("WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second." % time_taken,
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


    def action_code_to_action(self, policy, game, unit=None, city_tile=None, team=None, dest=[]):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        for action_code in np.argsort(policy)[::-1]:
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
                    action =  self.actions_units[action_code](
                        game=game,
                        unit_id=unit.id if unit else None,
                        unit=unit,
                        city_id=city_tile.city_id if city_tile else None,
                        citytile=city_tile,
                        team=team,
                        x=x,
                        y=y
                    )
                if hasattr(action, 'direction'):
                    pos = unit.pos.translate(action.direction, 1)
                else:
                    pos = unit.pos 
                    
                if (pos not in dest) or in_city(pos, game, team):
                    return action, pos
            
            except Exception as e:
                # Not a valid action
                print(e)
                return None, None
            
            return None, None 



def in_city(pos, game, team):    
    try:
        citytile = game.map.get_cell(pos.x, pos.y).city_tile
        return citytile is not None and citytile.team == team
    except:
        return False




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
        order = [2,1,0,3,4]
        return action[order]

    def horizontal_convert_action(self, action):
        order = [0,3,2,1,4]
        return action[order]
    
    def all_convert_action(self, action):
        order = [2,3,0,1,4]
        return action[order]