import sys
import time
from functools import partial  # pip install functools
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from gym import spaces
from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
import torch 
import torch.nn as nn
import torch.nn.functional as F
import gym 
import random 

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
        self.n_obs_channel = 23
        self.action_space = spaces.Discrete(len(self.actions_units))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_obs_channel, 32, 32), dtype=np.float32)
        self.model = model
        self.tta = TTA()

    def torch_predict(self, obs):
        west_obs = np.rot90(obs, 1, axes=(1,2))
        south_obs = np.rot90(obs, 2, axes=(1,2))
        east_obs = np.rot90(obs, 3, axes=(1,2))
        obses = np.stack([obs, west_obs, south_obs, east_obs])

        with torch.no_grad():
          _policy = self.model(torch.from_numpy(obses))
        _policy = _policy.detach().numpy()

        center_policy = np.expand_dims(_policy[:,0].mean(axis=0), 0)# (32,32) 
        move_policy = _policy[:, 1]  # (4,32,32)
        bcity_policy = np.expand_dims(_policy[:, 2].mean(axis=0), 0)  # (32,32) 
        policy = np.concatenate([center_policy, move_policy, bcity_policy])  # (6,32,32)
        return policy
        
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
                        # 夜ならunitは作らない
                        # rpの更新直前はrpを優先する
                        if (unit_count < city_tile_count): #& \
                        #    (not game.is_night())& \
                        #    (not 45<=game.state["teamStates"][team]["researchPoints"]<50) & \
                        #    (not 190<=game.state["teamStates"][team]["researchPoints"]<200):
                            action = SpawnWorkerAction(team, None, x, y)
                            actions.append(action)
                            unit_count += 1
                        # # ウランの研究に必要な数のresearch pointを満たしていなければ研究をしてresearch pointを増やす
                        # unitが少ない場合(<3)はcooldownを温存して次のturn以降でworker buildをしたい
                        elif (game.state["teamStates"][team]["researchPoints"] < 200): #& \
                            #  (unit_count > 3):
                            action = ResearchAction(team, x, y, None)
                            actions.append(action)
                            game.state["teamStates"][team]["researchPoints"] += 1
        return actions 

    def process_unit_turn(self, game, team):
        actions = []
        units = game.get_teams_units(team)
        base_state = self.get_base_observation(game, team)
        for unit in units.values():
            if unit.can_act():
                state = self.get_observation(game, unit, team, base_state)
                policy = self.torch_predict(state)
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
        self.n = random.randint(-5, 5)
        self.m = random.randint(-5, 5)
        return np.roll(state, (self.n,self.m), axis=(1,2))

    def reverse_random_roll(self, state):
        return np.roll(state, (-self.n, -self.m), axis=(1,2)) 

    def vertical_convert_action(self, action):
        order = [0,3,2,1,4,5]
        return action[order]

    def horizontal_convert_action(self, action):
        order = [0,1,4,3,2,5]
        return action[order]
    
    def all_convert_action(self, action):
        order = [0,3,4,1,2,5]
        return action[order]