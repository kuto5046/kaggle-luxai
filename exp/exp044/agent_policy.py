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
import onnxruntime as ort 
import glob 
import random 


class ImitationAgent(Agent):
    def __init__(self, model=None, model_path=None, _n_obs_channel=23, n_stack=1) -> None:
        super().__init__()
        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            SpawnCityAction,
        ]
        self.action_space = spaces.Discrete(len(self.actions_units))
        self.n_stack = n_stack
        self._n_obs_channel = _n_obs_channel  #  28  # base obs
        self.n_obs_channel = self._n_obs_channel + (8 * (self.n_stack-1))  # base obs + last obs
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_obs_channel, 32, 32), dtype=np.float32)
        self.model = model
        self.model_path = model_path

    def set_model(self):
        # self.model_path
        models = glob.glob(self.model_path + '*.onnx')
        selected_model_path = random.choice(models)
        self.model = ort.InferenceSession(selected_model_path)
        print(f'[Imitation Agent] load model from {selected_model_path}')

    def onnx_predict(self, input):
        output = self.model.run(None, {"input.1": np.expand_dims(input, 0)})[0][0]
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
        self.units_last = 0
        self.city_tiles_last = 0
        self.fuel_collected_last = 0
        self.research_points_last = 0
        self.rewards = {}
        self.last_unit_obs = [np.zeros((8, 32, 32)) for i in range(self.n_stack-1)]

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

    def process_unit_turn(self, game, team, base_obs):
        actions = []
        units = game.get_teams_units(team)
        for unit in units.values():
            if unit.can_act():
                obs = self.get_observation(game, unit, None, unit.team, False, base_obs)
                policy = self.onnx_predict(obs)
                for action_code in np.argsort(policy)[::-1]:
                    # 夜でcity上にいない場合はbuild cityはしない
                    # if (action_code == 6)&(game.is_night())&(not game.game_map_by_pos(unit.pos).is_city_tile):
                    #     continue 
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
            self.set_model()

        base_obs = self.get_base_observation(game, team, self.last_unit_obs)
        unit_actions = self.process_unit_turn(game, team, base_obs)
        city_actions = self.process_city_turn(game, team)
        actions = unit_actions + city_actions

        if self.n_stack > 1:
            self.get_last_observation(base_obs)
        time_taken = time.time() - start_time
        if time_taken > 1.0:  # Warn if larger than 0.5 seconds.
            print("[RL Agent]WARNING: Inference took %.3f seconds for computing actions. Limit is 3 second." % time_taken,
                  file=sys.stderr)
        return actions

    def get_last_observation(self, obs):
        current_unit_obs = obs[[2,4,5,7,8,9,11,12]]  # own_unit/opponent_unit/own_citytile/opponent_citytile
        # assert np.sum(current_unit_obs > 1) == 0
        self.last_unit_obs.append(current_unit_obs)
        if len(self.last_unit_obs)>=self.n_stack:  # 過去情報をn_stack分に保つ
            self.last_unit_obs.pop(0)
        assert len(self.last_unit_obs) == self.n_stack - 1
        
   
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
#             if game.is_night():
#                 loss = self.get_convert_fuel_loss(_unit)
#                 b[23, x,y] = loss / 156  # max is 4*(40-1)=156
        
            # b[26,x,y] = self.prob_unit_destroy_next_turn(game, _unit)

        for _unit in game.state["teamStates"][opponent_team]["units"].values():
            x = _unit.pos.x + x_shift
            y = _unit.pos.y + y_shift
            cooldown = _unit.cooldown / 6
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"] 
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / cap
            b[5:8, x,y] += (1, cooldown, resource)
            
#             b[27,x,y] = self.prob_unit_destroy_next_turn(game, _unit)

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
#                     if own_unit_count < own_city_tile_count:
#                         b[24, x,y] = 1
#                         own_unit_count += 1    
#                     elif game.state["teamStates"][team]["researchPoints"] < 200:
#                         own_incremental_rp += 1
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