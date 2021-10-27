import os 
import sys
import time
from functools import partial  # pip install functools
import random
import numpy as np

sys.path.append("../../")
from luxai2021.env.agent import Agent
from luxai2021.game.actions import *


class RandomAgent(Agent):
    def __init__(self) -> None:
        """
        Implements an agent opponent
        """
        self.team = None
        self.match_controller = None

        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER), 
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            SpawnCityAction,
            PillageAction,
        ]

        self.actions_cities = [
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction,
        ]

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
                action =  self.actions_cities[action_code](
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
            
            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference. Generally
        don't modify this part of the code.
        Returns: Array of actions to perform.
        """
        # start_time = time.time()
        new_turn = True
        actions = []

        # unit
        units = game.state["teamStates"][team]["units"].values()
        unit_count = len(units)
        for unit in units:
            if unit.can_act():
                action_code = random.choice(range(len(self.actions_units)))
                action = self.action_code_to_action(action_code, game, unit=unit)
                actions.append(action)
                new_turn = False

        # city
        cities = game.cities.values()
        for city in cities:
            if city.team == team:
                for cell in city.city_cells:
                    city_tile = cell.city_tile
                    if city_tile.can_act():
                        action_code = random.choice(range(len(self.actions_cities)))
                        action = self.action_code_to_action(action_code, game, city_tile=city_tile)
                        actions.append(action)
                        new_turn = False

        # time_taken = time.time() - start_time
        # if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
        #     print("WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second." % time_taken,
        #           file=sys.stderr)
        return actions
