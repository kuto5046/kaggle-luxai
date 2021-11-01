from imitation.data.types import Trajectory
import time
import pickle
from tqdm import tqdm
import json 
import os 
from pathlib import Path 
import logging
import numpy as np 
import pandas as pd 
from logging import INFO, DEBUG
import sys
import multiprocessing
from imitation.data import rollout

sys.path.append("../../LuxPythonEnvGym/")
from luxai2021.game.position import Position
from luxai2021.game.constants import Constants, LuxMatchConfigs_Replay
from luxai2021.game.game import Game 
from luxai2021.game.game_constants import GAME_CONSTANTS

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

def filter(episodes, target_sub_id, team_name, only_win):
    filtering_episodes = []
    for filepath in episodes: 
        with open(filepath) as f:
            json_load = json.load(f)

        if json_load['other']['SubmissionId'] != target_sub_id:
            continue
        win_index = np.argmax([r or 0 for r in json_load['rewards']])  # win or tie
        if only_win:  # 指定したチームが勝ったepisodeのみ取得
            if json_load['info']['TeamNames'][win_index] != team_name:
                continue
        else:  # 指定したチームの勝敗関わらずepisodeを取得
            if team_name not in json_load['info']['TeamNames']: 
                continue
        filtering_episodes.append(filepath)
    logger.info(f"Number of using episodes: {len(filtering_episodes)}")
    return filtering_episodes

def create_trajectories_dataset_from_json(episode_dir, output_dir, team_name='Toad Brigade', only_win=False): 
    logger.info(f"Team: {team_name}")
    episodes = [path for path in Path(episode_dir).glob('*.json') if 'output' not in path.name]

    submission_id_list = []
    latest_lb_list = []
    for filepath in episodes: 
        with open(filepath) as f:
            json_load = json.load(f)
            submission_id_list.append(json_load['other']['SubmissionId'])            
            latest_lb_list.append(json_load['other']['LatestLB'])            
    sub_df = pd.DataFrame([submission_id_list, latest_lb_list], index=['SubmissionId', 'LatestLB']).T
    target_sub_id = sub_df["SubmissionId"].value_counts().index[0]
    files = filter(episodes, target_sub_id, team_name, only_win)
    os.makedirs(output_dir, exist_ok=True)
    for filepath in tqdm(files):
        create_trajectory(filepath, output_dir, team_name)

def create_trajectory(filepath, output_dir, team_name):
    with open(filepath) as f:
        json_load = json.load(f)

    ep_id = json_load['info']['EpisodeId']
    team = json_load['info']['TeamNames'].index(team_name)  # 指定チームのindex

    if os.path.exists(output_dir + f"{ep_id}.pickle"):
        return None 

    actions = []
    infos = []
    observations = []
    num_steps = len(json_load['steps'])-1
    for idx, step in enumerate(range(num_steps)):
        if json_load['steps'][step][team]['status'] != 'ACTIVE':
            break

        game = get_game_state(json_load['steps'][step][0]['observation'])
        if step == 0:
            obs = get_cnn_observation(game, None, None, team)
            observations.append(obs)
            continue

        for action in json_load['steps'][step][team]['action']:
            # moveとbuild cityのaction labelのみが取得される?
            label, unit_id, tile_pos = to_label(action)
            if label is None:
                continue 
            if unit_id is not None:
                unit = game.state["teamStates"][team]["units"][unit_id]
                obs = get_cnn_observation(game, unit, None, team)
            elif tile_pos is not None:
                city_tile = game.map.map[tile_pos.x][tile_pos.y].city_tile
                obs = get_cnn_observation(game, None, city_tile, team)
            observations.append(obs)
            actions.append(label)
            infos.append({"step": step, "idx": idx})

    assert len(observations) == len(actions) + 1
    ts = Trajectory(obs=np.array(observations), acts=np.array(actions), infos=np.array(infos), terminal=True)
    with open(f"trajectory/{ep_id}.pickle", mode="wb") as f:
        pickle.dump(ts, f)

    
def to_label(action):
    """action記号をラベルに変換する関数
    扱っているのはunit系のactionのみでcity系のactionはNone扱い？
    unit系のactionにはtransferやpillageも含まれるがそれらのラベルはNone扱い？

    Args:
        action (list): [description]

    Returns:
        [type]: [description]
    
    ex)
    input: action=['m u_1 w']
    strs = ['m', 'u_1', 'w']
    strs[0] - action
    strs[1] - unit_id
    strs[2] - direction(if action is 'm')
    """
    label = None 
    tile_pos = None 
    unit_id = None 
    strs = action.split(' ')
    if strs[0] in ["m", "t", "bcity"]:
        unit_id = strs[1]
        if strs[0] == 'm':
            label_dict = {'c': 0, 'n': 1, 'w': 2, 's': 3, 'e': 4}
            label = label_dict[strs[2]]
        elif strs[0] == 't':
            label = 5
        elif strs[0] == 'bcity':
            label = 6
    elif strs[0] in ["r", "bw"]:
        x = int(strs[1])
        y = int(strs[2])
        tile_pos = Position(x,y)
        if strs[0] == "bw":
            label = 0
        elif strs[0] == "r":
            label = 1

    return label, unit_id, tile_pos

def get_cnn_observation(game, unit, city_tile, team):
    """
    Implements getting a observation from the current game for this unit or city
    0ch: target unit(worker) pos
    1ch: target unit(worker) resource
    2ch: target unit(cart) pos
    3ch: target unit(cart) pos
    4ch: own unit(worker) pos
    5ch: own unit(worker) cooldown
    6ch: own unit(worker) resource
    7ch: own unit(cart) pos
    8ch: own unit(cart) cooldown
    9ch: own unit(cart) resource
    10ch: opponent unit(worker) pos
    11ch: opponent unit(worker) cooldown
    12ch: opponent unit(worker) resource
    13ch: opponent unit(cart) pos
    14ch: opponent unit(cart) cooldown
    15ch: opponent unit(cart) resource
    16ch: target citytile pos
    17ch: target citytile fuel_ratio
    18ch: own citytile pos
    19ch: own citytile fuel_ratio
    20ch: own citytile cooldown
    21ch: opponent citytile pos
    22ch: opponent citytile fuel_ratio
    23ch: opponent citytile cooldown
    24ch: wood
    25ch: coal
    26ch: uranium
    27ch: own research points
    28ch: opponent research points
    29ch: road level
    30ch: cycle
    31ch: turn 
    32ch: map
    """

    height = game.map.height
    width = game.map.width

    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    n_obs_channel = 33
    b = np.zeros((n_obs_channel, 32, 32), dtype=np.float32)
    opponent_team = 1 - team
    # target unit
    if unit is not None:
        if unit.type == Constants.UNIT_TYPES.WORKER:
            x = unit.pos.x + x_shift
            y = unit.pos.y + y_shift
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            resource = (unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]) / cap
            b[:2, x,y] = (1, resource)
        elif unit.type == Constants.UNIT_TYPES.CART:
            x = unit.pos.x + x_shift
            y = unit.pos.y + y_shift 
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            resource = (unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]) / cap
            b[2:4, x,y] = (1, resource) 


    # unit
    for _unit in game.state["teamStates"][team]["units"].values():
        if unit is not None:
            if _unit.id == unit.id:
                continue
        x = _unit.pos.x + x_shift
        y = _unit.pos.y + y_shift
        if _unit.type == Constants.UNIT_TYPES.WORKER:
            # cooldown = _unit.cooldown / GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"]
            cooldown = _unit.cooldown / 6
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / cap
            b[4:7, x,y] = (1, cooldown, resource)
        elif _unit.type == Constants.UNIT_TYPES.CART:
            # cooldown = _unit.cooldown / GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["CART"]
            cooldown = _unit.cooldown / 6
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / cap
            b[7:10, x,y] = (1, cooldown, resource)   
    
    for _unit in game.state["teamStates"][opponent_team]["units"].values():
        x = _unit.pos.x + x_shift
        y = _unit.pos.y + y_shift
        if _unit.type == Constants.UNIT_TYPES.WORKER:
            # cooldown = _unit.cooldown / GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"]
            cooldown = _unit.cooldown / 6
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] 
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / cap
            b[10:13, x,y] = (1, cooldown, resource)
        elif _unit.type == Constants.UNIT_TYPES.CART:
            # cooldown = _unit.cooldown / GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["CART"]
            cooldown = _unit.cooldown / 6
            cap = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            resource = (_unit.cargo["wood"] + _unit.cargo["coal"] + _unit.cargo["uranium"]) / cap
            b[13:16, x,y] = (1, cooldown, resource)  
    
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
            if city_tile is not None:
                if (cell.city_tile.pos.x == city_tile.pos.x)&(cell.city_tile.pos.y == city_tile.pos.y):
                    b[16:18, x, y] = (1, fuel_ratio)
                    continue 
            
            if city.team == team:
                b[18:21, x, y] = (1, fuel_ratio, cooldown)
            else:
                b[21:24, x, y] = (1, fuel_ratio, cooldown)

    # resource
    resource_dict = {'wood': 24, 'coal': 25, 'uranium': 26}
    for cell in game.map.resources:
        x = cell.pos.x + x_shift
        y = cell.pos.y + y_shift
        r_type = cell.resource.type
        amount = cell.resource.amount / 800
        idx = resource_dict[r_type]
        b[idx, x, y] = amount
    
    # research points
    max_rp = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]
    b[27, :] = min(game.state["teamStates"][team]["researchPoints"], max_rp) / max_rp
    b[28, :] = min(game.state["teamStates"][opponent_team]["researchPoints"], max_rp) / max_rp
    
    # road
    for row in game.map.map:
        for cell in row:
            if cell.road > 0:
                x = cell.pos.x + x_shift
                y = cell.pos.y + y_shift
                b[29, x,y] = cell.road / 6


    # cycle
    cycle = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
    b[30, :] = game.state["turn"] % cycle / cycle
    b[31, :] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
    
    # map
    b[32, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    assert np.sum(b > 1) == 0
    return b 

game_state = None 
def get_game_state(observation):
    global game_state
    if observation["step"] == 0:
        configs = LuxMatchConfigs_Replay
        configs["width"] = observation["width"]
        configs["height"] = observation["height"]
        game_state = Game(configs)
        game_state.reset(observation["updates"])
        game_state.process_updates(observation["updates"][2:])
        game_state.id = observation["player"]
    else:
        game_state.process_updates(observation["updates"])
    return game_state


def main():
    episode_dir = '../../input/lux_ai_toad_episodes_1007/'
    output_dir = "./trajectory/"
    create_trajectories_dataset_from_json(episode_dir, output_dir, only_win=False)

if __name__ == '__main__':
    main()