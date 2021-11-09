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
from agent_policy import AgentPolicy
from agent import get_game_state

sys.path.append("../../LuxPythonEnvGym/")
from luxai2021.game.position import Position
from luxai2021.game.constants import Constants, LuxMatchConfigs_Replay
from luxai2021.game.game import Game 
from luxai2021.game.game_constants import GAME_CONSTANTS

agent = AgentPolicy(arche="cnn")

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
    for idx, filepath in enumerate(tqdm(files, total=len(files))):
        create_trajectory(filepath, output_dir, team_name)
        # if idx > 5:
        #     break

def create_trajectory(filepath, output_dir, team_name):
    with open(filepath) as f:
        json_load = json.load(f)

    ep_id = json_load['info']['EpisodeId']
    team = json_load['info']['TeamNames'].index(team_name)  # 指定チームのindex

    # if os.path.exists(output_dir + f"{ep_id}.pickle"):
    if os.path.exists(output_dir + f"{ep_id}_*.pickle"):
        return None 

    actions = []
    infos = []
    observations = []
    num_steps = len(json_load['steps'])-1
    idx = 0
    # print(f"\nnum_steps:{num_steps}")
    for step in range(num_steps):
        game = get_game_state(json_load['steps'][step][0]['observation'])
        # if json_load['steps'][step][team]['status'] != 'ACTIVE':
        #     break
        if step == 0:
            obs = agent.get_observation(game, None, None, team, False) 
            observations.append(obs)

        for action in json_load['steps'][step+1][team]['action']:
            # moveとbuild cityのaction labelのみが取得される?
            label, unit_id, tile_pos = to_label(action)
            if label is None:
                continue 
            if unit_id is not None:
                unit = game.state["teamStates"][team]["units"][unit_id]
                obs = agent.get_observation(game, unit, None, team, False)
                target = "unit"
            elif tile_pos is not None:
                city_tile = game.map.map[tile_pos.x][tile_pos.y].city_tile
                obs = agent.get_observation(game, None, city_tile, team, False)
                target = "city"
            observations.append(obs)
            actions.append(label)
            infos.append({"step": step, "idx": idx, "target": target})
            idx += 1

    assert len(observations) == len(actions) + 1
    ts = Trajectory(obs=np.array(observations), acts=np.array(actions), infos=np.array(infos), terminal=True)
    ts = rollout.flatten_trajectories([ts])
    for i, t in enumerate(ts):
        with open(f"trajectory/{ep_id}_{i}.pickle", mode="wb") as f:
            pickle.dump(t, f)  
    # with open(f"trajectory/{ep_id}.pickle", mode="wb") as f:
    #     pickle.dump(ts, f)

    
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


def main():
    episode_dir = '../../input/lux_ai_toad_episodes_1007/'
    output_dir = "./trajectory/"
    create_trajectories_dataset_from_json(episode_dir, output_dir, only_win=False)

if __name__ == '__main__':
    main()