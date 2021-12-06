import os
import time
import pickle
import math
import numpy as np
import time
import numpy as np
import copy 
import random 
import sys
from luxai2021.game.actions import *


class MCTS():
    """Montec Carlo Tree Search
    """
    def __init__(self, agent, eps=1e-8, cpuct=1.0):
        self.agent = agent
        self.eps = eps
        self.cpuct = cpuct

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Vs = {}  # stores game.getValidMoves for board s

        self.n_actions = len(self.agent.actions_units)
        self.n_teams = 2

    def get_action(self, obs, game, timelimit=3.0):
        """obsから適切な行動を探索し確率で表す

        Args:
            obs ([type]): [description]
            timelimit (float, optional): [description]. Defaults to 1.0.

        Returns:
            [type]: [description]
        """
        start_time = time.time()
        self.searched_game = copy.deepcopy(game)

        # 設定時間で探索を打ち切る
        while time.time() - start_time < timelimit:
            self.search()

        s = self.create_state(game)
        team = obs['player']
        # 行動aが状態sで選択された数からその行動を取りうる確率を算出する
        actions = []
        x_shift = (32 - game.map.width) // 2
        y_shift = (32 - game.map.height) // 2
        obs = self.agent.get_observation(game, team=team)
        policy_map = self.agent.onnx_predict(obs)
        units = game.get_teams_units(team)
        for unit in units.values():
            if unit.can_act():
                try:
                    counts = [self.Nsa[(s, unit.id, a)] if (s, unit.id, a) in self.Nsa else 0 for a in range(self.n_actions)]
                    prob = counts / np.sum(counts)
                    action_code = np.argmax(prob)
                    for action_code in np.argsort(prob)[::-1]:
                        action = self.agent.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=team)
                        if action.is_valid(game, actions):
                            actions.append(action)
                            break 
                except:
                    x = unit.pos.x + x_shift
                    y = unit.pos.y + y_shift
                    policy = policy_map[:,x,y]
                    for action_code in np.argsort(policy)[::-1]:
                        action = self.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=unit.team)
                        if action.is_valid(game, actions):
                            actions.append(action)
                            break

        city_actions = self.agent.process_city_turn(game, team)
        actions += city_actions
        return actions 

    def get_valid_action(self, unit, actions=[]):
        valid = []
        for action_code in range(self.n_actions):
            action = self.agent.action_code_to_action(action_code, self.searched_game, unit, None, unit.team)
            if action.is_valid(self.searched_game, actions):
                valid.append(1)
            else:
                valid.append(0)
        return valid

    def initial_observation(self, s):
        values = {}
        for team in range(self.n_teams):
            x_shift = (32 - self.searched_game.map.width) // 2
            y_shift = (32 - self.searched_game.map.height) // 2
            obs = self.agent.get_observation(self.searched_game, None, None, team)
            policy_map, value = self.agent.onnx_predict(obs)
            units = self.searched_game.get_teams_units(team)
            values[team] = value  # 状態価値
            for unit in units.values():
                x = unit.pos.x + x_shift
                y = unit.pos.y + y_shift
                self.Ps[s, unit.id] = policy_map[:,x,y]
                valids = self.get_valid_action(unit)
                self.Ps[s, unit.id] = self.Ps[s, unit.id] * valids
                sum_Ps_s = np.sum(self.Ps[s, unit.id])
                if sum_Ps_s > 0:
                    self.Ps[s, unit.id] /= sum_Ps_s  # renormalize
                self.Vs[s, unit.id] = valids
        self.Ns[s] = 0
        return values 

    def decide_action_by_Q(self, s):
        actions = []
        best_acts = {}
        for team in range(self.n_teams):
            unit_count = len(self.searched_game.get_teams_units(team))
            for unit in self.searched_game.get_teams_units(team).values():
                try:
                    valids = self.Vs[s, unit.id]  # 状態sにおける各行動が選択可能かを表す
                except:
                    valids = self.get_valid_action(unit, actions)
                
                try:
                    # 初期値を振る
                    cur_best = -float('inf')
                    best_act = 0 # 任意の行動
                    # pick the action with the highest upper confidence bound(UCB)
                    for a in range(self.n_actions):
                        # 行動aが選択可能なら
                        # 状態sにおける行動価値
                        # UCBはa,sの組み合わせの経験が少ない場合は不確実性を高く持ち
                        # uは行動aの不確実度
                        if valids[a]:
                            if (s, unit.id, a) in self.Qsa:
                                u = self.Qsa[(s, unit.id, a)] + self.cpuct * self.Ps[s, unit.id][a] * math.sqrt(
                                        self.Ns[s]) / (1 + self.Nsa[(s, unit.id, a)])
                            else:
                                u = self.cpuct * self.Ps[s, unit.id][a] * math.sqrt(
                                    self.Ns[s] + self.eps)  # Q = 0 ?
                            # 行動aの不確実度が最大のもの(経験が少ない状態行動)を最適行動として選択する
                            if u > cur_best:
                                cur_best = u
                                best_act = a        
                except:
                    # random
                    valid_actions = [i for i, x in enumerate(valids) if x == 1]
                    best_act = random.choice(valid_actions)

                best_acts[unit.id] = best_act
                action = self.agent.action_code_to_action(best_act, game=self.searched_game, unit=unit, city_tile=None, team=unit.team)
                if action.is_valid(self.searched_game, actions):
                    actions.append(action)

            # city
            city_actions = []
            city_tile_count = 0
            for city in self.searched_game.cities.values():
                for cell in city.city_cells:
                    if city.team == team:
                        city_tile_count += 1

            # Inference the model per-city
            cities = self.searched_game.cities.values()
            tmp_rp = 0
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
                                city_actions.append(action)
                                unit_count += 1
                            # # ウランの研究に必要な数のresearch pointを満たしていなければ研究をしてresearch pointを増やす
                            elif self.searched_game.state["teamStates"][team]["researchPoints"]+tmp_rp < 200:
                                action = ResearchAction(team, x, y, None)
                                city_actions.append(action)
                                tmp_rp += 1

            actions += city_actions
        return actions, best_acts

    def update_by_search_result(self, s, best_acts, values):
        for team in range(self.n_teams):
            for unit in self.searched_game.get_teams_units(team).values():
                if unit.id in list(best_acts.keys()):
                    a = best_acts[unit.id]  # 1 turn前の行動
                else:
                    a = 0  # 新しく作られたunitの場合best actがない場合がある
                v = values[team]

                # 探索結果を用いてQ,Nを更新
                if (s, unit.id, a) in self.Qsa:
                    self.Qsa[(s, unit.id, a)] = (self.Nsa[(s, unit.id, a)] * self.Qsa[
                        (s, unit.id, a)] + v) / (self.Nsa[(s, unit.id, a)] + 1)
                    self.Nsa[(s, unit.id, a)] += 1
                else:
                    self.Qsa[(s, unit.id, a)] = v
                    self.Nsa[(s, unit.id, a)] = 1
        self.Ns[s] += 1

    def create_state(self, game):
        s = game.map.get_map_string()
        s += str(game.state['teamStates'][0]['researched']['coal'] * 1)
        s += str(game.state['teamStates'][0]['researched']['uranium'] * 1)
        s += str(game.is_night()*1)
        s += str(game.state["turn"])
        return s
    
    def search(self):
        s = self.create_state(self.searched_game)
        if s not in self.Ns:
            values = self.initial_observation(s)
            return values
        actions, best_acts = self.decide_action_by_Q(s)
        is_game_over = self.searched_game.run_turn_with_actions(actions)
        values = self.search()
        self.update_by_search_result(s, best_acts, values)
        return values