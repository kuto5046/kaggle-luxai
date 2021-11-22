import os
import time
import pickle
import math
import numpy as np
import time
import numpy as np
import copy 
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
        # self.Vs = {}  # stores game.getValidMoves for board s


    def get_action(self, obs, game, timelimit=3.0):
        """obsから適切な行動を探索し確率で表す

        Args:
            obs ([type]): [description]
            timelimit (float, optional): [description]. Defaults to 1.0.

        Returns:
            [type]: [description]
        """
        self.searched_game = copy.deepcopy(game)
        start_time = time.time()
        actions = []
        # 設定時間で探索を打ち切る
        while time.time() - start_time < timelimit:
            self.search()

        s = game.map.get_map_string()
        team = obs['player']
        # 行動aが状態sで選択された数からその行動を取りうる確率を算出する
        units = game.get_teams_units(team)
        unit_count = len(units)
        for unit in units.values():
            if unit.can_act():
                counts = [self.Nsa[(s, unit.id, a)] if (s, unit.id, a) in self.Nsa else 0 for a in range(7)]
                prob = counts / np.sum(counts)
                action_code = np.argmax(prob)
                actions.append(self.model.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=unit.team))

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
                        if game.worker_unit_capeached:
                            action = SpawnWorkerAction(team, None, x, y)
                            actions.append(action)
                            unit_count += 1
                        # # ウランの研究に必要な数のresearch pointを満たしていなければ研究をしてresearch pointを増やす
                        if game.state["teamStates"][team]["researchPoints"] < 200:
                            action = ResearchAction(team, x, y, None)
                            actions.append(action)
                            game.state["teamStates"][team]["researchPoints"] += 1
        return actions 

    def get_valid_action(self, unit, actions=[]):
        valid = []
        for action_code in range(7):
            action = self.agent.action_code_to_action(action_code, self.searched_game, unit, None, unit.team)
            if action.is_valid(self.searched_game, actions):
                valid.append(1)
            else:
                valid.append(0)
        return valid

    def search(self):
        s = self.searched_game.map.get_map_string()
        # 一度も状態sを観測したことがない場合, agentの推定で行動を決定する
        if s not in self.Ns:
            self.Ns[s] = 0
            values = {}
            for team in range(2):
                base_obs = self.agent.get_base_observation(self.searched_game, team, self.agent.last_unit_obs)
                units = self.searched_game.get_teams_units(team)
                for unit in units.values():
                    # if unit.can_act():
                    obs = self.agent.get_observation(self.searched_game, unit, None, unit.team, False, base_obs)
                    self.Ps[s, unit.id], values[unit.id] = self.agent.onnx_predict(obs)
                    valids = self.get_valid_action(unit)
                    self.Ps[s, unit.id] = self.Ps[s, unit.id] * valids
                    sum_Ps_s = np.sum(self.Ps[s, unit.id])
                    if sum_Ps_s > 0:
                        self.Ps[s, unit.id] /= sum_Ps_s  # renormalize
                    # self.Vs[s, unit.id] = valids
            return values

        # 一度以上状態sを観測している場合Q値から行動を決定する
        actions = []
        best_acts = {}
        for team in range(2):
            unit_actions = []
            for unit in self.searched_game.get_teams_units(team).values():
                # valids = self.Vs[s, unit.id]  # 状態sにおける各行動が選択可能かを表す
                valids = self.get_valid_action(unit, actions)
                # 初期値を振る
                cur_best = -float('inf')
                best_act = None  # 任意の行動
                # pick the action with the highest upper confidence bound(UCB)
                for a in range(7):
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
                
                if best_act is not None:
                    best_acts[unit.id] = best_act
                    unit_actions.append(self.agent.action_code_to_action(best_act, game=self.searched_game, unit=unit, city_tile=None, team=unit.team))
                    actions += unit_actions

            city_actions = self.agent.process_city_turn(self.searched_game, team)
            actions += city_actions

        # 決定した行動をもとに次のturnへ移行
        is_game_over = self.searched_game.run_turn_with_actions(actions)

        # 再帰関数
        # 上限時間or初めてのsの場合値が返されるためsearch-loopから抜ける
        values = self.search()

        # searchを抜けた後の処理
        for team in range(2):
            for unit in self.searched_game.get_teams_units(team).values():
                # if unit.can_act():
                a = best_acts[unit.id]
                v = values[unit.id]

                # 探索結果を用いてQ,Nを更新
                if (s, unit.id, a) in self.Qsa:
                    self.Qsa[(s, unit.id, a)] = (self.Nsa[(s, unit.id, a)] * self.Qsa[
                        (s, unit.id, a)] + v) / (self.Nsa[(s, unit.id, a)] + 1)
                    self.Nsa[(s, unit.id, a)] += 1
                else:
                    self.Qsa[(s, unit.id, a)] = v
                    self.Nsa[(s, unit.id, a)] = 1 
        self.Ns[s] += 1
        return values


