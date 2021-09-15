# contains designed heuristics
# which could be fine tuned

import numpy as np
import builtins as __builtin__

from typing import List
from lux import game

from lux.game import Game, Unit
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_position import Position
from lux.game_constants import GAME_CONSTANTS


def find_best_cluster(game_state: Game, unit: Unit, distance_multiplier = -0.5, DEBUG=False):
    """最も良いクラスタを見つける関数
    報酬関数に近い

    Args:
        game_state (Game): [description]
        unit (Unit): [description]
        distance_multiplier (float, optional): [description]. Defaults to -0.5.
        DEBUG (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    if DEBUG: print = __builtin__.print
    else: print = lambda *args: None

    # ユニットのtravel rangeを計算
    unit.compute_travel_range((game_state.turns_to_night, game_state.turns_to_dawn, game_state.is_day_time),)

    # for printing
    score_matrix_wrt_pos = game_state.init_zero_matrix()

    best_position = unit.pos
    best_cell_value = -1

    # only consider other cluster if the current cluster has more than one agent mining
    consider_different_cluster = False
    # 現在の位置からresource group idを見つける場合True
    current_leader = game_state.xy_to_resource_group_id.find(tuple(unit.pos))
    if current_leader:
    
        # もし現在のクラスタが１つ以上のmining agentを持っていたら他のクラスタを考慮する
        units_mining_on_current_cluster = game_state.resource_leader_to_locating_units[current_leader]
        if len(units_mining_on_current_cluster) > 1:
            consider_different_cluster = True

    # give very slight preference to richer matrices
    matrix = game_state.convolved_rate_matrix**0.01

    # セルを探索
    for y in range(game_state.map_height):
        for x in range(game_state.map_width):
            # 走査中のセルがすでにtargetとしてset済みor敵もしくは味方のcity tile画存在している場合はスキップ
            if (x,y) in game_state.targeted_xy_set:
                continue
            if (x,y) in game_state.opponent_city_tile_xy_set:
                continue
            if (x,y) in game_state.player_city_tile_xy_set:
                continue

            # if the targeted cluster is not targeted and mined
            # prefer to target the other cluster
            # よくわからないけどある条件下でbonusを渡している
            target_bonus = 1
            if consider_different_cluster:
                target_leader = game_state.xy_to_resource_group_id.find((x,y))
                if target_leader:
                    units_targeting_or_mining_on_target_cluster = \
                        game_state.resource_leader_to_locating_units[target_leader] | \
                        game_state.resource_leader_to_targeting_units[target_leader]
                    if len(units_targeting_or_mining_on_target_cluster) == 0:
                        target_bonus = 50

            # prefer empty tile because you can build afterwards
            empty_tile_bonus = 1
            if game_state.distance_from_resource[y,x] == 1:
                empty_tile_bonus = 2

            # scoring function
            if matrix[y,x] > 0:
                # using simple distance
                # 距離(この距離の名前ってついてるんだっけ？)
                distance = abs(unit.pos.x - x) + abs(unit.pos.y - y)
                distance = max(0.9, distance)  # prevent zero error
                
                # 対象までの距離がtravel range内に収まっていれば集積されたボーナスでそのセルをスコアリング
                if distance <= unit.travel_range:
                    cell_value = empty_tile_bonus * target_bonus * matrix[y,x] * distance ** distance_multiplier
                    score_matrix_wrt_pos[y,x] = int(cell_value)

                    if cell_value > best_cell_value:
                        best_cell_value = cell_value
                        best_position = Position(x,y)

    # print(travel_range)
    # print(score_matrix_wrt_pos)

    return best_position, best_cell_value
