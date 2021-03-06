import os
import numpy as np
import torch
from lux.game import Game
from lux.game_map import Position


path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'
model = torch.jit.load(f'{path}/best.pth')
model.eval()


# Input for Neural Network
def make_input(obs, unit_id, n_obs_channel):
    """obs情報をnnが学習しやすい形式に変換する関数
    全て0~1に正規化されている
    1 ch: 全部のunitの位置  
    2 ch: 全部のunitが持つresourceの合計量(/100で正規化？)(3つまとめて良い？)  

    3 ch: 自チームのworker-unitの位置 
    4 ch: cooldownの状態(/6で正規化)
    5 ch: resourceの合計量(/100で正規化？)(3つまとめて良い？)
    
    6 ch: 敵チームのworker-unitの位置 
    7 ch: cooldownの状態(/6で正規化) (workerはmax=2/cargoはmax=3という認識)
    8 ch: resourceの合計量(/100で正規化？)(3つまとめて良い？)
    
    9 ch: 自チームのcitytileの位置
    10ch: 自チームのcitytileの夜間生存期間
    11ch: cooldown(/10)
    
    12ch: 敵チームのcitytileの位置
    13ch: 敵チームのcitytileの夜間生存期間
    14ch: cooldown(/10)

    15ch: wood量
    16ch: coal量
    17ch: uranium量
    
    18ch: 自チームのresearch point(位置情報はなし)
    19ch: 敵チームのresearch point(位置情報はなし)
    
    20ch: road level

    21ch: 何cycle目かを表す
    22ch: 何step目かを表す
    23ch: map
    """
    width, height = obs['width'], obs['height']

    # mapのサイズを調整するためにshiftするマス数
    # width=20の場合は6 width=21の場合5
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    # (c, w, h)
    # mapの最大サイズが(32,32)なのでそれに合わせている
    b = np.zeros((n_obs_channel, 32, 32), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if unit_id == strs[3]:
                # Position and Cargo
                b[:2, x, y] = (
                    1,
                    (wood + coal + uranium) / 100
                )
            else:
                # Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 2 + (team - obs['player']) % 2 * 3
                b[idx:idx + 3, x, y] = (
                    1,
                    cooldown / 6,
                    (wood + coal + uranium) / 100
                )
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            cooldown = int(strs[5])
            idx = 8 + (team - obs['player']) % 2 * 2
            b[idx:idx + 3, x, y] = (
                1,
                cities[city_id],
                cooldown / 10
            )
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 14, 'coal': 15, 'uranium': 16}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[17 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
        elif input_identifier == "ccd":
            x = int(strs[1]) + x_shift
            y = int(strs[2]) + y_shift
            road_level = float(strs[3])
            b[19, x, y] =  road_level / 6
    
    # Day/Night Cycle
    b[20, :] = obs['step'] % 40 / 40
    # Turns
    b[21, :] = obs['step'] / 360
    # Map Size
    b[22, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return b


game_state = None
def get_game_state(observation):
    global game_state
    
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation["player"]
    else:
        game_state._update(observation["updates"])
    return game_state


def in_city(pos):    
    try:
        city = game_state.map.get_cell_by_pos(pos).citytile
        return city is not None and city.team == game_state.id
    except:
        return False


def call_func(obj, method, args=[]):
    return getattr(obj, method)(*args)


def get_adjacent_units_and_unit_resource(unit, obs, own_team):
    adjacent_units = []
    unit_resource = {}
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        # unitのobsに注目
        if input_identifier == 'u':
            if int(strs[2])==own_team: 
                unit_id = strs[3]
                pos = Position(int(strs[4]), int(strs[5]))
                # 対象unitに隣接するunit_idを記録
                if unit.pos.is_adjacent(pos):
                    adjacent_units.append(unit_id)
                # 対象unitの保有するresource_typeとamountを取得
                if unit.id == unit_id:
                    unit_resource['wood'] = int(strs[6])
                    unit_resource['coal'] = int(strs[7])
                    unit_resource['uranium'] = int(strs[8])
                
    return adjacent_units, unit_resource


# unit_actions = [('move', 'n'), ('move', 's'), ('move', 'w'), ('move', 'e'), ('build_city',)]
unit_actions = [('move', 'n'), ('move', 's'), ('move', 'w'), ('move', 'e'), ('build_city',), ('pillage',), ('transfer', )]
def get_action(policy, unit, dest, obs, own_team):

    # 行動確率の高い順に考える
    for label in np.argsort(policy)[::-1]:
        act = unit_actions[label]
        pos = unit.pos.translate(act[-1], 1) or unit.pos  # moveの場合移動pos/それ以外の場合現在のpos

        # 既に決定している他のunitと移動先が被っていないorcity内であれば
        if pos not in dest or in_city(pos):
            
            if act[0] == 'transfer':
                adjacent_units, unit_resource = get_adjacent_units_and_unit_resource(unit, obs, own_team)
                # resourceを保有していないor隣接するunitがいない場合はtransferはしない
                if (sum(list(unit_resource.values())) == 0)or(len(adjacent_units)==0):
                    continue
                #adjacent_units: transfer先の候補となる隣接unit
                # 優先順位としてはwood -> coal -> uranium
                # 1つでも保有していればそれをtransfer用のresourceとする
                for resource_type, amount in unit_resource.items():
                    if amount > 0:
                        break
                assert amount > 0

                transfer_unit = adjacent_units[0]  # とりあえず仮でこうしておく
                # actというtupleにdest_id, resource_type, amountを追加
                # act = ('transfer', transfer_unit, resource_type, amount)d
                return unit.transfer(transfer_unit, resource_type, amount), pos
            
            return call_func(unit, *act), pos         
            
    return unit.move('c'), unit.pos


def agent(observation, configuration):
    global game_state
    game_state = get_game_state(observation)    
    player = game_state.players[observation.player]
    actions = []
    
    # City Actions
    unit_count = len(player.units)
    for city in player.cities.values():
        for city_tile in city.citytiles:
            if city_tile.can_act():
                # 保有unit数(worker)よりもcity tileの数が多いならworkerを追加
                if unit_count < player.city_tile_count: 
                    actions.append(city_tile.build_worker())
                    unit_count += 1
                # ウランの研究に必要な数のresearch pointを満たしていなければ研究をしてresearch pointを増やす
                elif not player.researched_uranium():
                    actions.append(city_tile.research())
                    player.research_points += 1
    
    # Worker Actions
    dest = []
    for unit in player.units:
        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos)):
            state = make_input(observation, unit.id, n_obs_channel=23)
            with torch.no_grad():
                p, v = model(torch.from_numpy(state).unsqueeze(0))

            policy = p.squeeze(0).numpy()
            value = v.item()

            action, pos = get_action(policy, unit, dest, observation, player.team)
            actions.append(action)
            dest.append(pos)

    return actions