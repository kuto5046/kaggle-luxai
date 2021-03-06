import os
import numpy as np
import torch
import torch.nn.functional as F
from lux.game import Game
from lux.game_map import Position


path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'
unit_model = torch.jit.load(f'{path}/unit_best.pth')
unit_model.eval()
city_model = torch.jit.load(f'{path}/city_best.pth')
city_model.eval()


# Input for Neural Network
# Input for Neural Network
def make_input(obs, target_id, n_obs_channel, target):
    """obs情報をnnが学習しやすい形式に変換する関数
    全て0~1に正規化されている
    1 ch: actionの対象のunit(city)の位置  
    2 ch: actionの対象のunitが持つresourceの合計量(/100で正規化？)(3つまとめて良い？)  

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

    if target == "city":
        # x:target_id[0], y:target_id[1]
        b[1, target_id[0], target_id[1]] = 1  # 0chは何も情報がない

    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])

            if target_id == strs[3]:
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


city_actions = [('noaction',), ('research',), ('build_worker', )]
def get_city_action(policy, city_tile, unit_count, player):
    # 行動確率の高い順に考える
    for label in np.argsort(policy)[::-1]:

        act = city_actions[label]
        if (act[0] == "noaction"):
            continue

        # if (act=="build_worker")&(unit_count < player.city_tile_count): 
        if unit_count < player.city_tile_count: 
            unit_count += 1
            return city_tile.build_worker(), unit_count
        # elif (act=="research")&(not player.researched_uranium()):
        elif not player.researched_uranium():
            player.research_points += 1
            return city_tile.research(), unit_count
    return None, unit_count


unit_actions = [('noaction',), ('move', 'n'), ('move', 's'), ('move', 'w'), ('move', 'e'), ('build_city',), ('transfer', )]
def get_unit_action(policy, unit, dest, obs, own_team):

    # 行動確率の高い順に考える
    for label in np.argsort(policy)[::-1]:
        act = unit_actions[label]
        pos = unit.pos.translate(act[-1], 1) or unit.pos  # moveの場合移動pos/それ以外の場合現在のpos

        # 既に決定している他のunitと移動先が被っていないorcity内であれば
        if pos not in dest or in_city(pos):
            if act[0] == 'noaction':
                if policy[label] > 0.9:
                    return None, pos
                else:
                    continue
            elif act[0] == 'transfer':
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
                state = make_input(observation, (city_tile.pos.x, city_tile.pos.y), n_obs_channel=23, target="city")
                with torch.no_grad():
                    p, v = city_model(torch.from_numpy(state).unsqueeze(0))
                policy = p.squeeze(0).numpy()
                value = v.item()
                action, unit_count = get_city_action(policy, city_tile, unit_count, player)
                if action is not None:
                    actions.append(action)
    
    # Worker Actions
    dest = []
    for unit in player.units:
        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos)):
            state = make_input(observation, unit.id, n_obs_channel=23, target="unit")
            with torch.no_grad():
                p, v = unit_model(torch.from_numpy(state).unsqueeze(0))

            policy = F.softmax(p).squeeze(0).numpy()
            # print(policy)
            value = v.item()

            action, pos = get_unit_action(policy, unit, dest, observation, player.team)
            if action is not None:
                actions.append(action)
                dest.append(pos)

    return actions