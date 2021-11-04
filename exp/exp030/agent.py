from stable_baselines3 import PPO  # pip install stable-baselines3


import sys
sys.path.append("../../LuxPythonEnvGym/")
from luxai2021.env.agent import AgentFromStdInOut
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default, LuxMatchConfigs_Replay
from agent_policy import AgentPolicy
import glob 
from luxai2021.game.game import Game 
from imitation.algorithms import bc
from stable_baselines3.common import policies


arche = "cnn"
pretrained_path = "bc_policy"
model = bc.reconstruct_policy(pretrained_path)
_agent = AgentPolicy(mode="inference", arche=arche, model=model)

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

def agent(observation, configuration):
    global game_state
    game_state = get_game_state(observation)
    player = observation.player
    _actions = _agent.process_turn(game_state, player)
    actions = []
    for action_object in _actions:
        action_str = action_object.to_message(game_state)
        actions.append(action_str)
    # print(actions)
    return actions
