from stable_baselines3 import PPO  # pip install stable-baselines3


import sys
sys.path.append("../../LuxPythonEnvGym/")
from luxai2021.env.agent import AgentFromStdInOut
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default, LuxMatchConfigs_Replay
from agent_policy import AgentPolicy
import glob 
import os 
from luxai2021.game.game import Game 
from imitation.algorithms import bc
from stable_baselines3.common.policies import ActorCriticCnnPolicy


path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'
# pretrained_model = path + "/bc_model.zip"
# model = PPO.load(pretrained_model, custom_objects={"policy_class": ActorCriticCnnPolicy})
model = bc.reconstruct_policy(path + '/models/bc_policy')
_agent = AgentPolicy(mode="inference", model=model, n_stack=1)


game_state = None
def get_game_state(observation):
    global game_state
    
    if observation["step"] == 0:
        configs = LuxMatchConfigs_Replay
        configs["width"] = observation["width"]
        configs["height"] = observation["height"]
        game_state = Game(configs)
        game_state.reset(observation["updates"])
        # game_state.id = observation["player"]
    else:
        game_state.reset(observation["updates"], increment_turn=True)
        # game_state.process_updates(observation["updates"])
        # game_state.state["turn"] += 1
    return game_state


def agent(observation, configuration):
    global game_state
    game_state = get_game_state(observation)
    player = observation.player
    if observation["step"] == 0:
        _agent.game_start(game_state)
    _actions = _agent.process_turn(game_state, player)
    actions = []
    for action_object in _actions:
        action_str = action_object.to_message(game_state)
        actions.append(action_str)
    # print(actions)
    return actions
