import os 
import sys
sys.path.append("../../LuxPythonEnvGym/")
from luxai2021.env.agent import AgentFromStdInOut
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default, LuxMatchConfigs_Replay
from agent_policy import AgentPolicy
import glob 
import time 
import onnxruntime as ort
from luxai2021.game.game import Game 
from mcts import MCTS 

path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'
pretrained_model = path + '/models/rl_cnn_model_1080000_steps.onnx'
print(pretrained_model)
model = ort.InferenceSession(pretrained_model)
# model = PPO.load(pretrained_model)
_agent = AgentPolicy(mode="inference", model=model)
mcts = MCTS(_agent)

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
    start_time = time.time()
    global game_state
    game_state = get_game_state(observation)
    if observation["step"] == 0:
        _agent.game_start(game_state)
    # if observation.step > 50 and observation.remainingOverageTime > 5:
    #     timelimit = configuration.actTimeout+observation.remainingOverageTime/(configuration.episodeSteps-observation.step)
    # else: 
    #     timelimit = configuration.actTimeout - 0.1
    timelimit = 1000 # 2.9
    _actions = mcts.get_action(observation, game_state, timelimit)
    # _actions = _agent.process_turn(game_state, player)
    actions = []
    for action_object in _actions:
        action_str = action_object.to_message(game_state)
        actions.append(action_str)
    print(f"turn {observation['step']}: {time.time() - start_time}s / {observation.remainingOverageTime}s")
    return actions
