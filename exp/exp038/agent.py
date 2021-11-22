
import sys
sys.path.append("../../LuxPythonEnvGym/")
from luxai2021.game.constants import LuxMatchConfigs_Default, LuxMatchConfigs_Replay
from agent_policy import AgentPolicy
import glob 
from luxai2021.game.game import Game 
from mcts import MCTS 
import onnxruntime as ort

# models = glob.glob(f'./models/rl_cnn_model_*_steps.zip')
# pretrained_model = sorted(models, key=lambda x: int(x.split('_')[-2]), reverse=True)[0]
# policy_kwargs = dict(
#             features_extractor_class=LuxNet,
#             features_extractor_kwargs=dict(features_dim=64),
#         )
# model = PPO.load(pretrained_model, custom_objects={"policy_class": ActorCriticCnnPolicy, "policy_kwargs": policy_kwargs})
pretrained_model = './models/bc_policy.onnx'
model = ort.InferenceSession(pretrained_model)
_agent = AgentPolicy(mode="inference", model=model, _n_obs_channel=28, n_stack=4)
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
    else:
        game_state.reset(observation["updates"], increment_turn=True)
    return game_state


def agent(observation, configuration):
    global game_state
    game_state = get_game_state(observation)
    if observation["step"] == 0:
        mcts.agent.game_start(game_state)
    timelimit = 1000
    actions = mcts.get_action(observation, game_state, timelimit)
    return actions
