from stable_baselines3 import PPO  # pip install stable-baselines3

from luxai2021.env.agent import AgentFromStdInOut
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
from agent_policy import AgentPolicy
import glob 
from luxai2021.game.game import Game 

models = glob.glob(f'./models/rl_model_*_steps.zip')
pretrained_model = sorted(models, key=lambda x: int(x.split('_')[-2]), reverse=True)[0]
model = PPO.load(pretrained_model)
_agent = AgentPolicy(mode="inference", model=model)


game_state = None
def get_game_state(observation):
    global game_state
    
    if observation["step"] == 0:
        game_state = Game(LuxMatchConfigs_Default)
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

    return actions
