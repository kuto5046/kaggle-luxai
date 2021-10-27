import argparse
import glob
import os
import sys
import random
import traceback
import numpy as np 
from pathlib import Path 
from functools import partial  # pip install functools
import logging 
from logging import DEBUG, INFO
from stable_baselines3.common import callbacks
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import wandb  
from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn, get_device
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, configure
from wandb.integration.sb3 import WandbCallback

from agent_policy import AgentPolicy
sys.path.append("../../")
from agents.imitation.agent_policy import ImitationAgent
from agents.random.agent_policy import RandomAgent

sys.path.append("../../LuxPythonEnvGym")
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.game.game import Game

# Neural Network for Lux AI
class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h

    
class LuxNet(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(LuxNet, self).__init__(observation_space, features_dim)
        layers, filters = 12, 32
        n_obs_channel = observation_space.shape[0]
        self.conv0 = BasicConv2d(n_obs_channel, filters, (3, 3), False)
        self.num_actions = features_dim
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])

        self.head_p = nn.Linear(filters, self.num_actions, bias=False)
        # self.head_v = nn.Linear(filters*2, 1, bias=False)  # for value(reward)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        # h = (batch, c, w, h) -> (batch, c) 
        # h:(64,32,32,32)
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)  # h=head: (64, 32)
        # h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)  # h_avg: (64, 32)
        p = self.head_p(h_head)  # policy
        # v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], dim=1)))  # value
        return p # , v.squeeze(dim=1)



def get_logger(level=INFO, out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    # logger.info("logger set up")
    return logger

logger = get_logger(level=INFO, out_file='results.log')
# logger = configure("./logs", ["stdout", "log", "tensorboard"])


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=SubprocVecEnv#multiprocessing-unleashing-the-power-of-vectorized-environments
def make_env(local_env, rank, seed=0):
    """
    Utility function for multi-processed env.

    :param local_env: (LuxEnvironment) the environment 
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        local_env.seed(seed + rank)
        return local_env

    set_random_seed(seed)
    return _init


def main():

    ############
    #  config
    ############
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    seed = config["basic"]["seed"]
    n_steps = config["basic"]["n_steps"]
    n_envs = config["basic"]["n_envs"]
    step_count = config["basic"]["step_count"]
    pretrained_path = config["basic"]["pretrained_path"]

    model_update_step_freq = config["model"]["model_update_step_freq"]
    model_arche = config["model"]["model_arche"]

    is_resume = config['resume']['is_resume']
    resume_num_timesteps = config['resume']['resume_num_timesteps']
    run_id = config['resume']['run_id']

    ckpt_params = config['callbacks']['checkpoints']
    eval_params = config['callbacks']['eval']
    model_params = config["model"]["params"]
    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]
    
    if not is_resume:
        run_id = None 

    run = wandb.init(
        project='lux-ai', 
        entity='kuto5046', 
        config=config, 
        group=EXP_NAME, 
        resume=is_resume,
        id = run_id,
        sync_tensorboard=True,# auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )
    # Run a training job
    configs = LuxMatchConfigs_Default

    ##############
    #    agent
    ##############
    # Create a default opponent agent and a RL agent in training mode
    
    opponents = {
        "random": RandomAgent(),
        "imitation": ImitationAgent(),
    }

    if os.path.exists(pretrained_path):
        old_model = PPO.load(pretrained_path, device="cpu")

        # Create a default opponent agent and a RL agent in training mode
        opponents["self-play"] = AgentPolicy(mode="inference", arche=model_arche, model=old_model)

        player = AgentPolicy(mode="train", arche=model_arche, model=old_model)
        # environmnet
        if n_envs == 1:
            env = LuxEnvironment(configs=configs,
                                learning_agent=player,
                                opponent_agents=opponents, 
                                model_update_step_freq=model_update_step_freq)
        else:
            env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                            learning_agent=AgentPolicy(mode="train", arche=model_arche, model=old_model),
                                                            opponent_agents=opponents,
                                                            model_update_step_freq=model_update_step_freq), i) for i in range(n_envs)])
    else:
        opponents["self-play"] = AgentPolicy(mode="inference", arche=model_arche)
        player = AgentPolicy(mode="train", arche=model_arche)
        #  environmnet
        if n_envs == 1:
            env = LuxEnvironment(configs=configs,
                                learning_agent=player,
                                opponent_agents=opponents, 
                                initial_opponent_policy="random",
                                model_update_step_freq=model_update_step_freq)
        else:
            env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                            learning_agent=AgentPolicy(mode="train", arche=model_arche),
                                                            opponent_agents=opponents,
                                                            initial_opponent_policy="random",
                                                            model_update_step_freq=model_update_step_freq), i) for i in range(n_envs)])
        
    if model_arche == "mlp":
        model = PPO("MlpPolicy", env, **model_params)
    elif model_arche == "cnn":
        # change custom network
        policy_kwargs = dict(
            features_extractor_class=LuxNet,
            features_extractor_kwargs=dict(features_dim=player.action_space.n),
        )
        # Attach a ML model from stable_baselines3 and train a RL model
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, **model_params)


    #############
    #  callback
    #############
    callbacks = []
    callbacks.append(WandbCallback())
    callbacks.append(CheckpointCallback(**ckpt_params))
    
    # Since reward metrics don't work for multi-environment setups, we add an evaluation logger
    # for metrics.
    env_eval = None
    if n_envs > 1:
        # An evaluation environment is needed to measure multi-env setups. Use a fixed 4 envs.
        opponents = {"imitation": ImitationAgent()}
        env_eval = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                     learning_agent=AgentPolicy(mode="train", arche=model_arche),
                                                     opponent_agents=opponents), i) for i in range(1)])
        callbacks.append(EvalCallback(env_eval, **eval_params))

    ###########
    # train
    ###########
    logger.info("Training model...")
    try:
        if is_resume:
            model.num_timesteps = resume_num_timesteps
            model.learn(total_timesteps=step_count, callback=callbacks, reset_num_timesteps=False)
        else:
            model.learn(total_timesteps=step_count, callback=callbacks)
 
        model.save(path=f'models/rl_model_{step_count}_steps.zip')
        logger.info(f"Done training model.  this: {step_count}(steps), total: {model.num_timesteps}(steps)")
    except:
        model.save(path=f'models/rl_model_{model.num_timesteps}_steps.zip')
        logger.info(f"There are something errors. Finish training model. total: {model.num_timesteps}(steps)")
        traceback.print_exc()

    run.finish()


if __name__ == "__main__":
    main()

