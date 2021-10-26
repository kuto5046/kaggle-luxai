import argparse
import glob
import os
import sys
import random
import numpy as np 
from pathlib import Path 
from functools import partial  # pip install functools
import logging 
from logging import DEBUG, INFO
from stable_baselines3.common import callbacks
import torch
import yaml
import wandb  
from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn, get_device
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from agent_policy import AgentPolicy
from agents.imitation.agent_policy import ImitationAgent
from agents.random.agent_policy import RandomAgent
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, configure
from luxai2021.game.game import Game


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
    model_update_step_freq = config["basic"]["model_update_step_freq"]

    is_resume = config['resume']['is_resume']
    resume_num_timesteps = config['resume']['resume_num_timesteps']
    run_id = config['resume']['run_id']

    ckpt_params = config['callbacks']['checkpoints']
    eval_params = config['callbacks']['eval']
    model_params = config["model"]
    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]
    
    if not is_resume:
        run_id = None 

    run = wandb.init(
        project='lux-ai', 
        entity='kuto5046', 
        config=config, 
        group=EXP_NAME, 
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
        opponents["self-play"] = AgentPolicy(mode="inference", model=old_model)

        player = AgentPolicy(mode="train", model=old_model)
        # environmnet
        if n_envs == 1:
            env = LuxEnvironment(configs=configs,
                                learning_agent=player,
                                opponent_agents=opponents, 
                                model_update_step_freq=model_update_step_freq)
        else:
            env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                            learning_agent=AgentPolicy(mode="train", model=old_model),
                                                            opponent_agents=opponents,
                                                            model_update_step_freq=model_update_step_freq), i) for i in range(n_envs)])
    else:
        opponents["self-play"] = AgentPolicy(mode="inference")
        player = AgentPolicy(mode="train")
        #  environmnet
        if n_envs == 1:
            env = LuxEnvironment(configs=configs,
                                learning_agent=player,
                                opponent_agents=opponents, 
                                model_update_step_freq=model_update_step_freq)
        else:
            env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                            learning_agent=AgentPolicy(mode="train"),
                                                            opponent_agents=opponents,
                                                            model_update_step_freq=model_update_step_freq), i) for i in range(n_envs)])
        
    model = PPO("MlpPolicy", env, **model_params)

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
                                                     learning_agent=AgentPolicy(mode="train"),
                                                     opponent_agents=opponents), i) for i in range(1)])
        callbacks.append(EvalCallback(env_eval, **eval_params))

    ###########
    # train
    ###########
    logger.info("Training model...")
    if is_resume:
        model.num_timesteps = resume_num_timesteps
        model.learn(total_timesteps=step_count, callback=callbacks, reset_num_timesteps=False)
    else:
        model.learn(total_timesteps=step_count, callback=callbacks)
        
    if not os.path.exists(f'models/rl_model_{step_count}_steps.zip'):
        model.save(path=f'models/rl_model_{step_count}_steps.zip')
    logger.info("Done training model.")


    #############
    # Inference 
    #############
    # kaggleのreplayをself-playで残したい
    run.finish()


if __name__ == "__main__":
    main()

