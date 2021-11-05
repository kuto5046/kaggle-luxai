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
import yaml
import wandb  
from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn, get_device
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from agent_policy import AgentPolicy, LuxNet
sys.path.append("../../")
from agents.imitation.agent_policy import ImitationAgent
from agents.random.agent_policy import RandomAgent

sys.path.append("../../LuxPythonEnvGym")
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback, StackObsWrapper
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.game.game import Game


def get_logger(level=INFO, out_file=None):
    logger = logging.getLogger("Log")
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

# logger = get_logger(level=INFO, out_file='results.log')


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
        if local_env.learning_agent.n_stack > 1:
            return StackObsWrapper(local_env)
        else:
            return local_env

    set_random_seed(seed)
    return _init

def load_model_params(model, model_params):
    for k, v in model_params.items():
        if hasattr(model, k):
            setattr(model, k, v)
    return model 

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
    debug = config["basic"]["debug"]

    model_update_step_freq = config["model"]["model_update_step_freq"]
    n_stack = config["model"]["n_stack"]

    is_resume = config['resume']['is_resume']
    resume_num_timesteps = config['resume']['resume_num_timesteps']
    run_id = config['resume']['run_id']

    ckpt_params = config['callbacks']['checkpoints']
    eval_params = config['callbacks']['eval']
    model_params = config["model"]["params"]
    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]
    
    if (not is_resume) or (run_id == "None"):
        run_id = wandb.util.generate_id()

    mode = None
    if debug:
        mode = "disabled"

    run = wandb.init(
        project='lux-ai', 
        entity='kuto5046', 
        config=config, 
        group=EXP_NAME, 
        resume="allow",
        id=run_id,
        mode=mode,
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
        "self-play": AgentPolicy(mode="inference", n_stack=n_stack)
    }

    env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                    learning_agent=AgentPolicy(mode="train", n_stack=n_stack),
                                                    opponent_agents=opponents,
                                                    initial_opponent_policy="random",
                                                    model_update_step_freq=model_update_step_freq), i) for i in range(n_envs)])

    if os.path.exists(pretrained_path):
        print(f"\npretrained {pretrained_path}\n")
        model = PPO.load(pretrained_path, device="cuda")
        model.set_env(env)
        model = load_model_params(model, model_params)
    else:
        policy_kwargs = dict(
            features_extractor_class=LuxNet,
            features_extractor_kwargs=dict(features_dim=env.action_space.n),
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
    # An evaluation environment is needed to measure multi-env setups. Use a fixed 4 envs.
    opponents = {"imitation": ImitationAgent()}  # for eval opponent
    env_eval = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                    learning_agent=AgentPolicy(mode="train", n_stack=n_stack),
                                                    opponent_agents=opponents), i) for i in range(1)])
    callbacks.append(EvalCallback(env_eval, **eval_params))

    ###########
    # train
    ###########
    print("Training model...")
    try:
        if is_resume:
            model.num_timesteps = resume_num_timesteps
            model.learn(total_timesteps=step_count, callback=callbacks, reset_num_timesteps=False)
        else:
            model.learn(total_timesteps=step_count, callback=callbacks)
 
        model.save(path=f'models/rl_cnn_model_{step_count}_steps.zip')
        print(f"Done training model.  this: {step_count}(steps), total: {model.num_timesteps}(steps)")
    except:
        model.save(path=f'models/tmp_rl_cnn_model_{model.num_timesteps}_steps.zip')
        print(f"There are something errors. Finish training model. total: {model.num_timesteps}(steps)")
        traceback.print_exc()
    
    run.finish()


if __name__ == "__main__":
    main()

