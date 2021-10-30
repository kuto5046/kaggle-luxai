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
import pathlib
import pickle
import tempfile
import torch
import yaml
import wandb  
from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn, get_device
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, configure
from wandb.integration.sb3 import WandbCallback
from imitation.algorithms import bc
from imitation.algorithms.adversarial import airl, gail
from imitation.data import rollout
from imitation.util import logger, util

from agent_policy import AgentPolicy, LuxNet
sys.path.append("../../")
from agents.imitation.agent_policy import ImitationAgent
from agents.random.agent_policy import RandomAgent

sys.path.append("../../LuxPythonEnvGym")
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
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


    # jsonデータからtrajectoryを作成
    trajectories = None

    # Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
    # This is a more general dataclass containing unordered
    # (observation, actions, next_observation) transitions.
    transitions = rollout.flatten_trajectories(trajectories)

    env = LuxEnvironment(configs=configs,
                          learning_agent=None,
                          opponent_agents=None, 
                          model_update_step_freq=model_update_step_freq)

    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    # Train BC on expert data.
    # BC also accepts as `demonstrations` any PyTorch-style DataLoader that iterates over
    # dictionaries containing observations and actions.
    bc_logger = logger.configure(tempdir_path / "BC/")
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        custom_logger=bc_logger,
    )
    bc_trainer.train(n_epochs=10)

    run.finish()


if __name__ == "__main__":
    main()

