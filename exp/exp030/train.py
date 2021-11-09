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
from numpy.lib.function_base import extract
from stable_baselines3.common import callbacks
import pathlib
import pickle
import tempfile
import torch
import yaml
import wandb  
import ast 
import pandas as pd 
from tqdm import tqdm
from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn, get_device
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common import policies
from wandb.integration.sb3 import WandbCallback
from imitation.algorithms.bc import BC, ConstantLRSchedule
from imitation.algorithms.adversarial import airl, gail
from imitation.data import rollout
from imitation.util import logger, util
from torch.utils.data import Dataset, DataLoader
from agent_policy import AgentPolicy, LuxNet

sys.path.append("../../LuxPythonEnvGym")
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default, LuxMatchConfigs_Replay
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


class LuxDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(file, mode="rb") as f:
            traj = pickle.load(f)
        return traj

def main():

    ############
    #  config
    ############
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    seed = config["basic"]["seed"]
    n_envs = config["basic"]["n_envs"]
    model_arche = config["model"]["model_arche"]
    run_id = config['basic']['run_id']
    debug = config["basic"]["debug"]
    trajectory_dir = config["basic"]["trajectory_dir"]
    step_count = config["basic"]["step_count"]
    method = config["basic"]["method"]
    ckpt_params = config["callbacks"]["checkpoints"]
    model_params = config["model"]["params"]
    batch_size = config["model"]["batch_size"]
    
    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]

    mode = None
    if debug:
        mode = "disabled"
    # run = wandb.init(
    #     project='lux-ai', 
    #     entity='kuto5046', 
    #     config=config, 
    #     group=EXP_NAME, 
    #     id = run_id,
    #     mode=mode,
    #     sync_tensorboard=True,# auto-upload sb3's tensorboard metrics
    #     monitor_gym=False,  # auto-upload the videos of agents playing the game
    #     save_code=False,  # optional
    # )
    # Run a training job
    configs = LuxMatchConfigs_Default

    # jsonデータからtrajectoryを作成
    files = glob.glob(trajectory_dir + "*.pickle")
    data_loader = DataLoader(
        LuxDataset(files), 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    
    agent = AgentPolicy(mode="train", arche="cnn")
    opponents = {"self-play":AgentPolicy(mode="train", arche=model_arche)}
    env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                          learning_agent=agent,
                          opponent_agents=opponents), i) for i in range(n_envs)])
    observation_space = agent.observation_space
    action_space = agent.action_space
    # callbacks = []
    # callbacks.append(WandbCallback())
    # callbacks.append(CheckpointCallback(**ckpt_params))

    if method == "BC":
        policy = policies.ActorCriticCnnPolicy(
            observation_space=observation_space, 
            action_space=action_space, 
            lr_schedule=ConstantLRSchedule(torch.finfo(torch.float32).max),
            features_extractor_class=LuxNet,
            features_extractor_kwargs=dict(features_dim=action_space.n)
            )
        # logger.configure("./")
        bc_trainer = BC(
            observation_space=observation_space,
            action_space=action_space,
            policy=policy,
            demonstrations=data_loader,
            batch_size=batch_size,
        )
        
        bc_trainer.train(n_epochs=20, log_interval=100000)
        bc_trainer.save_policy('bc_policy')

    elif method == "GAIL":
        policy_kwargs = dict(
            features_extractor_class=LuxNet,
            features_extractor_kwargs=dict(features_dim=action_space.n),
        )
        # gail_logger = logger.configure("./")
        gail_trainer = gail.GAIL(
            venv=env,
            demonstrations=data_loader,
            demo_batch_size=batch_size,
            gen_algo=PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, **model_params),
            allow_variable_horizon=True,
        )
        gail_trainer.train(
            total_timesteps=step_count)
        gail_trainer.gen_algo.save("gail_policy")
        # elif method == "AIRL":
    #     pass
    
    # run.finish()



if __name__ == "__main__":
    main()

