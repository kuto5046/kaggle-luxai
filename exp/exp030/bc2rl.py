import argparse
import glob
import logging
import os
import random
import sys
import traceback
from functools import partial  # pip install functools
from logging import DEBUG, INFO
from pathlib import Path
import gym 
from imitation.algorithms import bc
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common import callbacks
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import (get_device, get_schedule_fn,
                                            set_random_seed)
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnv, sync_envs_normalization,
                                              DummyVecEnv, VecEnv, VecMonitor,
                                              is_vecenv_wrapped)

from wandb.integration.sb3 import WandbCallback

import wandb
from agent_policy import AgentPolicy, LuxNet
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy

sys.path.append("../../")
from agents.imitation.agent_policy import ImitationAgent
from agents.random.agent_policy import RandomAgent

sys.path.append("../../LuxPythonEnvGym")
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import (LuxEnvironment, SaveReplayAndModelCallback,
                                   CustomEnvWrapper)
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.game.game import Game
from stable_baselines3.common import base_class


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
        return CustomEnvWrapper(local_env)

    set_random_seed(seed)
    return _init


def main():
    env = SubprocVecEnv([make_env(LuxEnvironment(configs=LuxMatchConfigs_Default,
                                                learning_agent=AgentPolicy(mode="train", n_stack=1),
                                                opponent_agents={"self-play": AgentPolicy(mode="inference", n_stack=1)}), rank=0)])

    policy_kwargs = dict(
        features_extractor_class=LuxNet,
        features_extractor_kwargs=dict(features_dim=128),
    )
    # Attach a ML model from stable_baselines3 and train a RL model
    # class CopyPolicy(ActorCriticCnnPolicy):
    #     def __new__(cls, *args, **kwargs):
    #         return bc.reconstruct_policy("bc_policy")

    model = PPO(CopyPolicy, env, policy_kwargs=policy_kwargs)
    model.save(path='bc_model.zip')


if __name__ == "__main__":
    main()

