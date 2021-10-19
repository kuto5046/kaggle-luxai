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
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from agent_policy import AgentPolicy
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, configure


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

class SelfPlayCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, model_type, load_freq_rollout, verbose=0):
        super(SelfPlayCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.model_type = model_type
        self.load_freq_rollout = load_freq_rollout
        self.num_rollout = 0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True


    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        if (self.num_rollout > 0)&(self.num_rollout % self.load_freq_rollout == 0):
            models = glob.glob(f'./models/rl_model_*_steps.zip')
            if (self.model_type == "latest")&(len(models)>0):
                pretrained_model = sorted(models, key=lambda x: int(x.split('_')[-2]), reverse=True)[0]
            elif self.model_type == "best":
                pretrained_model = './models/best_model'
            
            logger.info(f"[STEP:{self.num_timesteps} ROLLOUT:{self.num_rollout}] Update the opponent by {self.model_type} model for self-play...")
            self.model.load(pretrained_model)
    
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.num_rollout += 1
            
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
    resume = config["basic"]["resume"]
    pretrained_path = config["basic"]["pretrained_path"]
    ckpt_params = config['callbacks']['checkpoints']
    eval_params = config['callbacks']['eval']
    selfplay_params = config['callbacks']['selfplay']
    model_params = config["model"]
    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]

    run = wandb.init(
        project='lux-ai', 
        entity='kuto5046', 
        config=config, 
        group=EXP_NAME, 
        sync_tensorboard=True,# auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )
    # Run a training job
    configs = LuxMatchConfigs_Default


    ##############
    #    agent
    ##############
    # Create a default opponent agent and a RL agent in training mode

    if resume:
        # agent
        opponent = AgentPolicy(mode="train")
        player = AgentPolicy(mode="train")

        #  environmnet
        if n_envs == 1:
            env = LuxEnvironment(configs=configs,
                                learning_agent=player,
                                opponent_agent=opponent)
        else:
            env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                        learning_agent=AgentPolicy(mode="train"),
                                                        opponent_agent=opponent), i) for i in range(n_envs)])

        # by default previous model params are used (lr, batch size, gamma...)
        model = PPO.load(pretrained_path)
        model.set_env(env=env)
        # Update the learning rate
        model.lr_schedule = get_schedule_fn(model_params["learning_rate"])
    else:
        # agent
        opponent = Agent()
        player = AgentPolicy(mode="train")

        #  environmnet
        if n_envs == 1:
            env = LuxEnvironment(configs=configs,
                                learning_agent=player,
                                opponent_agent=opponent)
        else:
            env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                        learning_agent=AgentPolicy(mode="train"),
                                                        opponent_agent=opponent), i) for i in range(n_envs)])
        # model
        model = PPO("MlpPolicy", env, **model_params) 

    #############
    #  callback
    #############
    callbacks = []
    callbacks.append(WandbCallback())
    callbacks.append(CheckpointCallback(**ckpt_params))
    callbacks.append(SelfPlayCallback(**selfplay_params))

    # Save a checkpoint and 5 match replay files every 100K steps
    # callbacks.append(
    #     SaveReplayAndModelCallback(
    #         replay_env=LuxEnvironment(
    #             configs=configs,
    #             learning_agent=AgentPolicy(mode="inference", model=model),
    #             opponent_agent=Agent()
    #         ),
    #         **ckpt_params
    #     )
    # )    
    # Since reward metrics don't work for multi-environment setups, we add an evaluation logger
    # for metrics.
    env_eval = None
    if n_envs > 1:
        # An evaluation environment is needed to measure multi-env setups. Use a fixed 4 envs.
        env_eval = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                     learning_agent=AgentPolicy(mode="train"),
                                                     opponent_agent=opponent), i) for i in range(4)])
        callbacks.append(EvalCallback(env_eval, **eval_params))

    ###########
    # train
    ###########
    logger.info("Training model...")
    model.learn(total_timesteps=step_count, callback=callbacks)
        
    if not os.path.exists(f'models/rl_model_{step_count}_steps.zip'):
        model.save(path=f'models/rl_model_{step_count}_steps.zip')
    logger.info("Done training model.")


    #############
    # Inference 
    #############
    # kaggleのreplayをself-playで残したい
    run.finish()
    os.remove("log.txt")


if __name__ == "__main__":
    main()

