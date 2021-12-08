import argparse
import glob
import logging
import os
import random
import sys
import copy 
import time 
import traceback
from functools import partial  # pip install functools
from logging import DEBUG, INFO
from pathlib import Path
import gym
import cProfile
from imitation.algorithms import bc
import numpy as np
import torch
import torch.nn as nn 
from gym import spaces
import yaml
from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common import callbacks
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList, EventCallback,
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
from agent_policy import AgentPolicy, CustomActorCriticCnnPolicy, CustomFeatureExtractor

sys.path.append("../../")
# from agents.imitation.agent_policy import ImitationAgent
# from agents.imitation_.agent_policy import ImitationAgent_
from agents.unet_imitation.agent_policy import ImitationAgent
from agents.random.agent_policy import RandomAgent
from agents.eval_agent.agent_policy import EvalImitationAgent

sys.path.append("../../LuxPythonEnvGym")
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.game.game import Game
from stable_baselines3.common import base_class
from stable_baselines3.common.evaluation import evaluate_policy


class CustomEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(CustomEvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_rollout_start(self) -> bool:

        if self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
        # if self.n_calls % self.eval_freq == 0:
            start_time = time.time()
            print(f"[STEP {self.num_timesteps}]Eval {self.n_eval_episodes} episodes")
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # for k, v in episode_rewards_dict.items():
            #     self.logger.record(k, np.mean(v))

            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
            end_time = time.time()
            print(f"Eval spend time: {end_time-start_time}")

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

class RLPolicy(torch.nn.Module):
    def __init__(self, feature_extractor, mlp_extractor, value_net):
        super(RLPolicy, self).__init__()
        self.feature_extractor = feature_extractor
        self.mlp_extractor = mlp_extractor
        self.value_net = value_net

    def forward(self, observation):
        features = self.feature_extractor(observation)
        value_latent = self.mlp_extractor.forward_critic(features)
        return features[0], self.value_net(value_latent)

class CustomCheckpointCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save("./models/latest_checkpoint")
            self_play_model = RLPolicy(
                self.model.policy.features_extractor, 
                self.model.policy.mlp_extractor, 
                self.model.policy.value_net
            )
            dummy_obs = torch.rand(4, 17, 32, 32)
            dummy_global_obs = torch.rand(4,8,4,4)
            dummy_mask = torch.rand(6,32,32)  # not use
            copy_model = copy.deepcopy(self_play_model)  # cpuで保存すると元のmodelもcpuになってしまうのでdeepcopyする
            # traced = torch.jit.trace(copy_model.cpu(), {"obs":dummy_obs, "global_obs":dummy_global_obs, "mask": dummy_mask})
            # traced.save(f'{path}.pth')
            torch.onnx.export(copy_model.cpu(), {"obs": dummy_obs, "global_obs": dummy_global_obs, "mask": dummy_mask}, f"{path}.onnx", input_names=["obs", "global_obs", "mask"], opset_version=11)

            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True

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
    n_envs = config["basic"]["n_envs"]
    step_count = config["basic"]["step_count"]
    pretrained_il_path = config["basic"]["pretrained_il_path"]
    pretrained_rl_path = config["basic"]["pretrained_rl_path"]
    debug = config["basic"]["debug"]

    model_update_step_freq = config["model"]["model_update_step_freq"]

    is_resume = config['resume']['is_resume']
    resume_num_timesteps = config['resume']['resume_num_timesteps']
    run_id = config['resume']['run_id']

    ckpt_params = config['callbacks']['checkpoints']
    eval_params = config['callbacks']['eval']['params']
    eval_n_envs = config['callbacks']['eval']['n_envs']
    model_params = config["model"]["params"]
    model_save_path = Path(os.getcwd())/ckpt_params['save_path']
    seed_everything(seed)
    EXP_NAME = str(Path().resolve()).split('/')[-1]
    observation_space = spaces.Dict(
        {"obs":spaces.Box(low=0, high=1, shape=(4,17, 32, 32), dtype=np.float32), 
        "global_obs":spaces.Box(low=0, high=1, shape=(4, 8, 4, 4), dtype=np.float32),
        "mask":spaces.Box(low=0, high=1, shape=(6, 32, 32), dtype=np.long),
        })
    n_obs_channel = observation_space["obs"].shape[1]
    n_global_obs_channel = observation_space["global_obs"].shape[1]
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
    env = VecMonitor(SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                    learning_agent=AgentPolicy(mode="train"),
                                                    opponent_agents={
                                                        "imitation": ImitationAgent(),
                                                        # "self-play": AgentPolicy(mode="inference")
                                                        },
                                                    initial_opponent_policy="imitation",
                                                    model_update_step_freq=model_update_step_freq,
                                                    model_save_path=model_save_path), i) for i in range(n_envs)]))

    if os.path.exists(pretrained_rl_path):
        print(f"\npretrained {pretrained_rl_path}\n")
        model = PPO.load(pretrained_rl_path, device="cuda")
        model.set_env(env)
        model = load_model_params(model, model_params)
        model._setup_model()
    else:
        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256+n_global_obs_channel),  
        )
        model = PPO(CustomActorCriticCnnPolicy, env, policy_kwargs=policy_kwargs, **model_params)
        model.policy.load_state_dict(torch.load(pretrained_il_path))

    # fix param except outc
    for param in model.policy.features_extractor.parameters():
        param.requires_grad = False
    

    #############
    #  callback
    #############
    callbacks = []
    callbacks.append(WandbCallback())
    callbacks.append(CustomCheckpointCallback(**ckpt_params))
    
    # evaluate agent's score is 1550. 
    # env_eval = VecMonitor(SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
    #                                                 learning_agent=AgentPolicy(mode="train"),
    #                                                 opponent_agents={"evaluate": EvalImitationAgent()},
    #                                                 initial_opponent_policy="evaluate"), i) for i in range(eval_n_envs)]))
    # callbacks.append(CustomEvalCallback(env_eval, **eval_params))
    
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
        # torch.onnx.export(model.policy.cpu(), torch.rand(1, 23, 32, 32), f"models/rl_cnn_model_{step_count}_steps.onnx",input_names=['input_1'])
        print(f"Done training model.  this: {step_count}(steps), total: {model.num_timesteps}(steps)")
    except:
        model.save(path=f'models/tmp_rl_cnn_model_{model.num_timesteps}_steps.zip')
        # torch.onnx.export(model.policy.cpu(), torch.rand(1, 23, 32, 32), f"models/tmp_rl_cnn_model_{model.num_timesteps}_steps.onnx",input_names=['input_1'])
        print(f"There are something errors. Finish training model. total: {model.num_timesteps}(steps)")
        traceback.print_exc()
    
    run.finish()


if __name__ == "__main__":
    main()