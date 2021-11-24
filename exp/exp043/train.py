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
from agent_policy import AgentPolicy, CustomActorCriticCnnPolicy

sys.path.append("../../")
from agents.imitation.agent_policy import ImitationAgent
from agents.random.agent_policy import RandomAgent

sys.path.append("../../LuxPythonEnvGym")
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.game.game import Game
from stable_baselines3.common import base_class
# from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_rewards_dict = {}
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    current_rewards_dict = {}
    observations = env.reset()
    states = None
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        for k, v in infos[0].items():
            if k in current_rewards_dict.keys():
                current_rewards_dict[k] += v 
            elif "rew" in k:
                current_rewards_dict[k] = np.zeros(n_envs)
                current_rewards_dict[k] += v 

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]

                if callback is not None:
                    callback(locals(), globals())

                if (dones[i]):

                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode                            
                            episode_counts[i] += 1
                            for k, v in current_rewards_dict.items():
                                if k not in episode_rewards_dict.keys():
                                    episode_rewards_dict[k] = []
                                episode_rewards_dict[k].append(v[i])
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                        for k, v in current_rewards_dict.items():
                            if k not in episode_rewards_dict.keys():
                                episode_rewards_dict[k] = []
                            episode_rewards_dict[k].append(v[i])

                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    for k,v in current_rewards_dict.items():
                        current_rewards_dict[k][i] = 0
                    if states is not None:
                        states[i] *= 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_rewards_dict
    return mean_reward, std_reward


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
            episode_rewards, episode_lengths, episode_rewards_dict = evaluate_policy(
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
            for k, v in episode_rewards_dict.items():
                self.logger.record(k, np.mean(v))

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
    pretrained_bc_path = config["basic"]["pretrained_bc_path"]
    pretrained_path = config["basic"]["pretrained_path"]
    debug = config["basic"]["debug"]

    _n_obs_channel = config["model"]["_n_obs_channel"]
    # features_dim = config["model"]["features_dim"]
    model_update_step_freq = config["model"]["model_update_step_freq"]
    n_stack = config["model"]["n_stack"]

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
    
    # if (not is_resume) or (run_id == "None"):
    #     run_id = wandb.util.generate_id()

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
                                                    learning_agent=AgentPolicy(mode="train", _n_obs_channel=_n_obs_channel, n_stack=n_stack),
                                                    # opponent_agents={"imitation": ImitationAgent(id=i)},
                                                    opponent_agents={
                                                        "imitation": ImitationAgent(),
                                                        "self-play": AgentPolicy(mode="inference", _n_obs_channel=_n_obs_channel, n_stack=n_stack)
                                                        },
                                                    initial_opponent_policy="imitation",
                                                    model_update_step_freq=model_update_step_freq,
                                                    model_save_path=model_save_path), i) for i in range(n_envs)]))

    if os.path.exists(pretrained_path):
        print(f"\npretrained {pretrained_path}\n")
        model = PPO.load(pretrained_path, device="cuda")
        model.set_env(env)
        model = load_model_params(model, model_params)
        model._setup_model()
    else:
        # policy_kwargs = dict(
        #     features_extractor_class=LuxNet,
        #     features_extractor_kwargs=dict(features_dim=features_dim),  
        # )
        # Attach a ML model from stable_baselines3 and train a RL model
        class BCPolicy(CustomActorCriticCnnPolicy):
            def __new__(cls, *args, **kwargs):
                policy = bc.reconstruct_policy(pretrained_bc_path)

                # feature extractorのparamを固定
                # for param in policy.features_extractor.parameters():
                #     param.requires_grad = False

                # feature extractorの最終層だけ初期化して学習
                # policy.features_extractor.head = nn.Linear(32, 64, bias=False)
                # for param in policy.features_extractor.head.parameters():
                #     param.requires_grad = True
                
                return policy

        model = PPO(BCPolicy, env, **model_params)
        # model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, **model_params)
    #############
    #  callback
    #############
    callbacks = []
    callbacks.append(WandbCallback())
    callbacks.append(CheckpointCallback(**ckpt_params))
    
    # Since reward metrics don't work for multi-environment setups, we add an evaluation logger
    # # for metrics.
    # An evaluation environment is needed to measure multi-env setups. Use a fixed 4 envs.
    env_eval = VecMonitor(SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                    learning_agent=AgentPolicy(mode="train", _n_obs_channel=_n_obs_channel, n_stack=n_stack),
                                                    opponent_agents={"imitation": ImitationAgent()},
                                                    # opponent_agents={"random": RandomAgent()},
                                                    initial_opponent_policy="imitation"), i) for i in range(eval_n_envs)]))
    callbacks.append(CustomEvalCallback(env_eval, **eval_params))
    
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