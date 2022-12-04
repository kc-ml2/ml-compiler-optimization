import itertools
import os
import signal
import sys
from datetime import datetime
from netrc import netrc
from operator import itemgetter
from platform import platform
from pprint import pprint
from typing import Dict, Optional, Union

import numpy as np

import compiler_gym

import ray
from compiler_gym.util.registration import register
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.wrappers import (
    CommandlineWithTerminalAction, RandomOrderBenchmarks, TimeLimit
)
from ray.rllib.algorithms import Algorithm
from ray.rllib.env import EnvContext
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2

from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import EnvType, PolicyID
from ray.tune import register_env

from compopt.wrappers import RllibWrapper
import configs

PROJECT = 'mlco'
PROJECT_DIR = f'/data4/anthony/{PROJECT}'
RAW_ENV_ID = 'llvm-ic-v0'
ENV_ID = f'wrapped-{RAW_ENV_ID}'
ENV_CONFIG = dict(
    # compiler-gym
    observation_space="Programl",
    reward_space="IrInstructionCountOz",
)
MAX_TIMESTEPS = int(2e+7)

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# # helper methods
def register_all(config, model_cls=None):
    def make_env(dummy):
        env = compiler_gym.make(RAW_ENV_ID, **config.env_config)
        train_benchmarks = []
        for i in ['cbench-v1', 'npb-v0', 'chstone-v0']:
            train_benchmarks.extend(
                list(env.datasets[i].benchmark_uris())
            )
        env = CommandlineWithTerminalAction(env)
        env = TimeLimit(env, max_episode_steps=1024)
        env = RandomOrderBenchmarks(env, train_benchmarks)
        env = RllibWrapper(env)
        
        return env

    register_env(
        ENV_ID,
        make_env
    )
    if model_cls:
        ModelCatalog.register_custom_model(model_cls.__name__, model_cls)

class Callbacks(DefaultCallbacks):
    def on_sub_environment_created(
            self,
            *,
            worker: "RolloutWorker",
            sub_environment: EnvType,
            env_context: EnvContext,
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        pass

    def on_train_result(
            self,
            *,
            algorithm: "Algorithm",
            result: dict,
            **kwargs,
    ) -> None:
        pass

    def on_algorithm_init(
            self,
            *,
            algorithm: "Algorithm",
            **kwargs,
    ) -> None:
        print(algorithm.get_policy().model)

    def on_create_policy(self, *, policy_id: PolicyID, policy: Policy) -> None:
        print(policy_id, policy.model)

    def on_episode_start(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        bms = [e.benchmark for e in base_env.get_sub_environments()]
        print(f'training {bms}')
        episode.media['trained_benchmarks'] = bms

    def on_episode_step(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            episode: Union[Episode, EpisodeV2],
            **kwargs,
    ) -> None:
        pass

    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2, Exception],
            **kwargs,
    ) -> None:
        pass

class Workspace:
    def __init__(
            self,
            config,
            ckpt_dir,
            model_cls=None,
            ckpt_interval=sys.maxsize,
            verbose=False
    ):
        self.config = config
        self.ckpt_dir = ckpt_dir
        self.obs_type = config.env_config['observation_space']
        self.ckpt_interval = ckpt_interval
        self.verbose = verbose

        register_all(
            config=config,
            model_cls=model_cls
        )
        self.algo = config.build()

    def save(self, ckpt_dir=None):

        self.algo.save(self.ckpt_dir)


    def train(self, train_iter=int(1e+5)):

        for i in range(train_iter):
            result = self.algo.train()

            if self.verbose:
                pprint(result)
            # if i % self.ckpt_interval == 0:
            #     self.algo.save(self.ckpt_dir)


    def eval(self):
        eval_llvm_instcount_policy(self._eval)

    def _eval(self, env):
        done = False
        env = CommandlineWithTerminalAction(env)
        while not done:
            ac = self.algo.compute_single_action(env.observation[self.obs_type])
            obs, _, done, _ = env.step(ac)


# algo.restore('/home/anthony/ray_results/PPO_llvm-ic-v0_2022-11-13_00-51-04euk5qbze/checkpoint_000033')


if __name__ == '__main__':
    from compiler_gym.leaderboard.llvm_instcount import FLAGS
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    FLAGS.leaderboard_results = os.path.join(PROJECT_DIR, f'{ts}_llvm_instcount-results.csv')
    FLAGS.leaderboard_logfile = os.path.join(PROJECT_DIR, f'{ts}_llvm_instcount-results.log')

    ray.init()
    from compopt.rllib_model import Model as model_cls

    config = configs.get_ppo()
    config.training(
        model={'custom_model': model_cls.__name__}
    ).environment(
        env=ENV_ID,
        env_config=ENV_CONFIG,
        disable_env_checking=True
    ).callbacks(
        Callbacks
    )
    ws = Workspace(
        model_cls=model_cls,
        config=config,
        ckpt_dir=PROJECT_DIR,
        verbose=True
    )
    try:
        ws.train(5000)
    except KeyboardInterrupt as e:
        ws.algo.save()
        sys.exit(1)
    ws.eval()
    ws.train(5000)
    ws.eval()
    ws.algo.save(ws.ckpt_dir)
    ray.shutdown()
