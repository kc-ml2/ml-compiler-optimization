"""
these configs and methods are instantly changed, but used across experiments
"""
from typing import Dict, Union, Optional
import itertools
from platform import platform
from netrc import netrc
from operator import itemgetter

import compiler_gym
from compiler_gym.wrappers import RandomOrderBenchmarks, CommandlineWithTerminalAction, CycleOverBenchmarks
from ray.rllib.env import EnvContext

PROJECT = 'MLCO'

import numpy as np

from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID, EnvType

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'macOS' in platform():
    DEVICE = 'mps'

# logging and data path
DATA_DIR = '/data4/anthony/mlco'
LOG_DIR = f'{DATA_DIR}/ray-result'
try:
    WANDB = {
        'project': PROJECT,
        'api_key': netrc().authenticators('api.wandb.ai')[-1]
    }
except Exception as e:
    pass

# algorithm
# ENV_ID = 'MeentIndex-v0'  # 'MeentDirection-v0'
ENV_ID = 'llvm-ic-v0'
ENV_CONFIG = dict(
    # compiler-gym
    observation_space="Autophase",
    reward_space="IrInstructionCountOz",
)
MAX_TIMESTEPS = int(2e+7)


# helper methods
def make_env(**env_config):
    env = compiler_gym.make(ENV_ID, **env_config)
    train_benchmarks = []
    for i in ['cbench-v1', 'npb-v0', 'chstone-v0']:
        train_benchmarks.extend(
            list(env.datasets[i].benchmark_uris())
        )
    env = CommandlineWithTerminalAction(env)
    env = RandomOrderBenchmarks(env, train_benchmarks)
    # env = TimeLimit(env, max_episode_steps=128)
    return env


def register_all(config, model_cls=None):
    register_env(ENV_ID, lambda c: make_env(**config.env_config))
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
        # worker.env = RandomOrderBenchmarks(sub_environment)
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
        pass

    def on_create_policy(self, *, policy_id: PolicyID, policy: Policy) -> None:
        pass

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


def process_result(algo):
    bests = algo.workers.foreach_env(lambda env: env.best)

    bests.pop(0)
    bests = list(itertools.chain(*bests))

    best = max(bests, key=itemgetter(0))

    max_eff = best[0]
    img = best[1][np.newaxis, np.newaxis, :].repeat(32, axis=1)
    mean_eff = np.array([i[0] for i in bests]).mean()

    return {
        'scalar': {
            f'best_efficiency': max_eff,
            f'mean_efficiency': mean_eff,
        },
        'img': {
            f'best_structure': img,
        }
    }
