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
from ray.rllib.algorithms import Algorithm

from compopt.wrappers import RllibWrapper

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
ENV_ID = 'graph-v0'
ENV_CONFIG = dict(
    # compiler-gym
    observation_space="Programl",
    reward_space="IrInstructionCountOz",
)
MAX_TIMESTEPS = int(2e+7)