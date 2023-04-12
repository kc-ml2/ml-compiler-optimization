from pprint import pprint
from typing import Dict, Optional, Union

import numpy as np
import gym
import compiler_gym
import ray

from compiler_gym.wrappers import TimeLimit, CommandlineWithTerminalAction, \
    RandomOrderBenchmarks, RuntimePointEstimateReward, CompilerEnvWrapper, \
    ObservationWrapper
from ray import air
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ddppo import DDPPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID

from ray import tune
from ray.tune import register_env

RUNNABLE_BMS = [
    # 'benchmark://cbench-v1/bitcount',
    # 'benchmark://cbench-v1/blowfish',
    # 'benchmark://cbench-v1/bzip2',
    # 'benchmark://cbench-v1/crc32',
    # 'benchmark://cbench-v1/dijkstra',
    # 'benchmark://cbench-v1/gsm',
    # 'benchmark://cbench-v1/jpeg-c',
    # 'benchmark://cbench-v1/jpeg-d',
    # 'benchmark://cbench-v1/patricia',
    'benchmark://cbench-v1/qsort',
    # 'benchmark://cbench-v1/sha',
    # 'benchmark://cbench-v1/stringsearch',
    # 'benchmark://cbench-v1/stringsearch2',
    # 'benchmark://cbench-v1/susan',
    # 'benchmark://cbench-v1/tiff2bw',
    # 'benchmark://cbench-v1/tiff2rgba',
    # 'benchmark://cbench-v1/tiffdither',
    # 'benchmark://cbench-v1/tiffmedian'
]


class CustomCallbacks(DefaultCallbacks):
    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2, Exception],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        envs = base_env.get_sub_environments()
        runtimes = []
        for env in envs:
            runtimes.append(env.observation['Runtime'])
        avg_runtime = sum(runtimes) / len(runtimes)
        episode.custom_metrics['avg_runtime'] = avg_runtime


class CustomWrapper(CompilerEnvWrapper):
    def __init__(self, env):
        super(CustomWrapper, self).__init__(env)
        self.i = 0

    def reset(self):
        # only reset for runnable benchmarks
        bm = RUNNABLE_BMS[self.i % len(RUNNABLE_BMS)]
        obs = self.env.reset(benchmark=bm)
        self.i += 1
        return obs

class LogNormalizer(CompilerEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space.low = np.full_like(
            self.observation_space.low,
            -9223372036854775807
        )
        self.observation_space.dtype = np.dtype(np.float32)

    def reset(self):
        obs = self.env.reset()
        return np.log(obs + 1e-8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.log(obs + 1e-8), reward, done, info

# observation_space='InstCountNorm'
observation_space='Autophase'
def make_env():
    env = compiler_gym.make(
        "llvm-v0",
        observation_space=observation_space,
    )
    env = TimeLimit(env, max_episode_steps=256)
    env = CommandlineWithTerminalAction(env)
    env = RuntimePointEstimateReward(env)
    env = CustomWrapper(env)
    if observation_space == 'Autophase':
        env = LogNormalizer(env)

    return env


register_env(
    "llvm", lambda _: make_env()
)
algo = (
    SACConfig()
    .environment('llvm')
    .framework('torch')
    .training(
        train_batch_size=32,
        # sgd_minibatch_size=8,
        model={"fcnet_hiddens": [2048, 2048, 2048]}
    )
    .rollouts(
        num_envs_per_worker=16,
        num_rollout_workers=4
        # rollout_fragment_length=4,
    )
    .resources(num_gpus=2)
    .callbacks(CustomCallbacks)
    .debugging()
)

stop = {
    "timesteps_total": 200000,
}
tuner = tune.Tuner(
    'SAC',
    param_space=algo.to_dict(),
    # tune_config=tune.TuneConfig(), # for hparam search
    run_config=air.RunConfig(
        stop=stop,
        local_dir='/data1/llvm-runtime',
        name=observation_space,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end=True,
        ),
    ),
)

ray.init(local_mode=False)
results = tuner.fit()
ray.shutdown()
