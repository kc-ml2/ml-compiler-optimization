from typing import Dict, Optional, Union
import numpy as np
import compiler_gym
import gym
import ray
import torch
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from torch import nn
import torch_geometric as pyg
from compiler_gym.wrappers import TimeLimit, CommandlineWithTerminalAction, \
    RuntimePointEstimateReward, CompilerEnvWrapper
from ray import air, tune
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import PolicyID, ModelConfigDict
from ray.tune import register_env
from torch_geometric.nn import GATv2Conv

from compopt.constants import VOCAB, RUNNABLE_BMS
from compopt.utils import NumpyPreprocessor

observation_space = 'Programl'


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


def _sample():
    return [
               pyg.data.Data(
                   torch.FloatTensor(np.random.randn()),
                   torch.LongTensor(np.random.randint()).T,
                   torch.FloatTensor(np.random.randn())
               )
           ] * 32


class CustomWrapper(CompilerEnvWrapper):
    def __init__(self, env):
        super(CustomWrapper, self).__init__(env)
        self.i = 0
        self.observation_space.sample = _sample()

    def reset(self):
        # only reset for runnable benchmarks
        bm = RUNNABLE_BMS[self.i % len(RUNNABLE_BMS)]
        obs = self.env.reset(benchmark=bm)
        self.i += 1

        return obs


class CustomProcessor(gym.Wrapper):
    def __init__(self):
        self.preprocessor = NumpyPreprocessor(VOCAB)

    def _process(self, obs):
        x, edge_index, edge_attr = self.preprocessor.process(obs)
        data = pyg.data.Data(
            torch.FloatTensor(x),
            torch.LongTensor(edge_index).T,
            torch.FloatTensor(edge_attr)
        )

        return data

    def reset(self):
        obs = self.env.reset()

        return self._process(obs)

    def step(self, ac):
        obs, rew, done, info = self.env.step(ac)
        return self._process(obs), rew, done, info


def make_env():
    env = compiler_gym.make(
        'llvm-v0',
        observation_space=observation_space,
    )
    env = RuntimePointEstimateReward(env)
    env = TimeLimit(env, max_episode_steps=256)
    env = CommandlineWithTerminalAction(env)
    env = CustomWrapper(env)
    env = CustomProcessor(env)

    return env


class PolicyModel(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.c1 = GATv2Conv(
            7699,
            1024,
            edge_dim=3,
            heads=4,
            add_self_loops=True,
        )
        self.c2 = GATv2Conv(
            1024 * 4,
            1024,
            edge_dim=3,
            add_self_loops=True,
        )
        self.linear = nn.Linear(1024, num_outputs)

    def forward(self, obs):
        x = self.c1(obs.x, obs.edge_index, obs.edge_attr)
        x = self.c2(x, obs.edge_index, obs.edge_attr)
        x = self.linear(x)

        return x


class ValueModel(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.c1 = GATv2Conv(
            7699,
            1024,
            edge_dim=3,
            heads=4,
            add_self_loops=True,
        )
        self.c2 = GATv2Conv(
            1024 * 4,
            1024,
            edge_dim=3,
            add_self_loops=True,
        )
        self.linear = nn.Linear(1024, num_outputs)

    def forward(self, obs):
        x = self.c1(obs.x, obs.edge_index, obs.edge_attr)
        x = self.c2(x, obs.edge_index, obs.edge_attr)
        x = self.linear(x)

        return x


class CustomSAC(SACTorchModel):
    def build_policy_model(
        self,
        obs_space,
        num_outputs,
        policy_model_config,
        name
    ):

        return PolicyModel(num_outputs)

    def build_q_model(
        self,
        obs_space,
        action_space,
        num_outputs,
        q_model_config,
        name
    ):

        return ValueModel(num_outputs)


register_env(
    "llvm", lambda _: make_env()
)
algo = (
    SACConfig()
    .environment(
        'llvm',
        disable_env_checking=True
    )
    .framework('torch')
    .training(
        # q_model_config={'custom_model': Model},
        # policy_model_config={'custom_model': Model},
        train_batch_size=64,
        # sgd_minibatch_size=8,
        model={"fcnet_hiddens": [2048, 2048, 2048]}
    )
    .rollouts(
        # num_envs_per_worker=4,
        # num_rollout_workers=4,
        # batch_mode='complete_episodes',
        # rollout_fragment_length=4,
    )
    .resources(num_gpus=1)
    .callbacks(CustomCallbacks)
    .experimental(_disable_preprocessor_api=True)
    # .debugging()
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
            num_to_keep=5,
            checkpoint_score_attribute='episode_reward_mean',
            checkpoint_score_order='max',
        ),
    ),
)

ray.init(local_mode=False)
results = tuner.fit()
ray.shutdown()
