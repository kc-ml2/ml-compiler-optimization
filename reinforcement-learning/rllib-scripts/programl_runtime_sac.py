import compiler_gym
import compiler_gym
import gym
import numpy as np
import ray
import torch
import torch_geometric as pyg
from compiler_gym.wrappers import TimeLimit, CommandlineWithTerminalAction, \
    RuntimePointEstimateReward
from ray import air, tune
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune import register_env
from torch import nn
from torch_geometric.nn import GATv2Conv

from compopt.constants import VOCAB
from compopt.rllib_utils import CustomCallbacks
from compopt.utils import NumpyPreprocessor
from compopt.wrappers import RunnableWrapper
from compiler_gym.spaces import Sequence

observation_space = 'Programl'


def _sample():
    return [
       pyg.data.Data(
           torch.FloatTensor(np.random.randn(100, 7699)),
           torch.LongTensor(np.random.randint(100, 2)).T,
           torch.FloatTensor(np.random.randn(100, 3))
       )
   ] * 32


class CustomProcessor(gym.Wrapper):
    def __init__(self, env):
        super(CustomProcessor, self).__init__(env)
        self.preprocessor = NumpyPreprocessor(VOCAB)
        # self.env.observation_space = None
        # self.observation_space.sample = _sample
        # print(self.env.observation_space)
        # print(self.env.action_space)
        # print(self.env.reward_space)

    def _process(self, obs):
        x, edge_index, edge_attr = self.preprocessor.process(obs)
        data = {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}
        # data = pyg.data.Data(
        #     torch.FloatTensor(x),
        #     torch.LongTensor(edge_index).T,
        #     torch.FloatTensor(edge_attr)
        # )

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
    env = RunnableWrapper(env)
    env = CustomProcessor(env)

    return env


class PolicyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
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

    def forward(self, input_dict, state_in, seq_lens):
        obs = input_dict['obs']
        x, edge_index, edge_attr = obs['x'], obs['edge_index'], obs['edge_attr']
        x = self.c1(x, edge_index, edge_attr)
        x = self.c2(x, edge_index, edge_attr)
        x = self.linear(x)

        return x, []


class ValueModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

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

    def forward(self, input_dict, state_in, seq_lens):
        obs = input_dict['obs']
        x, edge_index, edge_attr = obs['x'], obs['edge_index'], obs['edge_attr']
        x = self.c1(x, edge_index, edge_attr)
        x = self.c2(x, edge_index, edge_attr)
        x = self.linear(x)

        return x, []


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
        q_model_config={'custom_model': ValueModel},
        policy_model_config={'custom_model': PolicyModel},
        train_batch_size=64,
        # sgd_minibatch_size=8,
        model={"fcnet_hiddens": [2048, 2048, 2048]}
    )
    .rollouts(
        # num_envs_per_worker=4,
        num_rollout_workers=0,
        # batch_mode='complete_episodes',
        # rollout_fragment_length=4,
        enable_connectors=True
    )
    .resources(num_gpus=0)
    .callbacks(CustomCallbacks)
    .experimental(
        _disable_preprocessor_api=True,
    )
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
