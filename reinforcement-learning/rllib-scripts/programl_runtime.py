import multiprocessing as mp

import compiler_gym
import ray
from compiler_gym.wrappers import TimeLimit, RuntimePointEstimateReward
from ray import air
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.tune import register_env
from torch import nn
from compopt.constants import RUNNABLE_BMS, VOCAB
from compopt.encoders import EdgeEncoder, GNN, NodeEncoder
from compopt.rllib_utils import CustomCallbacks
from compopt.wrappers import ObjectRefWrapper

observation_space = 'Programl'


def make_env(env_config):
    env = compiler_gym.make(
        "llvm-v0",
        **env_config,
    )
    env = TimeLimit(env, max_episode_steps=128)
    env = RuntimePointEstimateReward(env)
    env = ObjectRefWrapper(env)

    return env


register_env(
    "llvm", lambda c: make_env(c)
)


def main(bm):
    env_config = {
        'benchmark': bm,
        'observation_space': observation_space,
    }

    algo = (
        PPOConfig()
        .environment(
            'llvm',
            env_config=env_config,
            disable_env_checking=True
        )
        .framework('torch')
        .training(
            train_batch_size=70,
            sgd_minibatch_size=10,
            model={"fcnet_hiddens": [2048, 2048, 2048]}
        )
        .rollouts(
            num_envs_per_worker=1,
            num_rollout_workers=7,
            # batch_mode='complete_episodes',
            rollout_fragment_length=10,
        )
        .resources(num_gpus=2)
        .callbacks(CustomCallbacks)
        .experimental(_disable_preprocessor_api=True)
        # .debugging()
    )

    stop = {
        "timesteps_total": 1000000,
    }
    tuner = tune.Tuner(
        'PPO',
        param_space=algo.to_dict(),
        # tune_config=tune.TuneConfig(), # for hparam search
        run_config=air.RunConfig(
            stop=stop,
            local_dir='/data1/llvm-runtime-final',
            name=f"{env_config['benchmark'].split('/')[-1]}/{observation_space}",
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True,
                num_to_keep=5,
                checkpoint_frequency=5,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_score_order='max',
            ),
        ),
    )

    ray.init(local_mode=False)
    results = tuner.fit()
    ray.shutdown()


if __name__ == "__main__":
    with mp.Pool(len(RUNNABLE_BMS)) as pool:
        pool.map(main, RUNNABLE_BMS)
