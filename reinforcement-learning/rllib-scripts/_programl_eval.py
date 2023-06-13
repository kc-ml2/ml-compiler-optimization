import logging

import compiler_gym
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.wrappers import CommandlineWithTerminalAction, TimeLimit
from gym import Wrapper
from ray import tune
from ray.rllib.agents.ppo import ddppo, DDPPOTrainer, PPOTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_torch

from compopt.rllib_model import Model
from compopt.wrappers import compute_graph_space, parse_graph

torch, nn = try_import_torch()


class EvalWrapper(Wrapper):
    def __init__(
            self,
            env,
    ):
        super().__init__(env)
        self.observation_space = compute_graph_space()

    def step(self, ac):
        obs, rew, done, info = self.env.step(ac)
        obs = parse_graph(obs)

        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        obs = parse_graph(obs)

        return obs


def wrapper_stack(env):
    env.observation_space = 'Programl'
    env = CommandlineWithTerminalAction(env)
    env = TimeLimit(env, 1024)  # TODO: rough limit
    env = EvalWrapper(env)

    return env


def make_dummy_env(config):
    """Return an environment with correct observation space for rllib"""
    env = compiler_gym.make("llvm-ic-v0")
    env = wrapper_stack(env)

    return env


class MyPolicy:
    def __init__(self):
        ckpt = '/data1/anthony/eval/DDPPOTrainer_myenv-v0_66aa6_00000_0_2022-05-25_14-24-35/checkpoint_000768/checkpoint-768'
        config = {
            "disable_env_checking": True,
            "_disable_preprocessor_api": False,
            "num_workers": 1,
            # "evaluation_num_workers": 2,
            # "evaluation_interval": 1,
            # Run 10 episodes each time evaluation runs.
            "num_gpus": 0,
            "num_gpus_per_worker": 1,
            "evaluation_duration": 1,
            "num_cpus_per_worker": 32,
            "observation_filter": "NoFilter",
            "num_envs_per_worker": 1,
            "sgd_minibatch_size": 1,
            "rollout_fragment_length": 16,
            "num_sgd_iter": 4,
            "model": {
                "custom_model": "model",
                "vf_share_layers": False,
            },
            "log_level": "DEBUG",
            "ignore_worker_failures": True,
            "vf_loss_coeff": 0.5,
            # "grad_clip": 10,
            "clip_param": 0.1,
            "horizon": 1024,
            # "env_config": {
            #     "num_nodes": 100,
            #     "num_edges": 200,
            # }
        }
        tune.register_env("llvm-eval-v0", lambda c: make_dummy_env(c))
        ModelCatalog.register_custom_model("model", Model)

        # PPOTorchPolicy()
        config = ddppo.DDPPOTrainer.merge_trainer_configs(
            ddppo.DEFAULT_CONFIG,
            config
        )
        self.agent = DDPPOTrainer(config, "llvm-eval-v0")
        self.agent.restore(ckpt)

    def __call__(self, env):
        env = wrapper_stack(env)

        obs = env.reset()
        # obs = parse_graph(obs)
        done = False
        while not done:
            ac = self.agent.compute_single_action(obs)
            # print(ac)
            # logging.info(f'action: {ac}')
            obs, _, done, _ = env.step(ac)


if __name__ == '__main__':
    # env = compiler_gym.make('llvm-ic-v0')
    # env.observation_space = 'Programl'
    # MyPolicy()(env)
    eval_llvm_instcount_policy(MyPolicy())
