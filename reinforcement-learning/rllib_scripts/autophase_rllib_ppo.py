import numpy as np
import gym
import compiler_gym

from compiler_gym.wrappers import TimeLimit, CommandlineWithTerminalAction, \
    RandomOrderBenchmarks, RuntimePointEstimateReward, CompilerEnvWrapper, \
    ObservationWrapper
from ray.rllib.agents.ppo import PPOConfig
from ray.tune import register_env


class LogNormalizer(CompilerEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space.low = np.full_like(self.observation_space.low, -9223372036854775807)
        self.observation_space.dtype = np.dtype(np.float32)

    def reset(self):
        obs = self.env.reset()
        return np.log(obs + 1e-8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.log(obs + 1e-8), reward, done, info


def make_env():
    env = compiler_gym.make(
        "llvm-v0",
        observation_space='Autophase',
        reward_space='IrInstructionCountOz',
    )
    env = RandomOrderBenchmarks(
        env,
        env.datasets["benchmark://cbench-v1"].benchmarks()
    )
    env = TimeLimit(env, max_episode_steps=1024)
    env = CommandlineWithTerminalAction(env)
    # env = RuntimePointEstimateReward(env)
    env = LogNormalizer(env)

    return env


register_env(
    "llvm", lambda _: make_env()
)
algo = (
    PPOConfig()
    .environment('llvm', disable_env_checking=True)
    .framework('torch')
).build()

for i in range(1000):
    algo.train()
    if i % 100:
        algo.save()
