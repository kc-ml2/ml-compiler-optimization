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
        # self.observation_space.dtype = np.float32
    #     log_space = gym.spaces.Box(
    #         low=-9223372036854775807,
    #         high=9223372036854775807,
    #         shape=(env.observation_space.shape[0],),
    #         dtype=float,
    #     )
    #     setattr(log_space, 'name', 'LogSpace')
    #     # self.observation_space = log_space
    #     #
    #     self.env.unwrapped.observation.add_space(
    #         log_space
    #     )
    #     self.env.unwrapped.observation_space = log_space.name

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
    # env.observation_space = 'Autophase'
    # env.reward_space = 'IrInstructionCountOz'
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

for i in range(10000):
    algo.train()
