import os
import numpy as np
import torch
import gym
from compiler_gym.wrappers import TimeLimit, CommandlineWithTerminalAction, \
    RandomOrderBenchmarks, RuntimePointEstimateReward

os.environ["MUJOCO_GL"] = "egl"

import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.utils.utils import load_config, get_base_directory
from dreamer.envs.envs import make_dmc_env, make_atari_env, get_env_infos

import compiler_gym


class Normalizer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.k = 5
        self.history = np.zeros((self.env.observation_space.shape[0], self.k))
        self.i = 0

    def reset(self):
        obs = self.env.reset()
        return self.normalize(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.normalize(obs), reward, done, info

    def normalize(self, obs):
        # calculates simple moving average
        self.history[:, self.i % self.k] = obs
        self.i += 1

        return self.history.mean(1)


class LogNormalizer(gym.Wrapper):
    def reset(self):
        obs = self.env.reset()
        return np.log(obs + 1e-8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.log(obs + 1e-8), reward, done, info


def imagine(env, agent):
    posterior, deterministic = agent.rssm.recurrent_model_input_init(1)
    action = torch.zeros(1, agent.action_size).to(agent.device)

    observation = env.reset()
    print(env.benchmark)
    embedded_observation = agent.encoder(
        torch.from_numpy(observation).float().to(agent.device)
    )

    score = 0
    score_lst = np.array([])
    done = False

    deterministic = agent.rssm.recurrent_model(
        posterior, action, deterministic
    )
    prior_dist, prior = agent.rssm.transition_model(deterministic)
    posterior_dist, posterior = agent.rssm.representation_model(
        embedded_observation, deterministic
    )

    state = posterior.reshape(-1, agent.config.stochastic_size)
    deterministic = deterministic.reshape(-1, agent.config.deterministic_size)

    real_obses = []
    real_rewards = []
    rewal_runtimes = []
    acs, runtimes, recons = [], [], []
    rewards = []
    # continue_predictor reinit
    for t in range(agent.config.horizon_length):
        action = agent.actor(state, deterministic)
        deterministic = agent.rssm.recurrent_model(state, action, deterministic)
        _, state = agent.rssm.transition_model(deterministic)
        action = torch.argmax(action)
        ac = action.detach().cpu().item()
        obs, reward, done, _ = env.step(ac)
        real_obses.append(obs)
        real_rewards.append(reward)
        rewal_runtimes.append(env.state.walltime)
        acs.append(ac)
        runtimes.append(agent.runtime_predictor(state, deterministic).mean.detach().cpu().item())
        recons.append(agent.decoder(state, deterministic).mean.detach().cpu().numpy())
        rewards.append(agent.reward_predictor(state, deterministic).mean.detach().cpu().item())

    return acs, rewal_runtimes, runtimes, real_obses, recons, real_rewards, rewards


def main(config_file):
    config = load_config(config_file)

    # if config.environment.benchmark == "atari":
    #     env = make_atari_env(
    #         task_name=config.environment.task_name,
    #         seed=config.environment.seed,
    #         height=config.environment.height,
    #         width=config.environment.width,
    #         skip_frame=config.environment.frame_skip,
    #     )
    # elif config.environment.benchmark == "dmc":
    #     env = make_dmc_env(
    #         domain_name=config.environment.domain_name,
    #         task_name=config.environment.task_name,
    #         seed=config.environment.seed,
    #         visualize_reward=config.environment.visualize_reward,
    #         from_pixels=config.environment.from_pixels,
    #         height=config.environment.height,
    #         width=config.environment.width,
    #         frame_skip=config.environment.frame_skip,
    #     )
    env = compiler_gym.make(
        "llvm-v0",
        benchmark="benchmark://cbench-v1/qsort",
        observation_space="Autophase",
        # reward_space="IrInstructionCountOz",
    )
    env = RuntimePointEstimateReward(env)
    # env = RandomOrderBenchmarks(env, env.datasets[
    #     "benchmark://cbench-v1"].benchmarks())
    env = TimeLimit(env, max_episode_steps=1024)
    # env = CommandlineWithTerminalAction(env)
    env = LogNormalizer(env)
    obs_shape, discrete_action_bool, action_size = get_env_infos(env)

    log_dir = (
            get_base_directory()
            + "/runs/"
            + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + config.operation.log_dir
    )
    writer = SummaryWriter(log_dir)
    device = config.operation.device

    if config.algorithm == "dreamer-v1":
        agent = Dreamer(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
    elif config.algorithm == "plan2explore":
        agent = Plan2Explore(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
    agent.train(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="compiler-gym.yml",
        help="config file to run(default: dmc-walker-walk.yml)",
    )
    main(parser.parse_args().config)
