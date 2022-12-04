import torch
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import CommandlineWithTerminalAction, TimeLimit

from PPO import ActorCritic
from train import to_pyg
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy


def policy(env):
    #  __________________
    # < Your agent here! >
    #  ------------------
    #         \   ^__^
    #          \  (oo)\_______
    #             (__)\       )\/\
    #                 ||----w |
    #                 ||     ||
    #
    # Run your agent here, interacting with the input env. Once the agent
    # completed, return. Final reward is calculated based on the env state after
    # this function has returned.
    env = CommandlineWithTerminalAction(env)
    env = TimeLimit(env, env.action_space.n)
    env.observation_space = 'Programl'
    action_dim = env.action_space.n
    agent = ActorCritic(action_dim=action_dim)
    print('loading')
    agent.load_state_dict(
        torch.load(
            '/home/anthony/PPO-PyTorch/PPO_preTrained/llvm-ic-v0/model.pth',
            'cpu',
        )
    )
    print('loaded')
    # agent = agent.to('cuda:3')
    done = False
    obs = env.reset()
    obs = to_pyg(obs)  # .to('cuda:3')
    while not done:
        ac, _ = agent.act(obs)
        obs, rew, done, info = env.step(ac.item())
        obs = to_pyg(obs)  # .to('cuda:3')


eval_llvm_instcount_policy(policy)
