import json
import os
from pathlib import Path

import compiler_gym
import torch
from torch import nn

import wandb

from experiments import get_agent
from experiments import _Wrapper
from rl2.workers.base_worker import RolloutWorker

vocab_dir = os.path.join(Path.cwd(), '../../vocab.json')
with  open(vocab_dir, 'r') as fp:
    vocab = json.load(fp)
DEBUG = True
WANDB = False
WANDB_MODE = 'online' if WANDB else 'offline'
# WANDB = True
wandb.init(project="comp-opt", mode=WANDB_MODE, sync_tensorboard=True)


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, *x):
        for i in x:
            print(i.aaa)
        return x


with torch.cuda.device(0):
    env = compiler_gym.make(
        "llvm-ic-v0",
        observation_space="Programl",
        reward_space="IrInstructionCountOz",
    )
    env = _Wrapper(env)
    # dataset = env.datasets["benchmark://cbench-v1"]
    # env = CycleOverBenchmarks(
    #     env, dataset.benchmarks()
    # )

    agent = get_agent(
        vocab=vocab,
        action_n=env.action_space.n,
        buffer_size=8
    ).to('cuda:0')
    dataset = env.datasets["benchmark://cbench-v1"]

    for bm in dataset.benchmarks():
        print(f'training {bm}')
        obs = env.reset(benchmark=bm)
        initial_transition = [obs, agent.act([obs]), 0., False]

        worker = RolloutWorker(
            env=env,
            agent=agent,
            initial_transition=initial_transition
        )
        worker.set_train(train_interval=8, save_interval=-1, log_interval=8)
        try:
            worker.run(n=128)
        except Exception as e:
            print(e)
        finally:
            torch.save(agent.state_dict(), f'model{worker.global_steps}.pt')


    # agent = get_agent()
    # agent.load_state_dict(torch.load(f'model127.pt'))

    def policy(env):
        obs = env.observation['Programl']
        for i in range(16):
            ac = agent.act([obs])
            env.step(ac)
            obs = env.observation['Programl']


    # agent.load_state_dict(torch.load('model.pt'))
    from compiler_gym.leaderboard import llvm_instcount

    llvm_instcount.eval_llvm_instcount_policy(policy)
