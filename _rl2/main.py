import uuid
from collections import OrderedDict

from torch import nn
from torch.optim import Adam

from rl2.algorithms import ppo_v2, gae
from rl2.ctx import ctx_device
import gym
import numpy as np
import torch
import wandb
from gym import wrappers
from torch.utils.tensorboard import SummaryWriter

from rl2.buffer import NumpyBuffer
# from pettingzoo import magent
from rl2.debug import _p
from rl2.interfaces import OnPolicy
from rl2.workers.gym_worker import GymRolloutWorker

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ctx_device.set(device)

num_actions = None

max_length = 0.
max_return = 0.

DEBUG = True
WANDB = False
WANDB = True
WANDB_MODE = 'online' if WANDB else 'offline'


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)


if __name__ == '__main__':
    log_dir = '/tmp/' + str(uuid.uuid4())
    summary_writer = SummaryWriter(log_dir)
    wandb.init(project="ppo-v2", mode=WANDB_MODE)
    env = gym.make('CartPole-v1')
    # env = gym.make('Acrobot-v1')
    # env = gym.make("MountainCar-v0")
    # env = gym.make("ALE/Gravitar-v5")
    # env = gym.make("Taxi-v3") failed
    # env = gym.make("FrozenLake-v1") failed
    env = wrappers.Monitor(env, log_dir, video_callable=False)
    eg = OrderedDict({
        'obs': env.observation_space.sample(),
        'action': env.action_space.sample(),
        'reward': 0.,
        'done': False,
    })
    extras = OrderedDict({
        'value': np.float32,
        'log_prob': np.float32,
    })

    train_steps = int(1e8)
    train_interval = 128
    log_interval = 128
    lr = 1e-4

    buffer = NumpyBuffer(
        num_transitions=train_interval,
        example=eg,
        extras=extras
    )
    in_features = env.observation_space.shape[0]
    num_actions = env.action_space.n

    actor = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, num_actions),
        nn.Softmax(dim=-1)
    ).to(device)
    critic = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 1)
    ).to(device)

    actor.apply(_init_weights)
    critic.apply(_init_weights)
    optim = Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr
    )

    agent = OnPolicy(actor, critic, ppo_v2, gae, optim, buffer, device)

    # _p(env.__repr__())
    # _p(env.spec)
    # _p(env.observation_space)
    # _p(env.action_space)
    # _p(f'example transition\n{eg}')
    # _p(f'buffer spec\n{buffer.dtype}')
    # _p(agent)
    # summary_writer.add_graph(agent.actor, torch.FloatTensor(eg['obs']).to(
    # 'cuda'))
    # summary_writer.add_graph(agent.critic, torch.FloatTensor(eg['obs']).to(
    # 'cuda'))

    # num_epochs = 1000
    worker = GymRolloutWorker(env=env, agent=agent)
    # for epoch in range(num_epochs):
    while env.episode_id < 1000:
        info = worker.run(steps=train_interval)
        # agent.observe(*worker.last_transition())
        loss_dict = agent.learn()
        agent.buffer.reset()

        if DEBUG:
            _p(env.episode_id)
            _p(loss_dict)
            _p(info)
        wandb.log(loss_dict)
        wandb.log(info)
