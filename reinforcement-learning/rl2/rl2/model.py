from copy import deepcopy

import compiler_gym
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from compopt.constants import VOCAB
from compopt.encoders import GNN, Encoder, NodeEncoder, EdgeEncoder
from rl2.algorithms import ppo_v2, gae
from rl2.buffer import DequeBuffer

from rl2.interfaces import OnPolicy
from rl2.utils import _t, standardize

from compopt.wrappers import PygWrapper


class Agent(OnPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self):
        # buffer = self.buffer[:-1]  # omit last value
        # obs_arr = np.array(self.buffer.obs, dtype=object)
        obs_arr = list(self.buffer.obs)
        reward_arr = _t(self.buffer.reward)
        done_arr = _t(self.buffer.done)
        value_arr = _t(self.buffer.value)
        log_prob_arr = _t(self.buffer.log_prob)

        advs = self.estimator(
            reward_arr[:-1],  # 0~T-1
            value_arr,  # 0~T
            done_arr  # 0~T
        )
        advs = _t(advs)
        returns = _t(value_arr[:-1]) + advs
        advs = standardize(advs)
        # num_batches = 4
        # idxes = np.arange(0, len(buffer))  # misleading?
        # np.random.shuffle(idxes)
        # idxes = np.split(idxes, num_batches)

        # data_set = IterableDataset(buffer)
        # data_loader = DataLoader(data_set, pin_memory=True)
        # for batch in data_loader

        # for local_epoch in range(self.local_epochs):
        #     for batch_idx in idxes:
        # batch = buffer[batch_idx].as_tensor_dict(device=self.device)
        # inference
        ac_dist, batch_values, = self(obs_arr[:-1])

        # batch_returns = returns[batch_idx]

        acs = ac_dist.sample()
        log_probs = ac_dist.log_prob(acs).to(self.device)
        batch_ratios = torch.exp(log_probs - _t(log_prob_arr[:-1]))
        # batch_advs = advs[batch_idx]
        batch_entropy = ac_dist.entropy().mean()

        # loss

        loss, subloss_dict = self.loss_func(
            batch_ratios, advs,
            batch_values, returns,
            batch_entropy,
        )

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        loss_dict = {
            'loss': loss.item(),
            **subloss_dict,
        }
        return loss_dict


def get_agent(action_n, buffer_size):
    ne = NodeEncoder(len(VOCAB))
    ee = EdgeEncoder()
    encoder = Encoder(
        node_encoder=ne,
        edge_encoder=ee,
        gnn=GNN(ne.out_dim, ee.out_dim)
    )

    actor = nn.Sequential(
        encoder,
        nn.LazyLinear(action_n)
    )
    critic = nn.Sequential(
        deepcopy(encoder),
        nn.LazyLinear(1)
    )

    lr = 1e-4
    optim = Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr
    )
    buffer = DequeBuffer(buffer_size)
    agent = Agent(
        actor=actor,
        critic=critic,
        buffer=buffer,
        loss_func=ppo_v2,
        optim=optim,
        estimator=gae
    )  # .to(device)

    return agent