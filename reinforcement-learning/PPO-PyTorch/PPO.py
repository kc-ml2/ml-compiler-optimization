from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

################################## set device ##################################
print(
    "============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print(
    "============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


import numpy as np
import torch
from torch import nn
import torch_geometric.nn as pygnn


class NodeEncoder(nn.Module):
    def __init__(
            self,
            text_in,
            text_out=32,
            type_in=3,
            type_out=32,
    ):
        super(NodeEncoder, self).__init__()
        # must add 1 for OOV
        self.text_embedding = nn.Embedding(text_in + 1, text_out)
        self.type_embedding = nn.Embedding(type_in, type_out)
        self.out_dim = text_out + type_out  # concat

    def forward(
            self,
            x  # [B, num_nodes, channels]
    ) -> torch.Tensor:
        text, type_ = x[..., 0], x[..., 1]

        text = self.text_embedding(text)
        type_ = self.type_embedding(type_)

        x = torch.cat([text, type_], axis=-1)

        return x


class PositionEmbedding(nn.Module):
    def __init__(self, max_pos=5120, out_dim=64):
        # max_pos = 4096 https://github.com/ChrisCummins/phd/blob/aab7f16bd1f3546f81e349fc6e2325fb17beb851/programl/models/ggnn/messaging_layer.py#L38
        # increase embedding dim to 5120 just in case
        super().__init__()
        self.max_pos = max_pos
        in_dim = max_pos + 1 # add 1 for OOV
        positions = torch.arange(0, in_dim)
        inv_freq = 1 / (10000 ** (torch.arange(0.0, out_dim, 2.0) / out_dim))
        sinusoid_inp = torch.outer(positions, inv_freq)

        pos_emb = torch.zeros((in_dim, out_dim))
        evens = np.arange(0, int(out_dim), 2)
        odds = evens + 1
        pos_emb[:, odds] = torch.sin(sinusoid_inp)
        pos_emb[:, evens] = torch.cos(sinusoid_inp)
        self.pos_emb = pos_emb
        self.pos_emb.requires_grad = False

        self.f = nn.Linear(out_dim, out_dim)

    def forward(
            self,
            positions: torch.LongTensor,  # [B, num_edges]
    ):
        # replace OOV
        positions = positions.where(
            positions > self.max_pos,
            torch.LongTensor([self.max_pos]).to(positions.device)
        )

        embeds = torch.stack(
            [self.pos_emb[p] for p in positions]
        ).to(positions.device)
        x = self.f(embeds)

        return x


class EdgeEncoder(nn.Module):
    def __init__(self, flow_dim=3, out_dim=64):
        super(EdgeEncoder, self).__init__()

        self.flow_embedding = nn.Embedding(flow_dim, out_dim)
        self.pos_embedding = PositionEmbedding(out_dim=out_dim)
        self.out_dim = out_dim  # summed

    def forward(
            self,
            x  # [B, num_edges, channels]
    ) -> torch.Tensor:
        flow, pos = x[..., 0], x[..., 1]

        flow = self.flow_embedding(flow)
        pos = self.pos_embedding(pos)
        x = flow + pos

        return x


class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim=32):
        super().__init__()
        self.c1 = pygnn.GATv2Conv(node_dim, 256, edge_dim=edge_dim)
        self.c2 = pygnn.GATv2Conv(256, out_dim, edge_dim=edge_dim)
        # self.c2 = pygnn.GATv2Conv(256, 256, edge_dim=edge_dim)
        # self.c3 = pygnn.GATv2Conv(256, out_dim, edge_dim=edge_dim)
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_attr):
        x = self.c1(x, edge_index, edge_attr)
        x = self.c2(x, edge_index, edge_attr)
        # x = self.c2(x, edge_index, edge_attr)
        # [num_nodes, num_features]
        x = x.mean(0)

        return x


class Encoder(nn.Module):
    def __init__(
            self,
            node_encoder,
            edge_encoder,
            gnn,
    ):
        super().__init__()
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.gnn = gnn

    def forward(
            self,
            x,  # [B, num_nodes, num_features]
            edge_index,  # [B, 2, num_edges]
            edge_attr  # [B, num_edges, num_features]
    ):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # TODO: parallel execution
        x = self.gnn(x.float(), edge_index.long(), edge_attr.float())

        return x


import json
import os
from pathlib import Path

_dir = os.path.join(Path(__file__).parent, 'vocab.json')
with open(_dir, 'r') as fp:
    vocab = json.load(fp)


class Actor(nn.Module):
    def __init__(self, action_dim, encoder):
        super().__init__()
        self.encoder = encoder
        self.policy = nn.LazyLinear(action_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.encoder(x, edge_index, edge_attr)
        x = self.policy(x)
        x = F.softmax(x, -1)

        return x


class Critic(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.value = nn.LazyLinear(1)

    def forward(self, x, edge_index, edge_attr):
        x = self.encoder(x, edge_index, edge_attr)
        x = self.value(x)

        return x


class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()
        self.node_encoder = NodeEncoder(
            text_in=len(vocab),
            text_out=64,
            type_in=3,
            type_out=64
        )
        # node_output : text_out + type_out
        self.edge_encoder = EdgeEncoder(
            flow_dim=3,
            out_dim=128
        )
        # edge_output : out_dim
        self.gnn = GNN(
            node_dim=self.node_encoder.out_dim,
            edge_dim=self.edge_encoder.out_dim,
        )

        self.encoder = Encoder(
            node_encoder=self.node_encoder,
            edge_encoder=self.edge_encoder,
            gnn=self.gnn
        )

        self.actor = Actor(action_dim, self.encoder)

        self.critic = Critic(deepcopy(self.encoder))

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(
            state.x,
            state.edge_index,
            state.edge_attr,
        )
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, states, actions):
        action_logprobs, dist_entropies, state_values = [], [], []
        for state, action in zip(states, actions):
            action_prob = self.actor(
                state.x,
                state.edge_index,
                state.edge_attr,
            )
            dist = Categorical(action_prob)
            action_logprob = dist.log_prob(action)
            action_logprobs.append(action_logprob)
            dist_entropies.append(dist.entropy())
            state_value = self.critic(
                state.x,
                state.edge_index,
                state.edge_attr,
            )
            state_values.append(state_value)

        action_logprobs = torch.stack(action_logprobs)
        dist_entropies = torch.stack(dist_entropies)
        state_values = torch.stack(state_values)

        return action_logprobs, state_values, dist_entropies


class PPO:
    def __init__(self, action_dim, lr_actor, lr_critic, gamma, K_epochs,
                 eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.Loss = nn.SmoothL1Loss()

    def select_action(self, state):
        with torch.no_grad():
            state = state.to(device)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    # def gae(self, rews, vals, dones, gam=0.99, lamb=0.95):
    #     # rews : 0~t, vals, dones : 0~t+1
    #     advs = np.zeros_like(vals)
    #     masks = 1. - dones
    #     for i in reversed(range(len(vals) - 1)):
    #         delta = -vals[i] + (rews[i] + masks[i] * (gam * vals[i + 1]))
    #         advs[i] = delta + masks[i] * (gam * lamb * advs[i + 1])
    #
    #     return advs[:-1]

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
                reversed(self.buffer.rewards),
                reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = [s.detach().to(device) for s in self.buffer.states]
        # old_states = torch.squeeze(
        #     torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(
            torch.stack(self.buffer.actions, dim=0)
        ).detach().to(device)
        old_logprobs = torch.squeeze(
            torch.stack(self.buffer.logprobs, dim=0)
        ).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(
                ratios,
                1 - self.eps_clip,
                1 + self.eps_clip
            ) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + \
                   0.5 * self.Loss(state_values, rewards) - \
                   0.005 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=lambda storage, loc: storage
            )
        )
        self.policy.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=lambda storage,
                loc: storage
            )
        )
