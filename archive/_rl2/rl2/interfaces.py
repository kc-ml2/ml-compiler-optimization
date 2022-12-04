import numpy as np
import torch
import wandb
from torch import nn
from torch.distributions import Categorical

from rl2.utils import _t, standardize

from rl2.ctx import ctx_device


class Agent(nn.Module):
    def __init__(self):
        super().__init__()

    def act(self):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError


class OnPolicy(Agent):
    def __init__(
            self,
            actor,
            critic,
            loss_func,
            estimator,
            optim,
            buffer,
            lr=5e-4,
            local_epochs=4,
    ):
        super().__init__()
        self.buffer = buffer
        self.device = ctx_device.get()

        self.lr = lr
        self.local_epochs = local_epochs
        self.estimator = estimator

        self.actor = actor
        self.critic = critic
        self.loss_func = loss_func
        self.optim = optim

        self.log_prob = 0.
        self.value = 0.

    def forward(self, x):
        action_dist = self._policy(x)
        value_logits = self._value(x)

        return action_dist, value_logits

    def act(self, obs):
        with torch.no_grad():
            action_dist = self._policy(obs)
            self.value = self._value(obs).cpu().numpy()

        ac = action_dist.sample()
        self.log_prob = action_dist.log_prob(ac).cpu().numpy()

        return ac.item()

    def _value(self, obs):
        # if not isinstance(obs, torch.Tensor):
        #     obs = _t(obs)
        logits = self.critic(obs).squeeze()

        return logits

    def _policy(self, obs):
        # if not isinstance(obs, torch.Tensor):
        #     obs = _t(obs)
        logits = self.actor(obs)
        dist = Categorical(logits=logits)

        return dist

    def observe(self, obs, action, reward, done):
        if done:
            self.log_prob = 0.
            self.value = 0.

        self.buffer.append(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            log_prob=self.log_prob,
            value=self.value,
        )

    def learn(self):
        buffer = self.buffer[:-1]  # omit last value
        advs = self.estimator(
            buffer['reward'],  # 0~T-1
            self.buffer['value'],  # 0~T
            self.buffer['done']  # 0~T
        )
        advs = _t(advs)
        returns = _t(buffer['value']) + advs
        advs = standardize(advs)
        num_batches = 4
        idxes = np.arange(0, len(buffer))  # misleading?
        np.random.shuffle(idxes)
        idxes = np.split(idxes, num_batches)

        # data_set = IterableDataset(buffer)
        # data_loader = DataLoader(data_set, pin_memory=True)
        # for batch in data_loader

        for local_epoch in range(self.local_epochs):
            for batch_idx in idxes:
                batch = buffer[batch_idx].as_tensor_dict(device=self.device)
                # inference
                ac_dist, batch_values = self(batch['obs'])
                batch_returns = returns[batch_idx]

                acs = ac_dist.sample()
                log_probs = ac_dist.log_prob(acs).to(self.device)

                batch_ratios = torch.exp(log_probs - batch['log_prob'])
                batch_advs = advs[batch_idx]
                batch_entropy = ac_dist.entropy().mean()

                # loss

                loss, subloss_dict = self.loss_func(
                    batch_ratios, batch_advs,
                    batch_values, batch_returns,
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



class OffPolicy(Agent):
    def __init__(
            self,
            behavior_net,
            target_net,
            loss_func,
    ):
        self.loss_func = loss_func
        self.behavior_net = behavior_net
        self.target_net = target_net
        self.device = ctx_device.get()

    def forward(self, x):
        pass

    def act(self, obs):
        with torch.no_grad():
            ac = self._policy(obs)

        return ac

    def _policy(self, obs):
        ac = self.behavior_net(obs).argmax(-1)

        return ac
