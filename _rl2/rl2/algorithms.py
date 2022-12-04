import torch
from torch.nn import functional as F

from rl2.utils import _dynamic_zeros_like


def gae(rews, vals, dones, gam=0.99, lamb=0.95):
    # rews : 0~t, vals, dones : 0~t+1
    advs = _dynamic_zeros_like(vals)
    masks = 1. - dones
    for i in reversed(range(len(vals) - 1)):
        delta = -vals[i] + (rews[i] + masks[i] * (gam * vals[i + 1]))
        advs[i] = delta + masks[i] * (gam * lamb * advs[i + 1])

    return advs[:-1]


def surrogate_loss(ratios, advs, clip_param=0.1):
    clipped_ratios = torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
    policy_gradients = torch.min(clipped_ratios * advs, ratios * advs)

    return policy_gradients.mean()


def ppo_v2(
        ratios, advs,
        values, returns,
        entropy,
        vf_coef=0.5,
        ent_coef=0.005,
):
    actor_loss = surrogate_loss(ratios, advs) + ent_coef * entropy
    critic_loss = vf_coef * F.smooth_l1_loss(values.to('cuda:0'), returns.to('cuda:0'))
    utility = actor_loss - critic_loss
    loss = -utility

    # TODO: rather than return, report this to context
    loss_info = {
        'loss': loss.item(),
        'actor_loss': actor_loss.item(),
        'critic_loss': critic_loss.item(),
        'entropy': entropy.item()
    }
    print(loss_info)
    return loss, loss_info
