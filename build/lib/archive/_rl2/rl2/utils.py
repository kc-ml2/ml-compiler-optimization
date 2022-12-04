import random

import numpy as np
import scipy
import torch
from rl2.ctx import ctx_device

_device = ctx_device.get()


def generator(buffer, num_batch=1):
    chunks = np.split(buffer, num_batch)

    for chunk in chunks:
        yield chunk


def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def discounted_cumsum(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and
    # advantage estimates
    return scipy.signal.lfilter(
        [1],
        [1, float(-discount)],
        x[::-1],
        axis=0
    )[::-1]


def manual_seed(env, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)


def _avg(l):
    return sum(l) / len(l)


def _t(x, device=_device, cls=torch.FloatTensor):
    return cls(np.asarray(x, order='C')).to(device)


def _dynamic_zeros_like(x):
    if isinstance(x, np.ndarray):
        return np.zeros_like(x)
    elif isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    else:
        raise RuntimeError('Unknown type')
