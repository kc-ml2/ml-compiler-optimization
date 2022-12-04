from typing import OrderedDict

import numpy as np
import torch


def _infer_dtype(example):
    l = []
    for k, v in example.items():
        if hasattr(v, 'dtype'):
            _dtype = v.dtype
        else:
            _dtype = type(v)
            if _dtype == bool:
                _dtype = np.uint8

        if hasattr(v, 'shape'):
            _shape = v.shape
        else:
            # consider it's scalar, edge case?
            _shape = ()

        l.append((k, _dtype, _shape))

    return l


class NumpyBuffer(np.ndarray):
    # (s, r, d) -> env -> a pair
    default_keys = ['obs', 'action', 'reward', 'done']

    """
       implements circular queue ADT
       pre-allocates memory
       overwrite from 0 index when curr_index hits maxsize

       must provide example
       example = OrderedDict({
           'obs': env.observation_space.sample(),
           'action': env.action_space.sample(),
           'reward': 0.,
           'done': False,
       })
       dtype is implicitly determined via example,
       so example must be in correct shape, dtype

       optionally extra data structure
       extras = OrderedDict({
           'value': np.float32,
           'log_prob': np.float32,
       })
       """

    def __new__(
            cls,
            num_transitions: int,
            num_envs=1,
            num_agents=1,
            spec: OrderedDict = None,
            example: OrderedDict = None,
            extras: OrderedDict = {},
            seed=42,
            overwrite=True,
            *args,
            **kwargs
    ):
        assert spec or example, \
            'you must provide one of following arguments. spec, example'

        assert all([k in example.keys() for k in NumpyBuffer.default_keys])

        # type check too strong?
        # assert isinstance(example['obs'], np.ndarray)
        # or example['obs'].__class__.__module__ == 'builtins'

        dtype_list = _infer_dtype(example)
        extra_dtype_list = list(extras.items())

        dtype = np.dtype(dtype_list + extra_dtype_list)

        # add 1 for last value
        shape = (num_transitions + 1,)
        if num_envs > 1:
            shape = (*shape, num_envs)
        if num_agents > 1:
            shape = (*shape, num_agents)

        obj = super().__new__(cls, shape=shape, dtype=dtype)
        obj.seed = seed
        obj.overwrite = overwrite
        obj.num_transitions = num_transitions
        obj.next_idx = 0

        return obj

    def __array_finalize__(self, obj):
        pass

    def __init__(self, *args, **kwargs):
        self.full_action = self.reset if self.overwrite else self._raise_error

    def _raise_error(self):
        raise RuntimeError('buffer full! call reset explicitly')

    def reset(self):
        # reset will contain last transition from previous truncated episode
        # self[0] = self[-1]  # .copy()
        self.next_idx = 0

    def is_full(self):
        return self.next_idx == len(self)

    def _set(self, idx, transition):
        for k, v in transition.items():
            self[k][idx] = v

    def append(self, **kwargs):
        if self.is_full():
            self.full_action()

        self._set(self.next_idx, kwargs)
        self.next_idx += 1

    @property
    def all_keys(self):
        return list(self.dtype.fields.keys())

    def _as_dict(self, cls_callback):
        d = {}
        for k in self.all_keys:
            d[k] = cls_callback(self[k])
        return d

    def as_dict(self):
        callback = np.array

        return self._as_dict(callback)

    def as_tensor_dict(self, device='cpu'):

        return self._as_dict(
            lambda x: torch.from_numpy(np.array(x)).to(device)
        )

    def get_tensor(self, key, device='cpu'):

        return torch.from_numpy(np.array(self[key])).to(device)
