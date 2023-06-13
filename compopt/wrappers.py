import pickle

import networkx as nx
import numpy as np
import ray
import torch_geometric as pyg
import compiler_gym
from compiler_gym.datasets import Datasets
from compiler_gym.wrappers import CompilerEnvWrapper, \
    CommandlineWithTerminalAction, TimeLimit
from gym import Env
from gym import Wrapper
from gym.spaces import Box
from ray import ObjectRef

from compopt.constants import (
    NODE_FEATURES,
    EDGE_FEATURES,
    MAX_TEXT, MAX_TYPE,
    MAX_FLOW, MAX_POS, RUNNABLE_BMS
)
from compopt.nx_utils import parse_nodes, parse_edges
from compopt.spaces import compute_graph_space

MAX_NODES = int(1e4)
MAX_EDGES = int(1e4)
DATA = [
    'cbench-v1',
    'mibench-v1',
    'blas-v0',
    'npb-v0'
]


class RunnableWrapper(CompilerEnvWrapper):
    def __init__(self, env):
        super(RunnableWrapper, self).__init__(env)
        self.i = 0
        # self.observation_space.sample = _sample()

    def reset(self):
        # only reset for runnable benchmarks
        bm = RUNNABLE_BMS[self.i % len(RUNNABLE_BMS)]
        obs = self.env.reset(benchmark=bm)
        self.i += 1

        return obs


def make_env(config):
    # TODO: what if env object is given?
    env = compiler_gym.make(
        "llvm-ic-v0",
        observation_space="Programl",
        reward_space="IrInstructionCountOz",
    )
    # env = TimeLimit(env, 128)
    env = CommandlineWithTerminalAction(env)
    env = TimeLimit(env, 1024)  # TODO: rough limit
    # TODO: conflicts with rllib's evaluator
    # env = CycleOverBenchmarks(
    #     env, dataset.benchmarks()
    # )
    env = RllibWrapper(env, DATA)

    return env


class ActionHistogram(CompilerEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space._shape = (
            self.observation_space.shape[0] + env.action_space.n,
        )

    def reset(self):
        obs = self.env.reset()
        self.histogram = np.zeros(self.env.action_space.n, dtype=int)
        obs = np.concatenate([obs, self.histogram])
        return obs

    def step(self, action):
        self.histogram[action] += 1
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate([obs, self.histogram])

        return obs, reward, done, info


class LogNormalizer(CompilerEnvWrapper):
    """
    Autophase Lognormzlier
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space.low = np.full_like(
            self.observation_space.low,
            -9223372036854775807
        )
        self.observation_space.dtype = np.dtype(np.float32)

    def reset(self):
        obs = self.env.reset()
        return np.log(obs + 1e-8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.log(obs + 1e-8), reward, done, info


class RunnableWrapper(CompilerEnvWrapper):
    def __init__(self, env):
        super(RunnableWrapper, self).__init__(env)
        self.i = 0
        # self.observation_space.sample = _sample()

    def reset(self):
        # only reset for runnable benchmarks
        bm = RUNNABLE_BMS[self.i % len(RUNNABLE_BMS)]
        obs = self.env.reset(benchmark=bm)
        self.i += 1

        return obs


class PygWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # TODO: add patience
        # TODO: obs space doesn't match with actual obs

    def _parse_graph(self, g):
        # TODO: want to avoid for loop
        parse_nodes(g.nodes)
        parse_edges(g.edges)

        g = nx.DiGraph(g)
        g = pyg.utils.from_networkx(
            g,
            group_node_attrs=NODE_FEATURES,
            group_edge_attrs=EDGE_FEATURES
        )

        return g

    def step(self, ac):
        obs, rew, done, info = self.env.step(ac)
        obs = self._parse_graph(obs)

        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        obs = self._parse_graph(obs)

        return obs


def parse_graph(g):
    # TODO: want to avoid for loop
    x = parse_nodes(g.nodes, return_attr=True)
    edge_attr = parse_edges(g.edges, return_attr=True)

    g = nx.DiGraph(g)
    edge_index = list(g.edges)

    return x, edge_index, edge_attr


def compute_space():
    return Tuple([
        _Repeated(
            Box(
                low=np.array([0, 0]),
                high=np.array([MAX_TEXT, MAX_TYPE]),
                shape=(2,),
                dtype=int
            ),
            max_len=MAX_NODES * 2,  # TODO: find exact bound
        ),
        _Repeated(
            Box(
                low=np.array([0, 0]),
                high=np.array([MAX_NODES, MAX_NODES]),
                # high=np.array([num_nodes - 1, num_nodes - 1]),
                shape=(2,),
                dtype=int
            ),
            max_len=MAX_EDGES * 2,  # TODO: find exact bound
        ),
        _Repeated(
            Box(
                low=np.array([0, 0]),
                high=np.array([MAX_FLOW, MAX_POS]),
                shape=(2,),
                dtype=int
            ),
            max_len=MAX_EDGES * 2,  # TODO: find exact bound
        )
    ])


class RllibWrapper(Wrapper):
    def __init__(
            self,
            env,
            dataset_ids=None
    ):
        super().__init__(env)
        if dataset_ids:
            datasets = [env.datasets[i] for i in dataset_ids]
            self.env.datasets = Datasets(datasets)
        self.observation_space = compute_graph_space()

    def step(self, ac):
        obs, rew, done, info = self.env.step(ac)
        obs = parse_graph(obs)

        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        # self.observation_space = self._compute_space()  # if not child of gym.Space, setter may not work
        # TODO: can fail
        for i in range(128):
            self.env.benchmark = self.env.datasets.random_benchmark(
                weighted=True
            )

            obs = self.env.reset(*args, **kwargs)
            # x, edge_index, edge_attr = parse_graph(obs)
            # num_nodes, num_edges = len(x), len(edge_attr)
            # self.env.observation_space = compute_space(num_nodes, num_edges)

            if obs.number_of_nodes() > MAX_NODES or \
                    obs.number_of_edges() > MAX_EDGES:
                continue
            else:
                obs = parse_graph(obs)

                return obs


class ObjRefSpace(Box):
    def __init__(self, dummy_obs):
        super().__init__(shape=(28,), low=0, high=255, dtype=np.uint8)

        dummy_obs = parse_graph(dummy_obs)
        obj_ref = ray.put(dummy_obs)
        self.dummy = ObjectRefWrapper.encrypt(obj_ref)

    def sample(self):
        return self.dummy

    def contains(self, x):
        return True


class ObjectRefWrapper(Wrapper):
    def __init__(self, env: Env) -> None:
        if not ray.is_initialized():
            raise RuntimeError(
                'Ray must be initialized before using RllibWrapper'
            )

        super().__init__(env)
        # Repeated Values too inefficient
        # So we bypass by storing object into ObjectStore
        obs = env.reset()
        self.observation_space = ObjRefSpace(obs)

        # self.flag = True

        # TODO: check if ray server is up

    @staticmethod
    def encrypt(obj_ref: ray.ObjectRef) -> np.ndarray:
        # obj_ref.binary() == b'\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\x00\x00\x00\x03'

        arr = np.array([i for i in obj_ref.binary()], dtype=np.uint8)

        return arr

    @staticmethod
    def decrypt(arr: np.ndarray) -> ObjectRef:
        # obj = ObjectRef(arr.tobytes())
        obj = ObjectRef(arr.tobytes())

        return obj

    def _process_obs(self, obs):
        obs = parse_graph(obs)

        # e.g. ObjectRef(00ffffffffffffffffffffffffffffffffffffff0100000003000000)
        obj_ref = ray.put(obs)

        arr = ObjectRefWrapper.encrypt(obj_ref)

        return arr

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        return self._process_obs(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self._process_obs(obs)

        return obs, rew, done, info


class PickleWrapper(Wrapper):
    def __init__(self, env):
        super(PickleWrapper, self).__init__(env)
        self.max_pkl_len = 1000000
        self.env.observation_space._shape = (self.max_pkl_len,)

    def pad(self, obs):
        obs = np.array(obs)
        original_len = len(obs)
        obs = np.pad(obs, (0, self.max_pkl_len - len(obs)))
        obs[-1] = original_len

        return obs

    def reset(self):
        obs = self.env.reset()
        obs = pickle.dumps(obs)
        obs = self.pad(obs)

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = pickle.dumps(obs)
        obs = self.pad(obs)

        return obs, rew, done, info
