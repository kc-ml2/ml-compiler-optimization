import numpy as np

import gym
from compiler_gym.spaces import Sequence
from gym import Wrapper
from gym.spaces import Box, Tuple

import compiler_gym
from compiler_gym.datasets import Datasets
from compiler_gym.wrappers import CommandlineWithTerminalAction, TimeLimit, CompilerEnvWrapper

import networkx as nx
import walker
import torch_geometric as pyg

from ray.rllib.utils.spaces.repeated import Repeated

from compopt.constants import (
    VOCAB, NODE_FEATURES,
    EDGE_FEATURES,
    MAX_TEXT, MAX_TYPE,
    MAX_FLOW, MAX_POS
)

MAX_NODES = int(1e4)
MAX_EDGES = int(5e4)
DATA = [
    'cbench-v1',
    'mibench-v1',
    'blas-v0',
    'npb-v0'
]


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


class Dynamic(Repeated):
    def __init__(self, child_space: gym.Space, max_len):
        super(Dynamic, self).__init__(child_space=child_space, max_len=max_len)

    def sample(self):
        # trick to bypass rllib ModelV2 init
        return [
            self.child_space.sample()
            for _ in range(32)
        ]

    def contains(self, x):
        return True


def parse_nodes(ns, return_attr=False):
    # in-place
    x = []
    for nid in ns:
        n = ns[nid]
        n.pop('function', None)
        n.pop('block', None)
        n.pop('features', None)
        n['text'] = VOCAB.get(n['text'], MAX_TEXT)

        if return_attr:
            x.append(np.array([n['text'], n['type']]))

    return x


def parse_edges(es, return_attr=False):
    # in-place
    x = []
    for eid in es:
        e = es[eid]
        e['position'] = min(e['position'], MAX_POS)

        if return_attr:
            x.append(np.array([e['flow'], e['position']]))

    return x


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


def compute_space(num_nodes=MAX_NODES, num_edges=MAX_EDGES):
    return Tuple([
        Dynamic(
            Box(
                low=np.array([0, 0]),
                high=np.array([MAX_TEXT, MAX_TYPE]),
                shape=(2,),
                dtype=int
            ),
            max_len=num_nodes * 5,  # TODO: find exact bound
        ),
        Dynamic(
            Box(
                low=np.array([0, 0]),
                high=np.array([num_nodes, num_nodes]),
                # high=np.array([num_nodes - 1, num_nodes - 1]),
                shape=(2,),
                dtype=int
            ),
            max_len=num_edges * 5,  # TODO: find exact bound
        ),
        Dynamic(
            Box(
                low=np.array([0, 0]),
                high=np.array([MAX_FLOW, MAX_POS]),
                shape=(2,),
                dtype=int
            ),
            max_len=num_edges * 5,  # TODO: find exact bound
        )
    ])


class RllibWrapper(Wrapper):
    def __init__(
            self,
            env,
            dataset_ids=None
    ):
        self.env = env

        # if dataset_ids:
        #     datasets = [env.datasets[i] for i in dataset_ids]
        #     self.env.datasets = Datasets(datasets)
        # obs_space = compute_space()
        # setattr(obs_space, 'name', 'repeated')
        # self.env.observation.spaces['repeated'] = obs_space
        self.observation_space = compute_space()

    def step(self, ac):
        obs, rew, done, info = self.env.step(ac)
        obs = parse_graph(obs)

        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        # self.observation_space = self._compute_space()  # if not child of gym.Space, setter may not work
        # TODO: can fail
        # for i in range(128):
        #     self.env.benchmark = self.env.datasets.random_benchmark(
        #         weighted=True
        #     )
        
        obs = super().reset(*args, **kwargs)
        # print(obs)
        # obs = parse_graph(obs)
        # self.env.observation_space = compute_space()
        # walks = walker.random_walks(obs, n_walks=15, walk_len=10)
        TRIALS = 10
        for i in range(TRIALS):
            if obs.number_of_nodes() < MAX_NODES or \
                    obs.number_of_edges() < MAX_EDGES:
                break
            
        obs = parse_graph(obs)

        return obs
