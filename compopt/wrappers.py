from gym import Wrapper

import networkx as nx
import torch_geometric as pyg
from compiler_gym.datasets import Datasets
from compiler_gym.wrappers import CompilerEnvWrapper
from gym import Wrapper

from compopt.constants import (
    NODE_FEATURES,
    EDGE_FEATURES,
    RUNNABLE_BMS, MAX_NODES, MAX_EDGES
)
from compopt.nx_utils import parse_nodes, parse_edges, parse_graph
from compopt.spaces import compute_space


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
        self.observation_space = compute_space()

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
