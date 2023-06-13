import numpy as np

from gym.spaces import Box, Tuple

from ray.rllib.utils.spaces.repeated import Repeated

from compopt.constants import (
    MAX_TEXT, MAX_TYPE, MAX_FLOW, MAX_POS, MAX_NODES, MAX_EDGES
)


class XSpace(Repeated):
    def __init__(self):
        super().__init__(
            Box(
                low=np.array([0, 0]),
                high=np.array([MAX_TEXT, MAX_TYPE]),
                shape=(2,),
                dtype=int
            ),
            max_len=MAX_NODES,  # TODO: find exact bound
        )

    def sample(self):
        pass


class EdgeIndexSpace(Repeated):
    def __init__(self):
        super().__init__(
            Box(
                low=np.array([0, 0]),
                high=np.array([MAX_NODES, MAX_NODES]),
                # high=np.array([num_nodes - 1, num_nodes - 1]),
                shape=(2,),
                dtype=int
            ),
            max_len=MAX_EDGES * 2,  # TODO: find exact bound
        )

    def sample(self):
        pass


class EdgeFeatureSpace(Repeated):
    def __init__(self):
        super().__init__(
            Box(
                low=np.array([0, 0]),
                high=np.array([MAX_FLOW, MAX_POS]),
                shape=(2,),
                dtype=int
            ),
            max_len=MAX_EDGES * 2,  # TODO: find exact bound
        )

    def sample(self):
        pass


def compute_graph_space():
    return Tuple([
        XSpace(),
        EdgeIndexSpace(),
        EdgeFeatureSpace()
    ])
