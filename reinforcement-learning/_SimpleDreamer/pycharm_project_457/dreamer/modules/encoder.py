import torch
import torch.nn as nn
import os
os.environ['CUDA_VIDISABLE_DEVICES'] = '1'

from dreamer.utils.utils import (
    pixel_normalization,
    initialize_weights,
    horizontal_forward,
)


class Encoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.encoder

        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape

        # self.network = nn.Sequential(
        #     nn.Conv2d(
        #         self.observation_shape[0],
        #         self.config.depth * 1,
        #         self.config.kernel_size,
        #         self.config.stride,
        #     ),
        #     activation,
        #     nn.Conv2d(
        #         self.config.depth * 1,
        #         self.config.depth * 2,
        #         self.config.kernel_size,
        #         self.config.stride,
        #     ),
        #     activation,
        #     nn.Conv2d(
        #         self.config.depth * 2,
        #         self.config.depth * 4,
        #         self.config.kernel_size,
        #         self.config.stride,
        #     ),
        #     activation,
        #     nn.Conv2d(
        #         self.config.depth * 4,
        #         self.config.depth * 8,
        #         self.config.kernel_size,
        #         self.config.stride,
        #     ),
        #     activation,
        # )
        self.network = nn.Sequential(
            nn.LazyLinear(4096),
            activation,
            nn.LazyLinear(2048),
            activation,
            nn.LazyLinear(1024),
            activation,
        )
        self.network.apply(initialize_weights)

    def forward(self, x):
        # x = pixel_normalization(x)
        return horizontal_forward(self.network, x, input_shape=self.observation_shape)
