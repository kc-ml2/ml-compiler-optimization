from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
import torch_geometric.nn as pygnn
from compopt.constants import VOCAB
from compopt.encoders import NodeEncoder, EdgeEncoder, GNN

torch, nn = try_import_torch()


# class GNN(nn.Module):
#     def __init__(self, node_dim, edge_dim, out_dim=32):
#         super().__init__()
#         self.c1 = pygnn.GATv2Conv(node_dim, 128, edge_dim=edge_dim, heads=3)
#         self.c2 = pygnn.GATv2Conv(128, 128, edge_dim=edge_dim, heads=3)
#         self.c4 = pygnn.GATv2Conv(128, out_dim, edge_dim=edge_dim, heads=3)
#         self.out_dim = out_dim
#
#     def forward(self, x, edge_index, edge_attr):
#         for conv in self.convs:
#             x = conv(x, edge_index, edge_attr)
#         # [num_nodes, num_features]
#         x = x.mean(0)
#
#         return x


class Model(TorchModelV2, nn.Module):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
    ):
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,

        )
        nn.Module.__init__(self)

        self.node_encoder = NodeEncoder(
            text_in=len(VOCAB),
            text_out=64,
            type_in=3,
            type_out=64
        )
        self.edge_encoder = EdgeEncoder(
            flow_dim=3,
            out_dim=128
        )

        self.gnn = GNN(
            self.node_encoder.out_dim,
            self.edge_encoder.out_dim,
            out_dim=256,
        )

        self.policy = nn.Linear(self.gnn.out_dim, action_space.n)
        self.value = nn.Linear(self.gnn.out_dim, 1)

        print(self)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # [B, num_nodes, num_node_features] [B, num_edges, 2] [B, num_edges, num_edge_features]
        x, edge_index, edge_attr = input_dict['obs']

        # [B, num_nodes, num_node_features] [B, 2, num_edges] [B, num_edges, num_edge_features]
        x, edge_index, edge_attr = x.values, edge_index.values.mT, edge_attr.values

        x = self.node_encoder(x.long())
        edge_attr = self.edge_encoder(edge_attr.long())

        x = torch.stack([
            self.gnn(i.float(), j.long(), k.float()) for i, j, k in
            zip(x, edge_index, edge_attr)
        ])

        action_logits = self.policy(x)
        self._value_logits = self.value(x)

        return action_logits, []

    @override(TorchModelV2)
    def value_function(self):
        return self._value_logits.squeeze(1)
