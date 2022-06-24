import numpy as np
import torch
from torch import nn
import torch_geometric.nn as pygnn


class NodeEncoder(nn.Module):
    def __init__(
            self,
            text_in,
            text_out=32,
            type_in=3,
            type_out=32,
    ):
        super(NodeEncoder, self).__init__()
        # must add 1 for OOV
        self.text_embedding = nn.Embedding(text_in + 1, text_out)
        self.type_embedding = nn.Embedding(type_in, type_out)
        self.out_dim = text_out + type_out  # concat

    def forward(
            self,
            x  # [B, num_nodes, channels]
    ) -> torch.Tensor:
        text, type_ = x[..., 0], x[..., 1]

        text = self.text_embedding(text)
        type_ = self.type_embedding(type_)

        x = torch.cat([text, type_], axis=-1)

        return x


class PositionEmbedding(nn.Module):
    def __init__(self, max_pos=5120, out_dim=64):
        # max_pos = 4096 https://github.com/ChrisCummins/phd/blob/aab7f16bd1f3546f81e349fc6e2325fb17beb851/programl/models/ggnn/messaging_layer.py#L38
        # increase embedding dim to 5120 just in case
        super().__init__()
        self.max_pos = max_pos
        in_dim = max_pos + 1  # add 1 for OOV
        positions = torch.arange(0, in_dim)
        inv_freq = 1 / (10000 ** (torch.arange(0.0, out_dim, 2.0) / out_dim))
        sinusoid_inp = torch.outer(positions, inv_freq)

        pos_emb = torch.zeros((in_dim, out_dim))
        evens = np.arange(0, int(out_dim), 2)
        odds = evens + 1
        pos_emb[:, odds] = torch.sin(sinusoid_inp)
        pos_emb[:, evens] = torch.cos(sinusoid_inp)
        self.pos_emb = pos_emb
        self.pos_emb.requires_grad = False

        self.f = nn.Linear(out_dim, out_dim)

    def forward(
            self,
            positions: torch.LongTensor,  # [B, num_edges]
    ):
        # replace OOV TODO : cpu - gpu maybe bottleneck, preprocess instead
        # positions = positions.where(
        #     positions > self.max_pos,
        #     torch.LongTensor([self.max_pos]).to(positions.device)
        # )

        embeds = torch.stack(
            [self.pos_emb[p] for p in positions]
        ).to(positions.device)
        x = self.f(embeds)

        return x


class EdgeEncoder(nn.Module):
    def __init__(self, flow_dim=3, out_dim=64):
        super(EdgeEncoder, self).__init__()

        self.flow_embedding = nn.Embedding(flow_dim, out_dim)
        self.pos_embedding = PositionEmbedding(out_dim=out_dim)
        self.out_dim = out_dim  # summed

    def forward(
            self,
            x  # [B, num_edges, channels]
    ) -> torch.Tensor:
        flow, pos = x[..., 0], x[..., 1]

        flow = self.flow_embedding(flow)
        pos = self.pos_embedding(pos)
        x = flow + pos

        return x


class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim=32):
        super().__init__()
        self.convs = nn.ModuleList([
            pygnn.GATv2Conv(node_dim, 512, edge_dim=edge_dim),
            pygnn.GATv2Conv(512, 512, edge_dim=edge_dim),
            pygnn.GATv2Conv(512, 256, edge_dim=edge_dim),
            pygnn.GATv2Conv(256, out_dim, edge_dim=edge_dim),
        ])
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        # [num_nodes, num_features]
        x = x.mean(0)

        return x


class Encoder(nn.Module):
    def __init__(
            self,
            node_encoder,
            edge_encoder,
            gnn,
    ):
        super().__init__()
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.gnn = gnn

    def forward(
            self,
            x,  # [B, num_nodes, num_features]
            edge_index,  # [B, 2, num_edges]
            edge_attr  # [B, num_edges, num_features]
    ):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # TODO: parallel execution
        x = torch.stack([
            self.gnn(i.float(), j.long(), k.float()) for i, j, k in
            zip(x, edge_index, edge_attr)
        ])

        return x
