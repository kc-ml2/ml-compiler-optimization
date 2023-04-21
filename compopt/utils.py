import numpy as np
import networkx as nx


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    # https://gist.github.com/foowaa/5b20aebd1dff19ee024b6c72e14347bb

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return sinusoid_table


class NumpyPreprocessor:
    def __init__(self, vocab):
        self.type_onehot = np.eye(3)
        self.token_onehot = np.eye(len(vocab) + 1)
        self.flow_onehot = np.eye(3)
        self.pos_table = get_sinusoid_encoding_table(1024 * 8, 3)
        self.vocab = vocab

    def process(self, obs):
        token_arr = []
        for v in nx.get_node_attributes(obs, 'text').values():
            try:
                token_arr.append(self.vocab[v])
            except:
                token_arr.append(len(self.vocab))  # OOV
        token_arr = np.array(token_arr, dtype=int)

        type_arr = np.array(
            list(nx.get_node_attributes(obs, 'type').values()),
            dtype=int
        )

        x = np.concatenate([
            self.type_onehot[type_arr],
            self.token_onehot[token_arr],
        ], axis=1)

        flow_arr = np.array(list(nx.get_edge_attributes(obs, 'flow').values()),
                            dtype=int)
        position_arr = np.array(
            list(nx.get_edge_attributes(obs, 'position').values()), dtype=int)

        edge_index = np.array(list(obs.edges))[:, :2]

        edge_attr = self.flow_onehot[flow_arr] + self.pos_table[position_arr]

        return x, edge_index.T, edge_attr