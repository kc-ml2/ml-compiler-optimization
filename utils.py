import json
import torch
import scipy.sparse
import networkx as nx
import torch_geometric

def from_json(filepath):
    with open(filepath, "r") as f:
        json_data = json.load(f)
    G = nx.readwrite.json_graph.node_link_graph(json_data)
    G = nx.convert_node_labels_to_integers(G)

    edge_index = torch.LongTensor(list(G.edges())).t().contiguous()
    data = {}
    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    return(data)