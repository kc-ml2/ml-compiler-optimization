import numpy as np
import networkx as nx

from compopt.constants import VOCAB, MAX_TEXT, MAX_POS


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


def parse_graph(g):
    # TODO: want to avoid for loop
    x = parse_nodes(g.nodes, return_attr=True)
    edge_attr = parse_edges(g.edges, return_attr=True)

    g = nx.DiGraph(g)
    edge_index = list(g.edges)

    return x, edge_index, edge_attr
