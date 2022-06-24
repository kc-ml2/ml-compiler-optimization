import json
import os
from pathlib import Path

_dir = os.path.join(Path(__file__).parent, 'vocab.json')
with open(_dir, 'r') as fp:
    VOCAB = json.load(fp)

NODE_FEATURES = ['text', 'type']
MAX_TEXT, MAX_TYPE = len(VOCAB), 3

EDGE_FEATURES = ['flow', 'position']
MAX_FLOW, MAX_POS = 3, 5120

# deprecated
node_high = 7696
edge_high = 4099
edge_idx_high = None
max_num_nodes = 831232
max_num_edges = 1501132

