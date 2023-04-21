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

MAX_NODES = int(1e6)
MAX_EDGES = int(1e7)

RUNNABLE_BMS = [
    'benchmark://cbench-v1/bitcount',
    'benchmark://cbench-v1/blowfish',
    'benchmark://cbench-v1/bzip2',
    'benchmark://cbench-v1/crc32',
    'benchmark://cbench-v1/dijkstra',
    'benchmark://cbench-v1/gsm',
    'benchmark://cbench-v1/jpeg-c',
    'benchmark://cbench-v1/jpeg-d',
    'benchmark://cbench-v1/patricia',
    'benchmark://cbench-v1/qsort',
    'benchmark://cbench-v1/sha',
    'benchmark://cbench-v1/stringsearch',
    'benchmark://cbench-v1/stringsearch2',
    'benchmark://cbench-v1/susan',
    'benchmark://cbench-v1/tiff2bw',
    'benchmark://cbench-v1/tiff2rgba',
    'benchmark://cbench-v1/tiffdither',
    'benchmark://cbench-v1/tiffmedian'
]

# deprecated
# node_high = 7696
# edge_high = 4099
# edge_idx_high = None
# max_num_nodes = 831232
# max_num_edges = 1501132

