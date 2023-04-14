from pathlib import Path
import compiler_gym

import networkx as nx
DATA_DIR = str(Path(__file__).parent.parent) + '/data'

env = compiler_gym.make(
    "llvm-ic-v0",
    observation_space='Programl',
)

ds = env.datasets['benchmark://cbench-v1']
bms = list(ds.benchmark_uris())
print('total bms: ', len(bms))
for bm in bms:
    obs = env.reset(benchmark=bm)
    file = '-'.join(bm.split('/')[2:])
    print('saving ', file)
    nx.write_gexf(obs, f'{DATA_DIR}/{file}.gexf')
    print('done')