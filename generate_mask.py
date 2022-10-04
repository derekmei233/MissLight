from utils.mask_pos import random_mask
from utils.preparation import build_relation
from utils.control import create_world
import os


pattern = ['non_neighbor', 'neighbor']
num = [[1,2,3],[2,3,4]]
config_file = f'cityflow_syn4x4.cfg'
world = create_world(config_file)
relation = build_relation(world)

record = []
root_dir = os.path.join('data/output_data', f'syn4x4_test')
for i, p in enumerate(pattern):
    for j, n in enumerate(num[i]):
        record.append(random_mask(n, p, relation))

if not os.path.isdir(root_dir):
    os.mkdir(root_dir)
with open(os.path.join(root_dir, 'mask_pos.txt'), 'w') as f:
    for i in record:
        f.write(str(i))
        f.write('\n')
