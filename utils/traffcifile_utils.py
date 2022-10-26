import random
import json
from utils.agent_preparation import create_world
from utils.preparation import create_relation


def augment_traffic(path, seed, max_interval, interval):
    f = open(path, 'rb')
    random.seed(seed)
    contents = json.load(f)
    for i in contents:
        lasting = random.randint(0, max_interval)
        i['endTime'] = i['startTime'] + lasting
        intv = random.randint(1, min(interval, random.randint(0, lasting)) + 1)
        i['interval'] = intv
    df = path[:-5] + f'seed{seed}_maxinterval{max_interval}_interval{interval}.json'
    df = open(df, 'wb')
    json.dump(df)


def flow_heapmap(file):
    world = create_world(file)
    relation = build_relation(world)
    with open(file, 'rb') as f:
        contents = json.load(f)
    flow = contents['dir'] + contents['flowFile']
    with open(flow, 'rb') as f:
        contents = json.load(f)
    mapping = dict()
    for idx, inter in enumerate(world.intersection_ids):
        mapping.update({inter: idx})
    reverse_mapping = {}
    for inter in world.intersections:
        roads = inter.in_roads
        for r in roads:
            reverse_mapping.update({r['id']: inter.id})
