import random
import json


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
    json.dump(df, contents)


def static_traffic(path):
    f = open(path, 'rb')
    contents = json.load(f)
    traffic = 0
    for i in contents:
        intv = i['endTime'] - i['startTime']
        count = int(intv / i['interval']) + 1
        traffic += count
    return traffic
