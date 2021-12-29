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


def convert_traffic(path, seed, fluc):
    f = open(path, 'rb')
    random.seed(seed)
    contents = json.load(f)
    index = []
    for i in contents:
        shift = random.randint(-fluc, fluc)
        index.append([i['interval'],max(0, i['startTime'] + shift),
                      max(0, i['endTime'] + shift)])
    random.shuffle(index)
    for idx, j in enumerate(contents):
        j['interval'] = index[idx][0]
        j['startTime'] = index[idx][1]
        j['endTime'] = index[idx][2]
    df = path[:-5] + f'seed{seed}fluc{fluc}.json'
    df = open(df, 'w')
    json.dump(contents, df)
    return contents


def static_traffic(path):
    f = open(path, 'rb')
    contents = json.load(f)
    traffic = 0
    for i in contents:
        intv = i['endTime'] - i['startTime']
        count = int(intv / i['interval']) + 1
        traffic += count
    print(traffic)
    return traffic
