import numpy as np
from utils.preparation import build_relation
import random


def random_mask(num, policy, relation):
    # its grid style road network, and we need to keep mask position at every position with relatively same visit probability. Balanced exhaustive enumeration, sequential sampling method is better.
    seed = random.seed(random.randint(1, 1000))
    neighbor = relation['neighbor_idx']
    n_intersections = len(neighbor)
    candidate = set(np.arange(0, n_intersections))
    neighbor_pool = set()
    neighbor_non_pool = set(candidate)
    sampled = set()

    if policy == "allowed":
        neighbor_flag = 0 
        count = 0
        assert num < n_intersections
        while (count < num):
            if (neighbor_flag == 0) and (count == num - 1):
                tmp = random.choice(neighbor_pool)
                sampled.add(tmp)
                candidate.pop(tmp)
            else:
                tmp = random.choice(candidate)
                sampled.add(tmp)
                candidate.pop(tmp)
                if tmp in neighbor_pool:
                    neighbor_flag = 1
                new_neighbor = neighbor[tmp]
                neighbor_pool = neighbor_pool or new_neighbor
                neighbor_non_pool.pop(new_neighbor)


    elif policy == "forbidden":
        assert num < 7
        count=0
        masks=[]
        while(count<num):
            tmp=random.randint(0,15)
            if tmp+1 not in masks and tmp-1 not in masks and tmp not in masks and tmp+4 not in masks and tmp-4 not in masks:
                #print(tmp)
                masks.append(tmp)
                count+=1
    return masks

def random_mask(num, policy):
    random.seed(random.randint(1, 1000))
    if policy == "allowed":
        assert num < 16
        masks = random.sample(range(0,16),num)
    elif policy == "forbidden":
        assert num < 7
        count=0
        masks=[]
        while(count<num):
            tmp=random.randint(0,15)
            if tmp+1 not in masks and tmp-1 not in masks and tmp not in masks and tmp+4 not in masks and tmp-4 not in masks:
                #print(tmp)
                masks.append(tmp)
                count+=1
    return masks 


if __name__ == '__main__':
    from world import World
    config_file = 'cityflow_hz4x4.cfg'
    world = World(config_file, thread_num=8)
    relation = build_relation(world)
    mask = random_mask(2, 'allowed', relation)