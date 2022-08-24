import numpy as np
from preparation import build_relation
import random


def random_mask(num, policy, relation):
    # its grid style road network, and we need to keep mask position at every position with relatively same visit probability. Balanced exhaustive enumeration, sequential sampling method is better.
    seed = random.seed(random.randint(1, 1000))
    neighbor = relation['neighbor_idx']
    n_intersections = len(neighbor)
    candidate = list(np.arange(0, n_intersections))
    neighbor_pool = list()
    neighbor_non_pool = list(candidate)
    sampled = list()

    if policy == "neighbor":
        neighbor_flag = 0
        count = 0
        assert num < n_intersections
        while (count < num):
            if (neighbor_flag == 0) and (count == num - 1):
                tmp = random.choice(neighbor_pool)
                sampled.append(tmp)
                candidate.remove(tmp)
            else:
                tmp = random.choice(candidate)
                #print(tmp)
                sampled.append(tmp)
                candidate.remove(tmp)
                if tmp in neighbor_pool:
                    neighbor_flag = 1
                new_neighbor = neighbor[tmp]
                neighbor_pool = neighbor_pool or new_neighbor
                for item in new_neighbor:
                    if item in neighbor_non_pool:
                       neighbor_non_pool.remove(item)
            count+=1


    elif policy == "non_neighbor":
        assert num <= n_intersections/2
        count=0
        #print(neighbor)
        while(count<num):
            tmp=random.choice(candidate)
            sampled.append(tmp)
            candidate.remove(tmp)
            for item in neighbor[tmp]:
                if item in candidate:
                   candidate.remove(item)
            #assert num-count<=len(candidate),"please try a lower num"
            #tmp=random.randint(0,15)
            count+=1
            assert num - count <= len(candidate), "please try a lower num"
    return sampled


if __name__ == '__main__':
    from world import World
    config_file = 'cityflow_hz4x4.cfg'
    world = World(config_file, thread_num=8)
    relation = build_relation(world)
    mask = random_mask(2, 'neighbor', relation)
