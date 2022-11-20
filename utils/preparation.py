import numpy as np
import os
import random
import json
import re
from world import World


def fork_config(load_file, save_dir):
    # process and redirect .cfg file into current working directory
    oList = re.split('\_|\.', load_file)
    network = oList[1] # network name
    # change .cfg settings according to the shell inputs
    with open(load_file) as f:
        contents = json.load(f)
    save_file = os.path.join(save_dir, load_file)
    contents['saveReplay'] = True
    contents['roadnetLogFile'] = os.path.join(save_dir.lstrip('data/'), contents['roadnetLogFile'].split('/')[-1])
    contents['replayLogFile'] = os.path.join(save_dir.lstrip('data/'), contents['replayLogFile'].split('/')[-1])
    with open(save_file, 'w') as f:
        json.dump(contents, f, indent=2)
    return save_file

def one_hot(phase, num_class):
    assert num_class > phase.max()
    one_hot = np.zeros((len(phase), num_class))
    one_hot[range(0, len(phase)), phase.squeeze()] = 1
    # one_hot = one_hot.reshape(*phase.shape, num_class)
    return one_hot.squeeze()

def build_relation(world):
    '''
    build the relation between intersections and roads.
    For each intersection,it has in_roads,out_roads.
    TODO: this could be managed in a graphic way
    '''
    length = len(world.intersection_ids)
    data = world.intersection_ids[length-1].split("_")
    shape = []
    shape.append(int(data[2]))
    shape.append(int(data[1]))
    net_shape = shape
    intersections = world.intersections
    roads = world.roadnet['roads']
    all_inters = world.roadnet['intersections']
    inter_dict_id2inter = {}
    inter_dict_inter2id = {}
    road_dict_id2road = {}
    road_dict_road2id = {}
    inter_in_roads = {}
    inter_out_roads = {}
    road_links = {}
    inter_nb_num = {}
    neighbor_num = {}

    # create mapping of roads
    for idx, road_dic in enumerate(roads):
        road_dict_id2road[idx] = road_dic['id']
        road_dict_road2id[road_dic['id']] = idx

    # create mapping of inter and its roads
    for idx, inter_dic in enumerate(intersections):
        inter_dict_id2inter[idx] = inter_dic.id
        inter_dict_inter2id[inter_dic.id] = idx
        inter_in_roads[inter_dic.id] = [road['id']
                                        for road in inter_dic.in_roads]
        inter_out_roads[inter_dic.id] = [road['id']
                                         for road in inter_dic.out_roads]
        road_links[inter_dic.id] = []
        for roadlinks_dic in inter_dic.roadlinks:
            start = roadlinks_dic[0]
            end = roadlinks_dic[1]
            for all_inter_dic in all_inters:
                if all_inter_dic['id'] == inter_dic.id:
                    for road_link_dic in all_inter_dic['roadLinks']:
                        if road_link_dic['startRoad'] == start and road_link_dic['endRoad'] == end:
                            road_links[inter_dic.id].append(
                                roadlinks_dic + tuple([road_link_dic['type']]) + tuple([road_link_dic['direction']]))


        inter_nb_num[inter_dic.id] = []
        for in_road_dic in inter_dic.in_roads:
            inter_nb_num[inter_dic.id].append(in_road_dic['startIntersection'])

    for inter_dic in inter_nb_num:
        num = 0
        for neighbor in inter_nb_num[inter_dic]:
            if neighbor in inter_dict_inter2id:
                num += 1
        if num not in neighbor_num:
            neighbor_num[num] = []
        neighbor_num[num].append(inter_dict_inter2id[inter_dic])

    neighbor_idx = dict()
    for inter_dic in inter_nb_num:
        neighbor_idx.update({inter_dict_inter2id[inter_dic]: []})
        for nb in inter_nb_num[inter_dic]:
            if inter_dict_inter2id.get(nb) is not None:
                neighbor_idx[inter_dict_inter2id[inter_dic]].append(inter_dict_inter2id[nb])

    net_info = {'inter_dict_id2inter': inter_dict_id2inter, 'inter_dict_inter2id': inter_dict_inter2id,
                'road_dict_id2road': road_dict_id2road, 'road_dict_road2id': road_dict_road2id,
                'inter_in_roads': inter_in_roads, 'inter_out_roads': inter_out_roads,
                'road_links': road_links, 'neighbor_idx': neighbor_idx, 'net_shape': net_shape}
    print("relationship generated")
    return net_info

def get_road_adj(relation):
    """
    generate adjacent matrix for road node map
    """
    road_info = relation['road_links']
    road_dict_road2id = relation['road_dict_road2id']
    num_roads = len(relation['road_dict_id2road'])
    adjacency_matrix = np.zeros(
        (int(num_roads), int(num_roads)), dtype=np.float32)

    for inter_dic in road_info:
        for link_dic in road_info[inter_dic]:
            source = link_dic[0]
            target = link_dic[1]
            adjacency_matrix[road_dict_road2id[source]
                             ][road_dict_road2id[target]] = 1
    return adjacency_matrix

"""
def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    # ensure the num of nodes is the same
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]
    mean = train[:, :, :3, :].mean(axis=(0, 1, 3), keepdims=True)
    std = train[:, :, :3, :].std(axis=(0, 1, 3), keepdims=True)

    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)

    def normalize(x):
        r = np.nan_to_num((x[:, :, :3, :] - mean) / std)
        r = np.concatenate((r, x[:, :, 3:, :]), axis=2)
        t = x[:, :, 3:, :]
        return r
        # return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm
"""



def mask_op(data, mask_matrix, adj_matrix, data_construct):
    '''
    :param data:(N,F)
    :param mask_matrix:(N,)
    :param adj_matrix:(N,N)
    :param data_construct:select or random
    :return:(N,F): inference value( 0 at observable position)
    '''
    # avoid inplace replacement
    # data_or = -1 * np.ones_like(data, dtype=np.float32) for debug use only
    data_or = np.zeros_like(data, dtype=np.float32)
    if data_construct == 'select':
        for mask_id, value in enumerate(mask_matrix):
            if value != 1:
                neighbors = []
                for col_id, x in enumerate(adj_matrix[:, mask_id]):
                    if x == 1:
                        neighbors.append(col_id)
                neighbor_all = np.zeros_like(data[0, :3])
                if len(neighbors) != 0:
                    for node in neighbors:
                        neighbor_all = data[node, :3] + neighbor_all
                    data_or[mask_id, :3] = neighbor_all / len(neighbors)
                else:
                    rand_id = random.randint(0, len(mask_matrix)-1)
                    while mask_matrix[rand_id] != 1:
                        rand_id = random.randint(0, len(mask_matrix)-1)
                    data_or[mask_id, :3] = data[rand_id, :3]
                if value == 0:
                    # set virtual node's phase
                    rand_id = random.sample(neighbors, 1)[0]
                    data_or[mask_id, 3:] = data[rand_id, 3:]
    else:
        for mask_id, value in enumerate(mask_matrix):
            if value != 1:
                rand_id = random.randint(0, len(mask_matrix)-1)
                while mask_matrix[rand_id] != 1:
                    rand_id = random.randint(0, len(mask_matrix)-1)
                data_or[mask_id, :3] = data[rand_id, :3]
                if value == 0:
                    data_or[mask_id, 3:] = data[rand_id, 3:]
    return data_or

def mask_with_truth(data, mask_matrix, adj_matrix, data_construct):
    '''
    :param data:(B,N,F,T)
    :param mask_matrix:(N,)
    :param adj_matrix:(N,N)
    :param data_construct:select or random
    :return:(B,N,F,T): inference value( with truth value at observable position) for data construction
    '''
    # avoid inplace replacement
    # data_or = -1 * np.ones_like(data, dtype=np.float32) for debug use only
    mask_data = data.copy()
    if data_construct == 'select':
        for mask_id, value in enumerate(mask_matrix):
            if value != 1:
                neighbors = []
                for col_id, x in enumerate(adj_matrix[:, mask_id]):
                    if x == 1:
                        neighbors.append(col_id)
                neighbor_all = np.zeros_like(data[:, 0, :3, :])
                if len(neighbors) != 0:
                    for node in neighbors:
                        neighbor_all = data[:, node, :3, :] + neighbor_all
                    mask_data[:, mask_id, :3, :] = neighbor_all / len(neighbors)
                else:
                    rand_id = random.randint(0, len(mask_matrix)-1)
                    while mask_matrix[rand_id] != 1:
                        rand_id = random.randint(0, len(mask_matrix)-1)
                    mask_data[:, mask_id, :3, :] = data[:, rand_id, :3, :]
                if value == 0:
                    # set virtual node's phase
                    rand_id = random.sample(neighbors, 1)[0]
                    mask_data[:, mask_id, 3:, :] = data[:, rand_id, 3:, :]
    else:
        for mask_id, value in enumerate(mask_matrix):
            if value != 1:
                rand_id = random.randint(0, len(mask_matrix)-1)
                while mask_matrix[rand_id] != 1:
                    rand_id = random.randint(0, len(mask_matrix)-1)
                mask_data[:, mask_id, :3, :] = data[:, rand_id, :3, :]
                if value == 0:
                    mask_data[:, mask_id, 3:, ] = data[:, rand_id, 3:, :]
    return mask_data

def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    # ensure the num of nodes is the same
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]
    mean = train.mean(axis=(0, 1, 3), keepdims=True)[:,:,0:3,:]
    std = train.std(axis=(0, 1, 3), keepdims=True)[:,:,0:3,:]

    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)


    train_norm = normalize(train, mean, std)
    val_norm = normalize(val, mean, std)
    test_norm = normalize(test, mean, std)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm

def normalize(x, mean, std):
    noramlized_x = np.nan_to_num((x[:,:,0:3,:] - mean) / std)
    x[:,:,0:3,:] = noramlized_x
    return x


def inter2edge_slice(relation, states, phases, mask_pos):
    """
    convert feature from intersection-node formation to edge-node formation
    """
    inter_dict_id2inter = relation['inter_dict_id2inter']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']
    num_roads = len(road_dict_road2id)
    
    # states: [N_node, N_features] -> edge: [N_edge, 11]
    # only support phase space equal to 8
    # masked_x = -1 * np.ones((num_roads, 11), dtype=np.float32) for debug only
    masked_x = np.zeros((num_roads, 11), dtype=np.float32)
    phases_oh = one_hot(phases, 8)
    for id_node, ob_length in enumerate(states):
        if id_node in mask_pos:
            phase = phases_oh[id_node]
            inter = inter_dict_id2inter[id_node]
            in_roads = inter_in_roads[inter]
            for id_road, road in enumerate(in_roads):
                road_id = road_dict_road2id[road]
                masked_x[road_id, 3:] = phase
        else:
            phase = phases_oh[id_node]
            direction = []
            direction.append(np.concatenate([ob_length[0:3], phase]))
            direction.append(np.concatenate([ob_length[3:6], phase]))
            direction.append(np.concatenate([ob_length[6:9], phase]))
            direction.append(np.concatenate([ob_length[9:], phase]))
            # convert begin
            inter = inter_dict_id2inter[id_node]
            in_roads = inter_in_roads[inter]
            for id_road, road in enumerate(in_roads):
                road_id = road_dict_road2id[road]
                masked_x[road_id] = direction[id_road]
    return masked_x

def get_mask_matrix(relation, mask_pos):
    inter_dict_id2inter = relation['inter_dict_id2inter']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']
    num_roads = len(road_dict_road2id)
    mask_matrix = np.zeros(num_roads)
    for id_node in range(len(inter_dict_id2inter)):
        if id_node not in mask_pos:            
            inter = inter_dict_id2inter[id_node]
            in_roads = inter_in_roads[inter]
            for id_road, road in enumerate(in_roads):
                road_id = road_dict_road2id[road]
                mask_matrix[road_id] = 1
        else:
            inter = inter_dict_id2inter[id_node]
            in_roads = inter_in_roads[inter]
            for id_road, road in enumerate(in_roads):
                road_id = road_dict_road2id[road]
                mask_matrix[road_id] = 2
    return mask_matrix


def reconstruct_data_slice(data, phases, relation):
    '''
    convert data from road graph to intersection graph
    input shape:(N_road,3)
    output shape:(N_inter,12)
    '''
    inter_dict_inter2id = relation['inter_dict_inter2id']
    road_dict_road2id = relation['road_dict_road2id']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']

    # (B,N,F,T)->(B,T,N,F)
    if phases is not None:
        inter_feature = np.zeros((len(inter_dict_inter2id), 12), dtype=np.float32)
        phase_oh = one_hot(phases, 8)
        for inter_dic in inter_in_roads:
            idx = inter_dict_inter2id[inter_dic]
            #phase = phase_oh[idx]
            in_roads = inter_in_roads[inter_dic]
            in_roads_id = [road_dict_road2id[road] for road in in_roads]
            N = data[in_roads_id[0], :3]
            E = data[in_roads_id[1], :3]
            S = data[in_roads_id[2], :3]
            W = data[in_roads_id[3], :3]
            inter_feature[inter_dict_inter2id[inter_dic]] = np.concatenate([N, E, S, W])
            # no need to recover phase 
    else:
        inter_feature = np.zeros((len(inter_dict_inter2id), 12), dtype=np.float32)
        for inter_dic in inter_in_roads:
            idx = inter_dict_inter2id[inter_dic]
            in_roads = inter_in_roads[inter_dic]
            in_roads_id = [road_dict_road2id[road] for road in in_roads]
            N = data[in_roads_id[0], :3]
            E = data[in_roads_id[1], :3]
            S = data[in_roads_id[2], :3]
            W = data[in_roads_id[3], :3]
            inter_feature[inter_dict_inter2id[inter_dic]] = np.concatenate([N, E, S, W])
    return inter_feature


if __name__ == '__main__':
    config_file = 'cityflow_hz4x4.cfg'
    world = World(config_file, thread_num=8)
    relation = build_relation(world)
    adj_matrix = get_road_adj(relation)
    mask_pos = [2, 9]
    mask_matrix = get_mask_matrix(relation, mask_pos)
    states = np.random.randint(0, 10, [16, 12])
    phases = np.random.randint(0, 8, [16])
    masked = inter2edge_slice(relation, states, phases, mask_pos)
    infer = mask_with_truth(masked[np.newaxis,:,:,np.newaxis], mask_matrix, adj_matrix, 'select').squeeze()
    infer_op = mask_op(masked, mask_matrix, adj_matrix, 'select')
    final = infer_op + masked
    prediction = reconstruct_data_slice(final, phases, relation)
    prediction_dire = reconstruct_data_slice(infer.squeeze(), phases, relation)
    print('prediction: ', prediction)

