import pickle
import numpy as np
import random
import argparse
import configparser
import os
import torch


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
    mean = train.mean(axis=(0, 1, 3), keepdims=True)
    std = train.std(axis=(0, 1, 3), keepdims=True)

    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)

    def normalize(x):
        return np.nan_to_num((x - mean) / std)
        # return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm


def get_world_shape(world):
    length = len(world.intersection_ids)
    data = world.intersection_ids[length-1].split("_")
    shape = []
    shape.append(int(data[2]))
    shape.append(int(data[1]))
    return shape


def build_relation_intersection_road(world, save_dir):
    '''
    build the relation between intersections and roads.
    For each intersection,it has in_roads,out_roads.
    '''
    net_shape = get_world_shape(world)
    intersections = world.intersections
    roads = world.roadnet['roads']
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
        road_links[inter_dic.id] = inter_dic.roadlinks

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

    net_info = {'inter_dict_id2inter': inter_dict_id2inter, 'inter_dict_inter2id': inter_dict_inter2id,
                'road_dict_id2road': road_dict_id2road, 'road_dict_road2id': road_dict_road2id,
                'inter_in_roads': inter_in_roads, 'inter_out_roads': inter_out_roads,
                'road_links': road_links, 'neighbor_num': neighbor_num, 'net_shape': net_shape}
    with open(save_dir, 'wb') as fo:
        pickle.dump(net_info, fo)

    print("save relation done")


def load_graphdata_channel(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True):
    '''
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注: 从文件读入的数据，x是最大最小归一化的，但是y是真实值
    这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
    :param graph_signal_matrix_filename: str, graph_signal_matrix_filename = ./roadgraph/hz/state_4x4.pkl
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    # file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    # dirpath = os.path.dirname(graph_signal_matrix_filename)

    # filename = os.path.join(dirpath,
    #                         file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '_astcgn'

    print('load file:', graph_signal_matrix_filename)
    pkl_file = open(graph_signal_matrix_filename, 'rb')
    file_data = pickle.load(pkl_file)

    # mask:0 means random feature
    mask_matrix = file_data['node_update']
    train_x = file_data['train_x']  # (72, 80, 3, 30)
    # train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (72, 80, 3,30)

    val_x = file_data['val_x']
    # val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    # test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']

    # mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    # std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)
    mean = file_data['mean']  # (1, 1, 3, 1)
    std = file_data['std']  # (1, 1, 3, 1)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(
        torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(
        torch.FloatTensor).to(DEVICE)  # (B, N, F, T)

    train_dataset = torch.utils.data.TensorDataset(
        train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(
        torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(
        torch.FloatTensor).to(DEVICE)  # (B, N, F, T)

    val_dataset = torch.utils.data.TensorDataset(
        val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(
        torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(
        torch.FloatTensor).to(DEVICE)  # (B, N, F, T)

    test_dataset = torch.utils.data.TensorDataset(
        test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std, mask_matrix


def get_road_adj(filename):
    with open(filename, 'rb') as fo:
        result = pickle.load(fo)
    road_info = result['road_links']
    road_dict_road2id = result['road_dict_road2id']
    num_roads = len(result['road_dict_id2road'])
    adjacency_matrix = np.zeros(
        (int(num_roads), int(num_roads)), dtype=np.float32)

    for inter_dic in road_info:
        for link_dic in road_info[inter_dic]:
            source = link_dic[0]
            target = link_dic[1]
            adjacency_matrix[road_dict_road2id[source]
                             ][road_dict_road2id[target]] = 1

    return adjacency_matrix


def phase_to_onehot(phase, num_class):
    assert num_class > phase.max()
    one_hot = np.zeros((1, num_class))
    one_hot[-1][phase.reshape(-1)] = 1.
    # one_hot = one_hot.reshape(*phase.shape, num_class)
    return one_hot


# TODO: improve efficiency
def inter2state(relation_file, mask_pos, deq, idx, num_of_vertices, in_channels, len_input, state_t, phase_t):
    with open(relation_file, 'rb') as f_re:
        relation = pickle.load(f_re)
    inter_dict_id2inter = relation['inter_dict_id2inter']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']
    num_roads = len(road_dict_road2id)
    net_shape = relation['net_shape']
    neighbor_num = relation['neighbor_num']
    mask_inter = mask_pos
    masked_x = np.zeros(num_of_vertices, in_channels, len_input)
    # TODO: use state in deque to infer next_t state and substitute masked state. So infer state at t based on deque -> complete state at t
    # TODO: (current state not utilized)
    idx = min(idx, 29)
    if idx == 0 and state_t is None:
        return masked_x
    else:
        for i in range(-1, -idx - 1, -1):
            tmp = deq[i]
            road_feature = np.zeros((int(num_roads), 11), dtype=np.float32)
            for id_node, node_dict in enumerate(tmp):
                obs = node_dict[0]
                phase = phase_to_onehot(node_dict[1], 8)[0]
                direction = []
                if obs.shape[-1] == 12:
                    direction.append(np.concatenate([obs[0:3], phase]))
                    direction.append(np.concatenate([obs[3:6], phase]))
                    direction.append(np.concatenate([obs[6:9], phase]))
                    direction.append(np.concatenate([obs[9:], phase]))
                inter = inter_dict_id2inter[id_node]
                in_roads = inter_in_roads[inter]
                for id_road, road in enumerate(in_roads):
                    road_id = road_dict_road2id[road]
                    # TODO: double check here
                    road_feature[road_id] = direction[id_road]


def convert_road_state(relation_file, state_file, neighbor_node, mask_pos, save_dir):
    with open(relation_file, 'rb') as f_re:
        relation = pickle.load(f_re)
    inter_dict_id2inter = relation['inter_dict_id2inter']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']
    num_roads = len(road_dict_road2id)
    net_shape = relation['net_shape']
    neighbor_num = relation['neighbor_num']
    mask_inter = mask_pos

    # road_update:0:roads related to virtual inter,1:unmasked,2:masked
    road_update = np.zeros(int(num_roads), dtype=np.int32)

    all_road_feature = []

    for state_dic in state_file:
        with open(state_dic, "rb") as f_ob:
            state = pickle.load(f_ob)
        # only update 64 road_node,
        # It has 16 roads'states which start_intersection is virtual are not known.

        # add phase:road_feature(,,11) or no phase:road_feature(,,3)
        road_feature = np.zeros(
            (len(state), int(num_roads), 11), dtype=np.float32)

        for id_time, step_dict in enumerate(state):
            for id_node, node_dict in enumerate(step_dict):
                if id_node in mask_inter:
                    if id_time == 0:
                        in_roads = inter_in_roads[inter_dict_id2inter[id_node]]
                        for id_road, road in enumerate(in_roads):
                            road_id = road_dict_road2id[road]
                            road_update[road_id] = 2
                else:
                    obs = node_dict[0][0]
                    phase = phase_to_onehot(node_dict[1], 8)[0]
                    direction = []
                    if obs.shape[-1] == 12:

                        '''
                        3 dims:left,straight,right
                        list order:N,E,S,W
                        N = obs[0:3]
                        E = obs[3:6]
                        S = obs[7:10]
                        W = obs[10:13]
                        '''

                        direction.append(np.concatenate([obs[0:3], phase]))
                        direction.append(np.concatenate([obs[3:6], phase]))
                        direction.append(np.concatenate([obs[6:9], phase]))
                        direction.append(np.concatenate([obs[9:], phase]))
                    inter = inter_dict_id2inter[id_node]

                    # the order of in_roads are the same as list order
                    in_roads = inter_in_roads[inter]
                    for id_road, road in enumerate(in_roads):
                        road_id = road_dict_road2id[road]
                        road_feature[id_time][road_id] = direction[id_road]
                        if id_time == 0:
                            road_update[road_id] = 1
            for update_id, update in enumerate(road_update):
                # unknow road_node
                if update != 1:
                    rand_id = random.randint(0, num_roads-1)
                    while road_update[rand_id] != 1:
                        rand_id = random.randint(0, num_roads-1)
                    road_feature[id_time][update_id] = road_feature[id_time][rand_id]
        all_road_feature.append(road_feature)
    road_info = {'road_feature': all_road_feature, 'road_update': road_update}
    with open(save_dir, 'wb') as fw:
        pickle.dump(road_info, fw)
    print("save road_node_info done")

def get_mask_pos(relation_file, neighbor_node, mask_num):
    with open(relation_file, 'rb') as f_re:
        relation = pickle.load(f_re)
    inter_dict_id2inter = relation['inter_dict_id2inter']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']
    num_roads = len(road_dict_road2id)
    net_shape = relation['net_shape']
    neighbor_num = relation['neighbor_num']
    mask_inter = random.sample(neighbor_num[int(neighbor_node)], int(mask_num))
    return mask_inter

"""
def build_road_state(relation_file, state_file, neighbor_node, mask_num, save_dir):
    with open(relation_file, 'rb') as f_re:
        relation = pickle.load(f_re)
    inter_dict_id2inter = relation['inter_dict_id2inter']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']
    num_roads = len(road_dict_road2id)
    net_shape = relation['net_shape']
    neighbor_num = relation['neighbor_num']
    mask_inter = random.sample(neighbor_num[int(neighbor_node)], int(mask_num))

    # road_update:0:roads related to virtual inter,1:unmasked,2:masked
    road_update = np.zeros(int(num_roads), dtype=np.int32)

    all_road_feature = []

    for state_dic in state_file:
        with open(state_dic, "rb") as f_ob:
            state = pickle.load(f_ob)
        # only update 64 road_node,
        # It has 16 roads'states which start_intersection is virtual are not known.

        # add phase:road_feature(,,11) or no phase:road_feature(,,3)
        road_feature = np.zeros(
            (len(state), int(num_roads), 11), dtype=np.float32)

        for id_time, step_dict in enumerate(state):
            for id_node, node_dict in enumerate(step_dict):
                if id_node in mask_inter:
                    if id_time == 0:
                        in_roads = inter_in_roads[inter_dict_id2inter[id_node]]
                        for id_road, road in enumerate(in_roads):
                            road_id = road_dict_road2id[road]
                            road_update[road_id] = 2
                else:
                    obs = node_dict[0][0]
                    phase = phase_to_onehot(node_dict[1], 8)[0]
                    direction = []
                    if obs.shape[-1] == 12:

                        '''
                        3 dims:left,straight,right
                        list order:N,E,S,W
                        N = obs[0:3]
                        E = obs[3:6]
                        S = obs[7:10]
                        W = obs[10:13]
                        '''

                        direction.append(np.concatenate([obs[0:3], phase]))
                        direction.append(np.concatenate([obs[3:6], phase]))
                        direction.append(np.concatenate([obs[6:9], phase]))
                        direction.append(np.concatenate([obs[9:], phase]))
                    inter = inter_dict_id2inter[id_node]

                    # the order of in_roads are the same as list order
                    in_roads = inter_in_roads[inter]
                    for id_road, road in enumerate(in_roads):
                        road_id = road_dict_road2id[road]
                        road_feature[id_time][road_id] = direction[id_road]
                        if id_time == 0:
                            road_update[road_id] = 1
            for update_id, update in enumerate(road_update):
                # unknow road_node
                if update != 1:
                    rand_id = random.randint(0, num_roads-1)
                    while road_update[rand_id] != 1:
                        rand_id = random.randint(0, num_roads-1)
                    road_feature[id_time][update_id] = road_feature[id_time][rand_id]
        all_road_feature.append(road_feature)
    road_info = {'road_feature': all_road_feature, 'road_update': road_update}
    with open(save_dir, 'wb') as fw:
        pickle.dump(road_info, fw)
    print("save road_node_info done")
    return mask_inter
"""

def search_data(sequence_length, num_of_depend, label_start_idx,
                len_input, num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,num_of_hours
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + len_input
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, len_input, num_for_predict, points_per_hour):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    # if num_of_weeks > 0:
    #     week_indices = search_data(data_sequence.shape[0], num_of_weeks,
    #                                label_start_idx, num_for_predict,
    #                                7 * 24, points_per_hour)
    #     if not week_indices:
    #         return None, None, None, None

    #     week_sample = np.concatenate([data_sequence[i: j]
    #                                   for i, j in week_indices], axis=0)

    # if num_of_days > 0:
    #     day_indices = search_data(data_sequence.shape[0], num_of_days,
    #                               label_start_idx, num_for_predict,
    #                               24, points_per_hour)
    #     if not day_indices:
    #         return None, None, None, None

    #     day_sample = np.concatenate([data_sequence[i: j]
    #                                  for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, len_input, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, len_input, num_for_predict,
                              points_per_hour, save=False):
    with open(graph_signal_matrix_filename, "rb") as f_ob:
        all_data = pickle.load(f_ob)
    data_all = all_data['road_feature']
    node_update = all_data['road_update']
    all_samples = []
    for data_seq in data_all:
        for idx in range(data_seq.shape[0]):
            sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                        num_of_hours, idx, len_input, num_for_predict,
                                        points_per_hour)
            if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
                continue
            # hour_sample(30,80,11)
            # target(30,80,11)
            week_sample, day_sample, hour_sample, target = sample

            # [(week_sample),(day_sample),(hour_sample),target,time_sample]
            sample = []

            # if num_of_weeks > 0:
            #     week_sample = np.expand_dims(week_sample, axis=0).transpose(
            #         (0, 2, 3, 1))  # (1,N,F,T)
            #     sample.append(week_sample)

            # if num_of_days > 0:
            #     day_sample = np.expand_dims(day_sample, axis=0).transpose(
            #         (0, 2, 3, 1))  # (1,N,F,T)
            #     sample.append(day_sample)

            if num_of_hours > 0:
                # before:hour_sample:(30,80,4)
                # after expand:(1,30,80,4)
                # after transpose:(1,80,4,30)
                hour_sample = np.expand_dims(hour_sample, axis=0).transpose(
                    (0, 2, 3, 1))  # (1,N,F,T)
                sample.append(hour_sample)

            target = np.expand_dims(target, axis=0).transpose(
                (0, 2, 3, 1))  # (1,N,F,T)
            target = target[:, :, [0, 1, 2]]
            sample.append(target)

            time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
            sample.append(time_sample)

            all_samples.append(
                sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,F,Tpre),(1,1)]

    random.shuffle(all_samples)
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,F,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]
    # 切片操作[start_idx,end_idx,step],符号表示切片方向，“-”表示从右往左
    # training_set[:-2]:取[0,倒数第二个）个数据，即：training_set[0]
    # axis=-1:倒数第一个维度
    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]  # (B,N,F,Tpre)
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(
        train_x, val_x, test_x)

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        },
        'node_update': node_update
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data _mean :', stats['_mean'].shape, stats['_mean'])
    print('train data _std :', stats['_std'].shape, stats['_std'])
    print('node update matrix :', all_data['node_update'].shape)

    if save:
        # file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        # dirpath = os.path.dirname(graph_signal_matrix_filename)
        # filename = os.path.join(dirpath, file + '_r' + str(num_of_hours) +
        #                         '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '_astcgn'
        filename = graph_signal_matrix_filename.split('.')[0]+'_dataset.pkl'
        print('save file:', filename)
        dataset_info = {'train_x': all_data['train']['x'], 'train_target': all_data['train']['target'],
                        'train_timestamp': all_data['train']['timestamp'],
                        'val_x': all_data['val']['x'], 'val_target': all_data['val']['target'],
                        'val_timestamp': all_data['val']['timestamp'],
                        'test_x': all_data['test']['x'], 'test_target': all_data['test']['target'],
                        'test_timestamp': all_data['test']['timestamp'],
                        'mean': all_data['stats']['_mean'], 'std': all_data['stats']['_std'],
                        'node_update': all_data['node_update']}
        with open(filename, 'wb') as fw:
            pickle.dump(dataset_info, fw)
    return all_data


def read_output(output_file, relation_file, save_file):
    # save_len = len(save_file)
    num = 0
    for output in output_file:
        with open(output, 'rb') as f_op:
            output_data = pickle.load(f_op)
        re_data = {}
        for op_dic in output_data:
            if op_dic == 'input':
                continue
            state_data = reconstruct_data(
                output_data[op_dic], relation_file)
            re_data[op_dic] = state_data
        with open(save_file[num], 'wb') as f_sf:
            pickle.dump(state_data, f_sf)
        num += 1
    print("Convert Done!")


def reconstruct_data(data, relation_file):
    '''
    convert data from road graph to intersection graph
    input shape:(B,N_road,3,T)
    output shape:(B,N_inter,12,T)
    '''
    with open(relation_file, 'rb') as f_r:
        relation = pickle.load(f_r)
    # inter_dict_id2inter = relation['inter_dict_id2inter']
    inter_dict_inter2id = relation['inter_dict_inter2id']
    road_dict_road2id = relation['road_dict_road2id']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']
    # num_roads = len(road_dict_road2id)
    # net_shape = relation['net_shape']
    # neighbor_num = relation['neighbor_num']

    # (B,N,F,T)->(B,T,N,F)
    data = np.transpose(data, (0, 3, 1, 2))
    inter_feature = np.zeros((data.shape[0], data.shape[1], len(
        inter_dict_inter2id), 12), dtype=np.float32)
    for inter_dic in inter_in_roads:
        in_roads = inter_in_roads[inter_dic]
        in_roads_id = [road_dict_road2id[road] for road in in_roads]
        N = data[:, :, [in_roads_id[0]]]
        E = data[:, :, [in_roads_id[1]]]
        S = data[:, :, [in_roads_id[2]]]
        W = data[:, :, [in_roads_id[3]]]
        inter_feature[:, :, [inter_dict_inter2id[inter_dic]]
                      ] = np.concatenate([N, E, S, W], axis=3)
    return inter_feature


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/HZ_4x4_astgcn.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config))
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']

    num_of_vertices = int(data_config['num_of_vertices'])
    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])
    len_input = int(data_config['len_input'])
    dataset_name = data_config['dataset_name']
    num_of_weeks = int(training_config['num_of_weeks'])
    num_of_days = int(training_config['num_of_days'])
    num_of_hours = int(training_config['num_of_hours'])
    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
    relation_filename = data_config['relation_filename']
    neighbor_node = data_config['neighbor_node']
    mask_num = data_config['mask_num']
    state_basedir = data_config['state_basedir']

    if not os.path.exists(state_basedir):
        os.makedirs(state_basedir)
    if not os.path.exists(os.path.dirname(graph_signal_matrix_filename)):
        os.makedirs(os.path.dirname(graph_signal_matrix_filename))

    # read state of intersections,convert it into state which road graph needed,save.
    state_file = ['rawstate_4x4_199_or.pkl', 'rawstate_4x4_199.pkl']
    state_file_list = [os.path.join(state_basedir, s_dic)
                       for s_dic in state_file]
    graph_signal_matrix_filename = graph_signal_matrix_filename.split(
        '.')[0]+'_s'+str(points_per_hour)+'_p'+str(num_for_predict)+'_n'+str(neighbor_node)+'_m'+str(mask_num)+'.pkl'
    build_road_state(relation_filename, state_file_list, neighbor_node, mask_num,
                     save_dir=graph_signal_matrix_filename)

    # according to file of task above, generate train set,val set and test set.
    all_data = read_and_generate_dataset(
        graph_signal_matrix_filename, 0, 0, num_of_hours, len_input, num_for_predict, points_per_hour=points_per_hour, save=True)

