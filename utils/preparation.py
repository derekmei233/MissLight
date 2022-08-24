import numpy as np
import os
import configparser
import pickle
import random


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

def convert_road_state(relation, state_file, mask_pos, save_dir):

    inter_dict_id2inter = relation['inter_dict_id2inter']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']
    num_roads = len(road_dict_road2id)
    mask_inter = mask_pos
    print("mask_inter", mask_inter, '\n')

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
                    obs = node_dict[0]
                    # set for multiple action space
                    phase = one_hot(node_dict[1], 8)
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
    adj_road = get_road_adj(relation)
    road_info = {'road_feature': all_road_feature, 'road_update': road_update, 'adj_road': adj_road}

    print(print("road_update", np.where(np.array(road_update) == 2), '/n'))
    with open(save_dir, 'wb') as fw:
        pickle.dump(road_info, fw)
    print("save road_node_info done")

def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, len_input, num_for_predict,
                              points_per_hour, save=False):
    with open(graph_signal_matrix_filename, "rb") as f_ob:
        all_data = pickle.load(f_ob)
    data_all = all_data['road_feature']
    node_update = all_data['road_update']
    adj_road = all_data['adj_road']
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
        'node_update': node_update,
        'adj_road': adj_road
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
        dataset_info = {'train_x': all_data['train']['x'], 'train_target': all_data['train']['target'],
                        'train_timestamp': all_data['train']['timestamp'],
                        'val_x': all_data['val']['x'], 'val_target': all_data['val']['target'],
                        'val_timestamp': all_data['val']['timestamp'],
                        'test_x': all_data['test']['x'], 'test_target': all_data['test']['target'],
                        'test_timestamp': all_data['test']['timestamp'],
                        'mean': all_data['stats']['_mean'], 'std': all_data['stats']['_std'],
                        'node_update': all_data['node_update'], 'adj_road': all_data['adj_road']}
        with open(filename, 'wb') as fw:
            pickle.dump(dataset_info, fw)
        print('save file:', filename)
    return all_data

def run_preparation(config_file, mask_pos, graph_signal_matrix_filename, relation, state_file):
    config = configparser.ConfigParser()
    config.read(config_file)

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
    graph_signal_matrix_filename = graph_signal_matrix_filename

    neighbor_node = data_config['neighbor_node']

    mask_num = len(mask_pos)

    if not os.path.exists(os.path.dirname(graph_signal_matrix_filename)):
        os.makedirs(os.path.dirname(graph_signal_matrix_filename))

    # read state of intersections,convert it into state which road graph needed,save.

    graph_signal_matrix_filename = graph_signal_matrix_filename.split(
        '.')[0]+'_s'+str(points_per_hour)+'_p'+str(num_for_predict)+'_n'+str(neighbor_node)+'_m'+str(mask_num)+'.pkl'
    convert_road_state(relation, state_file, mask_pos, save_dir=graph_signal_matrix_filename)

    # according to file of task above, generate train set,val set and test set.
    all_data = read_and_generate_dataset(
        graph_signal_matrix_filename, 0, 0, num_of_hours, len_input, num_for_predict, points_per_hour=points_per_hour, save=True)


def mask_op(data, mask_matrix, adj_matrix, data_construct):
    '''
    :param data:(N,F)
    :param mask_matrix:(N,)
    :param adj_matrix:(N,N)
    :param data_construct:select or random
    :return:(N,F): inference value( -1 at observable position)
    '''
    # avoid inplace replacement
    data_or = -1 * np.ones_like(data, dtype=np.float32)
    if data_construct == 'select':
        for mask_id, value in enumerate(mask_matrix):
            if value != 1:
                neighbors = []
                for col_id, x in enumerate(adj_matrix[:, mask_id]):
                    if x == 1.:
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
    if phases is not None:
        masked_x = -1 * np.ones((num_roads, 11), dtype=np.float32)
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
                # convert finished, -1 at masked positions
    else:
        masked_x = -1 * np.ones((num_roads, 3), dtype=np.float32)
        for id_node, ob_length in enumerate(states):
            if id_node in mask_pos:
                inter = inter_dict_id2inter[id_node]
                in_roads = inter_in_roads[inter]
                for id_road, road in enumerate(in_roads):
                    road_id = road_dict_road2id[road]
            else:
                direction = []
                direction.append(np.concatenate([ob_length[0:3]]))
                direction.append(np.concatenate([ob_length[3:6]]))
                direction.append(np.concatenate([ob_length[6:9]]))
                direction.append(np.concatenate([ob_length[9:]]))
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
            phase = phase_oh[idx]
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
    from world import World
    config_file = 'cityflow_hz4x4.cfg'
    world = World(config_file, thread_num=8)
    relation = build_relation(world)
    adj_matrix = get_road_adj(relation)
    mask_pos = [9]
    mask_matrix = get_mask_matrix(relation, mask_pos)
    states = np.random.randint(0, 10, [16, 12])
    phases = np.random.randint(0, 8, [16])
    masked = inter2edge_slice(relation, states, None, [9])
    infer = mask_op(masked, mask_matrix, adj_matrix, 'select')
    final = infer * masked * -1
    prediction = reconstruct_data_slice(final, phases, relation)
    print('prediction: ', prediction)
