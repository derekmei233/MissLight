import pickle
import numpy as np
import random
import argparse
import configparser
import os


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
    # mean = train.mean(axis=0, keepdims=True)
    # std = train.std(axis=0, keepdims=True)
    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm


def build_road_state(relation_file, state_file, neighbor_node, mask_num, save_dir):
    with open(relation_file, 'rb') as f_re:
        relation = pickle.load(f_re)

    with open(state_file, "rb") as f_ob:
        state = pickle.load(f_ob)

    inter_dict_id2inter = relation['inter_dict_id2inter']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']
    num_roads = len(road_dict_road2id)
    net_shape = relation['net_shape']
    neighbor_num = relation['neighbor_num']

    mask_inter = random.sample(neighbor_num[int(neighbor_node)],int(mask_num))

    # only update 64 road_node,
    # It has 16 roads'states which start_intersection is virtual are not known.

    # add phase:road_feature(,,4) or no phase:road_feature(,,3)
    road_feature = np.zeros((len(state), int(num_roads), 4), dtype=np.float32)
    road_update = np.zeros(int(num_roads), dtype=np.int32)

    for id_time, step_dict in enumerate(state):
        for id_node, node_dict in enumerate(step_dict):
            if id_node in mask_inter:
                continue
            obs = node_dict[0][0]
            phase = node_dict[1][0]
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

                direction.append(np.concatenate([obs[0:3],phase]))
                direction.append(np.concatenate([obs[3:6],phase]))
                direction.append(np.concatenate([obs[6:9],phase]))
                direction.append(np.concatenate([obs[9:],phase]))
            inter = inter_dict_id2inter[id_node]

            # the order of in_roads are the same as list order
            in_roads = inter_in_roads[inter]
            for id_road, road in enumerate(in_roads):
                road_id = road_dict_road2id[road]
                road_feature[id_time][road_id] = direction[id_road]
                if id_time == 0:
                    road_update[road_id] = 1
        for update in road_update:
            # unknow road_node
            if update == 0:
                rand = 0
                while rand == 0:
                    rand = random.randint(0, num_roads-1)
                road_feature[id_time][update] = road_feature[id_time][rand]

    road_info = {'road_feature': road_feature, 'road_update': road_update}
    with open(save_dir, 'wb') as fw:
        pickle.dump(road_info, fw)
    print("save road_node_info done")


# def get_mask_inter(net_shape, neighbor_node, mask_num, neighbor_num):
#     '''generate masked intersections and its id
#     params:
#     net_shape: shape of road network,eg:(4,4),(1,5)...
#     neighbor_node: mask nodes which neighbors' num is neighbor_node
#     mask_num: the total num of mask nodes
#     neighbor_num: record the neighbors' num of each node 
#     '''

#     mask_id = random.sample(neighbor_num[int(neighbor_node)],int(mask_num))
#     return mask_id

def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
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
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour):
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

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour, save=False):
    with open(graph_signal_matrix_filename, "rb") as f_ob:
        all_data = pickle.load(f_ob)
    data_seq = all_data['road_feature']
    node_update = all_data['road_update']
    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue
        # hour_sample(30,80,4)
        # target(30,80,4)
        week_sample, day_sample, hour_sample, target = sample

        # [(week_sample),(day_sample),(hour_sample),target,time_sample]
        sample = []

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose(
                (0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose(
                (0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            # before:hour_sample:(30,80,4)
            # after expand:(1,30,80,4)
            # after transpose:(1,80,4,30)
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose(
                (0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose(
            (0, 2, 3, 1))  # (1,N,F,T)
        sample.append(target)

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(
            sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,F,Tpre),(1,1)]
    # len(all_samples):121
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
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath, file + '_r' + str(num_of_hours) +
                                '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '_astcgn'
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

    # read state of intersections,convert it into state which road graph needed,save.
    state_file = "./roadgraph/hz/rawstate_4x4.pkl"
    build_road_state(relation_filename, state_file, neighbor_node, mask_num,
                     save_dir=graph_signal_matrix_filename)

    # according to file of task above, generate train set,val set and test set.
    all_data = read_and_generate_dataset(
        graph_signal_matrix_filename, 0, 0, num_of_hours, num_for_predict, points_per_hour=points_per_hour, save=True)
