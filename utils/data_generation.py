import numpy as np
import torch
import random
import pickle as pkl
from utils.preparation import mask_with_truth, one_hot, get_road_adj, normalization


def store_reshaped_data(data, information):
    # store information into [N_agents, N_features] formation
    actions = np.array(information[2])
    state_t = np.stack((information[0][0][0], information[0][1][0]))
    phase_t = np.stack((information[0][0][1], information[0][1][1]))
    reward = np.stack((information[1][0], information[1][1]))
    action = np.stack((actions[0], actions[1]))
    state_tp = np.stack((information[3][0][0], information[3][1][0]))
    phase_tp = np.stack((information[3][0][1], information[3][1][1]))
    for i in range(2, len(information[0])):
        state_t = np.concatenate((state_t, information[0][i][0][np.newaxis, :]), axis = 0)
        phase_t = np.concatenate((phase_t, information[0][i][1][np.newaxis, :]), axis =0)
        reward = np.concatenate((reward, information[1][i][np.newaxis]), axis = 0)
        action = np.concatenate((action, actions[i][np.newaxis]), axis=0)
        state_tp = np.concatenate((state_tp, information[3][i][0][np.newaxis, :]), axis = 0)
        phase_tp = np.concatenate((phase_tp, information[3][i][1][np.newaxis, :]), axis = 0)
    data.append([state_t, phase_t, reward, action, state_tp, phase_tp])

def store_reshaped_data_hetero(data, information):
    # store information into [N_agents, N_features] formation
    state_t = np.stack((information[0][0], information[0][1]))
    movement = np.stack((information[1][0], information[1][1]))
    reward = np.stack((information[2][0], information[2][1]))
    for i in range(2, len(information[0])):
        state_t = np.concatenate((state_t, information[0][i][np.newaxis, :]), axis=0)
        movement = np.concatenate((movement, information[1][i][np.newaxis]), axis=0)
        reward = np.concatenate((reward, information[2][i][np.newaxis]), axis=0)
    data.append([state_t, movement, reward])

def generate_reward_dataset(file, phases=8, infer='NN_st'):
    # prepare data for training reward_inference model
    # data formation [N_samples, [state_t, phase_t, reward, state_tp, phase_tp]]
    # TODO: need aggregated and separate rewards
    with open(file, 'rb') as f:
        contents = pkl.load(f)
    if infer == 'NN_st':
        # training sample [state_t, onehot(phase_t)], target [reward]
        feature = list()
        target = list()
        for sample in contents:
            feature_t = np.concatenate((sample[0], one_hot(sample[1], phases)), axis=1)
            feature.append(feature_t)
            target.append(np.mean(sample[2], axis = 1))
    elif infer == 'NN_stp':
        feature = list()
        target = list()
        for sample in contents:
            feature.append(sample[4], axis=1)
            target.append(np.mean(sample[2], axis=1))
    elif infer == 'NN_sta':
        # training sample [state_t, onehot(phase_t)], target [reward]
        feature = list()
        target = list()
        for sample in contents:
            feature_t = np.concatenate((sample[0], one_hot(sample[3], phases)), axis=1)
            feature.append(feature_t)
            target.append(np.mean(sample[2], axis = 1))

    feature= np.concatenate(feature)
    target = np.concatenate(target)
    total_idx = len(target)
    sample_idx = range(total_idx)
    sample_idx = random.sample(sample_idx, len(sample_idx))
    x_train = feature[sample_idx[: int(0.8 * total_idx)]]
    y_train = target[sample_idx[: int(0.8 * total_idx)]]
    x_test = feature[sample_idx[int(0.8 * total_idx) :]]
    y_test = target[sample_idx[int(0.8 * total_idx) :]]
    dataset = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
    return dataset

def generate_reward_dataset_hetero(file):
    # prepare data for training reward_inference model
    # data formation [N_samples, [state_t, phase_t, reward, state_tp, phase_tp]]
    # TODO: need aggregated and separate rewards
    with open(file, 'rb') as f:
        contents = pkl.load(f)
    # training sample [state_t, onehot(phase_t)], target [reward]
    feature = list()
    target = list()
    for sample in contents:
        feature_t = np.concatenate((sample[0], sample[1]), axis=1)
        feature.append(feature_t)
        target.append(sample[2])

    feature= np.concatenate(feature)
    target = np.concatenate(target)
    total_idx = len(target)
    sample_idx = range(total_idx)
    sample_idx = random.sample(sample_idx, len(sample_idx))
    x_train = feature[sample_idx[: int(0.8 * total_idx)]]
    y_train = target[sample_idx[: int(0.8 * total_idx)]]
    x_test = feature[sample_idx[int(0.8 * total_idx) :]]
    y_test = target[sample_idx[int(0.8 * total_idx) :]]
    dataset = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
    return dataset


def build_road_state(raw_state, relation, mask_pos):
    '''
    only support RIGHT, 4 road 3 lane, 8 phase intersections
    '''
    #TODO: CHECK
    inter_dict_id2inter = relation['inter_dict_id2inter']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']

    num_roads = len(road_dict_road2id)
    net_shape = relation['net_shape']

    # mask position and convertion from intersection to road structure
    adj_road = get_road_adj(relation)
    road_update = np.zeros(int(num_roads), dtype=np.int8)
    all_road_feature = []
    for epoch_state in raw_state:
        road_feature = np.zeros((len(epoch_state), int(num_roads), 11), dtype=np.float32)
        for  id_time, step_dict in enumerate(epoch_state):
            for id_node, node_dict in enumerate(step_dict):
                obs = node_dict[0]
                phase = one_hot(node_dict[1], 8)
                direction = []
                assert(len(obs)==12), 'only support 3 lanes road in [left, strait, right] order and [N,E,S,W] order'
                direction.append(np.concatenate([obs[0:3], phase]))
                direction.append(np.concatenate([obs[3:6], phase]))
                direction.append(np.concatenate([obs[6:9], phase]))
                direction.append(np.concatenate([obs[9:], phase]))

                in_roads = inter_in_roads[inter_dict_id2inter[id_node]]
                for id_road, road in enumerate(in_roads):
                    road_id = road_dict_road2id[road]
                    road_feature[id_time][road_id] = direction[id_road]
                    if id_time == 0:
                        if id_node in mask_pos:
                            road_update[road_id] = 2
                        else:
                            road_update[road_id] = 1
        all_road_feature.append(road_feature)
    road_info = {'road_feature': np.array(all_road_feature, dtype=np.float32), 'road_update': road_update, 'adj_road': adj_road}
    return road_info

def time_helper(cur_t, history_t):
    '''
    notice cur_t is the time shift, t=0 is the start point
    end index = real pos + 1
    start index = max(real pos + 1 - interval, 0)
    '''
    if cur_t - history_t + 1 <= 0:
        start_t = 0
    else:
        start_t = cur_t - history_t + 1
    end_t = cur_t + 1
    return start_t, end_t

def generate_state_dataset(state_file, history_t, pattern):
    '''
    This function is used for generating data directly used for training GraphWaveNet-r-f
    tans_config stored the hyperparameter used for generating dataset
    read in sample [ E, E_T, N, F]
    return sample should be [1, N, F, T], default = [1, 80, 4, 10]
    process: [sample, target] -> norm -> data for model_new -> (only normalize on first 3 features)
    '''
    with open(state_file, 'rb') as f:
        seq_data = np.load(f).transpose(0,2,3,1)
        update_rule = np.load(f)
        adj_matrix = np.load(f)
    all_sample = []
    for epoch_data in seq_data:
        for cur_t in range(epoch_data.shape[-1] - 1): #iterate on T axis
            sample = np.zeros((1, 80, 11, history_t), dtype=np.float32)
            start, end = time_helper(cur_t, history_t)
            sample[:, :, :, start-end:] = epoch_data[:, :, start:end]
            label = np.zeros((1, 80, 11, 1), dtype=np.float32)
            label[:, :, :, :] = epoch_data[:, :, end:end+1]
            all_sample.append((sample, label))
    # train， test, val split
    random.shuffle(all_sample)
    split_1 = int(len(all_sample) * 0.6)
    split_2 = int(len(all_sample) * 0.8)
    train_x, train_label = [np.concatenate(entry, axis=0) for entry in zip(*all_sample[:split_1])] # [B, N, F, T]
    val_x, val_label = [np.concatenate(entry, axis=0) for entry in zip(*all_sample[split_1:split_2])]
    test_x, test_label = [np.concatenate(entry, axis=0) for entry in zip(*all_sample[split_2:])]
    # TODO: imputate at the same time step now
    train_x = mask_with_truth(train_x, update_rule, adj_matrix, pattern)
    val_x = mask_with_truth(val_x, update_rule, adj_matrix, pattern)
    test_x = mask_with_truth(test_x, update_rule, adj_matrix, pattern)
    train_label = train_label[:, :, 0:3, :]
    test_label = test_label[:, :, 0:3, :]
    val_label = val_label[:, :, 0:3, :]
    (stats, train_x, val_x, test_x) = normalization(train_x, val_x, test_x)
    all_data = {
        'train': {
            'x': train_x,
            # 'x_phase':train_x[:,:,3:,:],
            'target': train_label,
        },
        'val': {
            'x': val_x,
            # 'x_phase':val_x[:,:,3:,:],
            'target': val_label,
        },
        'test': {
            'x': test_x,
            # 'x_phase':test_x[:,:,3:,:],
            'target': test_label,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        },
        'node_update': update_rule,
        'adj_road':adj_matrix
    }
    return all_data

def load_inference_data(file_data, DEVICE='cpu', batch_size=64, shuffle=True):

    mask_matrix = file_data['node_update']

    train_x = file_data['train']['x']  # (396, 80, 11, 30)
    # train_x_phase = file_data['train_x_phase']
    train_target = file_data['train']['target'][:, :, :3]  # (396, 80, 3,1)

    val_x = file_data['val']['x']
    # val_x_phase = file_data['val_x_phase']
    val_target = file_data['val']['target'][:, :, :3]

    test_x = file_data['test']['x']
    # test_x_phase = file_data['test_x_phase']
    test_target = file_data['test']['target'][:, :, :3]

    # mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    # std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)
    mean = file_data['stats']['_mean']  # (1, 1, 3, 1)
    std = file_data['stats']['_std']  # (1, 1, 3, 1)

    # tmp_train = re_normalization(train_x,mean,std)

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
    
    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std, mask_matrix, 


