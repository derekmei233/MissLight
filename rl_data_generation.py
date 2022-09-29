import numpy as np
import random
import pickle as pkl
from utils.preparation import one_hot


def store_reshaped_data(data, information):
    # store information into [N_agents, N_features] formation
    state_t = np.stack((information[0][0][0], information[0][1][0]))
    phase_t = np.stack((information[0][0][1], information[0][1][1]))
    reward = np.stack((information[1][0], information[1][1]))
    state_tp = np.stack((information[2][0][0], information[2][1][0]))
    phase_tp = np.stack((information[2][0][1], information[2][1][1]))
    for i in range(2, len(information[0])):
        state_t = np.concatenate((state_t, information[0][i][0][np.newaxis, :]), axis = 0)
        phase_t = np.concatenate((phase_t, information[0][i][1][np.newaxis, :]), axis =0)
        reward = np.concatenate((reward, information[1][i][np.newaxis]), axis = 0)
        state_tp = np.concatenate((state_tp, information[2][i][0][np.newaxis, :]), axis = 0)
        phase_tp = np.concatenate((phase_tp, information[2][i][1][np.newaxis, :]), axis = 0)
    data.append([state_t, phase_t, reward, state_tp, phase_tp])

def generate_dataset(file, phases=8, infer='st'):
    # prepare data for training inference model
    # data formation [N_samples, [state_t, phase_t, reward, state_tp, phase_tp]]
    # TODO: need aggregated and separate rewards
    with open(file, 'rb') as f:
        contents = pkl.load(f)
    if infer == 'st':
        # training sample [state_t, onehot(phase_t)], target [reward]
        feature = list()
        target = list()
        for sample in contents:
            feature_t = np.concatenate((sample[0], one_hot(sample[1], phases)), axis=1)
            feature.append(feature_t)
            target.append(np.mean(sample[2], axis = 1))
    elif infer == 'stp':
        feature = list()
        target = list()
        for sample in contents:
            feature.append(sample[3], axis=1)
            target.append(np.mean(sample[2], axis=1))

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
