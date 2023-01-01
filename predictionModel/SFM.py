from utils.preparation import inter2edge_slice, mask_op, mask_with_truth, reconstruct_data_slice
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

# sfm_prediction only needs one time slice, redesign mask_op and construct_data
class SFM_dataset(Dataset):
    def __init__(self, feature, target):
        self.len = len(feature)
        self.features = feature
        self.target = target

    def __getitem__(self, idx):
        return self.features[idx, :], self.target[idx]

    def __len__(self):
        return self.len

def masked_mae(preds, labels, mask):
    loss = np.abs(preds-labels)
    loss = loss * mask
    return np.sum(loss) / len(np.where(mask == 1))


class SFM_predictor(object):
    def __init__(self, mask_matrix, adj_matrix, pattern):
        super(SFM_predictor, self).__init__()
        self.mask_matrix = mask_matrix
        self.mask = np.where(mask_matrix == 1,1,0)[:, np.newaxis].repeat(3, axis=1).T[np.newaxis,:,:,np.newaxis]
        self.eval_mask = np.where(mask_matrix == 2,1,0)[:, np.newaxis].repeat(3, axis=1).T[np.newaxis,:,:,np.newaxis]
        self.adj_matrix = adj_matrix
        self.pattern = pattern
        self.criterion = masked_mae
        self.name = self.__class__.__name__

    def predict(self, states, phases, relation, mask_pos, mask_matrix, adj_matrix, mode='select'):
        y_eval = inter2edge_slice(relation, states, phases, [])
        y_true_numpy = inter2edge_slice(relation, states, phases, mask_pos)

        infer = mask_op(y_true_numpy, mask_matrix, adj_matrix, mode)
        #edge_feature = infer * y_true_numpy * -1 for debug use only
        edge_feature = infer + y_true_numpy
        prediction = reconstruct_data_slice(edge_feature, phases, relation)
        loss = self.criterion(edge_feature.T[np.newaxis,0:3,:,np.newaxis], y_eval.T[np.newaxis,0:3,:,np.newaxis], self.eval_mask)
        return prediction, loss

    def make_model(self, **kwargs):
        return self

    def eval(self, x_test, y_test):
        y_pred = mask_with_truth(y_test, self.mask_matrix, self.adj_matrix, self.pattern)
        test_dataset = SFM_dataset(y_pred.transpose(0, 2, 1, 3), y_test.transpose(0, 2, 1, 3))
        test_loader = DataLoader(test_dataset, batch_size=64)
        test_loss = 0.0
        for i, data in enumerate(test_loader):
            y_pred, y_true = data
            loss = self.criterion(y_pred.numpy(), y_true.numpy(), self.eval_mask)
            test_loss += loss
        print(f'test average loss {test_loss / i}.')


class SFMHetero_predictor(object):
    def __init__(self, converter):
        super(SFMHetero_predictor, self).__init__()
        self.converter = converter
        self.mask_matrix = converter.lane_mask
        self.eval_matrix = 1 - converter.lane_mask
        self.connectivity = converter.phased_connection
        self.mask = np.where(self.mask_matrix == 1)[0]
        self.eval_mask = np.where(self.mask_matrix == 0)[0]
        self.criterion = masked_mae
        self.name = self.__class__.__name__

    def predict(self, last_state, action, states):
        true_lanes = self.converter.state2lane(states)
        masked_true_lanes = true_lanes * self.eval_matrix

        if last_state is None:
            # first step use average over all intersections
            recovered_lanes = deepcopy(masked_true_lanes)
            recovered_lanes[self.mask] = np.sum(masked_true_lanes) / len(self.eval_mask)
        else:
            connectivity = self.converter.get_phase_connectivity(action)
            inferred_lanes = np.dot((self.converter.state2lane(last_state) * self.eval_matrix), connectivity) * self.mask_matrix
            recovered_lanes = inferred_lanes + masked_true_lanes
            recovered_lanes = self.converter.fill_zero_connectivity(connectivity, recovered_lanes)
        recovered_states = self.converter.lane2state(recovered_lanes)
        # evaluate on masked positions
        loss = self.criterion(recovered_lanes, true_lanes, self.mask_matrix)
        return recovered_states, loss


    def make_model(self, **kwargs):
        return self

    def eval(self, x_test, y_test):
        y_pred = mask_with_truth(y_test, self.mask_matrix, self.adj_matrix, self.pattern)
        test_dataset = SFM_dataset(y_pred.transpose(0, 2, 1, 3), y_test.transpose(0, 2, 1, 3))
        test_loader = DataLoader(test_dataset, batch_size=64)
        test_loss = 0.0
        for i, data in enumerate(test_loader):
            y_pred, y_true = data
            loss = self.criterion(y_pred.numpy(), y_true.numpy(), self.eval_mask)
            test_loss += loss
        print(f'test average loss {test_loss / i}.')