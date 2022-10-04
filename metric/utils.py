import os
import numpy as np
import torch
import torch.utils.data
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_squared_error
#from scipy.sparse.linalg import eigs
from metric.metrics import masked_mape_test, masked_mae_test, masked_rmse_test
import pickle



def re_normalization(x, mean, std):
    r = x[:, :, :3, :] * std + mean
    r = np.concatenate((r, x[:, :, 3:, :]), axis=2)
    return r


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

def compute_val_loss_mstgcn(net, val_loader, criterion,  masked_flag, missing_value, sw, epoch, mask_matrix, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss
        type = 'val'

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            if masked_flag:
                loss = criterion(outputs, labels, mask_matrix, type,missing_value)
            else:
                loss = criterion(outputs, labels)

            tmp.append(loss.item())
            # if batch_index % 100 == 0:
            print('validation batch %s / %s, loss: %.2f' %
                    (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss

def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method, _mean, _std, params_path, mask_matrix, type):
    '''

    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []  # 存储所有batch的output

        input = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, labels = batch_data

            input.append(encoder_inputs.cpu().numpy())  # (batch, T', 1)

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' %
                      (batch_index + 1, loader_length))

        input = np.concatenate(input, 0)

        input = re_normalization(input, _mean, _std)

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(
            params_path, 'output_epoch_%s_%s.pkl' % (global_step, type))
        origin_data = {'input':input,'prediction':prediction,'data_target_tensor':data_target_tensor}
        with open(output_filename, 'wb') as fw:
            pickle.dump(origin_data, fw)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[-1]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            # unmask or mask
            if metric_method == 'mask':
                # data_target_tensor = np.reshape(
                #     data_target_tensor, (data_target_tensor.shape[0], -1, data_target_tensor.shape[-1]))
                # prediction = np.reshape(
                #     prediction, (prediction.shape[0], -1, prediction.shape[-1]))
                # mae = mean_absolute_error(
                #     data_target_tensor[:, :, i], prediction[:, :, i])
                # rmse = mean_squared_error(
                #     data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
                # smape = smape_np(
                #     data_target_tensor[:, :, i], prediction[:, :, i], 0)
                mae = masked_mae_test(
                    data_target_tensor[:, :, :, i], prediction[:, :, :, i], prediction[:, :, :, i].shape, mask_matrix, 0.0)
                rmse = masked_rmse_test(
                    data_target_tensor[:, :, :, i], prediction[:, :, :, i], prediction[:, :, :, i].shape, mask_matrix, 0.0)
                mape = masked_mape_test(
                    data_target_tensor[:, :, :, i], prediction[:, :, :, i], prediction[:, :, :, i].shape, mask_matrix, 0)
            # else:
            #     data_target_tensor = np.reshape(
            #         data_target_tensor, (data_target_tensor.shape[0], -1, data_target_tensor.shape[-1]))
            #     prediction = np.reshape(
            #         prediction, (prediction.shape[0], -1, prediction.shape[-1]))
            #     mae = mean_absolute_error(
            #         data_target_tensor[:, :, i], prediction[:, :, i])
            #     rmse = mean_squared_error(
            #         data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            #     smape = masked_smape_np(
            #         data_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        if metric_method == 'mask':
            # mae = mean_absolute_error(
            #     data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
            # rmse = mean_squared_error(
            #     data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
            # smape = smape_np(
            #     data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
            mae = masked_mae_test(
                data_target_tensor, prediction, prediction.shape, mask_matrix, 0.0)
            rmse = masked_rmse_test(
                data_target_tensor, prediction, prediction.shape, mask_matrix, 0.0)
            mape = masked_mape_test(
                data_target_tensor, prediction, prediction.shape, mask_matrix, 0.0)
        # else:
        #     mae = mean_absolute_error(
        #         data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        #     rmse = mean_squared_error(
        #         data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        #     smape = masked_smape_np(
        #         data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)
