import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import eigs
from metric.metrics import masked_smape_np,  smape_np,masked_mae, masked_mse, masked_rmse, masked_mae_test, masked_rmse_test
import pickle


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


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

def get_world_shape(world):
    length = len(world.intersection_ids)
    data = world.intersection_ids[length-1].split("_")
    shape = []
    shape.append(int(data[1]))
    shape.append(int(data[2]))
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

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '_astcgn'

    print('load file:', filename)
    pkl_file = open(filename, 'rb')
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

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            if masked_flag:
                loss = criterion(outputs, labels, mask_matrix, missing_value)
            else:
                loss = criterion(outputs, labels)

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' %
                      (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss


# def evaluate_on_test_mstgcn(net, test_loader, test_target_tensor, sw, epoch, _mean, _std):
#     '''
#     for rnn, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.
#
#     :param net: model
#     :param test_loader: torch.utils.data.utils.DataLoader
#     :param test_target_tensor: torch.tensor (B, N_nodes, T_output, out_feature)=(B, N_nodes, T_output, 1)
#     :param sw:
#     :param epoch: int, current epoch
#     :param _mean: (1, 1, 3(features), 1)
#     :param _std: (1, 1, 3(features), 1)
#     '''
#
#     net.train(False)  # ensure dropout layers are in test mode
#
#     with torch.no_grad():
#
#         test_loader_length = len(test_loader)
#
#         test_target_tensor = test_target_tensor.cpu().numpy()
#
#         prediction = []  # 存储所有batch的output
#
#         for batch_index, batch_data in enumerate(test_loader):
#
#             encoder_inputs, labels = batch_data
#
#             outputs = net(encoder_inputs)
#
#             prediction.append(outputs.detach().cpu().numpy())
#
#             if batch_index % 100 == 0:
#                 print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))
#
#         prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
#         prediction_length = prediction.shape[2]
#
#         for i in range(prediction_length):
#             assert test_target_tensor.shape[0] == prediction.shape[0]
#             print('current epoch: %s, predict %s points' % (epoch, i))
#             mae = mean_absolute_error(test_target_tensor[:, :, i], prediction[:, :, i])
#             rmse = mean_squared_error(test_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
#             mape = masked_mape_np(test_target_tensor[:, :, i], prediction[:, :, i], 0)
#             print('MAE: %.2f' % (mae))
#             print('RMSE: %.2f' % (rmse))
#             print('MAPE: %.2f' % (mape))
#             print()
#             if sw:
#                 sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
#                 sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
#                 sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)


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
            params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input, prediction=prediction,
                 data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[-1]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            # unmask or mask
            if metric_method == 'mask':
                data_target_tensor = np.reshape(
                    data_target_tensor, (data_target_tensor.shape[0], -1, data_target_tensor.shape[-1]))
                prediction = np.reshape(
                    prediction, (prediction.shape[0], -1, prediction.shape[-1]))
                mae = mean_absolute_error(
                    data_target_tensor[:, :, i], prediction[:, :, i])
                rmse = mean_squared_error(
                    data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
                smape = smape_np(
                    data_target_tensor[:, :, i], prediction[:, :, i], 0)
                # mae = masked_mae_test(
                #     data_target_tensor[:, :, :, i], prediction[:, :, :, i], prediction[:, :, :, i].shape, mask_matrix, 0.0)
                # rmse = masked_rmse_test(
                #     data_target_tensor[:, :, :, i], prediction[:, :, :, i], prediction[:, :, :, i].shape, mask_matrix, 0.0)
                # smape = masked_smape_np(
                #     data_target_tensor[:, :, :, i], prediction[:, :, :, i], prediction[:, :, :, i].shape, mask_matrix, 0)
            else:
                data_target_tensor = np.reshape(
                    data_target_tensor, (data_target_tensor.shape[0], -1, data_target_tensor.shape[-1]))
                prediction = np.reshape(
                    prediction, (prediction.shape[0], -1, prediction.shape[-1]))
                mae = mean_absolute_error(
                    data_target_tensor[:, :, i], prediction[:, :, i])
                rmse = mean_squared_error(
                    data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
                smape = masked_smape_np(
                    data_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('SMAPE: %.2f' % (smape))
            excel_list.extend([mae, rmse, smape])

        # print overall results
        if metric_method == 'mask':
            mae = mean_absolute_error(
                data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
            rmse = mean_squared_error(
                data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
            smape = smape_np(
                data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
            # mae = masked_mae_test(
            #     data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), prediction.shape, mask_matrix, 0.0)
            # rmse = masked_rmse_test(
            #     data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), prediction.shape, mask_matrix, 0.0)
            # smape = masked_smape_np(
            #     data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), prediction.shape, mask_matrix, 0)
        else:
            mae = mean_absolute_error(
                data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
            rmse = mean_squared_error(
                data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
            smape = masked_smape_np(
                data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all SMAPE: %.2f' % (smape))
        excel_list.extend([mae, rmse, smape])
        print(excel_list)
