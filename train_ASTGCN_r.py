#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from predictionModel.ASTGCN_r import make_model
from metric.utils import compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from prepareData import load_graphdata_channel, get_road_adj, read_output
#from tensorboardX import SummaryWriter
from metric.metrics import masked_mse


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/hz4x4_astgcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

relation_filename = data_config['relation_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
neighbor_node = int(data_config['neighbor_node'])
mask_num = int(data_config['mask_num'])

model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])

folder_dir = 's%d_p%d_n%d_m%d' % (
    points_per_hour, num_for_predict, neighbor_node, mask_num)
print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('params_path:', params_path)
# graph_signal_matrix_filename = ./roadgraph/hz/state_4x4.pkl
"""
graph_signal_matrix_filename = graph_signal_matrix_filename.split(
    '.')[0]+'_s'+str(points_per_hour)+'_p'+str(num_for_predict)+'_n'+str(neighbor_node)+'_m'+str(mask_num)+'_dataset.pkl'

train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std, mask_matrix = load_graphdata_channel(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)

adj_mx = get_road_adj(relation_filename)

net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                 num_for_predict, len_input, num_of_vertices)
"""


def train_main(inference_net, start_epoch, graph_signal_matrix_filename, relation_filename):

    train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std, mask_matrix = load_graphdata_channel(
        graph_signal_matrix_filename, num_of_hours,
        num_of_days, num_of_weeks, DEVICE, batch_size)
    if start_epoch != 0:
        start_epoch = start_epoch

    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    masked_flag = 0
    criterion = nn.L1Loss().to(DEVICE)
    criterion_masked = masked_mse
    if loss_function == 'masked_mse':
        criterion_masked = masked_mse  # nn.MSELoss().to(DEVICE)
        masked_flag = 1
    # elif loss_function == 'masked_mae':
    #     criterion_masked = masked_mae
    #     masked_flag = 1
    # elif loss_function == 'mae':
    #     criterion = nn.L1Loss().to(DEVICE)
    #     masked_flag = 0
    # elif loss_function == 'rmse':
    #     criterion = nn.MSELoss().to(DEVICE)
    #     masked_flag = 0
    optimizer = optim.Adam(inference_net.parameters(), lr=learning_rate)
    sw = SummaryWriter(log_dir=params_path, flush_secs=5)
    #print(inference_net)
    """
    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in inference_net.state_dict():
        print(param_tensor, '\t', inference_net.state_dict()[param_tensor].size())
        total_param += np.prod(inference_net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])
    """
    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    if start_epoch > 0:

        params_filename = os.path.join(
            params_path, 'epoch_%s.params' % start_epoch)

        inference_net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    # train model
    for epoch in range(start_epoch, start_epoch + epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        if masked_flag:
            val_loss = compute_val_loss_mstgcn(
                inference_net, val_loader, criterion_masked, masked_flag, missing_value, sw, epoch, mask_matrix)
        else:
            val_loss = compute_val_loss_mstgcn(
                inference_net, val_loader, criterion, masked_flag, missing_value, sw, epoch, mask_matrix)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(inference_net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)

        inference_net.train()  # ensure dropout layers are in train mode

        type = 'train'

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, labels = batch_data

            optimizer.zero_grad()

            outputs = inference_net(encoder_inputs)

            if masked_flag:
                loss = criterion_masked(
                    outputs, labels, mask_matrix, type, missing_value)
            else:
                loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

            if global_step % 1000 == 0:

                print('global step: %s, training loss: %.2f, time: %.2fs' %
                      (global_step, training_loss, time() - start_time))

    print('best epoch:', best_epoch)

    # apply the best model on the test set
    predict_main(inference_net, best_epoch, test_loader, test_target_tensor,
                 metric_method, _mean, _std, mask_matrix, 'test', relation_filename)

    best_param_path = os.path.join(
        params_path, 'epoch_%s.params' % best_epoch)
    inference_net.load_state_dict(torch.load(best_param_path))
    final_param_path = os.path.join(
        params_path, 'epoch_%s.params' % (start_epoch + epochs))
    torch.save(inference_net.state_dict(), final_param_path)

    return inference_net


def predict_main(inference_net, global_step, data_loader, data_target_tensor, metric_method, _mean, _std, mask_matrix, type, relation_filename):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 4, 1)
    :param std: (1, 1, 4, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(
        params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    inference_net.load_state_dict(torch.load(params_filename))

    predict_and_save_results_mstgcn(inference_net, data_loader, data_target_tensor,
                                    global_step, metric_method, _mean, _std, params_path, mask_matrix, type)
    
    # convert feature from road to intersection 
    output_dir = os.path.join('experiments', dataset_name,
                              's'+str(points_per_hour)+'_p'+str(num_for_predict)+'_n'+str(neighbor_node)+'_m'+str(mask_num))
    output_file = ['output_epoch_'+str(global_step)+'_test.pkl']
    output_file_list = [os.path.join(output_dir, output_dic)
                        for output_dic in output_file]

    save_file = ['reconstruct_epoch_'+str(global_step)+'_test.pkl']
    save_file_list = [os.path.join(output_dir, save_dic)
                      for save_dic in save_file]
    read_output(output_file_list, relation_filename, save_file_list)


if __name__ == "__main__":
    print("")

    # predict_main(33, test_loader, test_target_tensor,
    #              metric_method, _mean, _std, mask_matrix, 'test')

