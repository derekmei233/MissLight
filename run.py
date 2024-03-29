from torch.cuda import is_available
from predictionModel.NN import NN_predictor
from predictionModel.SFM import SFM_predictor
from predictionModel.GraphWN import GraphWN_predictor

import pickle as pkl
from utils.preparation import build_relation, get_road_adj, get_mask_matrix, fork_config
from utils.agent_preparation import create_env, create_world, create_fixedtime_agents, create_preparation_agents, create_app1maxp_agents, create_independent_agents, create_maxp_agents, create_shared_agents, create_model_based_agents
from utils.model_utils import get_road_adj_phase
from utils.data_generation import generate_reward_dataset, build_road_state, generate_state_dataset
from utils.control import fixedtime_execute, app1_trans_train, app1maxp_train, app2_conc_train, app2_shared_train, model_based_shared_train, naive_train, maxp_execute, model_based_shared_train, app2_shared_train_v2, app1_trans_train_v2, model_based_shared_train_v2
from utils.mask_pos import random_mask
import argparse
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import torch


REWARD_TYPE = 'NN_st'
SAVE_RATE = 5
EPOCHS = 50
HISTORY_LENGTH = 12 # GraphWN should change block and layer accordingly
IN_DIM = {'NN_st': 20, 'NN_stp': 12, 'NN_sta': 20}
if torch.has_cuda:
    INFER_DEVICE = torch.device('cuda')
elif torch.has_mps:
    INFER_DEVICE = torch.device('mps')
else:
    INFER_DEVICE = torch.device('cpu')
DEVICE = torch.device('cpu')


# mps may be problematic in training GraphWN.

# TODO: test on different reward impute(t or pt) first
# TODO: var = [Imputation/Agent/Control/prefix]

parser = argparse.ArgumentParser(description='IDQN - FixedTime generate dataset for reward inference model')
parser.add_argument('--config', type=str, default='syn4x4', help='network working on')

parser.add_argument('--action_interval', type=int, default=10, help='how often agent make decisions')
parser.add_argument('--fix_time', type=int, default=40, help='how often fixtime agent change phase')
parser.add_argument('--episodes', type=int, default=10, help='training episodes')

parser.add_argument('-impute', default='sfm', choices=['sfm', 'gwn'])
parser.add_argument('-agent', default='DQN',choices=['DQN','FRAP'], help='test on flexible agents FRAP')
parser.add_argument('-control', default='I-F', choices=['F-F','I-F','I-M','M-M','S-S-A','S-S-O', 'I-I', 'S-S-O-model_based'])
parser.add_argument('--prefix', default='1,7,15', type=str, help='')

parser.add_argument('--debug', action='store_true')
parser.add_argument('--mask_pos', default='1,7,15', type=str)


if __name__ == "__main__":
    # Prepare configuration for this run
    args = parser.parse_args()
    config = args.config
    saveReplay = True if args.debug else False
    config_file = f'configs/cityflow_{config}.cfg'
    action_interval = args.action_interval
    episodes = args.episodes
    fixed = args.fix_time
    agent_type = args.agent
    prefix = args.prefix
    control = args.control
    imputation = args.impute

    # Working directory preparation start here
    model_dir = 'model'
    state_dir = 'dataset'
    replay_dir = 'replay'
    log_dir = 'logging'
    param_dir = Path.joinpath(Path(imputation), agent_type, control)
    root_dir = Path.joinpath(Path('data/output_data'), config, prefix)
    cur_working_dir = Path.joinpath(root_dir, param_dir)
    model_dir = Path.joinpath(cur_working_dir, model_dir)
    log_dir = Path.joinpath(cur_working_dir, log_dir)
    replay_dir = Path.joinpath(cur_working_dir, replay_dir)

    # Saving simulation data of I-F and stored for later use by other control method under same setting
    dataset_root = Path.joinpath(root_dir, 'sfm', agent_type, 'I-F')
    state_dir = Path.joinpath(dataset_root, state_dir)
    reward_model_dir = Path.joinpath(dataset_root, 'model')
    state_model_dir = Path.joinpath(dataset_root, 'model')
    if not Path.exists(model_dir):
        model_dir.mkdir(parents=True)
    if not Path.exists(state_dir):
        state_dir.mkdir(parents=True)
    if not Path.exists(log_dir):
        log_dir.mkdir(parents=True)
    if not Path.exists(replay_dir):
        replay_dir.mkdir(parents=True)

    # File preparation start here
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(Path.joinpath(
    log_dir, datetime.now().strftime('%Y_%m_%d-%H_%M_%S') + ".log"))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    save_reward_file = Path.joinpath(state_dir, f'state_reward.pkl')
    save_state_file = Path.joinpath(state_dir, f'state_phase.pkl')
    save_state_dataset = Path.joinpath(state_dir, f'state_dataset.npy')
    graphwn_dataset = Path.joinpath(state_dir, f'masked_graphwn_dataset.pkl')
    if saveReplay:
        config_file = fork_config(config_file, str(replay_dir))

    # Create world configuration
    world = create_world(config_file)
    relation = build_relation(world) # Create world to extract intersections relationship. 

    # Pick missing positions
    if args.mask_pos == '':
        mask_pos=[]
    else:
        mask_pos = args.mask_pos.split(',')
        mask_pos = [int(i) for i in mask_pos]

    logger.info(f"mask_pos: {mask_pos}")

    if control =='F-F':
        # Conventional control method 1: Unobs - Fixedtime, Obs - Fixedtime
        agents = create_fixedtime_agents(world, time=fixed)
        env = create_env(world, agents)
        fixedtime_execute(logger, env, agents, action_interval)

    elif control == 'I-F':
        # Conventional control method 2 naive appraoch: Unobs - Fixedtime, Obs - IDQN
        gen_agents = create_preparation_agents(world, mask_pos, fixed, agent_type, DEVICE)
        env = create_env(world, gen_agents)

        if not Path.exists(save_reward_file):
            print('start test nn predictor \n')
            # Generate state reward pairs for later state model (GraphWN) and reward model (Neural Network) training
            reward_info, raw_state = naive_train(logger, env, gen_agents, episodes, action_interval, SAVE_RATE, agent_type)
            # save reward training raw data
            with open(save_reward_file, 'wb') as f:
                pkl.dump(reward_info, f)
            # save state training data
            with open(save_state_file, 'wb') as f:
                pkl.dump(raw_state, f)
        if not Path.exists(save_state_dataset):
            with open(save_state_file, 'rb') as f:
                c = pkl.load(f)
            raw_state = c 
            state_info = build_road_state(raw_state, relation, mask_pos) # save road level data and mask information
            with open(save_state_dataset, 'wb') as f:
                np.save(f, state_info['road_feature'])
                np.save(f, state_info['road_update'])
                np.save(f, state_info['adj_road'])
        reward_dataset = generate_reward_dataset(save_reward_file, 8, infer=REWARD_TYPE) # default setting infer == 'st'
        # Train reward inference model
        net = NN_predictor(IN_DIM[REWARD_TYPE], 1, DEVICE, reward_model_dir, REWARD_TYPE) # generate reward inference model at model_dir
        if not net.is_model():
            net.train(reward_dataset['x_train'], reward_dataset['y_train'], reward_dataset['x_test'], reward_dataset['y_test'], epochs=EPOCHS)
        else:
            net.load_model()
        if imputation == "gwn":
            # TODO: clean gwn branch in next version
            adj_matrix, phase_adj = get_road_adj_phase(relation)
            data = generate_state_dataset(save_state_dataset, history_t=HISTORY_LENGTH, pattern='select')
            with open(graphwn_dataset, 'wb') as f:
                pkl.dump(data, f)
            N = data['adj_road'].shape[0]
            state_net = GraphWN_predictor(N, data['node_update'], adj_matrix, data['stats'], 11, 3, INFER_DEVICE, state_model_dir)
            if not state_net.is_model():
                state_net.train(data['train']['x'], data['train']['target'], data['val']['x'], data['val']['target'], EPOCHS) # TODO: 3 for debug

    elif control == 'M-M':
        # MaxPressure control test
        agents = create_maxp_agents(world)
        env = create_env(world, agents)
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        state_inference_net = SFM_predictor(mask_matrix, adj_matrix, 'select')
        maxp_execute(logger, env, agents, action_interval, state_inference_net, mask_pos, relation,
             mask_matrix, adj_matrix)

    elif control == 'I-M':
        agents = create_app1maxp_agents(world, mask_pos, agent_type, DEVICE)
        env = create_env(world, agents)
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        state_inference_net = SFM_predictor(mask_matrix, adj_matrix, 'select')
        app1maxp_train(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, SAVE_RATE, agent_type)

    elif control == 'S-S-O':
        agents = create_shared_agents(world, mask_pos, agent_type, DEVICE)
        env = create_env(world, agents)
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        if imputation == 'sfm':
            state_inference_net = SFM_predictor(mask_matrix, adj_matrix, 'select')
            app1_trans_train_v2(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, DEVICE, SAVE_RATE, agent_type, HISTORY_LENGTH)
        elif imputation == 'gwn':
            # TODO: clean gwn branch in next version
            data = generate_state_dataset(save_state_dataset, history_t=HISTORY_LENGTH, pattern='select')
            with open(graphwn_dataset, 'rb') as f:
                data = pkl.load(f)
            N = data['adj_road'].shape[0]
            state_inference_net = GraphWN_predictor(N, data['node_update'], adj_matrix, data['stats'], 11, 3, DEVICE, state_model_dir)
            state_inference_net.load_model()
            app1_trans_train_v2(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, DEVICE, SAVE_RATE, agent_type, t_history=HISTORY_LENGTH)

    elif control == 'I-I':
        agents = create_independent_agents(world, agent_type, DEVICE)
        env = create_env(world, agents)
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        if imputation == 'sfm':
            state_inference_net = SFM_predictor(mask_matrix, adj_matrix, 'select')
            app2_conc_train(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, reward_model_dir, REWARD_TYPE, DEVICE, SAVE_RATE, agent_type)
        else:
            # TODO: check graphwn in next version
            data = generate_state_dataset(save_state_dataset, history_t=HISTORY_LENGTH, pattern='select')
            with open(graphwn_dataset, 'rb') as f:
                data = pkl.load(f)
            N = data['adj_road'].shape[0]
            state_inference_net = GraphWN_predictor(N, data['node_update'], adj_matrix, data['stats'], 11, 3, DEVICE, state_model_dir)
            state_inference_net.load_model()
            app2_conc_train(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, reward_model_dir, REWARD_TYPE, DEVICE, SAVE_RATE, agent_type)

    elif control == 'S-S-A':
        agents = create_shared_agents(world, [], agent_type, DEVICE)
        env = create_env(world, agents)
        adj_matrix, phase_adj = get_road_adj_phase(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        if imputation == 'sfm':
            state_inference_net = SFM_predictor(mask_matrix, adj_matrix, 'select')
            data = generate_state_dataset(save_state_dataset, history_t=HISTORY_LENGTH, pattern='select')
            app2_shared_train_v2(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, reward_model_dir, REWARD_TYPE, DEVICE, SAVE_RATE, agent_type, HISTORY_LENGTH)
        elif imputation == 'gwn':
            # TODO: check graphwn in next version
            data = generate_state_dataset(save_state_dataset, history_t=HISTORY_LENGTH, pattern='select')
            with open(graphwn_dataset, 'rb') as f:
                data = pkl.load(f)
            N = data['adj_road'].shape[0]
            state_inference_net = GraphWN_predictor(N, data['node_update'], adj_matrix, data['stats'], 11, 3, DEVICE, state_model_dir)
            state_inference_net.load_model()
            app2_shared_train_v2(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, reward_model_dir, REWARD_TYPE, DEVICE, SAVE_RATE, agent_type, HISTORY_LENGTH)

    elif control == 'S-S-O-model_based':
        agents = create_model_based_agents(world, mask_pos, DEVICE, agent_type)
        env = create_env(world, agents)
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        if imputation == 'sfm':
            state_inference_net = SFM_predictor(mask_matrix, adj_matrix, 'select')
            model_based_shared_train_v2(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, reward_model_dir, REWARD_TYPE, DEVICE, SAVE_RATE, agent_type, HISTORY_LENGTH, update_times=10)
        elif imputation == 'gwn':
            # TODO: check graphwn in next version
            dj_matrix, phase_adj = get_road_adj_phase(relation)
            data = generate_state_dataset(save_state_dataset, history_t=HISTORY_LENGTH, pattern='select')
            with open(graphwn_dataset, 'rb') as f:
                data = pkl.load(f)
            N = data['adj_road'].shape[0]
            state_inference_net = GraphWN_predictor(N, data['node_update'], adj_matrix, data['stats'], 11, 3, DEVICE, state_model_dir)
            state_inference_net.load_model()
            model_based_shared_train_v2(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, reward_model_dir, REWARD_TYPE, DEVICE, SAVE_RATE, agent_type, HISTORY_LENGTH, update_times=10)
