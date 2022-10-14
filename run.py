from predictionModel.NN import NN_predictor
from predictionModel.SFM import SFM_predictor

import pickle as pkl
from utils.preparation import build_relation, get_road_adj, get_mask_matrix, fork_config
from utils.agent_preparation import create_env, create_world, create_fixedtime_agents, create_preparation_agents, create_app1maxp_agents, create_independent_agents,\
    create_maxp_agents, create_shared_agents
from utils.data_generation import generate_reward_dataset
from utils.control import fixedtime_execute, app1_trans_train, app1maxp_train, app2_conc_train, app2_shared_train, naive_train, maxp_execute
from utils.mask_pos import random_mask
import argparse
import os
from datetime import datetime
import logging
import torch


REWARD_TYPE = 'NN_st'
SAVE_RATE = 10
EPOCHS = 50

# TODO: test on different reward impute(t or pt) first
# TODO: var = [Imputation/Agent/Control/prefix]
parser = argparse.ArgumentParser(description='IDQN - FixedTime generate dataset for reward inference model')
parser.add_argument('--config', type=str, default='hz4x4', help='network working on')

parser.add_argument('--action_interval', type=int, default=10, help='how often agent make decisions')
parser.add_argument('--fix_time', type=int, default=30, help='how often fixtime agent change phase')
parser.add_argument('--episodes', type=int, default=100, help='training episodes')

parser.add_argument('-impute', default='sfm')
parser.add_argument('-agent', default='DQN',choices=['DQN','FRAP'])
parser.add_argument('-control', default='I-F', choices=['F-F','I-F','I-M','M-M','S_S_A','S_S_O'])
parser.add_argument('--prefix', default='working', type=str)

parser.add_argument('--debug', action='store_true')
parser.add_argument('--mask_pos', default='4,7,12', type=str)


if __name__ == "__main__":
    # save replay if debug == True, default False
    args = parser.parse_args()
    config = args.config
    saveReplay = True if args.debug else False

    config_file = f'cityflow_{config}.cfg'
    action_interval = args.action_interval
    episodes = args.episodes

    # prepare working directory
    model_dir = 'model'
    state_dir = 'dataset'
    replay_dir = 'replay'
    log_dir = 'logging'
    param_dir = os.path.join(args.impute, args.agent, args.control).replace('\\','/')
    root_dir = os.path.join('data/output_data', args.config, args.prefix).replace('\\','/')
    cur_working_dir = os.path.join(root_dir, param_dir).replace('\\','/')
    model_dir = os.path.join(cur_working_dir, model_dir).replace('\\','/')
    state_dir = os.path.join(cur_working_dir, state_dir).replace('\\','/')
    log_dir = os.path.join(cur_working_dir, log_dir).replace('\\','/')
    replay_dir = os.path.join(cur_working_dir, replay_dir).replace('\\','/')

    reward_model_dir = os.path.join(root_dir, args.impute, args.agent, 'I-F', 'model').replace('\\','/')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(replay_dir):
        os.makedirs(replay_dir)       

    # working dir preparation finished, file preparation start
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(
    log_dir, datetime.now().strftime('%Y_%m_%d-%H_%M_%S') + ".log"))
    fh.setLevel(logging.DEBUG)
    #sh = logging.StreamHandler()
    #sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    #logger.addHandler(sh)
    save_reward_file = os.path.join(state_dir, f'state_reward.pkl').replace('\\','/')
    save_state_file = os.path.join(state_dir, f'state_phase.pkl').replace('\\','/')

    if saveReplay:
        config_file = fork_config(config_file, replay_dir)

    # create world and relationship of intersections
    world = create_world(config_file)
    relation = build_relation(world) # no need to reset this while creating new environment since we only need the relationship not object.
    if args.mask_pos == '':
        mask_pos=[]
        #mask_pos = random_mask(3, 'non_neighbor', relation)
    else:
        mask_pos = args.mask_pos.split(',')
        mask_pos = [int(i) for i in mask_pos]
    logger.info(f"mask_pos: {mask_pos}")

    if args.control == 'I-F':
        gen_agents = create_preparation_agents(world, mask_pos,time=args.fix_time,agent=args.agent)
        env = create_env(world, gen_agents)
        # environment preparation, in_dim == 20 [lanes:3 * roads:4 + phases:8] = 20
        input_dim = 20

        if not os.path.isfile(save_reward_file):
            print('start test nn predictor \n')
            info, raw_state = naive_train(logger, env, gen_agents, episodes, action_interval, save_rate=SAVE_RATE,agent_name=args.agent)
            # save inference training raw data
            with open(save_reward_file, 'wb') as f:
                pkl.dump(info, f)
            
            with open(save_state_file, 'wb') as f:
                pkl.dump(raw_state, f)

        reward_dataset = generate_reward_dataset(save_reward_file, 8, infer=REWARD_TYPE) # default setting infer == 'st'
        #state_dataset = generate_state_dataset()
        net = NN_predictor(input_dim, 1, 'cpu', model_dir) # generate reward inference model at model_dir
        net.train(reward_dataset['x_train'], reward_dataset['y_train'], reward_dataset['x_test'], reward_dataset['y_test'], epochs=EPOCHS)
        test = net.predict(torch.from_numpy(reward_dataset['x_train'][11]).to('cpu'))

    elif args.control =='F-F':
        agents = create_fixedtime_agents(world, time=args.fix_time)
        env = create_env(world, agents)
        fixedtime_execute(logger, env, agents, action_interval, relation)

    elif args.control == 'M-M':
        agents = create_maxp_agents(world)
        env = create_env(world, agents)
        state_inference_net = SFM_predictor()
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        maxp_execute(logger, env, agents, action_interval, state_inference_net, mask_pos, relation,
             mask_matrix, adj_matrix)

    elif args.control == 'I-M':
        agents = create_app1maxp_agents(world, mask_pos,agent=args.agent)
        env = create_env(world, agents)
        state_inference_net = SFM_predictor()
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        app1maxp_train(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, SAVE_RATE,agent_name=args.agent)

    elif args.control == 'S-S-O':
        agents = create_shared_agents(world, mask_pos,agent=args.agent)
        env = create_env(world, agents)
        state_inference_net = SFM_predictor()
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        app1_trans_train(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, SAVE_RATE,agent_name=args.agent)
    
    elif args.control == 'I-I':
        agents = create_independent_agents(world,agent=args.agent)
        env = create_env(world, agents)
        state_inference_net = SFM_predictor()
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        app2_conc_train(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, reward_model_dir, reward_type=REWARD_TYPE, save_rate=SAVE_RATE,agent_name=args.agent)

    elif args.control == 'S_S_A':
        agents = create_shared_agents(world, mask_pos,agent=args.agent)
        env = create_env(world, agents)
        state_inference_net = SFM_predictor()
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        app2_shared_train(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, reward_model_dir, reward_type=REWARD_TYPE,save_rate=SAVE_RATE,agent_name=args.agent)

'''
    elif args.control == 'FRAP-S-A':
        agents = create_shared_agents(world, mask_pos,agent=args.agent)
        env = create_env(world, agents)
        state_inference_net = SFM_predictor()
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        app2_fsa_train(logger, env, agents, episodes, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, reward_model_dir, reward_type=REWARD_TYPE,save_rate=SAVE_RATE)

'''
