from torch.cuda import is_available
from predictionModel.NN import NN_predictor
from predictionModel.SFM import SFMHetero_predictor

import pickle as pkl
from utils.preparation import fork_config, state_lane_convertor
from utils.agent_preparation import create_env, create_world, create_fixedtime_agents, create_preparation_agents_hetero, create_app1maxp_agents_hetero, create_shared_agents_hetero

from utils.data_generation import generate_reward_dataset_hetero
from utils.control import fixedtime_execute, naive_train_hetero, app1maxp_train_hetero, app1_trans_train_hetero, app2_shared_train_hetero

import argparse
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import torch
import os


REWARD_TYPE = 'NN_st'
SAVE_RATE = 1
EPOCHS = 50
IN_DIM = 12 + 12
# if torch.has_cuda:
#     DEVICE = torch.device('cuda')
# elif torch.has_mps:
#     DEVICE = torch.device('mps')
# else:
#     DEVICE = torch.device('cpu')
DEVICE = torch.device('cpu')

# TODO: test on different reward impute(t or pt) first
# TODO: var = [Imputation/Agent/Control/prefix]
parser = argparse.ArgumentParser(description='FRAP - FixedTime generate dataset for reward inference model')
parser.add_argument('--config', type=str, default='atlanta1x5', help='network working on')

parser.add_argument('--action_interval', type=int, default=10, help='how often agent make decisions')
parser.add_argument('--fix_time', type=int, default=40, help='how often fixtime agent change phase')
parser.add_argument('--episodes', type=int, default=100, help='training episodes')
parser.add_argument('--prefix', default='hetero', type=str)
parser.add_argument('--debug', action='store_true')

parser.add_argument('--mask_pos', default='3', type=str) # -1 if no mask position
parser.add_argument('-control', default='S-S-A', choices=['F-F','I-F','I-M','S-S-A','S-S-O'])


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
    param_dir = Path(args.control)
    root_dir = Path.joinpath(Path('data/output_data'), args.config, args.prefix, Path(args.mask_pos.replace(',', '_')))
    cur_working_dir = Path.joinpath(root_dir, param_dir)
    model_dir = Path.joinpath(cur_working_dir, model_dir)
    log_dir = Path.joinpath(cur_working_dir, log_dir)
    replay_dir = Path.joinpath(cur_working_dir, replay_dir)

    dataset_root = Path.joinpath(root_dir, 'I-F')
    state_dir = Path.joinpath(dataset_root, state_dir)
    reward_model_dir = Path.joinpath(dataset_root, 'model')

    if not Path.exists(model_dir):
        model_dir.mkdir(parents=True)
    if not Path.exists(state_dir):
        state_dir.mkdir(parents=True)
    if not Path.exists(log_dir):
        log_dir.mkdir(parents=True)
    if not Path.exists(replay_dir):
        replay_dir.mkdir(parents=True)

    # working dir preparation finished, file preparation start
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(Path.joinpath(
    log_dir, datetime.now().strftime('%Y_%m_%d-%H_%M_%S') + ".log"))
    fh.setLevel(logging.DEBUG)
    #sh = logging.StreamHandler()
    #sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    #logger.addHandler(sh)
    save_reward_file = Path.joinpath(state_dir, f'state_reward.pkl')

    if saveReplay:
        config_file = fork_config(config_file, str(replay_dir))

    # create world and relationship of intersections
    world = create_world(config_file)
    #relation = build_relation(world) # no need to reset this while creating new environment since we only need the relationship not object.
    if args.mask_pos == '-1':
        mask_pos=[]
    else:
        mask_pos = args.mask_pos.split(',')
        mask_pos = [int(i) for i in mask_pos]
    logger.info(f"mask_pos: {mask_pos}")

    if args.control == 'I-F':
        gen_agents = create_preparation_agents_hetero(world, mask_pos,time=args.fix_time, device=DEVICE)
        env = create_env(world, gen_agents)
        # environment preparation, in_dim == 20 [lanes:3 * roads:4 + phases:8] = 20

        if not Path.exists(save_reward_file):
            print('start test nn predictor \n')
            reward_info = naive_train_hetero(logger, env, gen_agents, episodes, action_interval, save_rate=SAVE_RATE)
            # save inference training raw data
            with open(save_reward_file, 'wb') as f:
                pkl.dump(reward_info, f)

        reward_dataset = generate_reward_dataset_hetero(save_reward_file) # default setting infer == 'st'
        #state_dataset = generate_state_dataset()
        net = NN_predictor(IN_DIM, 1, DEVICE, reward_model_dir, REWARD_TYPE) # generate reward inference model at model_dir
        if not net.is_mode():
            net.train(reward_dataset['x_train'], reward_dataset['y_train'], reward_dataset['x_test'], reward_dataset['y_test'], epochs=EPOCHS)
        else:
            net.load_model()

    elif args.control =='F-F':
        agents = create_fixedtime_agents(world, time=args.fix_time)
        env = create_env(world, agents)
        fixedtime_execute(logger, env, agents, action_interval)

    elif args.control == 'I-M':
        agents = create_app1maxp_agents_hetero(world, mask_pos, device=DEVICE)
        env = create_env(world, agents)

        converter = state_lane_convertor(agents, mask_pos)
        state_inference_net = SFMHetero_predictor(converter)
        app1maxp_train_hetero(logger, env, agents, episodes, action_interval, state_inference_net, SAVE_RATE)

    elif args.control == 'S-S-O':
        agents = create_shared_agents_hetero(world, mask_pos, device=DEVICE)
        env = create_env(world, agents)

        converter = state_lane_convertor(agents, mask_pos)
        state_inference_net = SFMHetero_predictor(converter)
        app1_trans_train_hetero(logger, env, agents, episodes, action_interval, state_inference_net, SAVE_RATE)

    elif args.control == 'S-S-A':
        agents = create_shared_agents_hetero(world, [], device=DEVICE)
        env = create_env(world, agents)

        converter = state_lane_convertor(agents, mask_pos)
        state_inference_net = SFMHetero_predictor(converter)
        reward_inference_net = NN_predictor(IN_DIM, 1, DEVICE, reward_model_dir, REWARD_TYPE)
        reward_inference_net.load_model()
        app2_shared_train_hetero(logger, env, agents, episodes, action_interval, state_inference_net, reward_inference_net, SAVE_RATE)
