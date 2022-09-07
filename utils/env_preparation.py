import gym
from predictionModel.NN import NN_predictor
from world import World
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
from environment import TSCEnv

from agent.idqn_agent import IDQNAgent
from agent.fixedtime_agent import FixedTimeAgent

from utils.preparation import build_relation
from rl_data_generation import store_reshaped_data, generate_dataset

import argparse
import os
from tqdm import tqdm
import pickle as pkl
from datetime import datetime
import logging
import numpy as np
import torch


parser = argparse.ArgumentParser(description='IDQN - FixedTime generate dataset for reward inference model')
parser.add_argument('--config', type=str, default='hz4x4', help='network working on')
parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=10, help='training episodes')
parser.add_argument('--save_dir', type=str, default="model", help='directory in which model should be saved')
parser.add_argument('--state_dir', type=str, default="state", help='directory in which state and road file should be saved')
parser.add_argument('--log_dir', type=str, default='logging', help='directory in which logging information should be saved')
parser.add_argument('--infer', type=str, default='st', choices=['st', 'stp'], help='choose inference model\'s input')


def create_world(config_file):
    return World(config_file, thread_num=8)

def create_env(world, agents):
    return TSCEnv(world, agents, None)

def create_preparation_agents(world, mask_pos):
    agents = []
    for idx, i in enumerate(world.intersections):
        action_space = gym.spaces.Discrete(len(i.phases))
        if idx not in mask_pos:
            agents.append(IDQNAgent(
                action_space,
                [
                    LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
                    IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
                ],
                LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
                i.id, idx
            ))
        else:
            agents.append(FixedTimeAgent(
                action_space, 
                [
                LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None), 
                IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
                ],
                LaneVehicleGenerator( world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
                i, idx
            ))
    return agents


def naive_train(env, agents, episode, fixedtime=30, action_interval=10):
    # take in environment and generate data for inference net
    # save t, phase_t, rewards_tp, state_tp, phase_tp(action_t) into dictionary
    information = []
    total_decision_num = 0
    best_att = np.inf
    for e in tqdm(range(episode)):
        # collect [state_t, phase_t, reward_t, state_tp, phase_tp(action_t)] pair at observable intersections
        last_obs = env.reset()
        episodes_rewards = [0 for _ in agents]
        episodes_decision_num = 0
        last_observable = []
        observable = []
        rewards_observable = []
        i = 0
        while i < 3600:
            if i % action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.learning_start:
                        actions.append(agent.choose(last_obs[agent_id]))
                    else:
                        actions.append(agent.sample())
                rewards_list = []
                for _ in range(action_interval):
                    obs, rewards, _, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards_train = np.mean(rewards_list, axis=0)
                rewards = np.mean(rewards_train, axis=1)
                for agent_id, agent in enumerate(agents):
                    if agent.name == 'FixedTimeAgent':
                        pass
                    else:
                        agent.remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
                        episodes_rewards[agent_id] += rewards[agent_id]
                        episodes_decision_num += 1
                        last_observable.append(last_obs[agent_id])
                        observable.append(obs[agent_id])
                        # S_{t+1}-> R_t
                        rewards_observable.append(rewards_train[agent_id])
                total_decision_num += 1
                last_obs = obs

            for agent_id, agent in enumerate(agents):
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()
        store_reshaped_data(information, [last_observable, rewards_observable, observable])
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        best_att = naive_execute(env, agents, best_att, information, fixedtime, action_interval)
    logger.info('-' * 50)
    return information

def naive_execute(env, agents, best_att, information, fixedtime, action_interval):
    i = 0
    last_obs = env.reset()
    last_states, last_phases = list(zip(*last_obs))
    last_states = np.array(last_states, dtype=np.float32)
    last_phases = np.array(last_phases, dtype=np.int8)
    episodes_rewards = [0 for i in agents]
    last_observable = []
    observable = []
    rewards_observable = []
    while i < 3600:
        if i % action_interval == 0:
            actions = []
            for agent_id, agent in enumerate(agents):
                actions.append(agent.get_action(last_states, last_phases))
            rewards_list = []
            for _ in range(action_interval):
                obs, rewards, _, _ = env.step(actions)
                i += 1
                rewards_list.append(rewards)
            rewards_train = np.mean(rewards_list, axis=0)
            rewards =  np.mean(rewards_train, axis=1)

            for agent_id, agent in enumerate(agents):
                if agent.name == 'FixedTimeAgent':
                    pass
                else:
                    episodes_rewards[agent_id] += rewards[agent_id]
                    last_observable.append(last_obs[agent_id])
                    observable.append(obs[agent_id])
                    # S_{t+1}-> R_t
                    rewards_observable.append(rewards_train[agent_id])

            store_reshaped_data(info, [last_obs, rewards, obs])
            last_obs = obs
            last_states, last_phases = list(zip(*last_obs))
            last_states = np.array(last_states, dtype=np.float32)
            last_phases = np.array(last_phases, dtype=np.int8)
    store_reshaped_data(information, [last_observable, rewards_observable, observable])
    att = env.eng.get_average_travel_time()
    if att < best_att:
        best_att = att
    logger.info("episode:{}, Test:{}".format(e, att))
    return best_att


if __name__ == "__main__":
    args = parser.parse_args()
    config = args.config
    config_file = f'cityflow_{args.config}.cfg'
    action_interval = args.action_interval
    episode = args.episodes
    action_interval = args.action_interval
    save_dir = args.save_dir
    state_dir = args.state_dir

    # prepare working directory
    root_dir = os.path.join('data/output_data', 'test')
    model_dir = os.path.join(root_dir, args.save_dir, args.infer)
    state_dir = os.path.join(root_dir, args.state_dir)
    log_dir = os.path.join(root_dir, args.log_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(
    log_dir, datetime.now().strftime('%Y_%m_%d-%H_%M_%S') + ".log"))
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    mask_pos = [2, 0]

    world = create_world(config_file)
    relation = build_relation(world)

    gen_agents = create_preparation_agents(world, mask_pos)
    env = create_env(world, gen_agents)

    # environment preparation
    save_file = os.path.join(state_dir, f'state_{args.infer}_reward.pkl')
    if args.infer == 'stp':
        input_dim = 12
    elif args.infer == 'st':
        input_dim = 20
    else:
        raise RuntimeError('infer mapping not implemented')
    if not os.path.isfile(save_file):
        print('start test nn predictor \n')
        info = naive_train(env, gen_agents, 10, 30, 10)
        # save inference training raw data
        with open(save_file, 'wb') as f:
            pkl.dump(info, f)
    dataset = generate_dataset(save_file, 'stp')
    net = NN_predictor(input_dim, 8, 'cpu', model_dir)
    net.train(dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test'], 10)
    test = net.predict(torch.from_numpy(dataset['x_train'][11]).to('cpu'))
    
    print('finished')
