import gym
from world import World
from environment import TSCEnv
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
from agent.idqn_agent import IDQNAgent

import argparse
import os
import numpy as np
import random
import pickle as pkl
from utils.preparation import one_hot
import torch
from tqdm import tqdm

from predictionModel.NN import NN_predictor

parser = argparse.ArgumentParser(description='IDQN generate dataset for inference model')
parser.add_argument('--config', type=str, default='hz4x4', help='network working on')
parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=10, help='training episodes')
parser.add_argument('--save_dir', type=str, default="model", help='directory in which model should be saved')
parser.add_argument('--state_dir', type=str, default="state", help='directory in which state and road file should be saved')


def create_preparation_agents(world):
    '''
    Use this function to create agent for later data generation, generated data will later be used in the inference model training
    '''
    agents = []
    for idx, i in enumerate(world.intersections):
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(IDQNAgent(
            action_space,
            [
                LaneVehicleGenerator(
                    world, i, ["lane_count"], in_only=True, average=None),
                IntersectionPhaseGenerator(world, i, ["phase"], targets=[
                                        "cur_phase"], negative=False),
            ],
            LaneVehicleGenerator(
                world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
            i.id, idx
        ))
    return agents

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
        reward = np.concatenate((reward, information[1][i][np.newaxis, :]), axis = 0)
        state_tp = np.concatenate((state_tp, information[2][i][0][np.newaxis, :]), axis = 0)
        phase_tp = np.concatenate((phase_tp, information[2][i][1][np.newaxis, :]), axis = 0)
    data.append([state_t, phase_t, reward, state_tp, phase_tp])

def create_world(config_file):
    return World(config_file, thread_num=8)

def create_env(world, agents):
    return TSCEnv(world, agents, None)

def idqn_train(env, agents, state_dir, episode, interval):
    # take in environment and generate data for inference net
    
    # save t, phase_t, rewards_tp, state_tp, phase_tp(action_t) into dictionary
    info = []
    total_decision_num = 0
    for e in tqdm(range(episode)):
        # collect [state_t, phase_t, reward_t, state_tp, phase_tp(action_t)] pair
        last_obs = env.reset()
        episodes_rewards = [0 for i in agents]
        episodes_decision_num = 0
        i = 0
        while i < 3600:
            if i % interval == 0:
                actions = []
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.learning_start:
                        actions.append(agent.choose(last_obs[agent_id]))
                    else:
                        actions.append(agent.sample())
                rewards_list = []
                for _ in range(interval):
                    obs, rewards, _, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)
                rewards_train = np.mean(rewards, axis=1)
                for agent_id, agent in enumerate(agents):
                    agent.remember(last_obs[agent_id], actions[agent_id], rewards_train[agent_id], obs[agent_id])
                    episodes_rewards[agent_id] += rewards[agent_id]
                    episodes_decision_num += 1
                total_decision_num += 1           
                store_reshaped_data(info, [last_obs, rewards, obs])
                last_obs = obs
            
            for agent_id, agent in enumerate(agents):
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()
        # test phase
        i = 0
        last_obs = env.reset()
        cur_obs, cur_phases = list(zip(*last_obs))
        cur_obs = np.array(cur_obs, dtype=np.float32)
        cur_phases = np.array(cur_phases, dtype=np.int8)
        episodes_rewards = [0 for i in agents]
        while i < 3600:
            if i % interval == 0:
                actions = []
                for agent_id, agent in enumerate(agents):
                    actions.append(agent.get_action(cur_obs, cur_phases))
                rewards_list = []
                for _ in range(interval):
                    obs, rewards, _, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)
                store_reshaped_data(info, [last_obs, rewards, obs])
                last_obs = obs
    return info

def generate_dataset(file, phases=8,  mod='intersection'):
    # prepare data for training inference model
    # data formation [N_samples, [state_t, phase_t, reward, state_tp, phase_tp]]
    with open(file, 'rb') as f:
        contents = pkl.load(f)
    if mod == 'intersection':
        # training sample [state_t, onehot(phase_t)], target [reward]
        feature = list()
        target = list()
        for sample in contents:
            feature_t = np.concatenate((sample[0], one_hot(sample[1], phases)), axis=1)
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


if __name__ == '__main__':
    args = parser.parse_args()
    config = args.config
    config_file = f'cityflow_{args.config}.cfg'
    interval = args.action_interval
    episode = args.episodes
    save_dir = args.save_dir
    state_dir = args.state_dir

    # prepare working directory
    root_dir = os.path.join('data/output_data', 'test')
    model_dir = os.path.join(root_dir, args.save_dir)
    state_dir = os.path.join(root_dir, args.state_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)

    # environment preparation
    world = create_world(config_file)
    agents = create_preparation_agents(world)
    env = create_env(world, agents)
    save_file = os.path.join(state_dir, 'raw_state_IDQN.pkl')
    if not os.path.isfile(save_file):
        info = idqn_train(env, agents, state_dir, episode, interval)
        # save inference training raw data
        with open(save_file, 'wb') as f:
            pkl.dump(info, f)
    dataset = generate_dataset(save_file)
    net = NN_predictor(20, 8, 'cpu', model_dir)
    net.train(dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test'], 10)
    test = net.predict(torch.from_numpy(dataset['x_train'][11]).to('cpu'))
    print('test')