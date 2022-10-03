import gym
from agent.max_pressure_agent import MaxPressureAgent
from agent.sdqn_agent import SDQNAgent, build_shared_model
from predictionModel.NN import NN_predictor
from predictionModel.SFM import SFM_predictor
from metric.maskMetric import MSEMetric
from world import World
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
from environment import TSCEnv

from agent.idqn_agent import IDQNAgent
from agent.fixedtime_agent import FixedTimeAgent

from utils.preparation import build_relation, get_road_adj, get_mask_matrix, one_hot
from rl_data_generation import store_reshaped_data, generate_dataset

import argparse
import os
from tqdm import tqdm
import pickle as pkl
from datetime import datetime
import logging
import numpy as np
import torch
import torch.optim as optim

# protocol: last_obs is returned from env, (last_stats, last_phases) is returned from imputation

REWARD_TYPE = 'NN_st'
SAVE_RATE = 10

parser = argparse.ArgumentParser(description='IDQN - FixedTime generate dataset for reward inference model')
parser.add_argument('--config', type=str, default='syn4x4', help='network working on')
parser.add_argument('--action_interval', type=int, default=10, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=100, help='training episodes')
parser.add_argument('--save_dir', type=str, default="model", help='directory in which model should be saved')
parser.add_argument('--state_dir', type=str, default="state", help='directory in which state and road file should be saved')
parser.add_argument('--log_dir', type=str, default='logging', help='directory in which logging information should be saved')
parser.add_argument('--infer', type=str, default='st', choices=['st', 'stp'], help='choose inference model\'s input')
parser.add_argument('-type', default='S-S-O', choices=['I-I', 'I-F', 'I-M','M-M','S-S-A','S-S-O', 'F-F'])
parser.add_argument('--prefix', default='test_fix')

def create_world(config_file):
    return World(config_file, thread_num=8)

def create_env(world, agents):
    return TSCEnv(world, agents, None)

def create_fixedtime_agents(world, time=30):
    agents = []
    for idx, i in enumerate(world.intersections):
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(FixedTimeAgent(
            action_space, 
            [
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None), 
            IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
            ],
            LaneVehicleGenerator( world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
            i.id, idx, time=time
        ))
    return agents

def fixedtime_execute(logger, env, agents, action_interval, relation):
    env.eng.set_save_replay(True)
    name = logger.handlers[0].baseFilename
    save_dir = name[name.index('/output_data'): name.index('/logging')]
    env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_0.txt"))
    logger.info(f"FixedTime - FixedTime control")
    i = 0
    obs = env.reset()
    states, phases = list(zip(*obs))
    states = np.array(states, dtype=np.float32)
    phases = np.array(phases, dtype=np.int8)
    for i in range(3600):
        if i % action_interval == 0:
            actions = []
            for ag in agents:
                action = ag.get_action(states, phases, relation)
                actions.append(action)
            for _ in range(action_interval):
                env.step(actions)
                i += 1
    att = env.eng.get_average_travel_time()
    logger.info(f'FixedTime time interval: {agents[0].time}')
    logger.info(f'FixedTime average travel time result: {att}')
    logger.info('-' * 50)
    return None


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
                i.id, idx
            ))
    return agents

def naive_train(logger, env, agents, episodes, action_interval):
    # take in environment and generate data for inference net
    # save t, phase_t, rewards_tp, state_tp, phase_tp(action_t) into dictionary
    logger.info(f" IDQN -  FixedTime control")
    information = []
    total_decision_num = 0
    best_att = np.inf
    for e in tqdm(range(episodes)):
        # collect [state_t, phase_t, reward_t, state_tp, phase_tp(action_t)] pair at observable intersections
        last_obs = env.reset()
        last_states, last_phases = list(zip(*last_obs))
        last_states = np.array(last_states, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
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
                        actions.append(agent.choose(last_states, last_phases))
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
                        # no imputation, use obs directly
                        agent.remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
                        episodes_rewards[agent_id] += rewards[agent_id]
                        episodes_decision_num += 1
                        last_observable.append(last_obs[agent_id])
                        observable.append(obs[agent_id])
                        # S_{t+1}-> R_t
                        rewards_observable.append(rewards_train[agent_id])
                total_decision_num += 1
                last_obs = obs
                last_states, last_phases = list(zip(*last_obs))
                last_states = np.array(last_states, dtype=np.float32)
                last_phases = np.array(last_phases, dtype=np.int8)

            for agent_id, agent in enumerate(agents):
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()
        store_reshaped_data(information, [last_observable, rewards_observable, observable])
        logger.info("episodes:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        best_att = naive_execute(logger, env, agents, e, best_att, information, action_interval)
    logger.info(f'naive average travel time result: {best_att}')
    logger.info('-' * 50)
    return information

def naive_execute(logger, env, agents, e, best_att, information, action_interval):
    if e % SAVE_RATE == SAVE_RATE - 1:
        env.eng.set_save_replay(True)
        name = logger.handlers[0].baseFilename
        save_dir = name[name.index('/output_data'): name.index('/logging')]
        env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_%s.txt" % e))
    else:
        env.eng.set_save_replay(False)
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

def create_maxp_agents(world):
    agents = []
    for idx, i in enumerate(world.intersections):
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(MaxPressureAgent(
            action_space,
            [
                LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
                IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
            ],
            LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
            i.id, idx
        ))
    return agents

def maxp_execute(logger, env, agents, action_interval, inference_net, mask_pos, relation, mask_matrix, adj_matrix):
    env.eng.set_save_replay(True)
    name = logger.handlers[0].baseFilename
    save_dir = name[name.index('/output_data'): name.index('/logging')]
    env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_0.txt"))
    logger.info(f"Max Pressure - Max Pressure control")
    i = 0
    record = MSEMetric('state mse', mask_pos)
    obs = env.reset()
    states, phases = list(zip(*obs))
    states = np.array(states, dtype=np.float32)
    phases = np.array(phases, dtype=np.int8)
    for i in range(3600):
        if i % action_interval == 0:
            # TODO: implement other State inference model later
            # SFM inference states
            actions = []
            recovered = inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
            record.add(phases, recovered)
            for ag in agents:
                action = ag.get_action(recovered, phases, relation)
                actions.append(action)
            for _ in range(action_interval):
                obs, _, _, _ = env.step(actions)
                i += 1
            states, phases = list(zip(*obs))
            states = np.array(states, dtype=np.float32)
            phases = np.array(phases, dtype=np.int8)
    att = env.eng.get_average_travel_time()
    mse = record.get_cur_result()
    record.update()

    logger.info(f'MaxPressure average travel time result: {att}')
    logger.info(f'Maxpressure MSE: {mse}')
    logger.info('-' * 50)
    return record

def create_app1maxp_agents(world, mask_pos):
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
            agents.append(MaxPressureAgent(
                action_space, 
                [
                LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None), 
                IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
                ],
                LaneVehicleGenerator( world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
                i.id, idx
            ))
    return agents

def app1maxp_train(logger, env, agents, episode, action_interval, inference_net, mask_pos, relation, mask_matrix, adj_matrix):
    logger.info(f" IDQN - Max Pressure control")
    total_decision_num = 0
    best_att = np.inf
    record = MSEMetric('state mse', mask_pos)
    for e in range(episode):
        # collect [state_t, phase_t, reward_t, state_tp, phase_tp(action_t)] pair at observable intersections
        last_obs = env.reset()
        last_states, last_phases = list(zip(*last_obs))
        last_states = np.array(last_states, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        episodes_rewards = [0 for _ in agents]
        i = 0
        episodes_decision_num = 0
        while i < 3600:
            if i % action_interval == 0:
                # TODO: implement other State inference model later
                # SFM inference states
                last_recovered = inference_net.predict(last_states, last_phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
                record.add(last_states, last_recovered)
                actions = []
                for agent_id, agent in enumerate(agents):
                    if agent.name == 'MaxPressureAgent':
                        action = agent.get_action(last_recovered, last_phases, relation)
                        actions.append(action)
                    else:
                        if total_decision_num > agent.learning_start:
                            actions.append(agent.choose(last_states, last_phases, relation))
                        else:
                            actions.append(agent.sample())
                rewards_list = []
                for _ in range(action_interval):
                    obs, rewards, _, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)
                rewards = np.mean(rewards, axis=1)
                for agent_id, agent in enumerate(agents):
                    if agent.name == 'MaxPressureAgent':
                        pass
                    else:
                        agent.remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id]) # no need to change obs
                        episodes_rewards[agent_id] += rewards[agent_id]
                        episodes_decision_num += 1
                total_decision_num += 1
                last_obs = obs
                last_states, last_phases = list(zip(*last_obs))
                last_states = np.array(last_states, dtype=np.float32)
                last_phases = np.array(last_phases, dtype=np.int32)
            for agent_id, agent in enumerate(agents):
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()
        cur_mse = record.get_cur_result()
        record.update()
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        logger.info("episode:{}, MSETrain:{}".format(e, cur_mse))
        best_att = app1maxp_execute(logger, env, agents, e, best_att, record, inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix)
    avg_mse = record.get_result()

    logger.info(f'approach 1: maxpressure average travel time result: {best_att}')
    logger.info(f'final mse is: {avg_mse}')
    logger.info('-' * 50)
    return record

def app1maxp_execute(logger, env, agents, e, best_att, record, inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix):
    if e % SAVE_RATE == SAVE_RATE - 1:
        env.eng.set_save_replay(True)
        name = logger.handlers[0].baseFilename
        save_dir = name[name.index('/output_data'): name.index('/logging')]
        env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_%s.txt" % e))
    else:
        env.eng.set_save_replay(False)
    i = 0
    last_obs = env.reset()
    last_states, last_phases = list(zip(*last_obs))
    last_states = np.array(last_states, dtype=np.float32)
    last_phases = np.array(last_phases, dtype=np.int8)
    while i < 3600:
        if i % action_interval == 0:
            # TODO: implement other State inference model later
            # SFM inference states
            last_recovered = inference_net.predict(last_states, last_phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
            record.add(last_states, last_recovered)
            actions = []
            for ag in agents:
                if ag.name == 'MaxPressureAgent':
                    action = ag.get_action(last_recovered, last_phases, relation)
                elif ag.name == 'IDQNAgent':
                    action = ag.get_action(last_states, last_phases)
                actions.append(action)
            for _ in range(action_interval):
                obs, _, _, _ = env.step(actions)
                i += 1
            
            last_obs = obs
            last_states, last_phases = list(zip(*last_obs))
            last_states = np.array(last_states, dtype=np.float32)
            last_phases = np.array(last_phases, dtype=np.int8)

    cur_mse = record.get_cur_result()
    record.update()
    att = env.eng.get_average_travel_time()
    if att < best_att:
        best_att = att
    logger.info("episode:{}, Test:{}".format(e, att))
    logger.info("episode:{}, MSETest:{}".format(e, cur_mse))
    return best_att


def create_sdqn_agents(world, mask_pos):
    agents = []
    obs_pos = list(set(range(len(world.intersections))) - set(mask_pos))
    iid = []
    ob_generator = []
    reward_generator = []
    for idx, inter in enumerate(world.intersections):
        ob_generator.append(
            [
                LaneVehicleGenerator(world, inter, ['lane_count'], in_only=True, average=None),
                IntersectionPhaseGenerator(world, inter, ["phase"], targets=['cur_phase'], negative=False)
            ])
        reward_generator.append(LaneVehicleGenerator(world, inter, ['lane_waiting_count'], in_only=True, average=None, negative=True))
        iid.append(inter.id)
    action_space = gym.spaces.Discrete(len(world.intersections[-1].phases))
    ob_length = ob_generator[0][0].ob_length + action_space.n
    q_model = build_shared_model(ob_length, action_space)
    target_q_model = build_shared_model(ob_length, action_space)
    optimizer = optim.RMSprop(q_model.parameters(), lr=0.001, alpha=0.9, centered=False, eps=1e-7)
    agents.append(SDQNAgent(action_space, ob_generator, reward_generator, iid, obs_pos, q_model, target_q_model, optimizer))
    return agents


def app1_trans_train(logger, env, agents, episode, action_interval, inference_net, mask_pos, relation, mask_matrix, adj_matrix):
    # this method is used in approach 1 ,transfer and approach 2, shared-parameter. 
    logger.info(f"SDQN - SDQN - O control") 
    total_decision_num = 0
    best_att = np.inf
    record = MSEMetric('state mse', mask_pos)
    for e in range(episode):
        last_obs = env.reset()
        last_states, last_phases = list(zip(*last_obs))
        last_states = np.array(last_states, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        # only recover state_t since we need this var to determine action t
        last_recovered = inference_net.predict(last_states, last_phases, relation, mask_pos, mask_matrix, adj_matrix)
        record.add(last_states, last_recovered)
        episodes_decision_num = 0
        episodes_rewards = [0 for _ in range(agents[0].sub_agents)]
        i = 0
        while i < 3600:
            if i % action_interval == 0:
                for ag in agents:
                    if total_decision_num > ag.learning_start:
                        actions = ag.choose(last_recovered, last_phases, relation)
                    else:
                        actions = ag.sample()
                rewards_list = []
                for _ in range(action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards_train = np.mean(rewards_list, axis=0)
                rewards = np.mean(rewards_train, axis=1)

                cur_states, cur_phases = list(zip(*obs))
                cur_states = np.array(cur_states, dtype=np.float32)
                cur_phases = np.array(cur_phases, dtype=np.int8)
                cur_recovered = inference_net.predict(cur_states, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
                record.add(cur_states, cur_recovered)
                for ag in (agents):
                    for idx in ag.idx:
                        ag.remember(
                            (last_recovered[idx], last_phases[idx]), actions[idx], rewards[idx], (cur_recovered[idx], cur_phases[idx]), idx)
                        episodes_rewards[idx] += rewards[idx]
                        episodes_decision_num += 1
                total_decision_num += agents[0].sub_agents
                last_obs = obs
                last_states = cur_states
                last_phases = cur_phases
                last_recovered = cur_recovered
            for ag in agents:
                # only use experiences at observable intersections
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_model_freq == ag.update_model_freq - 1:
                    ag.replay()
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_target_model_freq == ag.update_target_model_freq - 1:
                    ag.update_target_network()
        cur_mse = record.get_cur_result()
        record.update()
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        logger.info("episode:{}, MSETrain:{}".format(e, cur_mse))
        best_att = app1_trans_execute(logger, env, agents, e, best_att, record, inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix)
    avg_mse = record.get_result()
    logger.info(f'approach 1: transfer dqn average travel time result: {best_att}')
    logger.info(f'final mse is: {avg_mse}')
    logger.info('-' * 50)
    return record

def app1_trans_execute(logger, env, agents, e, best_att, record, inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix):
    if e % SAVE_RATE == SAVE_RATE - 1:
        env.eng.set_save_replay(True)
        name = logger.handlers[0].baseFilename
        save_dir = name[name.index('/output_data'): name.index('/logging')]
        env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_%s.txt" % e))
    else:
        env.eng.set_save_replay(False)
    i = 0
    obs = env.reset()
    states, phases = list(zip(*env.reset()))
    states = np.array(states, dtype = np.float32)
    phases = np.array(phases, dtype = np.int8)   
    while i < 3600:
        if i % action_interval == 0:
            # TODO: implement other State inference model later
            # SFM inference states
            recovered = inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
            record.add(states, recovered)
            for ag in agents:
                actions = ag.get_action(recovered, phases, relation)
            for _ in range(action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
            states, phases = list(zip(*obs))
            states = np.array(states, dtype = np.float32)
            phases = np.array(phases, dtype = np.int8)
    att = env.eng.get_average_travel_time()
    cur_mse = record.get_cur_result()
    record.update()
    if att < best_att:
        best_att = att
    logger.info("episode:{}, Test:{}".format(e, att))
    logger.info("episode:{}, MSETest:{}".format(e, cur_mse))
    return best_att

def create_idqn_agents(world):
    agents = []
    for idx, i in enumerate(world.intersections):
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(IDQNAgent(
            action_space,
            [
                LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
                IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
            ],
            LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
            i.id, idx
        ))
    return agents

def app2_conc_train(logger, env, agents, episode, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, model_dir, reward_type=REWARD_TYPE):
    logger.info(f"IDQN - IDQN control")
    logger.info(f"reward inference model: {reward_type}")
    if reward_type == 'SFM':
        reward_inference_net = SFM_predictor()
    elif reward_type == 'NN_st' or reward_type == 'NN_stp':
        reward_inference_net = NN_predictor(agents[0].ob_length, 1, 'cpu', model_dir)
        reward_inference_net.load_model()
    else:
        raise RuntimeError('not implemented yet')
    total_decision_num = 0
    best_att = np.inf
    record = MSEMetric('state mse', mask_pos)
    reward_record = MSEMetric('reward mse', mask_pos)
    for e in range(episode):
        last_obs = env.reset()
        last_states, last_phases = list(zip(*last_obs))
        last_states = np.array(last_states, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        last_recovered = state_inference_net.predict(last_states, last_phases, relation, mask_pos, mask_matrix, adj_matrix)
        record.add(last_states, last_recovered)
        episodes_decision_num = 0
        episodes_rewards = [0 for _ in agents]
        i = 0
        while i < 3600:
            if i % action_interval == 0:
                actions = []
                for ag in agents:
                    if total_decision_num > ag.learning_start:
                        action = ag.choose(last_recovered, last_phases)
                        actions.append(action)
                    else:
                        action = ag.sample()
                        actions.append(action)
                rewards_list = []
                for _ in range(action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards_train = np.mean(rewards_list, axis=0)
                rewards = np.mean(rewards_train, axis=1)
                if reward_type == 'SFM':
                    # TODO: check later 
                    rewards_predicted = reward_inference_net.predict(rewards_train, None, relation, mask_pos, mask_matrix, adj_matrix)
                    rewards_recovered = np.mean(rewards_predicted, axis=1)
                    reward_record.add(rewards, rewards_recovered)
                elif reward_type == 'NN_st':
                    rewards_recovered = rewards.copy()
                    for pos in mask_pos:
                        tmp = reward_inference_net.predict(torch.from_numpy(np.concatenate((last_recovered[pos], one_hot(last_phases[pos], agents[pos].action_space.n)))))
                        rewards_recovered[pos] = tmp
                    reward_record.add(rewards, rewards_recovered)
                elif reward_type == 'NN_stp':
                    # TODO: implement later
                    raise RuntimeError("not implemented")
                else:
                    raise RuntimeError("not implemented")
                cur_states, cur_phases = list(zip(*obs))
                cur_states = np.array(cur_states, dtype=np.float32)
                cur_phases = np.array(cur_phases, dtype=np.int8)
                cur_recovered = state_inference_net.predict(cur_states, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
                record.add(cur_states, cur_recovered)
                for agent_id, agent in enumerate(agents):
                    agent.remember(
                        (last_recovered[agent_id], last_phases[agent_id]), actions[agent_id], rewards_recovered[agent_id], (cur_recovered[agent_id], cur_phases[agent_id]))
                    episodes_rewards[agent_id] += rewards_recovered[agent_id]
                    episodes_decision_num += 1
                total_decision_num += 1
                last_obs = obs
                last_states = cur_states
                last_phases = cur_phases
                last_recovered = cur_recovered
            for agent_id, ag in enumerate(agents):
                # only use experiences at observable intersections
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_model_freq == ag.update_model_freq - 1:
                    ag.replay()
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_target_model_freq == ag.update_target_model_freq - 1:
                    ag.update_target_network()
        cur_mse = record.get_cur_result()
        reward_cur_mse = reward_record.get_cur_result()
        record.update()
        reward_record.update()
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        logger.info("episode:{}, MSETrain:{}".format(e, cur_mse))
        logger.info("episode:{}, Reward_MSETrain:{}".format(e, reward_cur_mse))        

        best_att = app1_trans_execute(logger, env, agents, e, best_att, record, state_inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix)
    avg_mse = record.get_result()
    reward_avg_mse = record.get_result()
    logger.info(f'approach 2: concurrent idqn average travel time result: {best_att}')
    logger.info(f'final mse is: {avg_mse}')
    logger.info(f'final reward mse is: {reward_avg_mse}')
    logger.info('-' * 50)
    return record


def app2_conc_execute(logger, env, agents, e, best_att, record, state_inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix):
    if e % SAVE_RATE == SAVE_RATE - 1:
        env.eng.set_save_replay(True)
        name = logger.handlers[0].baseFilename
        save_dir = name[name.index('/output_data'): name.index('/logging')]
        env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_%s.txt" % e))
    else:
        env.eng.set_save_replay(False)
    i = 0
    obs = env.reset()
    states, phases = list(zip(*obs))
    states = np.array(states, dtype = np.float32)
    phases = np.array(phases, dtype = np.int8)   
    while i < 3600:
        if i % action_interval == 0:
            # TODO: implement other State inference model later
            # SFM inference states
            recovered = state_inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
            record.add(states, recovered)
            actions = []
            for ag in agents:
                action = ag.get_action(recovered, phases, relation)
                actions.append(action)
            for _ in range(action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
            states, phases = list(zip(*obs))
            states = np.array(states, dtype = np.float32)
            phases = np.array(phases, dtype = np.int8)
    att = env.eng.get_average_travel_time()
    cur_mse = record.get_cur_result()
    record.update()
    if att < best_att:
        best_att = att
    logger.info("episode:{}, Test:{}".format(e, att))
    logger.info("episode:{}, MSETest:{}".format(e, cur_mse))
    return best_att

def app2_shared_train(logger, env, agents, episode, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, model_dir, reward_type=REWARD_TYPE):
    # this method is used in approach 1 ,transfer and approach 2, shared-parameter. 
    logger.info(f"SDQN - SDQN - A control")
    logger.info(f"reward inference model: {reward_type}")
    if reward_type == 'SFM':
        reward_inference_net = SFM_predictor()
    elif reward_type == 'NN_st' or reward_type == 'NN_stp':
        reward_inference_net = NN_predictor(agents[0].ob_length, 1, 'cpu', model_dir)
        reward_inference_net.load_model()
    else:
        raise RuntimeError('not implemented yet')

    total_decision_num = 0
    best_att = np.inf
    record = MSEMetric('state mse', mask_pos)
    reward_record = MSEMetric('reward mse', mask_pos)
    for e in range(episode):
        last_obs = env.reset()
        last_states, last_phases = list(zip(*last_obs))
        last_states = np.array(last_states, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        # only recover state_t since we need this var to determine action t
        last_recovered = state_inference_net.predict(last_states, last_phases, relation, mask_pos, mask_matrix, adj_matrix)
        record.add(last_states, last_recovered)
        episodes_decision_num = 0
        episodes_rewards = [0 for _ in range(agents[0].sub_agents)]
        i = 0
        while i < 3600:
            if i % action_interval == 0:
                for ag in agents:
                    if total_decision_num > ag.learning_start:
                        actions = ag.choose(last_recovered, last_phases, relation)
                    else:
                        actions = ag.sample()
                rewards_list = []
                for _ in range(action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards_train = np.mean(rewards_list, axis=0)
                rewards = np.mean(rewards_train, axis=1)
                if reward_type == 'SFM':
                    rewards_predicted = reward_inference_net.predict(rewards_train, None, relation, mask_pos, mask_matrix, adj_matrix)
                    rewards_recovered = np.mean(rewards_predicted, axis=1)
                    reward_record.add(rewards, rewards_recovered)
                elif reward_type == 'NN_st':
                    rewards_recovered = rewards.copy()
                    for pos in mask_pos:
                        tmp = reward_inference_net.predict(torch.from_numpy(np.concatenate((last_recovered[pos], one_hot(last_phases[pos], agents[0].action_space.n)))))
                        rewards_recovered[pos] = tmp
                    reward_record.add(rewards, rewards_recovered)
                elif reward_type == 'NN_stp':
                    # TODO: implement later
                    a = None
                else:
                    raise RuntimeError("not implemented")
                cur_states, cur_phases = list(zip(*obs))
                cur_states = np.array(cur_states, dtype=np.float32)
                cur_phases = np.array(cur_phases, dtype=np.int8)
                cur_recovered = state_inference_net.predict(cur_states, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
                record.add(cur_states, cur_recovered)
                for ag in agents:
                    for idx in ag.idx:
                        ag.remember(
                            (last_recovered[idx], last_phases[idx]), actions[idx], rewards_recovered[idx], (cur_recovered[idx], cur_phases[idx]), idx)
                        episodes_rewards[idx] += rewards[idx]
                        episodes_decision_num += 1
                total_decision_num += agents[0].sub_agents
                last_obs = obs
                last_states = cur_states
                last_phases = cur_phases
                last_recovered = cur_recovered
            for ag in agents:
                # only use experiences at observable intersections
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_model_freq == ag.update_model_freq - 1:
                    ag.replay()
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_target_model_freq == ag.update_target_model_freq - 1:
                    ag.update_target_network()
        cur_mse = record.get_cur_result()
        reward_cur_mse = reward_record.get_cur_result()
        record.update()
        reward_record.update()
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        logger.info("episode:{}, MSETrain:{}".format(e, cur_mse))
        logger.info("episode:{}, Reward_MSETrain:{}".format(e, reward_cur_mse))
        best_att = app2_shared_execute(logger, env, agents, e, best_att, record, state_inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix)
    avg_mse = record.get_result()
    logger.info(f'approach 2: shared dqn average travel time result: {best_att}')
    logger.info(f'final mse is: {avg_mse}')
    logger.info('-' * 50)
    return record

def app2_shared_execute(logger, env, agents, e, best_att, record, state_inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix):
    if e % SAVE_RATE == SAVE_RATE - 1:
        env.eng.set_save_replay(True)
        name = logger.handlers[0].baseFilename
        save_dir = name[name.index('/output_data'): name.index('/logging')]
        env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_%s.txt" % e))
    else:
        env.eng.set_save_replay(False)
    i = 0
    obs = env.reset()
    states, phases = list(zip(*env.reset()))
    states = np.array(states, dtype = np.float32)
    phases = np.array(phases, dtype = np.int8)   
    while i < 3600:
        if i % action_interval == 0:
            # TODO: implement other State inference model later
            # SFM inference states
            recovered = state_inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
            record.add(states, recovered)
            actions = []
            for ag in agents:
                actions = ag.get_action(recovered, phases, relation)
            for _ in range(action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
            states, phases = list(zip(*obs))
            states = np.array(states, dtype = np.float32)
            phases = np.array(phases, dtype = np.int8)
    att = env.eng.get_average_travel_time()
    cur_mse = record.get_cur_result()
    record.update()
    if att < best_att:
        best_att = att
    logger.info("episode:{}, Test:{}".format(e, att))
    logger.info("episode:{}, MSETest:{}".format(e, cur_mse))
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
    test_type = args.type

    # prepare working directory
    root_dir = os.path.join('data/output_data', f'{config}_test', args.prefix)
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
    #sh = logging.StreamHandler()
    #sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    #logger.addHandler(sh)
    save_file = os.path.join(state_dir, f'state_{args.infer}_reward.pkl')

    #mask_pos = [2, 0]

    logger.info(f"mask_pos: {mask_pos}")
    world = create_world(config_file)
    relation = build_relation(world)

    if test_type == 'I-F':
        gen_agents = create_preparation_agents(world, mask_pos)
        env = create_env(world, gen_agents)
        # environment preparation
        if args.infer == 'stp':
            input_dim = 12
        elif args.infer == 'st':
            input_dim = 20
        else:
            raise RuntimeError('infer mapping not implemented')
        if not os.path.isfile(save_file):
            print('start test nn predictor \n')
            info = naive_train(logger, env, gen_agents, 100, 10)
            # save inference training raw data
            with open(save_file, 'wb') as f:
                pkl.dump(info, f)
        dataset = generate_dataset(save_file, 8, args.infer)
        net = NN_predictor(input_dim, 1, 'cpu', model_dir)
        net.train(dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test'], 50)
        test = net.predict(torch.from_numpy(dataset['x_train'][11]).to('cpu'))


    elif test_type == 'M-M':
        agents = create_maxp_agents(world)
        env = create_env(world, agents)
        state_inference_net = SFM_predictor()
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        maxp_execute(logger, env, agents, action_interval, state_inference_net, mask_pos, relation,
             mask_matrix, adj_matrix)

    elif test_type == 'I-M':
        agents = create_app1maxp_agents(world, mask_pos)
        env = create_env(world, agents)
        state_inference_net = SFM_predictor()
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        app1maxp_train(logger, env, agents, episode, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix)

    elif test_type == 'S-S-O':
        agents = create_sdqn_agents(world, mask_pos)
        env = create_env(world, agents)
        state_inference_net = SFM_predictor()
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        app1_trans_train(logger, env, agents, episode, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix)
    
    elif test_type == 'I-I':
        agents = create_idqn_agents(world)
        env = create_env(world, agents)
        state_inference_net = SFM_predictor()
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        app2_conc_train(logger, env, agents, episode, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, model_dir)
        
    elif test_type == 'S-S-A':
        agents = create_sdqn_agents(world, mask_pos=[])
        env = create_env(world, agents)
        state_inference_net = SFM_predictor()
        adj_matrix = get_road_adj(relation)
        mask_matrix = get_mask_matrix(relation, mask_pos)
        app2_shared_train(logger, env, agents, episode, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, model_dir)
    elif test_type == 'F-F':
        agents = create_fixedtime_agents(world, time=10)
        env = create_env(world, agents)
        fixedtime_execute(logger, relation)
        agents = create_fixedtime_agents(world, time=20)
        env = create_env(world, agents)
        fixedtime_execute(logger, relation)
        agents = create_fixedtime_agents(world, time=30)
        env = create_env(world, agents)
        fixedtime_execute(logger, relation)
        agents = create_fixedtime_agents(world, time=40)
        env = create_env(world, agents)
        fixedtime_execute(logger, relation)
        agents = create_fixedtime_agents(world, time=50)
        env = create_env(world, agents)
        fixedtime_execute(logger, relation)
        agents = create_fixedtime_agents(world, time=60)
        env = create_env(world, agents)
        fixedtime_execute(logger, relation)
                


    print('finished')
