import gym
from world import World
from environment import TSCEnv
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
from agent.idqn_agent import IDQNAgent
from agent.sdqn_agent import SDQNAgent
from agent.max_pressure_agent import MaxPressureAgent
import argparse
import os
import numpy as np
import torch
import random
import logging
from datetime import datetime
import pickle as pkl
from utils.preparation import build_relation, get_road_adj, run_preparation, get_mask_matrix, one_hot
from predictionModel.SFM import SFM_predictor
from predictionModel.NN import NN_predictor
import rl_data_generation
from metric.maskMetric import MSEMetric

parser = argparse.ArgumentParser(description='DQN control test')
parser.add_argument('--config', type=str, default='hz4x4', help='network working on')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=10, help='training episodes')
parser.add_argument('--save_dir', type=str, default="model", help='directory in which model should be saved')
parser.add_argument('--state_dir', type=str, default="state", help='directory in which state and road file should be saved')
parser.add_argument('--log_dir', type=str, default='logging', help='directory in which logging information should be saved')
parser.add_argument('--replay_dir', type=str, default='replay', help='directory in which replay .txt should be saved')
parser.add_argument('--prefix', type=str, default="reward_test", help='root path of the logging folder')
parser.add_argument('--load_model', type=bool, default=False, help='directory from which to load model, None if not load')

args = parser.parse_args()

root_dir = os.path.join('data/output_data', args.prefix)
model_dir = os.path.join(root_dir, args.save_dir)
state_dir = os.path.join(root_dir, args.state_dir)
log_dir = os.path.join(root_dir, args.log_dir)
replay_dir = os.path.join(root_dir, args.replay_dir)

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

def create_agents(world, control, mask_pos):
    agents = []
    if control  == 'IDQN':
        for idx, inter in enumerate(world.intersections):
            action_space = gym.spaces.Discrete(len(inter.phases))
            agents.append(IDQNAgent(
            action_space,
            [
                LaneVehicleGenerator(
                    world, inter, ["lane_count"], in_only=True, average=None),
                IntersectionPhaseGenerator(world, inter, ["phase"], targets=["cur_phase"], negative=False),
            ],
            LaneVehicleGenerator(
                world, inter, ["lane_waiting_count"], in_only=True, average=None, negative=True),
            inter.id, idx
        ))
    elif control == 'SDQN':
        shared_model, shared_target_model, optimizer = None, None, None
        all_idx = list(range(len(world.intersections)))
        for idx, inter in enumerate(world.intersections):
            action_space = gym.spaces.Discrete(len(inter.phases))
            if shared_model is None:
                agents.append(SDQNAgent(action_space, 
                [
                    LaneVehicleGenerator(
                        world, inter, ["lane_count"], in_only=True, average=None),
                    IntersectionPhaseGenerator(world, inter, ["phase"], targets=["cur_phase"], negative=False),
                ],
                LaneVehicleGenerator(
                    world, inter, ["lane_waiting_count"], in_only=True, average=None, negative=True),
                inter.id, idx, all_idx
                ))
                shared_model, shared_target_model, optimizer = agents[-1].copy_model()
            else:
                agents.append(SDQNAgent(action_space, 
                [
                    LaneVehicleGenerator(
                        world, inter, ["lane_count"], in_only=True, average=None),
                    IntersectionPhaseGenerator(world, inter, ["phase"], targets=["cur_phase"], negative=False),
                ],
                LaneVehicleGenerator(
                    world, inter, ["lane_waiting_count"], in_only=True, average=None, negative=True),
                inter.id, idx, all_idx,
                shared_model, shared_target_model, optimizer
                ))
    else:
        raise RuntimeError(f'{control} is not implemented')
    return agents

def create_world(config_file):
    return World(config_file, thread_num=8)

def create_env(world, agents):
    return TSCEnv(world, agents, None)

def agents_train(env, agents, inference_net, infer, mask_pos, relation, mask_matrix, adj_matrix, reward_inference):

    if reward_inference == 'nn':
        reward_inference_model = NN_predictor(agents[0].ob_length, agents[0].action_space.n, 'cpu', model_dir)
        # nn model need training process
        generate_agents = rl_data_generation.create_preparation_agents(world)
        save_file = os.path.join(state_dir, 'raw_state_IDQN.pkl')
        if not os.path.isfile(save_file):
            info = rl_data_generation.idqn_train(env, generate_agents, state_dir, args.episodes, args.action_interval)
            with open(save_file, 'wb') as f:
                pkl.dump(info, f)
        else:
            with open(save_file, 'rb') as f:
                info = pkl.load(f)            
        dataset = rl_data_generation.generate_dataset(save_file)
        model_file = os.path.join(model_dir, "NN_inference.pt")
        if os.path.isfile(model_file):
             reward_inference_model.load_model()
        else:
            reward_inference_model.train(dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test'], 2)
        metric = MSEMetric('nn', world, mask_pos)
    elif reward_inference == 'sfm':
        reward_inference_model = SFM_predictor().make_model()
        metric = MSEMetric('sfm', world, mask_pos)
    else:
        metric = MSEMetric('state_diff', world, mask_pos)
    # for IDQN training
    total_decision_num = 0
    best_att = np.inf
    for e in range(args.episodes):
        # No replay during training
        last_obs, last_phases = list(zip(*env.reset()))
        last_obs = np.array(last_obs, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        episodes_rewards = [0 for i in agents]
        episodes_decision_num = 0
        i = 0
        if infer in ['sfm', 'net']:
            #Debug finished
            last_obs_true = last_obs
            last_obs_pred = inference_net.predict(last_obs_true, last_phases, relation, mask_pos, mask_matrix, adj_matrix)
            while i < 3600:
                if i % args.action_interval == 0:
                    actions = []
                    for agent_id, agent in enumerate(agents):
                        if total_decision_num > agent.learning_start:
                            # recovered phase
                            actions.append(agent.choose((last_obs_pred[agent_id], last_phases[agent_id])))
                        else:
                            actions.append(agent.sample())
                    rewards_list = []
                    for _ in range(args.action_interval):
                        # TODO: mask rewards
                        obs, rewards, dones, _ = env.step(actions)
                        i += 1
                        rewards_list.append(rewards)
                    rewards_true = np.mean(rewards_list, axis=0)

                    cur_obs_true, cur_phases = list(zip(*obs))
                    cur_obs_true = np.array(cur_obs_true, dtype=np.float32)
                    cur_phases = np.array(cur_phases, dtype=np.int8)
                    cur_obs_pred = inference_net.predict(cur_obs_true, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)

                    # rewards need mask here
                    if reward_inference == 'sfm':
                        rewards_pred = reward_inference_model.predict(rewards_true, None, relation, mask_pos, mask_matrix, adj_matrix)
                        rewards_pred = np.mean(rewards_pred, axis=1)
                        rewards_true = np.mean(rewards_true, axis=1)
                        rewards = rewards_pred
                        metric.update(rewards_pred, rewards_true)
                    elif reward_inference == 'state_diff':
                        # TODO: implementation of state_difference
                        rewards_true = np.mean(last_obs_true - cur_obs_true, axis=1)
                        rewards_pred = np.mean(last_obs_pred - cur_obs_pred, axis=1)
                        rewards = rewards_true.copy()
                        rewards[mask_pos] = rewards_pred[mask_pos]
                        metric.update(rewards_pred, rewards_true)
                    elif reward_inference == 'nn':
                        rewards_true = np.mean(rewards_true, axis=1)
                        rewards_pred = rewards_true.copy()
                        for pos in mask_pos:
                            tmp = reward_inference_model.predict(torch.from_numpy(np.concatenate((cur_obs_pred[pos], one_hot(cur_phases[pos], agents[pos].action_space.n)))))
                            rewards_pred[pos] = tmp
                        metric.update(rewards_pred, rewards_true)
                    else:
                        raise RuntimeError('reward inference not implemented')

                    for agent_id, agent in enumerate(agents):
                        agent.remember(
                            (last_obs_pred[agent_id], last_phases[agent_id]), actions[agent_id], rewards_pred[agent_id], (cur_obs_pred[agent_id], cur_phases[agent_id]))
                        episodes_rewards[agent_id] += rewards[agent_id]
                        episodes_decision_num += 1
                    total_decision_num += 1
                    last_obs_true = cur_obs_true
                    last_obs_pred = cur_obs_pred
                    last_phases = cur_phases
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                        agent.replay()
                    if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                        agent.update_target_network()
                if all(dones):
                    break
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        best_att = agents_train_test(e, env, agents, inference_net, infer, mask_pos, relation, mask_matrix, adj_matrix, best_att)
    logger.info(f"{metric.name}: {metric.get_result()}")
    logger.info('-----------------------------------------------------')

def agents_train_test(e, env, agents, inference_net, infer, mask_pos, relation, mask_matrix, adj_matrix, best_att):
    total_decision_num = 0
    cur_obs, cur_phases = list(zip(*env.reset()))
    cur_obs = np.array(cur_obs, dtype=np.float32)
    cur_phases = np.array(cur_phases, dtype=np.int8)
    i = 0
    if infer in ['sfm', 'net']:  
        cur_obs = inference_net.predict(cur_obs, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
        # No replay during testing
        while i < 3600:
            if i % args.action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(agents):
                    action = agent.get_action(cur_obs, cur_phases, relation)
                    actions.append(action)
                for _ in range(args.action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                
                cur_obs, cur_phases = list(zip(*obs))
                cur_obs = np.array(cur_obs, dtype=np.float32)
                cur_phases = np.array(cur_phases, dtype=np.int8)
                cur_obs = inference_net.predict(cur_obs, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
            if all(dones):
                break
    att = env.eng.get_average_travel_time()
    if att < best_att:
        best_att = att
        for ag in agents:
            ag.save_model(model_dir=os.path.join(model_dir, f'BEST_{control}'))
    logger.info("episode:{}, Test:{}".format(e, att))
    return best_att

def shared_agents_train(env, agents, inference_net, infer, control, mask_pos, relation, mask_matrix, adj_matrix, sample_method='all'):
    # for SDQN training, difference exists in create_agents() and agents.replay()
    if control == 'SDQN':
        zero_idx = 0 if sample_method == 'all' else min(set(range(len(env.world.intersections))) - set(mask_pos))
        update_idx = len(env.world.intersections) - 1 if sample_method == 'all' else\
             max(set(range(len(env.world.intersections))) - set(mask_pos))
    else:
        raise RuntimeError(f'control {control} not implemented')
    SDQNAgent.register_idx(zero_idx, update_idx)
    total_decision_num = 0
    best_att = np.inf
    for e in range(args.episodes):
        # No replay during training
        last_obs, last_phases = list(zip(*env.reset()))
        last_obs = np.array(last_obs, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        # only infer == 'sfm' or 'net' supports SDQN
        last_obs = inference_net.predict(last_obs, last_phases, relation, mask_pos, mask_matrix, adj_matrix)
        episodes_decision_num = 0
        episodes_rewards = [0 for i in agents]
        i = 0
        while i < 3600:
            if i % args.action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.learning_start:
                        # recovered phase
                        actions.append(agent.choose((last_obs[agent_id], last_phases[agent_id])))
                    else:
                        actions.append(agent.sample())
                rewards_list = []
                for _ in range(args.action_interval):
                    # TODO: mask rewards
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)
                #rewards need mask here
                if args.rewards == 'mask':
                    rewards = inference_net.predict(rewards, None, relation, mask_pos, mask_matrix, adj_matrix)
                    rewards = np.mean(rewards, axis=1)
                elif args.rewards == 'state_diff':
                    # TODO: implementation of state_difference
                    pass

                cur_obs, cur_phases = list(zip(*obs))
                cur_obs = np.array(cur_obs, dtype=np.float32)
                cur_phases = np.array(cur_phases, dtype=np.int8)
                cur_obs = inference_net.predict(cur_obs, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
                for agent_id, agent in enumerate(agents):
                    agent.remember(
                        (last_obs[agent_id], last_phases[agent_id]), actions[agent_id], rewards[agent_id], (cur_obs[agent_id], cur_phases[agent_id]))
                    episodes_rewards[agent_id] += rewards[agent_id]
                    episodes_decision_num += 1
                total_decision_num += 1
                last_obs = cur_obs
                last_phases = cur_phases
            # train SDQN or MaskSDQN
            if sample_method == 'all':
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                        agent.replay()
                    if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                        agent.update_target_network()
            # train SDQN
            elif sample_method == 'unmask':
                assert(control != 'MaskSDQN')
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1 and agent_id not in mask_pos:
                        agent.replay()
                    if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1 and agent_id not in mask_pos:
                        agent.update_target_network()
            else:
                raise RuntimeError(f'sample_method: {sample_method} not implemented')
            if all(dones):
                break
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        best_att = shared_agents_train_test(e, env, agents, inference_net, infer, control, mask_pos, relation, mask_matrix, adj_matrix, sample_method, best_att)
        logger.info('-----------------------------------------------------')

def shared_agents_train_test(e, env, agents, inference_net, infer, control, mask_pos, relation, mask_matrix, adj_matrix, sample_method, best_att):
    total_decision_num = 0
    cur_obs, cur_phases = list(zip(*env.reset()))
    cur_obs = np.array(cur_obs, dtype=np.float32)
    cur_phases = np.array(cur_phases, dtype=np.int8)
    if infer in ['sfm', 'net']:  
        cur_obs = inference_net.predict(cur_obs, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
        # No replay during testing
        i = 0
        while i < 3600:
            if i % args.action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(agents):
                    action = agent.get_action(cur_obs, cur_phases, relation)
                    actions.append(action)
                for _ in range(args.action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                cur_obs, cur_phases = list(zip(*obs))
                cur_obs = np.array(cur_obs, dtype=np.float32)
                cur_phases = np.array(cur_phases, dtype=np.int8)
                cur_obs = inference_net.predict(cur_obs, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
            if all(dones):
                break
    else:
        raise RuntimeError('inference type not implemented')
    att = env.eng.get_average_travel_time()
    if att < best_att:
        best_att = att
        for ag in agents:
            ag.save_model(model_dir=os.path.join(model_dir, f'BEST_{control}_{sample_method}'))
    logger.info("episode:{}, Test:{}".format(e, att))
    return best_att

if __name__ == '__main__':
    mask_pos = [[2, 0], [10, 11], [14, 13]]
    generate_choose = 'IDQN'
    infer_choose = ['sfm']  # ['net', 'sfm', 'no']
    control_choose = ['IDQN']  # ['maxpressure', 'CopyDQN', 'IDQN', 'SDQN', 'MaskSDQN']
    
    config_file = f'cityflow_{args.config}.cfg'
    configuration = f'configurations/{args.config}_astgcn.conf'
    graph_signal_matrix_filename = f'{state_dir}/state_{generate_choose}.pkl'

    agent_name = generate_choose
    world = create_world(config_file)
    # create relation
    relation = build_relation(world)
    reward_type = ['state_diff', 'nn', 'sfm']
    for mask in mask_pos:
        mask_pos = mask
        adj_matrix = get_road_adj(relation)   
        mask_matrix = get_mask_matrix(relation, mask_pos)
        logger.info('-------------------------------------')
        logger.info('-------------------------------------')
        logger.info(f'mask_pos = {mask_pos}')
        for infer in infer_choose:
            if infer == 'sfm':
                inference_net = SFM_predictor().make_model()

                for control in control_choose:
                    agents = create_agents(world, control, mask_pos)
                    env = create_env(world, agents)
                    # use inference model fill masked position then use IDQN control, train IDQN at all positions
                    if control == 'IDQN':
                        logger.info(f'infer = {infer}, control = {control}')
                        for reward_inference in reward_type:
                            agents_train(env, agents, inference_net, infer, mask_pos, relation, mask_matrix, adj_matrix, reward_inference)
                    # TODO: TEST REWARD INFERENCE ON SDQN
                    elif control == 'SDQN':
                        sample_method = 'all'
                        logger.info(f'infer = {infer}, control = {control}, sample_method = {sample_method}')
                        shared_agents_train(env, agents, inference_net, infer, control, mask_pos, relation, mask_matrix, adj_matrix, sample_method)
                        sample_method = 'unmask'
                        logger.info(f'infer = {infer}, control = {control}, sample_method = {sample_method}')
                        shared_agents_train(env, agents, inference_net, infer, control, mask_pos, relation, mask_matrix, adj_matrix, sample_method)

                    else:
                        raise RuntimeError(f'{infer} not implemented')

            else:
                raise RuntimeError('only implement sfm in reward test')
        logger.info('\n')
    print('experiments finished')