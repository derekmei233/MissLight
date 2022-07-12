# TODO: recall how to generate and convert graph data from node based to edge based
# TODO: provide dqn agent interface for sequential data 
# TODO: provide model training and loading interface to get model trained on fully observation

import gym
from world import World
from environment import TSCEnv
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
from agent.dqn_agent_torch import DQNAgent
from agent.max_pressure_agent import MaxPressureAgent
import argparse
import os
import numpy as np
import random
import logging
from datetime import datetime
import pickle
from utils.preparation import build_relation, get_road_adj, run_preparation, get_mask_matrix
from predictionModel.SFM import SFM_predictor


parser = argparse.ArgumentParser(description='DQN control test')
parser.add_argument('--config', type=str, default='hz4x4', help='network working on')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=100, help='training episodes')
parser.add_argument('--save_dir', type=str, default="model", help='directory in which model should be saved')
parser.add_argument('--state_dir', type=str, default="state", help='directory in which state and road file should be saved')
parser.add_argument('--log_dir', type=str, default='logging', help='directory in which logging information should be saved')
parser.add_argument('--prefix', type=str, default="0", help='root path of the logging folder')
parser.add_argument('--load_model', type=bool, default=True, help='directory from which to load model, None if not load')

args = parser.parse_args()

root_dir = os.path.join('data/output_data', args.prefix)
model_dir = os.path.join(root_dir, args.save_dir)
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


def create_preparation_agents(world):
    agents = []
    for idx, i in enumerate(world.intersections):
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(DQNAgent(
            action_space,
            [
                LaneVehicleGenerator(
                    world, i, ["lane_count"], in_only=True, average=None),
                IntersectionPhaseGenerator(world, i, ["phase"], targets=[
                                        "cur_phase"], negative=False),
            ],
            LaneVehicleGenerator(
                world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
            i.id, idx
        ))
    return agents

def create_agents(world, control):
    agents = []
    if control == 'maxpressure':
        for i in world.intersections:
            action_space = gym.spaces.Discrete(len(i.phases))
            agents.append(MaxPressureAgent(
                action_space,
                i,
                world,
                [
                    LaneVehicleGenerator(
                        world, i, ["lane_count"], in_only=True, average=None),
                    IntersectionPhaseGenerator(world, i, ["phase"], targets=[
                        "cur_phase"], negative=False),
                ],
            ))
    elif control == 'CopyDQN' or 'IDQN':
        for idx, i in enumerate(world.intersections):
            action_space = gym.spaces.Discrete(len(i.phases))
            agents.append(DQNAgent(
            action_space,
            [
                LaneVehicleGenerator(
                    world, i, ["lane_count"], in_only=True, average=None),
                IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
            ],
            LaneVehicleGenerator(
                world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
            i.id, idx
        ))
    
    else:
        raise RuntimeError(f'{control} is not implemented')
    return agents

def create_world(config_file):
    return World(config_file, thread_num=8)

def create_env(world, agents):
    return TSCEnv(world, agents, None)

def dqn_train(env, agents):
    total_decision_num = 0
    for e in range(100):
        last_obs = env.reset()
        # No replay during training
        episodes_rewards = [0 for i in agents]
        episodes_decision_num = 0
        i = 0
        while i < 3600:
            if i % args.action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.learning_start:
                        actions.append(agent.choose(last_obs[agent_id]))
                    else:
                        actions.append(agent.sample())
                rewards_list = []
                for _ in range(args.action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)

                for agent_id, agent in enumerate(agents):
                    agent.remember(
                        last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
                    episodes_rewards[agent_id] += rewards[agent_id]
                    episodes_decision_num += 1
                total_decision_num += 1

                last_obs = obs

            for agent_id, agent in enumerate(agents):
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()

            if all(dones):
                break
        print("episode:{}, average travel time:{}".format(e, env.eng.get_average_travel_time()))
    logger.info("generate model final episode:{}, average travel time:{}".format(e, env.eng.get_average_travel_time()))
    logger.info('----------------------------------------------------')
    logger.info('\n')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    for agent in agents:
        agent.save_model(model_dir, -1)
    return agents

def dqn_generate(env, agents, raw_state):
    save_state = []
    obs = env.reset()
    for agent in agents:
        agent.load_model(model_dir, -1)
    for i in range(args.steps):
        if i % args.action_interval == 0:
            actions = []
            for agent_id, agent in enumerate(agents):
                actions.append(agent.get_action(obs[agent_id]))
            save_state.append(obs)
        obs, rewards, dones, info = env.step(actions)
        
        if all(dones):
            break
    logger.info("Final Travel Time is %.4f" % env.eng.get_average_travel_time())
    raw_state_name  = f'{state_dir}/{raw_state}.pkl'
    with open(raw_state_name, 'wb') as fo:
        pickle.dump(save_state, fo)
    print("save done")
    logger.info('inference network training data preparation finished')
    logger.info('----------------------------------------------------')
    return raw_state_name

def agent_plan(env, agents, inference_net, mask_pos, relation, mask_matrix, adj_matrix):
    logger.info("thread:{}, action interval:{}".format(8, args.action_interval))
    env.reset()
    decision_num = 0
    tmp_action = {idx: -1 for idx in range(16) if idx in mask_pos}
    for s in range(3600):
        if s % args.action_interval == 0:
            states = []
            phases = []
            for j in agents:
                state, phase = j.get_ob()
                states.append(state)
                phases.append(phase)
            states = np.array(states, dtype=np.float32)
            phases = np.array(phases, dtype=np.int8)
            # inferred states result

            if inference_net is not None:
                actions = []
                # use inference to recover states, then optimize control
                recovered = inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
                for idx, I in enumerate(agents):
                    action = I.get_action(recovered, phases, relation)
                    actions.append(action)
            else:
                actions = []
                for idx, I in enumerate(agents):
                    # no inference, use FixedTime agent instead
                    if idx not in mask_pos:
                        action = I.get_action(states, phases, relation)
                        actions.append(action)
                    else:
                        # set FixedTime == 40 for convenience now
                        if s % 40 == 0:
                            action = (tmp_action[idx] + 1) % 8
                            tmp_action[idx] = action
                            actions.append(action)
                        else:
                            action = tmp_action[idx]
                            actions.append(action)
        obs, _, dones, _ = env.step(actions)
        if all(dones):
            break
    logger.info("runtime:{}, average travel time:{}".format(0, env.eng.get_average_travel_time()))
    logger.info('-----------------------------------------------------')

def agent_train(env, agents, inference_net, mask_pos, relation, mask_matrix, adj_matrix):
    total_decision_num = 0
    best_att = np.inf
    for e in range(100):
        last_obs, last_phases = list(zip(*env.reset()))
        last_obs = np.array(last_obs, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        last_obs = inference_net.predict(last_obs, last_phases, relation, mask_pos, mask_matrix, adj_matrix)
        # No replay during training
        episodes_rewards = [0 for i in agents]
        episodes_decision_num = 0
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
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)
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

            for agent_id, agent in enumerate(agents):
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()

            if all(dones):
                break
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        best_att = agents_train_test(e, env, agents, inference_net, mask_pos, relation, mask_matrix, adj_matrix, best_att)
    logger.info('-----------------------------------------------------')

def agents_train_test(e, env, agents, inference_net, mask_pos, relation, mask_matrix, adj_matrix, best_att):
    total_decision_num = 0
    cur_obs, cur_phases = list(zip(*env.reset()))
    cur_obs = np.array(cur_obs, dtype=np.float32)
    cur_phases = np.array(cur_phases, dtype=np.int8)
    cur_obs = inference_net.predict(cur_obs, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
    # No replay during testing
    episodes_rewards = [0 for i in agents]
    episodes_decision_num = 0
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
    att = env.eng.get_average_travel_time()
    if att < best_att:
        best_att = att
        for ag in agents:
            ag.save_model(model_dir=os.path.join(model_dir, 'BEST_IDQN'))
    logger.info("episode:{}, Test:{}".format(e, att))
    return best_att


if __name__ == '__main__':
    mask_num = [1,1,1,2,2,2,3,3,3]
    generate_choose = 'IDQN'
    infer_choose = ['sfm', 'no']  # ['no', 'sfm', 'net']
    control_choose = ['maxpressure', 'CopyDQN', 'IDQN']  #['maxpressure', 'SDQN', 'IDQN', 'MaskSDQN', 'MaskIDQN', 'CopyDQN']
    
    config_file = f'cityflow_{args.config}.cfg'
    configuration = f'configurations/{args.config}_astgcn.conf'
    graph_signal_matrix_filename = f'{state_dir}/state_{generate_choose}.pkl'
    
    agent_name = generate_choose
    world = create_world(config_file)
    # create relation
    relation = build_relation(world)

    for mask in mask_num:
        mask_pos = random.sample(range(np.product(relation['net_shape'])), mask)
        adj_matrix = get_road_adj(relation)   
        mask_matrix = get_mask_matrix(relation, mask_pos)
        logger.info('-------------------------------------')
        logger.info('-------------------------------------')
        logger.info(f'mask_pos = {mask_pos}')
        for infer in infer_choose:
            if infer == 'net':
                # need a dataset to train infer net
                # Right now support IDQN generation
                # store last model at epoch -1
                agents = create_preparation_agents(world)
                env = create_env(world, agents)
                if args.load_model:
                    for i, ag in enumerate(agents):
                        ag.load_model(model_dir, -1)
                else:
                    # train dqn agents here
                    agents = dqn_train(env, agents)
                # generate dataset here
                raw_state_name = dqn_generate(env, agents, f'raw_state_{generate_choose}')
                state_file = [raw_state_name]
                run_preparation(configuration, mask_pos, graph_signal_matrix_filename, relation, state_file)

                # train network then start agent training or planning

            elif infer == 'sfm':
                inference_net = SFM_predictor().make_model()
                for control in control_choose:
                    agents = create_agents(world, control)
                    env = create_env(world, agents)
                    if control == 'CopyDQN':
                        for i, ag in enumerate(agents):
                            ag.load_model(model_dir, -1)
                        # copy DQN does not need a training process
                        logger.info(f'infer = {infer}, control = {control}')
                        agent_plan(env, agents, inference_net, mask_pos, relation, mask_matrix, adj_matrix)

                    elif control == 'maxpressure': 
                        # copy DQN does not need a training process
                        logger.info(f'infer = {infer}, control = {control}')
                        agent_plan(env, agents, inference_net, mask_pos, relation, mask_matrix, adj_matrix)
                    
                    elif control == 'IDQN':
                        logger.info(f'infer = {infer}, control = {control}')
                        agent_train(env, agents, inference_net, mask_pos, relation, mask_matrix, adj_matrix)

                    else:
                        raise RuntimeError(f'{infer} not implemented')

            elif infer == 'no':
                inference_net = None
                for control in control_choose:
                    if control not in ['CopyDQN', 'maxpressure', 'IDQN']:
                        continue
                    agents = create_agents(world, control)
                    env = create_env(world, agents)
                    if control == 'CopyDQN':
                        for i, ag in enumerate(agents):
                            ag.load_model(model_dir, -1)
                        # copy DQN does not need a training process
                        logger.info(f'infer = {infer}, control = {control}')
                    
                    elif control == 'maxpressure':
                        # copy DQN does not need a training process
                        logger.info(f'infer = {infer}, control = {control}')
                    
                    # TODO: add a FixedTime IDQN control later
                    elif control == 'IDQN':
                        for i, ag in enumerate(agents):
                            ag.load_model(os.path.join(model_dir, 'BEST_IDQN'), -1)
                        logger.info(f'infer = {infer}, control = {control}')

                    agent_plan(env, agents, inference_net, mask_pos, relation, mask_matrix, adj_matrix)
        logger.info('\n')
    print('experiments finished')





