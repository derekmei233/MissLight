import gym
from environment import TSCEnv
from world import World
from agent.max_pressure_agent import MaxPressureAgent
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
from metric import TravelTimeMetric
import argparse
import configparser
import os
import shutil
import numpy as np
import logging
from datetime import datetime
from utils import *
import pickle
from prepareData import build_relation_intersection_road, inter2state, reconstruct_data, run_preparation
from predictionModel.ASTGCN_r import make_model
from prepareData import load_graphdata_channel, get_road_adj, read_output, get_mask_pos
from train_ASTGCN_r import train_main

from collections import deque
from tensorboardX import SummaryWriter
from metric.metrics import masked_mse
import torch
import torch.nn as nn
import torch.optim as optim
import random


# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--config_file', type=str, default='cityflow_hz_4x4.cfg', help='path of config file')
parser.add_argument("--config", default='configurations/HZ_4x4_astgcn.conf', type=str,
                    help="configuration file path")
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--test_steps', type=int, default=3600, help='number of steps for step')
parser.add_argument('--action_interval', type=int, default=10, help='how often agent make decisions')  #decided by self.t_min in MaxPressureAgent
parser.add_argument('--log_dir', type=str, default="log/maxpressure/hz_4x4", help='directory in which logs should be saved')

parser.add_argument('-rs','--replay_buffer_size', type=int, default=5000, help="size of replay buffer")
parser.add_argument('--state_dir', type=str, default="roadgraph/maxpressure/hz_4x4", help='directory in which roadgraph should be saved')
args = parser.parse_args()
config = configparser.ConfigParser()

config.read(args.config)
data_config = config['Data']
training_config = config['Training']

if not os.path.exists(args.state_dir):
    os.makedirs(args.state_dir)
relation_filename = os.path.join(args.state_dir, "roadnet_relation_hz4x4.pkl")
graph_signal_matrix_filename = os.path.join(args.state_dir, 'state_data/state_hz4x4.pkl')

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
print("CUDA:", USE_CUDA, DEVICE)

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



if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(args.log_dir, datetime.now().strftime('%Y%m%d-%H%M%S') + ".log"))
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)



# create world
world = World(args.config_file, thread_num=args.thread)
# create agents
agents = []
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
    # if len(agents) == 5:
    #     break

for agent in agents:
    print(agent.action_space)
# create metric
metric = TravelTimeMetric(world)
# create env
env = TSCEnv(world, agents, metric)


def generate(e, masked_pos, inference_net, state_dir, state_name):
    logger.info("thread:{}, action interval:{}".format(args.thread, args.action_interval))
    env.reset()
    decision_num = 0
    replay_buffer = deque(maxlen=args.replay_buffer_size)
    save_state = []
    for s in range(args.test_steps):
        if s % args.action_interval == 0:
            states = []
            phases = []
            for j in agents:
                state = j.get_ob()
                phase = j.get_phase()
                states.append(state)
                phases.append(phase)

            # generate state_mask
            state_t = np.array(states, dtype=np.float32)
            phase_t = np.array(phases, dtype=np.int8)
            masked_x = inter2state(relation, replay_buffer, decision_num, in_channels, len_input, None, None)
            # generate state_unmaksked
            road_unmasked_t = inference_net.forward(torch.tensor(masked_x[np.newaxis, :, :, :],
                                                                  dtype=torch.float32, device=DEVICE))
            road_unmasked_t = road_unmasked_t.detach().cpu().numpy()
            state_unmasked_t = reconstruct_data(road_unmasked_t, relation_filename)[0, 0, :, :]
            # inference state
            state_inference_t = apply_inference(state_t, state_unmasked_t, masked_pos)

            replay_buffer.append((state_inference_t, phase_t))
            decision_num += 1
            state_inference_t = state_inference_t.tolist()
            save_obs = np.expand_dims(state_inference_t, axis=1)
            save_phase = np.expand_dims(phases, axis=1)
            save_phase = np.expand_dims(save_phase, axis=1)
            # save_obs = np.expand_dims(list(zip(save_obs,save_phase)),axis=0)
            save_state.append(list(zip(save_obs, save_phase)))
            actions = []
            for idx, I in enumerate(agents):
                action = I.get_action(state_inference_t, relation, in_channels)
                actions.append(action)
        obs, _, dones, _ = env.step(actions)
        if all(dones):
            break
    with open(os.path.join(state_dir, state_name), 'wb') as fo:
        pickle.dump(save_state, fo)
    logger.info("runtime:{}, average travel time:{}".format(e, env.eng.get_average_travel_time()))

def plan(e, masked_pos, tmp_action):
    logger.info("thread:{}, action interval:{}".format(args.thread, args.action_interval))
    env.reset()
    decision_num = 0
    save_state = []
    for s in range(args.test_steps):
        if s % args.action_interval == 0:
            states = []
            phases = []
            for j in agents:
                state = j.get_ob()
                phase = j.get_phase()
                states.append(state)
                phases.append(phase)

            # generate state_mask
            state_t = np.array(states, dtype=np.float32)
            phase_t = np.array(phases, dtype=np.int8)
            actions = []
            for idx, I in enumerate(agents):
                if idx not in masked_pos:
                    action = I.get_action_org(state_t)
                    actions.append(action)
                else:
                    if s % 30 == 0:
                        action = (tmp_action[idx] + 1) % action_space.n
                        tmp_action[idx] = action
                        actions.append(action)
                    else:
                        action = tmp_action[idx]
                        actions.append(action)

        #if s % 30 == 0: print(actions)
        obs, _, dones, _ = env.step(actions)
        if all(dones):
            break
    logger.info("runtime:{}, average travel time:{}".format(e, env.eng.get_average_travel_time()))


def run(args, env):
    logger.info("thread:{}, acton interval:{}".format(args.thread, args.action_interval))
    for e in range(1):
        last_obs = env.reset()
        env.eng.set_save_replay(True)
        env.eng.set_replay_file("replay_%s.txt" % e)

        i = 0
        actions = []
        while i < args.steps:
            if i % args.action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(agents):
                    actions.append(agent.get_action(last_obs[agent_id]))
                for _ in range(args.action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1

                last_obs = obs    # will not use this in decision making process
            if all(dones):
                break
        logger.info("runtime:{}, average travel time:{}".format(e, env.eng.get_average_travel_time()))


def apply_inference(masked_state, inference_state, mask_pos):
    masked_state[mask_pos, :] = inference_state[mask_pos, :]
    return masked_state

if __name__ == '__main__':
    test = []
    for i in test:
        masked_pos = random.sample(range(16), i)
        tmp_action = {idx: -1 for idx in range(16) if idx in masked_pos}
        # build intersection and road net relationship
        build_relation_intersection_road(world, relation_filename)
        adj_mx = get_road_adj(relation_filename)
        #masked_pos = get_mask_pos(relation_filename, neighbor_node, mask_num)

        # masked position

        logger.info("masked position: {}".format(masked_pos))
        with open(relation_filename, 'rb') as f_re:
            relation = pickle.load(f_re)
        # tell its for uniform test
        logger.info("Maxpressure")
        inference_net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                                    num_for_predict, len_input, num_of_vertices)
        graph_signal_matrix_filename = graph_signal_matrix_filename
        graph_signal_matrix_filename_nd = graph_signal_matrix_filename.split(
            '.')[0] + '_s' + str(points_per_hour) + '_p' + str(num_for_predict) + '_n' + str(neighbor_node) + '_m' + str(
            mask_num) + '.pkl'
        graph_signal_matrix_filename_dataset = graph_signal_matrix_filename.split(
            '.')[0] + '_s' + str(points_per_hour) + '_p' + str(num_for_predict) + '_n' + str(neighbor_node) + '_m' + str(
            mask_num) + '_dataset.pkl'

        state_name_list = []

        sn_0 = "rawstate_hz4x4_0.pkl"
        state_name_list.append(sn_0)
        generate(0, masked_pos, inference_net, args.state_dir, sn_0)
        run_preparation(masked_pos, graph_signal_matrix_filename, relation_filename, args.state_dir, state_name_list)
        inference_net = train_main(inference_net, 0, graph_signal_matrix_filename_dataset, relation_filename)

        os.remove(graph_signal_matrix_filename_nd)
        os.remove(graph_signal_matrix_filename_dataset)
        sn_1 = "rawstate_hz4x4_1.pkl"
        state_name_list.append(sn_1)
        generate(1, masked_pos, inference_net, args.state_dir, sn_1)

        logger.info("Maxpressure&FixedTime")
        plan(0, masked_pos, tmp_action)
    masked_pos = list(range(16))
    tmp_action = {idx: -1 for idx in range(16) if idx in masked_pos}
    logger.info("Maxpressure&FixedTime")
    plan(0, masked_pos, tmp_action)