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
from prepareData import build_relation_intersection_road
from predictionModel.ASTGCN_r import make_model
from prepareData import load_graphdata_channel, get_road_adj,read_output

from collections import deque
from tensorboardX import SummaryWriter
from metric.metrics import masked_mse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--config_file', type=str, default='hz4x4', help='path of config file')
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

state_name = os.path.join(args.state_dir, "rawstate_hz4x4.pkl")
relation_filename = os.path.join(args.state_dir, "roadnet_relation_hz4x4.pkl")
graph_signal_matrix_filename = os.path.join(args.state_dir, 'state_data/state_4x4.pkl')

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
DEVICE = torch.device('cuda:0')
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
        world
    ))
    # if len(agents) == 5:
    #     break

for agent in agents:
    print(agent.action_space)
# create metric
metric = TravelTimeMetric(world)
# create env
env = TSCEnv(world, agents, metric)
replay_buffer = deque(maxlen=args.replay_buffer_size)

# train dqn_agent
def generate():
    save_state = []
    obs = env.reset()
    for i in range(args.test_steps):
        if i % args.action_interval == 0:
            actions = []
            for j in range(len(env.world.intersections)):




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



if __name__ == '__main__':
    # build intersection and road net relationship
    build_relation_intersection_road(world, relation_filename)
    # generate raw data in intersection format
    graph_signal_matrix_filename = graph_signal_matrix_filename.split(
        '.')[0] + '_s' + str(points_per_hour) + '_p' + str(num_for_predict) + '_n' + str(neighbor_node) + '_m' + str(
        mask_num) + '_dataset.pkl'

    train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std, mask_matrix = load_graphdata_channel(
        graph_signal_matrix_filename, num_of_hours,
        num_of_days, num_of_weeks, DEVICE, batch_size)

    adj_mx = get_road_adj(relation_filename)

    inference_net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                               num_for_predict, len_input, num_of_vertices)
    # start untrained inference



    run(args, env)