import random
import gym
from environment_colight import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent.colight_pytorch_agent import CoLightAgent
from metric import TravelTimeMetric
import argparse
import configparser
import os
import numpy as np
import logging
from datetime import datetime
from utils import *
import pickle
from prepareData import build_relation_intersection_road, get_mask_pos, inter2state, reconstruct_data, run_preparation, get_road_adj
from predictionModel.ASTGCN_r import make_model
from metric.utils import compute_val_loss_mstgcn, predict_and_save_results_mstgcn
import torch
from train_ASTGCN_r import train_main
from collections import deque


# TODO: Update per training epoch
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--prefix', type=str, default='hz4x4', help='predix to determine file')
parser.add_argument('--pahse_expansion', action='store_false', default=True)


parser.add_argument('--config_file', type=str,default='hz4x4', help='path of config file')  #road net
parser.add_argument("--config", default='configurations/HZ_4x4_astgcn.conf', type=str,
                    help="configuration file path")
parser.add_argument('--thread', type=int, default=4,help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="-1",help='gpu to be used')  # choose gpu card
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help="learning rate")
parser.add_argument('-bs', '--batch_size', type=int, default=64, help="batch size")
parser.add_argument('-ls', '--learning_start', type=int, default=1000, help="learning start")
parser.add_argument('-rs', '--replay_buffer_size', type=int, default=5000, help="size of replay buffer")
parser.add_argument('-uf', '--update_target_model_freq', type=int, default=10, help="the frequency to update target q model")
parser.add_argument('-gc', '--grad_clip', type=float, default=5.0, help="clip gradients")
parser.add_argument('-ep', '--epsilon', type=float, default=0.8, help="exploration rate")
parser.add_argument('-ed', '--epsilon_decay', type=float, default=0.9995, help="decay rate of exploration rate")
parser.add_argument('-me', '--min_epsilon', type=float, default=0.01, help="the minimum epsilon when decaying")
parser.add_argument('--steps', type=int, default=3600, help='number of steps')  #per episodes
parser.add_argument('--test_steps', type=int, default=3600, help='number of steps for step')
parser.add_argument('--action_interval', type=int, default=10, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=60, help='training episodes')
#parser.add_argument('--test_episodes',type=int,default=10,help='testing episodes')

parser.add_argument('--load_model_dir', type=str, default="model/colight_torch/hz4x4", help='load this model to test')
parser.add_argument('--graph_info_dir', type=str,default="hz4x4",help='load infos about graph(i.e. mapping, adjacent)')
parser.add_argument('--train_model', action="store_false", default=True)
parser.add_argument('--test_model', action="store_true", default=False)
parser.add_argument('--save_model', action="store_false", default=True)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=1, help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model/colight_torch/hz4x4", help='directory in which model should be saved')
#parser.add_argument('--load_dir',type=str,default="model/colight",help='directory in which model should be loaded')
parser.add_argument('--log_dir', type=str, default="log/colight_torch/hz4x4", help='directory in which logs should be saved')
parser.add_argument('--vehicle_max', type=int, default=1, help='used to normalize node observayion')
parser.add_argument('--mask_type', type=int, default=0, help='used to specify the type of softmax')
parser.add_argument('--get_attention', action="store_true", default=False)
parser.add_argument('--test_when_train', action="store_false", default=True)
parser.add_argument('--state_dir', type=str, default="roadgraph/colight_torch/hz_4x4", help='directory in which roadgraph should be saved')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu
if not os.path.exists(args.state_dir):
    os.makedirs(args.state_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
file_prefix = args.prefix +"_"+"Colight_"+ str(args.graph_info_dir)+ "_" +str(args.learning_rate) + "_"\
              + str(args.epsilon) + "_" + str(args.epsilon_decay) + "_" +str(args.batch_size) + "_" + \
              str(args.learning_start) + "_" + str(args.replay_buffer_size) + \
              "_" +datetime.now().strftime('%Y%m%d-%H%M%S')
fh = logging.FileHandler(os.path.join(args.log_dir,file_prefix+".log"))
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)

# file of the relation between roads and intersections and other info
state_basename = args.state_dir
relation_filename = os.path.join(args.state_dir, "roadnet_relation_4x4.pkl")
graph_signal_matrix_filename = os.path.join(args.state_dir, 'state_data/state_hz4x4.pkl')

# create world
world = World("data/config_dir/config_"+args.config_file+".json", thread_num=args.thread)
graph_info_file_dir = "data/graphinfo/graph_info_"+args.graph_info_dir+".pkl"
graph_info_file = open(graph_info_file_dir, "rb")
res = pickle.load(graph_info_file)
net_node_dict_id2inter = res[0]
net_node_dict_inter2id =res[1]
net_edge_dict_id2edge=res[2]
net_edge_dict_edge2id=res[3]
node_degree_node=res[4]
node_degree_edge=res[5]
node_adjacent_node_matrix=res[6]
node_adjacent_edge_matrix=res[7]
edge_adjacent_node_matrix=res[8]
# net_node_dict_id2inter, net_node_dict_inter2id, net_edge_dict_id2edge, net_edge_dict_edge2id, \
#     node_degree_node,node_degree_edge, node_adjacent_node_matrix, node_adjacent_edge_matrix, \
#     edge_adjacent_node_matrix = pickle.load(graph_info_file)
graph_info_file.close()
#TODO:update the below dict (already done)
dic_traffic_env_conf = {
    "ACTION_PATTERN":"set",
    "NUM_INTERSECTIONS":len(net_node_dict_id2inter),  #used
    "NUM_ROADS":len(net_edge_dict_id2edge),  #used
    "MIN_ACTION_TIME":10,
    "YELLOW_TIME":5,
    "ALL_RED_TIME":0,
    "NUM_PHASES":8,  #used
    "NUM_LANES":1,  #used
    "ACTION_DIM":2,
    "MEASURE_TIME":10,
    "IF_GUI":True,
    "DEBUG":False,
    "INTERVAL":1,
    "THREADNUM":8,
    "SAVEREPLAY":True,
    "RLTRAFFICLIGHT":True,
    "DIC_FEATURE_DIM":dict(  #used
        D_LANE_QUEUE_LENGTH=(4, ),
        D_LANE_NUM_VEHICLE=(4, ),
        D_COMING_VEHICLE=(4, ),
        D_LEAVING_VEHICLE=(4, ),
        D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4, ),
        D_CUR_PHASE=(1, ), #used
        D_NEXT_PHASE=(1, ),
        D_TIME_THIS_PHASE=(1, ),
        D_TERMINAL=(1, ),
        D_LANE_SUM_WAITING_TIME=(4, ),
        D_VEHICLE_POSITION_IMG=(4,60,),
        D_VEHICLE_SPEED_IMG=(4,60,),
        D_VEHICLE_WAITING_TIME_IMG=(4,60,),
        D_PRESSURE=(1, ),
        D_ADJACENCY_MATRIX=(2, )),
    #used
    "LIST_STATE_FEATURE": [
        "cur_phase",
        # "time_this_phase",
        # "vehicle_position_img",
        # "vehicle_speed_img",
        # "vehicle_acceleration_img",
        # "vehicle_waiting_time_img",
        "lane_num_vehicle",
        # "lane_num_vehicle_been_stopped_thres01",
        # "lane_num_vehicle_been_stopped_thres1",
        # "lane_queue_length",
        # "lane_num_vehicle_left",
        # "lane_sum_duration_vehicle_left",
        # "lane_sum_waiting_time",
        # "terminal",

        # "coming_vehicle",
        # "leaving_vehicle",
        # "pressure",

        # "adjacency_matrix"
    ],
    "DIC_REWARD_INFO": {
        "flickering": 0,
        "sum_lane_queue_length": 0,
        "sum_lane_wait_time": 0,
        "sum_lane_num_vehicle_left": 0,
        "sum_duration_vehicle_left": 0,
        "sum_num_vehicle_been_stopped_thres01": 0,
        "sum_num_vehicle_been_stopped_thres1": -0.25,
        "pressure": 0,
    },
    "LANE_NUM": {
        "LEFT": 1,
        "RIGHT": 1,
        "STRAIGHT": 1
    },
    "PHASE": [
        'WSES',
        'NSSS',
        'WLEL',
        'NLSL',
        'WSWL',
        'ESEL',
        'NSNL',
        'SSSL',
    ],
}

dic_graph_setting = {
    "NEIGHBOR_NUM": 4,  # standard number of adjacent nodes of each node
    "NEIGHBOR_EDGE_NUM": 4,  # # standard number of adjacent edges of each node
    "N_LAYERS": 1,  # layers of MPNN
    "INPUT_DIM": [128,128], # input dimension of each layer of multiheadattention, the first value should == the last value of "NODE_EMB_DIM"
    "OUTPUT_DIM": [128,128], # output dimension of each layer of multiheadattention, the first value should == the last value of "NODE_EMB_DIM"
    "NODE_EMB_DIM": [128,128],  # the firsr two layer of dense to embedding the input
    "NUM_HEADS": [5,5],
    "NODE_LAYER_DIMS_EACH_HEAD":[16, 16],  # [input_dim,output_dim]
    "OUTPUT_LAYERS":[], #
    "NEIGHBOR_ID": node_adjacent_node_matrix,  # adjacent node id of each node
    "ID2INTER_MAPPING": net_node_dict_id2inter,  # id ---> intersection mapping
    "INTER2ID_MAPPING":net_node_dict_inter2id,  # intersection ----->id mapping
    "NODE_DEGREE_NODE": node_degree_node,  # number of adjacent nodes of node
    "EDGE_IDX": edge_adjacent_node_matrix # edge index of each graph
}

# create observation generator, which is used to construct sample
observation_generators = []
for node_dict in world.intersections:
    node_id = node_dict.id
    node_id_int = net_node_dict_inter2id[node_id]
    tmp_generator = LaneVehicleGenerator(world,
                                         node_dict, ["lane_count"],
                                         in_only=True,
                                         average=None)
    observation_generators.append((node_id_int, tmp_generator))
sorted(observation_generators, key=lambda x: x[0]) # sorted the ob_generator based on its corresponding id_int, increasingly
# create agent
action_space = gym.spaces.Discrete(len(world.intersections[0].phases))
colightAgent = CoLightAgent(
    action_space, observation_generators,
    LaneVehicleGenerator(world,world.intersections[0], ["lane_waiting_count"],in_only=True,average="all",negative=True), world, dic_traffic_env_conf,dic_graph_setting,args)
if args.load_model:
    colightAgent.load_model(args.load_dir, args.prefix, args.episodes)
print(colightAgent.ob_length)
print(colightAgent.action_space)
# create metric
metric = TravelTimeMetric(world)
env = TSCEnv(world, colightAgent, metric)

# build relationship between intersection to road
config = configparser.ConfigParser()
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

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


class TrafficLightDQN:
    def __init__(self, agent, env, args, logging_tool, fprefix, masked_pos, inference_net,
                 graph_signal_matrix_filename_nd, graph_signal_matrix_filename_dataset):
        self.agent = agent
        self.env = env
        self.world = world
        self.logging_tool = logging_tool
        self.yellow_time = self.world.intersections[0].yellow_phase_time
        self.args = args
        self.fprefix = fprefix
        self.masked_pos = masked_pos
        self.inference_net = inference_net
        # self.log_file = os.path.join(self.args.log_dir,self.args.prefix+ datetime.now().strftime('%Y%m%d-%H%M%S') + ".yzy.log")
        self.log_file = os.path.join(args.log_dir, self.fprefix + ".yzy.log")
        # self.log_file = file_prefix + ".yzy.log"
        log_handle = open(self.log_file, 'w')
        log_handle.close()
        self.replay_file_dir = "data/replay_dir/" + self.args.config_file
        if not os.path.exists(self.replay_file_dir):
            os.makedirs(self.replay_file_dir)
        self.replay_file_dir = "replay_dir/" + self.args.config_file

    def train(self):
        total_decision_num = 0
        state_name_list = []
        avg_time_list = []
        for e in range(self.args.episodes):
            last_obs = self.env.reset()  # observation dimension?
            if e % self.args.save_rate == self.args.save_rate - 1:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(self.replay_file_dir + "/%s_replay_%s.txt" % (self.fprefix, e))
            else:
                self.env.eng.set_save_replay(False)

            episodes_rewards = [0 for i in range(len(self.world.intersections))]
            episodes_decision_num = 0
            episode_loss = []
            i = 0
            while i < self.args.steps:
                if i % self.args.action_interval == 0:
                    actions = []
                    last_phase = []  # ordered by the int id of intersections
                    for j in range(len(self.world.intersections)):
                        node_id_str = self.agent.graph_setting["ID2INTER_MAPPING"][j]
                        node_dict = self.world.id2intersection[node_id_str]
                        last_phase.append(node_dict.current_phase)
                        # last_phase.append([self.world.intersections[j].current_phase])
                    state_t = np.array(last_obs, dtype=np.float32)
                    phase_t = np.array(last_phase, dtype=np.int8)
                    if i == 0:
                        masked_x = inter2state(relation, colightAgent.replay_buffer, total_decision_num, in_channels,
                                               len_input, None, None)
                        road_unmasked_t = inference_net.forward(torch.tensor(masked_x[np.newaxis, :, :, :],
                                                                             dtype=torch.float32, device=DEVICE))
                        road_unmasked_t = road_unmasked_t.detach().cpu().numpy()
                        state_unmasked_t = reconstruct_data(road_unmasked_t, relation_filename)[0, 0, :, :]
                        state_t = apply_inference(state_t, state_unmasked_t, self.masked_pos)
                        last_obs = state_t.tolist()
                    if (total_decision_num > self.agent.learning_start):
                        actions = self.agent.get_action(last_phase, last_obs)
                        # the retured dimension is [batch, agents],
                        # the batch is 1 when we get action, so we just get the first actions
                    else:
                        actions = self.agent.sample(s_size=self.agent.num_agents)

                    reward_list = []  # [intervals,agents,reward]
                    for _ in range(self.args.action_interval):
                        # TODO: reconstruct reward here
                        obs, _, dones, _ = env.step(actions)
                        i += 1
                    cur_phase = []
                    for j in range(len(self.world.intersections)):
                        node_id_str = self.agent.graph_setting["ID2INTER_MAPPING"][j]
                        node_dict = self.world.id2intersection[node_id_str]
                        cur_phase.append(node_dict.current_phase)
                    state_tp = np.array(obs, dtype=np.float32)
                    phase_tp = np.array(cur_phase, dtype=np.int8)
                    # TODO: check inference at t+1
                    masked_x = inter2state(relation, colightAgent.replay_buffer, total_decision_num, in_channels,
                                        len_input, state_t, phase_t)
                    road_unmasked_tp = inference_net.forward(torch.tensor(masked_x[np.newaxis, :, :, :],
                                                                         dtype=torch.float32, device=DEVICE))
                    road_unmasked_tp = road_unmasked_tp.detach().cpu().numpy()
                    state_unmasked_tp = reconstruct_data(road_unmasked_tp, relation_filename)[0, 0, :, :]
                    state_tp = apply_inference(state_tp, state_unmasked_tp, self.masked_pos)

                    rewards = np.mean(state_t - state_tp, axis=1)
                    for j in range(len(self.world.intersections)):
                        episodes_rewards[j] += rewards[j]
                    obs = state_tp.tolist()
                    # TODO: could cause problem in training
                    self.agent.remember(last_obs, phase_t, actions, rewards, obs, phase_tp)
                    episodes_decision_num += 1
                    total_decision_num += 1
                    last_obs = obs

                if total_decision_num > self.agent.learning_start and total_decision_num % self.agent.update_model_freq == self.agent.update_model_freq - 1:
                    cur_loss_q = self.agent.replay()
                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.agent.learning_start and total_decision_num % self.agent.update_target_model_freq == self.agent.update_target_model_freq - 1:
                    self.agent.update_target_network()

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            cur_travel_time = self.env.eng.get_average_travel_time()
            mean_reward = np.sum(episodes_rewards) / episodes_decision_num
            self.writeLog("TRAIN", e, cur_travel_time, mean_loss, mean_reward)
            self.logging_tool.info(
                "step:{}/{}, q_loss:{}, rewards:{}".format(i, self.args.steps, mean_loss, mean_reward))
            if e % self.args.save_rate == self.args.save_rate - 1:
                if not os.path.exists(self.args.save_dir):
                    os.makedirs(self.args.save_dir)
                self.agent.save_model(self.args.save_dir, prefix=args.prefix, e=e)
            self.logging_tool.info(
                "episode:{}/{}, average travel time:{}".format(e, self.args.episodes, cur_travel_time))
            for j in range(len(self.world.intersections)):
                self.logging_tool.debug(
                    "intersection:{}, mean_episode_reward:{}".format(j, episodes_rewards[j] / episodes_decision_num))
            if self.args.test_when_train:
                self.train_test(e, state_name_list=state_name_list, avg_time_list=avg_time_list)
                print(state_name_list[-5:])
                print(avg_time_list[-5:])

        self.agent.save_model(self.args.save_dir, prefix=args.prefix, e=self.args.episodes)


    def train_test(self, e, state_name_list, avg_time_list):
        obs = self.env.reset()
        state_name = 'rawstate_hz4x4_{}.pkl'.format(e)

        ep_rwds = [0 for i in range(len(self.world.intersections))]
        replay_buffer = deque(maxlen=args.replay_buffer_size)
        eps_nums = 0
        save_state = []
        for i in range(self.args.test_steps):
            if i % args.action_interval == 0:
                last_phase = []
                for j in range(len(self.world.intersections)):
                    node_id_str = self.agent.graph_setting["ID2INTER_MAPPING"][j]
                    node_dict = self.world.id2intersection[node_id_str]
                    last_phase.append(node_dict.current_phase)
                state_t = np.array(obs, dtype=np.float32)
                phase_t = np.array(last_phase, dtype=np.int8)

                masked_x = inter2state(relation, replay_buffer, eps_nums, in_channels, len_input, None, None)
                road_unmasked_t = self.inference_net.forward(torch.tensor(masked_x[np.newaxis, :, :, :],
                                                        dtype=torch.float32, device=DEVICE))
                road_unmasked_t = road_unmasked_t.detach().cpu().numpy()
                state_unmasked_t = reconstruct_data(road_unmasked_t, relation_filename)[0, 0, :, :]
                state_t = apply_inference(state_t, state_unmasked_t, self.masked_pos)
                replay_buffer.append((state_t, phase_t))
                state_t = state_t.tolist()
                save_obs = np.expand_dims(state_t, axis=1)
                save_phase = np.expand_dims(last_phase, axis=1)
                save_phase = np.expand_dims(save_phase, axis=1)
                save_state.append(list(zip(save_obs, save_phase)))
                # TODO: May need to change np to list
                actions = self.agent.get_action(last_phase, state_t, test_phase=True)
                actions = actions
                #rewards_list = []
                for _ in range(self.args.action_interval):
                    obs, _, dones, _ = self.env.step(actions)
                    i += 1
                # TODO: need reward or not?
                """
                rewards = state_t - state_tp
                for j in range(len(self.world.intersections)):
                    episodes_rewards[j] += rewards[j]
                for j in range(len(self.world.intersections)):
                    ep_rwds[j] += rewards[j]
                """

                eps_nums += 1
            if all(dones):
                break

        trv_time = self.env.eng.get_average_travel_time()
        if not avg_time_list:
            avg_time_list.append(trv_time)
            state_name_list.append(state_name)
        else:
            flag = 0
            for idx, t in enumerate(avg_time_list):
                if trv_time > t:
                    avg_time_list.insert(idx, trv_time)
                    state_name_list.insert(idx, state_name)
                    flag = 1
                    break
            if flag == 0:
                avg_time_list.append(trv_time)
                state_name_list.append(state_name)

        with open(os.path.join(args.state_dir, state_name), 'wb') as fo:
            pickle.dump(save_state, fo)

        if e % 5 == 0 or e == args.episodes - 1:
            if len(state_name_list) > 4:
                training_list = random.sample(state_name_list[-10:], 5)
            else:
                training_list = state_name_list
            run_preparation(masked_pos, graph_signal_matrix_filename, relation_filename, args.state_dir, training_list)
            self.inference_net = train_main(self.inference_net, int(e / 5) * epochs, graph_signal_matrix_filename_dataset, relation_filename)
            os.remove(graph_signal_matrix_filename_nd)
            os.remove(graph_signal_matrix_filename_dataset)

        # self.logging_tool.info("Final Travel Time is %.4f, and mean rewards %.4f" % (trv_time,mean_rwd))
        """
        self.logging_tool.info(
            "Test step:{}/{}, travel time :{}, rewards:{}".format(e, self.args.episodes, trv_time, mean_rwd))
        """
        self.logging_tool.info(
            "Test step:{}/{}, travel time :{}".format(e, self.args.episodes, trv_time))
        self.writeLog("TEST", e, trv_time, 100, -1)
        for i in self.agent.model.parameters():
            # print(i[0], i[1].data)
            break

        return trv_time

    def test(self, e, drop_load=False):
        save_state = []
        if not drop_load:
            model_name=self.args.load_model_dir
            # model_name='model/colight_torch_4x4_hz/colight_agent_hz4x4_199'
            if model_name is not None:
                # self.agent.load_model(model_name, args.predix, args.episodes)
                self.agent.load_model(mdir=model_name, prefix='hz4x4', e=args.episodes)
            else:
                raise ValueError("model name should not be none")
        attention_mat_list = []
        obs = self.env.reset()
        ep_rwds = [0 for i in range(len(self.world.intersections))]
        eps_nums = 0
        for i in range(self.args.test_steps):
            if i % args.action_interval == 0:
                last_phase = []
                for j in range(len(self.world.intersections)):
                    node_id_str = self.agent.graph_setting["ID2INTER_MAPPING"][j]
                    node_dict = self.world.id2intersection[node_id_str]
                    last_phase.append(node_dict.current_phase)
                if self.args.get_attention:
                    actions, att_step = self.agent.get_action(last_phase, obs,test_phase=True)
                    attention_mat_list.append(att_step[0])
                else:
                    actions = self.agent.get_action(last_phase, obs,test_phase=True)
                actions = actions
                save_obs = np.expand_dims(obs, axis=1)
                save_phase = np.expand_dims(last_phase, axis=1)
                save_phase = np.expand_dims(save_phase, axis=1)
                # save_obs = np.expand_dims(list(zip(save_obs,save_phase)),axis=0)
                save_state.append(list(zip(save_obs, save_phase)))
                rewards_list = []

                for _ in range(self.args.action_interval):
                    obs, _, dones, _ = self.env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)
                for j in range(len(self.world.intersections)):
                        ep_rwds[j] += rewards[j]
                eps_nums+=1
                #ep_rwds.append(rewards)

            # print(env.eng.get_average_travel_time())
            if all(dones):
                break
        mean_rwd = np.sum(ep_rwds)/eps_nums
        trv_time = self.env.eng.get_average_travel_time()
        self.logging_tool.info("Final Travel Time is %.4f, and mean rewards %.4f" % (trv_time,mean_rwd))
        if self.args.get_attention:
            tmpstr = self.args.load_model_dir
            tmpstr=tmpstr.split('/')[-1]
            att_file = "data/analysis/colight/"+tmpstr+"_att_ana.pkl"
            pickle.dump(attention_mat_list,open(att_file,"wb"))
            print("dump the attention matrix to ",att_file)
        with open(state_name, 'wb') as fo:
            pickle.dump(save_state, fo)
        print("save raw state done")
        return trv_time

    def writeLog(self, mode, step, travel_time, loss, cur_rwd):
        """
        :param mode: "TRAIN" OR "TEST"
        :param step: int
        """
        res = "CoLight" + '\t' + mode + '\t' + str(step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % loss +"\t"+ "%.2f" % cur_rwd
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

def apply_inference(masked_state, inference_state, mask_pos):
    masked_state[mask_pos, :] = inference_state[mask_pos, :]
    return masked_state

if __name__ == '__main__':
    build_relation_intersection_road(world, relation_filename)
    adj_mx = get_road_adj(relation_filename)
    #masked_pos = get_mask_pos(relation_filename, neighbor_node, mask_num)
    masked_pos = [5, 10]
    logger.info("masked position: {}".format(masked_pos))
    #logger.info("uniform test")
    with open(relation_filename, 'rb') as f_re:
        relation = pickle.load(f_re)

    inference_net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                               num_for_predict, len_input, num_of_vertices)
    graph_signal_matrix_filename_nd = graph_signal_matrix_filename.split(
        '.')[0] + '_s' + str(points_per_hour) + '_p' + str(num_for_predict) + '_n' + str(neighbor_node) + '_m' + str(
        mask_num) + '.pkl'
    graph_signal_matrix_filename_dataset = graph_signal_matrix_filename.split(
        '.')[0] + '_s' + str(points_per_hour) + '_p' + str(num_for_predict) + '_n' + str(neighbor_node) + '_m' + str(
        mask_num) + '_dataset.pkl'
    player = TrafficLightDQN(colightAgent, env, args, logger, file_prefix, masked_pos, inference_net,
                             graph_signal_matrix_filename_nd, graph_signal_matrix_filename_dataset)

    # generate raw data in intersection format
    player.train()

    # if args.train_model:
    #     print("begin to train model")
    #     player.train()
    #     player.test(True)
    # if args.test_model:
    #     print(args.load_model_dir)
    #     if (not args.train_model) and (args.load_model_dir is None):
    #         raise ValueError("invalid parameters, load_model_dir should not be None when the agent is not trained")
    #     print("begin to test model")
    #     player.test()
# simulate
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
# train(args, env)
# test()
# meta_test('/mnt/d/Cityflow/examples/config.json')


# TODO: 0. first determine which sections are masked and infer them with ASTGCN model --> if construct sample for ASTGCN is posible.
# this also means every sample in the deque is completed and filled by inference. / If NOT raise ERROR
# 1. find index of mask
# 2. provide mask and inference API for colight
# 3. ASTGCN construct graph
# 4. parameter of ASTGCN inference and exploit ASTGCN model from training process
# 5. TODO: reward tuning.
# 6. inference 2 t or 1 t???


# TODO: 1. construct inference network. ASTGCN


# TODO: 2. construct colight agent for environment

# TODO: 3. during training colight. decide how to choose actions when masked. (1/2 not enought to start inference. 2/2 enough for inference

# TODO: 4. finished i epoch. generate samples for ASTGCN by running test.

# TODO: 5. train ASTGCN and update model to do inference in next iteration.
