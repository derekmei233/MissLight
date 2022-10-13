# -*- coding: utf-8 -*-
import gym
import torch

from generator import LaneVehicleGenerator, IntersectionPhaseGenerator

from .rl_agent import RLAgent
import numpy as np
from collections import deque
import os
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random
from utils.preparation import build_relation, get_road_adj, get_mask_matrix
from utils.mask_pos import random_mask
from world import World
from environment import TSCEnv


def one_hot(phase, num_class):
    assert num_class > phase.max()
    one_hot = np.zeros((len(phase), num_class))
    one_hot[range(0, len(phase)), phase.squeeze()] = 1
    # one_hot = one_hot.reshape(*phase.shape, num_class)
    return one_hot

class FRAP(nn.Module):
    def __init__(self, size_in,output_shape, phase_pairs,
                 competition_mask):
        super(FRAP, self).__init__()
        self.dic_phase_expansion =  {
                    1: [0, 0, 1, 0, 0, 0, 1, 0],
                    2: [1, 0, 0, 0, 1, 0, 0, 0],
                    3: [0, 0, 0, 1, 0, 0, 0, 1],
                    4: [0, 1, 0, 0, 0, 1, 0, 0],
                    5: [0, 0, 0, 0, 0, 0, 1, 1],
                    6: [0, 0, 1, 1, 0, 0, 0, 0],
                    7: [0, 0, 0, 0, 1, 1, 0, 0],
                    8: [1, 1, 0, 0, 0, 0, 0, 0]
        }
        self.oshape = output_shape
        self.phase_pairs = phase_pairs
        self.comp_mask = competition_mask
        self.demand_shape = 1  # Allows more than just queue to be used
        self.one_hot = False
        self.d_out = 4  # units in demand input layer
        self.p_out = 4  # size of phase embedding
        self.lane_embed_units = 16
        relation_embed_size = 4

        self.p = nn.Embedding(2, self.p_out)
        self.d = nn.Linear(self.demand_shape, self.d_out)

        self.lane_embedding = nn.Linear(self.p_out + self.d_out, self.lane_embed_units)

        self.lane_conv = nn.Conv2d(2 * self.lane_embed_units, 20, kernel_size=(1, 1))

        self.relation_embedding = nn.Embedding(2, relation_embed_size)
        self.relation_conv = nn.Conv2d(relation_embed_size, 20, kernel_size=(1, 1))

        self.hidden_layer = nn.Conv2d(20, 20, kernel_size=(1, 1))
        self.before_merge = nn.Conv2d(20, 1, kernel_size=(1, 1))

    def _forward(self, states):
        '''
        :params states: [agents, ob_length]
        ob_length:concat[len(one_phase),len(intersection_lane)]
        '''
        #print(len(states))
        #if(len(states)==20):
            #states.resize_(1,12)
        #print(states)
        # if lane_num=12,then num_movements=12, but turning right do not be used
        num_movements = int((states.size()[1] - 1) / self.demand_shape) if not self.one_hot else int(
            (states.size()[1] - len(self.phase_pairs)) / self.demand_shape)
        batch_size = states.size()[0]
        acts = states[:, :1].to(torch.int64) if not self.one_hot else states[:, :len(self.phase_pairs)].to(torch.int64)
        states = states[:, 1:] if not self.one_hot else states[:, len(self.phase_pairs):]
        states = states.float()

        # Expand action index to mark demand input indices
        extended_acts = []
        if not self.one_hot:
            for i in range(batch_size):
                act_idx = acts[i]
                pair = self.phase_pairs[act_idx]
                zeros = torch.zeros(num_movements, dtype=torch.int64)
                zeros[pair[0]] = 1
                zeros[pair[1]] = 1
                extended_acts.append(zeros)
            extended_acts = torch.stack(extended_acts)
        else:
            extended_acts = acts
        phase_embeds = torch.sigmoid(self.p(extended_acts))

        phase_demands = []
        # if num_movements == 12:
        #     order_lane = [0,1,3,4,6,7,9,10] # remove turning_right phase
        # else:
        #     order_lane = [i for i in range(num_movements)]
        # for idx, i in enumerate(order_lane):
        for i in range(num_movements):
            # phase = phase_embeds[:, idx]  # size 4
            phase = phase_embeds[:, i]  # size 4
            demand = states[:, i:i + self.demand_shape]
            demand = torch.sigmoid(self.d(demand))  # size 4
            phase_demand = torch.cat((phase, demand), -1)
            phase_demand_embed = F.relu(self.lane_embedding(phase_demand))
            phase_demands.append(phase_demand_embed)
        phase_demands = torch.stack(phase_demands, 1)
        # phase_demands_old = torch.stack(phase_demands, 1)
        # # turn direction from NESW to ESWN
        # if num_movements == 8:
        #     phase_demands = torch.cat([phase_demands_old[:,2:,:],phase_demands_old[:,:2,:]],1)
        # elif num_movements == 12:
        #     phase_demands = torch.cat([phase_demands_old[:,3:,:],phase_demands_old[:,:3,:]],1)
        # phase_demands = torch.stack(phase_demands, 1)

        pairs = []
        for pair in self.phase_pairs:
            pairs.append(phase_demands[:, pair[0]] + phase_demands[:, pair[1]])

        rotated_phases = []
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                if i != j: rotated_phases.append(torch.cat((pairs[i], pairs[j]), -1))
        rotated_phases = torch.stack(rotated_phases, 1)
        rotated_phases = torch.reshape(rotated_phases,
                                       (batch_size, self.oshape, self.oshape - 1, 2 * self.lane_embed_units))
        rotated_phases = rotated_phases.permute(0, 3, 1, 2)  # Move channels up
        rotated_phases = F.relu(self.lane_conv(rotated_phases))  # Conv-20x1x1  pair demand representation

        # Phase competition mask
        competition_mask = self.comp_mask.repeat((batch_size, 1, 1))
        relations = F.relu(self.relation_embedding(competition_mask))
        relations = relations.permute(0, 3, 1, 2)  # Move channels up
        relations = F.relu(self.relation_conv(relations))  # Pair demand representation

        # Phase pair competition
        combine_features = rotated_phases * relations
        combine_features = F.relu(self.hidden_layer(combine_features))  # Phase competition representation
        combine_features = self.before_merge(combine_features)  # Pairwise competition result

        # Phase score
        combine_features = torch.reshape(combine_features, (batch_size, self.oshape, self.oshape - 1))
        q_values = (lambda x: torch.sum(x, dim=2))(combine_features)  # (b,8)
        return q_values

    def forward(self, states, train=True):
        if train:
            return self._forward(states)
        else:
            with torch.no_grad():
                return self._forward(states)




class FRAP_SH_Agent(RLAgent):
    """ the logic behind SDQN. Since SDQN is implemented on all intersections, the only difference is learning from
        all or observable intersections, we use it to process all information but only give it accessibility to observable ones
    """

    def __init__(self, action_space, ob_generator, reward_generator, iid, idx):
        super().__init__(action_space, ob_generator, reward_generator)
        self.iid = iid
        self.idx = idx  # learnable index
        self.sub_agents = len(iid)

        self.ob_generator = ob_generator
        ob_length = [self.ob_generator[0][0].ob_length, self.action_space.n]
        self.ob_length = sum(ob_length)

        self.phase_pairs=[[4, 10], [1, 7], [3, 9], [0, 6], [9, 10], [3, 4], [6, 7], [0, 1]]
        self.comp_mask = self.relation()
        self.dic_phase_expansion = None
        self.num_phases = len(self.phase_pairs)
        self.num_actions = len(self.phase_pairs)
        self.learnable = len(self.idx)
        self.memory = [deque(maxlen=10000) for i in range(self.learnable)]  # number of samples

        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.batch_size = 64

        self.criterion = nn.MSELoss(reduction='mean')
        self.model = self.build_shared_model()
        self.target_model = self.build_shared_model()
        #self.optimizer = optimizer
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.9, centered=False,
                                       eps=1e-7)

    def relation(self):
        comp_mask = []
        for i in range(len(self.phase_pairs)):
            zeros = np.zeros(len(self.phase_pairs) - 1, dtype=np.int)
            cnt = 0
            for j in range(len(self.phase_pairs)):
                if i == j: continue
                pair_a = self.phase_pairs[i]
                pair_b = self.phase_pairs[j]
                if len(list(set(pair_a + pair_b))) == 3: zeros[cnt] = 1
                cnt += 1
            comp_mask.append(zeros)
        comp_mask = torch.from_numpy(np.asarray(comp_mask))
        return comp_mask

    def choose(self, ob, phase, relation=None):
        if np.random.rand() <= self.epsilon:
            return self.sample()
        return self.get_action(ob, phase, relation)

    def get_action(self, ob, phase, relation=None):
        # get all observation now
        actions = []
        for idx in range(self.sub_agents):
            ob_oh = one_hot(phase[idx], self.action_space.n)
            obs = torch.tensor(np.concatenate((ob[idx:idx+1,:], ob_oh),axis=1)).float()
            act_values = self.model.forward(obs, train=False)
            actions.append(torch.argmax(act_values))
        return actions

    def build_shared_model(self):
        model=FRAP(self.ob_length,self.action_space.n,self.phase_pairs,self.comp_mask)
        return model

    def get_ob(self):
        obs = tuple([self.ob_generator[i][0].generate(), np.array(self.ob_generator[i][1].generate())]
                    for i in range(self.sub_agents))
        return obs

    def sample(self):
        return [self.action_space.sample() for _ in range(self.sub_agents)]

    def get_reward(self):
        rewards = tuple([self.reward_generator[i].generate() for i in range(self.sub_agents)])
        return rewards


    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        # only update model at idx == update_idx
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def remember(self, ob, action, reward, next_ob, idx):
        self.memory[self.idx.index(idx)].append((ob, action, reward, next_ob))

    def _encode_sample(self, minibatch):
        # TODO: check dimension
        obses_t, actions_t, rewards_t, obses_tp1 = list(zip(*minibatch))
        obs = [np.squeeze(np.stack(obs_i)) for obs_i in list(zip(*obses_t))]
        # expand action to one_hot
        obs_oh = one_hot(obs[1], self.action_space.n)
        obs = np.concatenate((obs[0], obs_oh), axis=1)
        next_obs = [np.squeeze(np.stack(obs_i)) for obs_i in list(zip(*obses_tp1))]
        # expand acton to one_hot
        next_obs_oh = one_hot(next_obs[1], self.action_space.n)
        next_obs = np.concatenate((next_obs[0], next_obs_oh), axis=1)
        rewards = np.array(rewards_t, copy=False)
        obs = torch.from_numpy(obs).float()
        rewards = torch.from_numpy(rewards).float()
        next_obs = torch.from_numpy(next_obs).float()
        return obs, actions_t, rewards, next_obs

    def replay(self):
        # sample from all buffers

        minibatch = self._sample()
        obs, actions, rewards, next_obs = self._encode_sample(minibatch)
        out = self.target_model.forward(next_obs, train=False)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model.forward(obs, train=False)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model.forward(obs, train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _sample(self):
        mini_batch = []
        for i in range(self.learnable):
            mini_batch.extend(random.sample(self.memory[i], self.batch_size))
        random.shuffle(mini_batch)
        return mini_batch

    def load_model(self, model_dir):
        # only load for idx == min(self.all_id)
        name = "sdqn.pt"
        model_name = os.path.join(model_dir, name)
        self.model = FRAP(self.ob_length, self.action_space.n,self.phase_pairs,self.comp_mask)
        self.model.load_state_dict(torch.load(model_name))
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        name = "sdqn.pt"
        model_name = os.path.join(model_dir, name)
        torch.save(self.model.state_dict(), model_name)


if __name__ == '__main__':
    config_file = f'cityflow_hz4x4.cfg'
    action_interval = 10
    episodes = 3600
    world = World(config_file, thread_num=8)
    relation = build_relation(world)
    mask_pos = random_mask(3, 'neighbor', relation)
    obs_pos = set(range(len(world.intersections))) - set(mask_pos)
    agents = []
    iid = []
    ob_generator = []
    reward_generator = []
    for idx, inter in enumerate(world.intersections):
        ob_generator.append(
            [
                LaneVehicleGenerator(world, inter, ['lane_count'], in_only=True, average=None),
                IntersectionPhaseGenerator(world, inter, targets=['cur_phase'], negative=False)
            ])
        reward_generator.append(
            LaneVehicleGenerator(world, inter, ['lane_waiting_count'], in_only=True, average=None, negative=True))
        iid.append(inter.id)
    action_space = gym.spaces.Discrete(len(world.intersections[-1].phases))
    ob_length = ob_generator[0][0].ob_length + action_space.n
    q_model = build_shared_model(ob_length, action_space)
    target_q_model = build_shared_model(ob_length, action_space)
    optimizer = optim.RMSprop(q_model.parameters(), lr=0.001, alpha=0.9, centered=False, eps=1e-7)
    agents.append(
        SDQNAgent(action_space, ob_generator, reward_generator, iid, obs_pos, q_model, target_q_model, optimizer))
    env = TSCEnv(world, agents, None)
    print('construction finished')
