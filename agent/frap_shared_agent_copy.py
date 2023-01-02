# -*- coding: utf-8 -*-
from ctypes import util
from . import RLAgent
import random
import numpy as np
from collections import deque
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
#from common.registry import Registry
import gym
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator
from torch.nn.utils import clip_grad_norm_
from . import BaseAgent
from copy import deepcopy
import os

def one_hot(phase, num_class):
    assert num_class > phase.max()
    one_hot = np.zeros((len(phase), num_class))
    one_hot[range(0, len(phase)), phase.squeeze()] = 1
    # one_hot = one_hot.reshape(*phase.shape, num_class)
    return one_hot


class FRAP_move(nn.Module):
    '''
    FRAP captures the phase competition relation between traffic movements through a modified network structure.
    '''

    def __init__(self, output_shape, phase2movements, competition_mask, device):
        super(FRAP_move, self).__init__()
        self.oshape = output_shape
        self.device = device
        self.phase2movements = phase2movements.to(self.device)
        self.comp_mask = competition_mask.to(self.device)
        self.demand_shape = 1  # Allows more than just queue to be used
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
        states: [agents, ob_length]
        ob_length:concat[len(one_phase),len(intersection_lane)]
        '''
        # if lane_num=12,then num_movements=12, but turning right do not be used
        num_movements = int((states.size()[1] - 1) / self.demand_shape)
        batch_size = states.size()[0]
        acts = states[:, :1].to(torch.int64)
        states = states[:, 1:].unsqueeze(1).repeat(1, self.phase2movements.shape[0], 1)
        states = states.float()

        # Expand action index to mark demand input indices
        extended_acts = []
        # for i in range(batch_size):
        # # index of phase
        #     act_idx = acts[i]
        #     connectivity = self.phase2movements[act_idx]
        #     extended_acts = torch.stack(connectivity)
        # phase_embeds = torch.sigmoid(self.p(extended_acts))

        connectivity = self.phase2movements[acts]
        phase_embeds = torch.sigmoid(self.p(connectivity))  # [B, 4, 3, 12]

        # if num_movements == 12:
        #     order_lane = [0,1,3,4,6,7,9,10] # remove turning_right phase
        # else:
        #     order_lane = [i for i in range(num_movements)]
        # for idx, i in enumerate(order_lane):

        # for i in range(num_movements):
        #     # phase = phase_embeds[:, idx]  # size 4
        #     phase = phase_embeds[:, i]  # size 4
        #     demand = states[:, i:i+self.demand_shape]
        #     demand = torch.sigmoid(self.d(demand))    # size 4
        #     phase_demand = torch.cat((phase, demand), -1)
        #     phase_demand_embed = F.relu(self.lane_embedding(phase_demand))
        #     phase_demands.append(phase_demand_embed)
        # phase_demands = torch.stack(phase_demands, 1)

        all_phase_demand = states * self.phase2movements  # [B, 3, 12] - checked
        all_phase_demand = torch.sigmoid(self.d(all_phase_demand.unsqueeze(-1)))  # [B, 3, 12, 4] - checked
        phase_demand = torch.cat((all_phase_demand, phase_embeds.repeat(1, self.phase2movements.shape[0], 1, 1)),
                                 -1)  # B, 3, 12, 8]
        phase_demand_embed = F.relu(self.lane_embedding(phase_demand))  # [B, 3, 12, 16]
        phase_demand_agg = torch.sum(phase_demand_embed, dim=2)  # [B, 3, 16]
        rotated_phases = []
        for i in range(phase_demand_agg.shape[-2]):
            for j in range(phase_demand_agg.shape[-2]):
                if i != j: rotated_phases.append(torch.cat((phase_demand_agg[:, i, :], phase_demand_agg[:, j, :]), -1))
        rotated_phases = torch.stack(rotated_phases, 1)  # [B, 2*3, 32]
        rotated_phases = torch.reshape(rotated_phases,
                                       (batch_size, self.oshape, self.oshape - 1,
                                        2 * self.lane_embed_units))  # [B, 3, 2, 32]
        rotated_phases = rotated_phases.permute(0, 3, 1, 2)  # [B, 32, 3, 2]
        rotated_phases = F.relu(self.lane_conv(rotated_phases))  # [B, 20, 3, 2]
        # pairs = []
        # for pair in self.phase_pairs:
        #     pairs.append(phase_demands[:, pair[0]] + phase_demands[:, pair[1]])

        # rotated_phases = []
        # for i in range(len(pairs)):
        #     for j in range(len(pairs)):
        #         if i != j: rotated_phases.append(torch.cat((pairs[i], pairs[j]), -1))
        # rotated_phases = torch.stack(rotated_phases, 1)
        # rotated_phases = torch.reshape(rotated_phases,
        #                                (batch_size, self.oshape, self.oshape - 1, 2 * self.lane_embed_units)) # [B, 3, 2, 16]
        # rotated_phases = rotated_phases.permute(0, 3, 1, 2)  # Move channels up
        # rotated_phases = F.relu(self.lane_conv(rotated_phases))  # Conv-20x1x1  pair demand representation

        # Phase competition mask
        competition_mask = self.comp_mask.repeat((batch_size, 1, 1))  # [B, 3, 2]
        relations = F.relu(self.relation_embedding(competition_mask.long()))  # [B, 3, 2, 4] ?
        relations = relations.permute(0, 3, 1, 2)  # [B, 4, 3, 2]
        relations = F.relu(self.relation_conv(relations))  # [B, 20, 3, 2]

        # Phase pair competition
        combine_features = rotated_phases * relations
        combine_features = F.relu(self.hidden_layer(combine_features))  # Phase competition representation
        combine_features = self.before_merge(combine_features)  # # [B, 1, 3, 2]

        # Phase score
        combine_features = torch.reshape(combine_features, (batch_size, self.oshape, self.oshape - 1))  # [B, 3, 2]
        q_values = (lambda x: torch.sum(x, dim=2))(combine_features)  # (B, 3)
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

    def __init__(self, action_space, ob_generator, reward_generator, iid, idx, q_model, target_q_model, optimizer,
                 device):
        super(FRAP_SH_Agent,self).__init__(action_space, ob_generator, reward_generator)

        self.inter_id = iid
        self.idx = idx  # learnable index
        self.learnable = len(self.idx)
        self.sub_agents = len(iid)
        self.name=self.__class__.__name__

        self.ob_generator = ob_generator
        ob_length = [self.ob_generator[0][0].ob_length, self.action_space.n]
        self.ob_length = sum(ob_length)

        self.lane_names = []
        [self.lane_names.extend(l) for l in self.ob_generator[0][0].lanes]
        self.directions = self.ob_generator[0].directions
        self.road_names = self.ob_generator[0].roads
        self.movements = [self._orthogonal_mapping(rad) for rad in self.directions]
        self.twelve_movements = ['N_L', 'N_T', 'N_R', 'E_L', 'E_T', 'E_R', 'S_L', 'S_T', 'S_R', 'W_L', 'W_T', 'W_R']

        self.world = self.ob_generator[0][0].world
        #self.inter_id = self.world.intersection_ids[self.idx]
        self.inter_obj = self.world.id2intersection[self.inter_id]

        self.phase = True
        self.one_hot = False
        self.phase = True
        assert self.phase is True
        self.one_hot = False
        assert self.one_hot is False

        self.inter_info = \
        [self.world.roadnet['intersections'][idx] for idx, i in enumerate(self.world.roadnet['intersections']) if
         i['id'] == self.inter_id][0]
        self.linkage_movement = {(i['startRoad'], i['endRoad']): i['type'] for i in self.inter_info['roadLinks']}
        self.phase2movements = self._phase_avail_movements()
        self.lane2movements = self._construct_lane2movement_mapping()
        self.phase2movements = torch.tensor(self.phase2movements).to(torch.int64)
        self.num_phases = self.phase2movements.shape[0]
        self.num_actions = self.phase2movements.shape[0]

        self.comp_mask = self.relation()
        self.dic_phase_expansion = None

        self.learnable = len(self.idx)
        self.memory = [deque(maxlen=10000) for _ in range(self.learnable)]  # number of samples
        self.memory_with_history = deque(maxlen=10000)

        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20
        self.grad_clip=5.0
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.device = device

        self.criterion = nn.MSELoss(reduction='mean')
        self.model = self._build_shared_model() # self.build_shared_model()

        self.target_model = target_q_model  # self.build_shared_model()
        self.update_target_network()
        # self.optimizer = optimizer
        self.optimizer = optimizer  # optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.9, centered=False,
        # eps=1e-7)

    def _construct_lane2movement_mapping(self):
        result = np.zeros([len(self.lane_names), len(self.twelve_movements)])
        mapping = self.inter_info['roadLinks']
        for r_link in mapping:
            tp = r_link['type']
            if tp == 'turn_left':
                #  only work for atlanta now, remove U-turn
                start_r = r_link['startRoad'].split('#')[0].replace('-', '')
                end_r = r_link['endRoad'].split('#')[0].replace('-', '')
                if start_r == end_r:
                    continue
                turn = 'L'
            elif tp == 'go_straight':
                turn = 'T'
            elif tp == 'turn_right':
                turn = 'R'
            else:
                raise ValueError
            for l_link in r_link['laneLinks']:
                idx = l_link['startLaneIndex']
                r_idx = self.lane_names.index(r_link['startRoad']+'_'+str(idx))
                c_idx = self.twelve_movements.index(self.movements[r_idx] + '_'+turn)
                result[r_idx, c_idx] = 1
        return result

    def _orthogonal_mapping(self, rad):
        if  rad > 5.49779 or rad < 0.785398:
            return 'N'
        elif rad >=0.785398 and rad < 2.35619:
            return 'E'
        elif rad >= 2.35619 and rad < 3.92699:
            return 'S'
        elif rad >= 3.92699 and rad < 5.49779:
            return 'W'
        else:
            raise ValueError


    def _phase_avail_movements(self):
        # no yellow phase
        result = np.zeros([self.action_space.n, len(self.twelve_movements)])
        for p in range(self.action_space.n):
            avail_road_links_id = self.inter_obj.phase_available_roadlinks[p]
            for l in avail_road_links_id:
                linkage = self.inter_obj.roadlinks[l]
                start = linkage[0]
                end = linkage[1]
                tp = self.linkage_movement[(start, end)]
                if tp == 'turn_left':
                    #  only work for atlanta now, remove U-turn
                    start_r = start.split('#')[0].replace('-', '')
                    end_r = end.split('#')[0].replace('-', '')
                    if start_r == end_r:
                        continue
                    turn = 'L'
                elif tp == 'go_straight':
                    turn = 'T'
                elif tp == 'turn_right':
                    turn = 'R'
                d = self.movements[self.road_names.index(start)]
                direction = self.twelve_movements.index(d + "_" + turn)
                result[p, direction] = 1
        return result


    def relation(self):
        '''
        relation
        Get the phase competition relation between traffic movements.

        :param: None
        :return comp_mask: matrix of phase competition relation
        '''
        comp_mask = []
        # remove connection at all phase, then compute if there is a same connection here
        removed_phase2movements = deepcopy(self.phase2movements)
        removed_phase2movements[:, np.sum(self.phase2movements, axis=0) == self.phase2movements.shape[0]] = 0
        for i in range(self.phase2movements.shape[0]):
            zeros = np.zeros(self.phase2movements.shape[0] - 1, dtype=np.int)
            cnt = 0
            for j in range(self.phase2movements.shape[0]):
                if i == j: continue

                pair_a = removed_phase2movements[i]
                pair_b = removed_phase2movements[j]
                if np.dot(pair_a, pair_b) >= 1: zeros[cnt] = 1
                cnt += 1
            comp_mask.append(zeros)
        comp_mask = torch.from_numpy(np.asarray(comp_mask))
        return comp_mask

    def choose(self, ob, phase, relation=None):
        if np.random.rand() <= self.epsilon:
            return self.sample()
        return self.get_action(ob, phase, relation)

    def get_action(self, ob, phase, test=False):
        # get all observation now
        if np.random.rand() <= self.epsilon:
            return self.sample()
        if self.phase2movements.shape[0] == 1:
            return [np.array(0) for _ in range(self.sub_agents)]
        actions = []
        for idx in range(self.sub_agents):
            feature = np.concatenate([phase[idx:idx+1,:].reshape(1, -1), ob.reshape(1, -1)], axis=1)
            observation = torch.tensor(feature, dtype=torch.float32).to(self.device)
            act_values = self.model(observation, train=False)
            act_values = act_values.to('cpu').clone().detach().numpy()
            actions.append(np.argmax(act_values, axis=1).squeeze())
        return actions

    def _build_shared_model(self):
        # Neural Net for Deep-Q learning Model
        model = FRAP_move(self.action_space.n,self.phase2movements,self.comp_mask, self.device).to(self.device)
        return model

    def get_ob(self):
        obs=[]
        for i in range(self.sub_agents):
            tmp=self.ob_generator[i][0].generate()
            obs.append([np.dot(tmp, self.lane2movements), np.array(self.ob_generator[i][1].generate())])

        return obs

    def get_delay(self):
        obs = np.array(list([np.mean(self.ob_generator[i][2].generate())] for i in range(self.sub_agents))).squeeze()
        return obs

    '''
    def get_movement(self, phase):
        movement = self.phase2movements[phase]
        return np.array(movement)
    '''

    def get_phase(self):
        phase_groups=[]
        for i in range(self.sub_agents):
            phase = []
            phase.append(self.ob_generator[i][1].generate())
            phase = (np.concatenate(phase)).astype(np.int8)
            phase_groups.append(phase)
        return phase_groups

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
        self.memory[self.idx.index(idx)].append((ob[0].reshape(1,-1), ob[1], action, reward, next_ob[0].reshape(1,-1), next_ob[1]))

    def remember_traj(self, last_recovered, last_phases, actions, cur_phases, ob_traj):
        self.memory_with_history.append((last_recovered, last_phases, actions, cur_phases, ob_traj))

    '''
    def _encode_sample(self, minibatch, batch_size):
        # TODO: check dimension
        obses_t, actions_t, rewards_t, obses_tp1 = list(zip(*minibatch))
        obs = [np.squeeze(np.stack(obs_i)) for obs_i in list(zip(*obses_t))]
        # expand action to one_hot
        # obs_oh = np.zeros((batch_size*self.learnable, 1))
        # for i in range(batch_size*self.learnable):
        # obs_oh[i] = obs[1][i]
        obs_oh = np.expand_dims(obs[1], axis=1)
        obs = np.concatenate((obs_oh, obs[0]), axis=1)
        next_obs = [np.squeeze(np.stack(obs_i)) for obs_i in list(zip(*obses_tp1))]
        # expand acton to one_hot
        # next_obs_oh = np.zeros((batch_size*self.learnable, 1))
        # for i in range(batch_size*self.learnable):
        # next_obs_oh[i] = next_obs[1][i]
        next_obs_oh = np.expand_dims(next_obs[1], axis=1)
        # next_obs_oh[0]=next_obs[1].squeeze()
        # next_obs_oh = one_hot(next_obs[1], self.action_space.n)
        next_obs = np.concatenate((next_obs_oh, next_obs[0]), axis=1)
        rewards = np.array(rewards_t, copy=False)
        obs = torch.from_numpy(obs).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        return obs, actions_t, rewards, next_obs
    '''

    def _batchwise(self, samples):
        '''
        _batchwise
        Reconstruct the samples into batch form(last state, current state, reward, action).

        :param samples: original samples record in replay buffer
        :return state_t, state_tp, rewards, actions: information with batch form
        '''
        # (batch_size,12)
        obs_t_all=[item[0] for item in samples] # last_obs(batch, 1, lane_num)
        obs_tp_all=[item[4] for item in samples] # cur_obs
        # obs_t = [utils.remove_right_lane(x) for x in obs_t_all]
        # obs_tp = [utils.remove_right_lane(x) for x in obs_tp_all]
        obs_t = obs_t_all
        obs_tp = obs_tp_all
        obs_t = np.concatenate(obs_t) # (batch,lane_num)
        obs_tp = np.concatenate(obs_tp) # (batch,lane_num)

        phase_t = np.concatenate([item[1].reshape(1,-1) for item in samples]) # (batch, 1)
        phase_tp = np.concatenate([item[5].reshape(1,-1) for item in samples])
        feature_t = np.concatenate([phase_t, obs_t], axis=1) # (batch,ob_length)
        feature_tp = np.concatenate([phase_tp, obs_tp], axis=1)
        # (batch_size, ob_length)

        state_t = torch.tensor(feature_t, dtype=torch.float32).to(self.device)
        state_tp = torch.tensor(feature_tp, dtype=torch.float32).to(self.device)
        # rewards:(64)
        rewards = torch.tensor(np.array([item[3] for item in samples]), dtype=torch.float32).to(self.device)  # TODO: BETTER WA
        # actions:(64,1)
        actions = torch.tensor(np.array([item[2] for item in samples]), dtype=torch.long).to(self.device)
        return state_t, state_tp, rewards, actions

    '''
    def replay(self):
        # sample from all buffers

        minibatch = self._sample(self.batch_size)
        obs, actions, rewards, next_obs = self._encode_sample(minibatch, self.batch_size)
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
    '''
    def replay(self):
        '''
        train
        Train the agent, optimize the action generated by agent.

        :param: None
        :return: value of loss
        '''
        if len(self.memory) < self.batch_size:
            return
        if self.action_space.n == 1:
            return np.array(0)
        samples = self._sample(self.batch_size)
        b_t, b_tp, rewards, actions = self._batchwise(samples)
        out = self.target_model(b_tp, train=False) # (batch_size,num_actions)
        target = rewards + self.gamma * torch.max(out, dim=1)[0] # (batch_size)
        target_f = self.model(b_t, train=False) # (batch_size,num_actions)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]

        loss = self.criterion(self.model(b_t, train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.to('cpu').clone().detach().numpy()

    def _encode_traj_sample(self, minibatch):
        last_recovered, last_phases, actions, cur_phases, ob_traj = list(zip(*minibatch))
        return last_recovered, last_phases, actions, cur_phases, ob_traj

    def _sample(self, batch_size):
        mini_batch = []
        for i in range(self.learnable):
            mini_batch.extend(random.sample(self.memory[i], batch_size))
        random.shuffle(mini_batch)
        return mini_batch

    def _sample_with_history(self, batch_size):
        mini_batch = []
        mini_batch.extend(random.sample(self.memory_with_history, batch_size))
        return mini_batch

    def get_latest_sample(self, infer='NN_st'):
        sample = []
        for idx, i in enumerate(self.idx):
            sample.append(self.memory[idx][-1])
        b_t, _, rewards,actions = self._batchwise(sample)
        '''
        states, phases = [np.stack(ob) for ob in list(zip(*obs))]
        if infer == 'NN_st':
            obs2 = one_hot(phases, self.action_space.n)
        elif infer == 'NN_sta':
            obs2 = one_hot(actions, self.action_space.n)
        x = torch.from_numpy(np.concatenate((obs2, states), axis=1)).float().to(self.device)
        target = torch.from_numpy(np.array(rewards)[:, np.newaxis]).float().to(self.device)
        '''
        x=b_t
        target=rewards.reshape((len(rewards),1))
        return x, target

    def replay_img(self, reward_model, update_times, infer='NN_st'):
        if update_times == 0:
            return
        minibatch = self._sample(update_times)
        #obs, actions, rewards, next_obs = self._encode_sample(minibatch, update_times)
        b_t,b_tp,rewards,actions=self._batchwise(minibatch)
        rewards = torch.squeeze(reward_model.predict(b_t), dim=1)
        out = self.target_model.forward(b_tp, train=False)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model.forward(b_t, train=False)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model.forward(b_t, train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        '''
        if infer == 'NN_st':
            '''
        #tensor->one_hot
        '''
            tmp = obs[:, :1].cpu().detach().numpy().astype(np.int8)
            states = obs[:, 1:].cpu().detach().numpy()
            ones = np.zeros((tmp.size, self.action_space.n)).astype(np.int8)
            for i in range(tmp.size):
                ones[i][tmp[i]] = 1
            x = torch.from_numpy(np.concatenate((states, ones), axis=1)).float().to(self.device)
        elif infer == 'NN_sta':
            obses_t, actions_t, _, _ = list(zip(*minibatch))
            tmp = [np.squeeze(np.stack(obs_i)) for obs_i in list(zip(*obses_t))]
            obs_oh = one_hot(actions_t, self.action_space.n)
            x = torch.from_numpy(np.concatenate((tmp[0], obs_oh), axis=1)).float().to(self.device)
        '''



    def load_model(self, model_dir):
        # TODO: add idqn
        name = "frapdqn_{}.pt".format(self.inter_id)
        model_name = Path.join(model_dir, name)
        self.model = FRAP_move(self.action_space.n, self.phase2movements, self.comp_mask, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = FRAP_move(self.action_space.n, self.phase2movements, self.comp_mask, self.device).to(
            self.device)
        self.target_model.load_state_dict(torch.load(model_name))


    def save_model(self, model_dir):
        if not Path.exists(model_dir):
            Path.mkdir(model_dir)
        name = "frapdqn_{}.pt".format(self.inter_id)
        model_name = Path.join(model_dir, name)
        torch.save(self.model.state_dict(), model_name)
