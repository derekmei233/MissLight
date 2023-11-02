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
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy


class FRAP_DQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator,reward_generator, iid, idx,device):
        super(FRAP_DQNAgent,self).__init__(action_space, ob_generator,reward_generator)

        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20
        self.gamma = 0.95
        self.grad_clip = 5.0
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.buffer_size = 5000
        self.memory = deque(maxlen=self.buffer_size)
        self.device=device

        self.sub_agents = 1

        self.iid=iid
        self.idx=idx
        self.name = self.__class__.__name__

        self.phase = True
        self.one_hot = False
        self.phase = True
        assert self.phase is True
        self.one_hot = False
        assert self.one_hot is False

        self.action_space = action_space
        self.ob_generator = ob_generator
        self.world = self.ob_generator[0].world
        #print(self.idx)
        self.inter_id = self.world.intersection_ids[self.idx]
        self.inter_obj = self.world.id2intersection[self.inter_id]

        ob_length = [self.ob_generator[0].ob_length, self.action_space.n]
        self.lane_names = []
        [self.lane_names.extend(l) for l in self.ob_generator[0].lanes]
        self.directions = self.ob_generator[0].directions
        self.road_names = self.ob_generator[0].roads
        self.movements = [self._orthogonal_mapping(rad) for rad in self.directions]
        self.twelve_movements = ['N_L', 'N_T', 'N_R','E_L', 'E_T', 'E_R', 'S_L', 'S_T', 'S_R', 'W_L', 'W_T', 'W_R']
        self.inter_info = [self.world.roadnet['intersections'][idx] for idx, i in enumerate(self.world.roadnet['intersections']) if i['id'] == self.inter_id][0]
        self.linkage_movement = {(i['startRoad'], i['endRoad']): i['type'] for i in self.inter_info['roadLinks']}
        self.lane2movements = self._construct_lane2movement_mapping()
        self.phase2movements = self._phase_avail_movements()
        self.comp_mask = self.relation()
        self.phase2movements = torch.tensor(self.phase2movements).to(torch.int64)
        self.num_phases = self.phase2movements.shape[0]
        self.num_actions = self.phase2movements.shape[0]



        self.ob_length = sum(ob_length)
        self.reward_generator = reward_generator

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=1e-7)
        self.criterion = nn.MSELoss(reduction='mean')


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

    def choose(self, ob, phase):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        return self.get_action(ob, phase)

    def get_ob(self):
        '''
        get_ob
        Get observation from environment.

        :param: None
        :return x_obs: observation generated by ob_generator
        '''
        tmp = self.ob_generator[0].generate()
        return [np.dot(tmp, self.lane2movements), np.array(self.ob_generator[1].generate())]
    
    def get_orig_ob(self):
        return self.ob_generator[0].generate()

    def get_delay(self):
        return np.mean(self.ob_generator[2].generate())
    
    def get_movement(self, phase):
        movement = self.phase2movements[phase]
        return np.array(movement)

    def get_reward(self):
        reward = self.reward_generator.generate()
        if len(reward) == 1:
            return reward[0]
        else:
            return reward

    def get_phase(self):
        phase = []
        phase.append(self.ob_generator[1].generate())
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase

    def get_action(self, ob, phase):
        '''
        get_action
        Generate action.

        :param ob: observation, the shape is (1,12)
        :param phase: current phase, the shape is (1,)
        :param test: boolean, decide whether is test process
        :return: action that has the highest score
        '''
        if self.phase2movements.shape[0] == 1:
            return np.array(0)

        feature = np.concatenate([phase[self.idx].reshape(1,-1), ob[self.idx].reshape(1,-1)], axis=1)
        observation = torch.tensor(feature, dtype=torch.float32).to(self.device)
        actions = self.model(observation, train=False)
        actions = actions.to('cpu').clone().detach().numpy()
        return np.argmax(actions, axis=1).squeeze()
    
    def get_movement_state(self, states):
        return np.dot(states, self.lane2movements)

    def sample(self):
        return self.action_space.sample()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = FRAP_move(self.action_space.n,self.phase2movements,self.comp_mask, self.device).to(self.device)
        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        weights = self.model.state_dict()
        #print(weights['param_group'])
        self.target_model.load_state_dict(weights)

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob[0].reshape(1,-1), ob[1], action, reward, next_ob[0].reshape(1,-1), next_ob[1]))

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
        samples = random.sample(self.memory, self.batch_size)
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

    def load_model(self, model_dir):
        # TODO: add idqn
        name = "frapdqn_{}.pt".format(self.iid)
        model_name = Path.join(model_dir, name)
        self.model = FRAP_move(self.action_space.n,self.phase2movements,self.comp_mask, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = FRAP_move(self.action_space.n,self.phase2movements,self.comp_mask, self.device).to(self.device)
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, model_dir):
        if not Path.exists(model_dir):
            Path.mkdir(model_dir)
        name = "frapdqn_{}.pt".format(self.iid)
        model_name = Path.join(model_dir, name)
        torch.save(self.model.state_dict(), model_name)



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
        self.demand_shape = 1   # Allows more than just queue to be used
        self.d_out = 4      # units in demand input layer
        self.p_out = 4      # size of phase embedding
        self.lane_embed_units = 16
        relation_embed_size = 4

        self.p = nn.Embedding(2, self.p_out)
        self.d = nn.Linear(self.demand_shape, self.d_out)

        self.lane_embedding = nn.Linear(self.p_out + self.d_out, self.lane_embed_units)

        self.lane_conv = nn.Conv2d(2*self.lane_embed_units, 20, kernel_size=(1, 1))

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
        num_movements = int((states.size()[1]-1)/self.demand_shape)
        batch_size = states.size()[0]
        acts = states[:, :1].to(torch.int64)
        states = states[:, 1:].unsqueeze(1).repeat(1,self.phase2movements.shape[0], 1)
        states = states.float()
        extended_acts = []

        connectivity = self.phase2movements[acts]
        phase_embeds = torch.sigmoid(self.p(connectivity)) # [B, 4, 3, 12]

        all_phase_demand = states * self.phase2movements # [B, 3, 12] - checked
        all_phase_demand = torch.sigmoid(self.d(all_phase_demand.unsqueeze(-1))) # [B, 3, 12, 4] - checked
        phase_demand = torch.cat((all_phase_demand, phase_embeds.repeat(1,self.phase2movements.shape[0],1,1)), -1) # B, 3, 12, 8]
        phase_demand_embed = F.relu(self.lane_embedding(phase_demand) ) # [B, 3, 12, 16]
        phase_demand_agg = torch.sum(phase_demand_embed, dim=2) # [B, 3, 16]
        rotated_phases = []
        for i in range(phase_demand_agg.shape[-2]):
            for j in range(phase_demand_agg.shape[-2]):
                if i != j: rotated_phases.append(torch.cat((phase_demand_agg[:,i,:], phase_demand_agg[:,j,:]), -1))
        rotated_phases = torch.stack(rotated_phases, 1) # [B, 2*3, 32]
        rotated_phases = torch.reshape(rotated_phases,
                                       (batch_size, self.oshape, self.oshape - 1, 2 * self.lane_embed_units)) # [B, 3, 2, 32]
        rotated_phases = rotated_phases.permute(0, 3, 1, 2)  # [B, 32, 3, 2]
        rotated_phases = F.relu(self.lane_conv(rotated_phases)) # [B, 20, 3, 2]

        # Phase competition mask
        competition_mask = self.comp_mask.repeat((batch_size, 1, 1)) # [B, 3, 2]
        relations = F.relu(self.relation_embedding(competition_mask.long())) # [B, 3, 2, 4] ?
        relations = relations.permute(0, 3, 1, 2)  # [B, 4, 3, 2]
        relations = F.relu(self.relation_conv(relations))  # [B, 20, 3, 2]

        # Phase pair competition
        combine_features = rotated_phases * relations
        combine_features = F.relu(self.hidden_layer(combine_features))  # Phase competition representation
        combine_features = self.before_merge(combine_features)  # # [B, 1, 3, 2]

        # Phase score
        combine_features = torch.reshape(combine_features, (batch_size, self.oshape, self.oshape - 1)) # [B, 3, 2]
        q_values = (lambda x: torch.sum(x, dim=2))(combine_features) # (B, 3)
        return q_values
        

    def forward(self, states, train=True):
        if train:
            return self._forward(states)
        else:
            with torch.no_grad():
                return self._forward(states)
