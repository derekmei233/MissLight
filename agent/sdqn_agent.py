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
from utils.preparation import one_hot, build_relation, get_road_adj, get_mask_matrix
from utils.mask_pos import random_mask
from world import World
from environment import TSCEnv



class SDQN(nn.Module):
    def __init__(self, size_in, size_out):
        super(SDQN, self).__init__()
        self.dense_1 = nn.Linear(size_in, 20)
        self.dense_2 = nn.Linear(20, 20)
        self.dense_3 = nn.Linear(20, size_out)

    def _forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)

def build_shared_model(ob_length, action_space):
    # Neural Net for Deep-Q learning Model
    model = SDQN(ob_length, action_space.n)
    return model


class SDQNAgent(RLAgent):
    """ the logic behind SDQN. Since SDQN is implemented on all intersections, the only difference is learning from
        all or observable intersections, we use it to process all information but only give it accessibility to observable ones
    """

    def __init__(self, action_space, ob_generator, reward_generator, iid, idx, q_model, target_q_model, optimizer):
        super().__init__(action_space, ob_generator, reward_generator)
        self.iid = iid
        self.idx = idx # learnable index
        self.sub_agents = len(iid)

        self.ob_generator = ob_generator
        ob_length = [self.ob_generator[0][0].ob_length, self.action_space.n]
        self.ob_length = sum(ob_length)

        self.learnable = len(self.idx)
        self.memory = [deque(maxlen=10000) for i in range(self.learnable)] # number of samples 

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
        self.model = q_model
        self.target_model = target_q_model
        self.optimizer = optimizer

    def choose(self, ob, phase, relation=None):
        if np.random.rand() <= self.epsilon:
            return self.sample()
        return self.get_action(ob, phase, relation)

    def get_action(self, ob, phase, relation=None):
        # get all observation now
        actions = []
        for idx in range(self.sub_agents):
            ob_oh = one_hot(phase[idx], self.action_space.n)
            obs = torch.tensor(np.concatenate((ob[idx], ob_oh))).float()
            act_values = self.model.forward(obs, train=False)
            actions.append(torch.argmax(act_values))
        return actions
    
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

    def remember(self, ob, action, reward, next_ob , idx):
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
        self.model = SDQN(self.ob_length, self.action_space.n)
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
                IntersectionPhaseGenerator(world, inter ,targets=['cur_phase'], negative=False)
            ])
        reward_generator.append(LaneVehicleGenerator(world, inter, ['lane_waiting_count'], in_only=True, average=None, negative=True))
        iid.append(inter.id)
    action_space = gym.spaces.Discrete(len(world.intersections[-1].phases))
    ob_length = ob_generator[0][0].ob_length + action_space.n
    q_model = build_shared_model(ob_length, action_space)
    target_q_model = build_shared_model(ob_length, action_space)
    optimizer = optim.RMSprop(q_model.parameters(), lr=0.001, alpha=0.9, centered=False, eps=1e-7)
    agents.append(SDQNAgent(action_space, ob_generator, reward_generator, iid, obs_pos, q_model, target_q_model, optimizer))
    env = TSCEnv(world, agents, None)
    print('construction finished')
    