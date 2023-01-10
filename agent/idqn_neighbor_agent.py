import gym
import torch

from .rl_agent import RLAgent
import numpy as np
from collections import deque
import os
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random
from utils.preparation import one_hot


class IDQN(nn.Module):
    def __init__(self, size_in, size_out):
        super(IDQN, self).__init__()
        self.dense_1 = nn.Linear(size_in, 20)
        self.dense_2 = nn.Linear(20, 20)
        self.dense_3 = nn.Linear(20, size_out)
        self.val_dense=nn.Linear(20,1)

    def _forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        act = self.dense_3(x)
        val=self.val_dense(x)
        Q = val + act - act.mean(-1).view(-1, 1)
        return Q


    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)

class NIDQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, iid, idx, device='cpu'):
        super().__init__(action_space, ob_generator, reward_generator)
        
        self.iid = iid
        self.sub_agents = 1
        self.idx = idx
        self.name = self.__class__.__name__
        self.device = device
        
        self.ob_generator = ob_generator
        ob_length = sum([ob.ob_length for ob in self.ob_generator[0]]) + self.action_space.n
        
        self.ob_length = ob_length

        self.memory = deque(maxlen=5000)
        self.learning_start = 1000
        self.update_model_freq = 1
        self.update_target_model_freq = 20
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.batch_size = 64

        self.criterion = nn.MSELoss()
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.9, centered=False, eps=1e-7)
        self.update_target_network()

    def choose(self, ob, phase, relation=None):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        return self.get_action(ob, phase, relation)

    def get_action(self, ob, oh_phase, relation=None):
        # get all observation now
        ob_oh = one_hot(oh_phase[self.idx], self.action_space.n)
        obs = torch.tensor(np.concatenate((ob[self.idx], ob_oh))).float().to(self.device)
        act_values = self.model.forward(obs, train=False)
        return torch.argmax(act_values).clone().numpy()

    def get_ob(self):
        obs = np.concatenate([ob.generate() for ob in self.ob_generator[0]])
        return [np.concatenate([ob_g.generate() for ob_g in self.ob_generator[0]]), np.array(self.ob_generator[1].generate())]

    def get_delay(self):
        return np.mean(self.ob_generator[2].generate())

    def sample(self):
        return self.action_space.sample()

    def get_reward(self):
        rewards = [r.generate() for r in self.reward_generator]
        reward = sum(rewards)
        return reward.squeeze()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = IDQN(self.ob_length, self.action_space.n)
        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob, action, reward, next_ob))

    def _encode_sample(self, minibatch):
        # TODO: check dimension
        obses_t, actions_t, rewards_t, obses_tp1 = list(zip(*minibatch))
        obs = [np.squeeze(np.stack(obs_i)) for obs_i in list(zip(*obses_t))]
        obs_oh = one_hot(obs[1], self.action_space.n)
        obs = np.concatenate((obs[0], obs_oh), axis=1)
        next_obs = [np.squeeze(np.stack(obs_i)) for obs_i in list(zip(*obses_tp1))]
        # expand acton to one_hot
        next_obs_oh = one_hot(next_obs[1], self.action_space.n)
        next_obs = np.concatenate((next_obs[0], next_obs_oh), axis=1)

        rewards = np.array(rewards_t, copy=False)
        obs = torch.from_numpy(obs).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        return obs, actions_t, rewards, next_obs

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        
        obs, actions, rewards, next_obs = self._encode_sample(minibatch)
        # 4 output 
        out = self.target_model.forward(next_obs, train=False)
        tmp = self.gamma * torch.max(out, dim=1)[0]
        target = rewards + tmp
        target_f = self.model.forward(obs, train=False)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model.forward(obs, train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, model_dir):
        # TODO: add idqn
        name = "idqn_{}.pt".format(self.iid)
        model_name = os.path.join(model_dir, name)
        self.model = IDQN(self.ob_length, self.action_space.n)
        self.model.load_state_dict(torch.load(model_name))
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        name = "idqn_{}.pt".format(self.iid)
        model_name = os.path.join(model_dir, name)
        torch.save(self.model.state_dict(), model_name)
