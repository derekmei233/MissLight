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
        self.dense_2 = nn.Linear(20, 20*12)
        self.dense_3 = nn.Linear(20, size_out)

    def _forward(self, x):
        x = F.relu(self.dense_1(x))
        x_all = F.relu(self.dense_2(x)).chunk(12,-1)

        x_out = [self.dense_3(x_all[i]) for i in range(12)]
        return x_out

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class IDQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, iid, idx,device):
        super().__init__(action_space, ob_generator, reward_generator)

        self.iid = iid
        self.idx = idx
        self.ob_generator = ob_generator
        ob_length = [self.ob_generator[0].ob_length, self.action_space.n]
        self.ob_length = sum(ob_length)
        self.dense=nn.Linear(96,8)
        self.sub_agents=1
        self.name = self.__class__.__name__

        self.memory = deque(maxlen=4000)
        self.learning_start = 200
        self.update_model_freq = 1
        self.update_target_model_freq = 20
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.criterion = nn.MSELoss()
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.9, centered=False, eps=1e-7)
        self.update_target_network()

    def choose(self, ob, phase, relation=None):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        return self.get_action(ob, phase, relation)

    def get_action(self, ob, phase, relation=None):
        # get all observation now
        ob_oh = one_hot(phase[self.idx], self.action_space.n)
        obs = torch.tensor(np.concatenate((ob[self.idx], ob_oh))).float()
        act_values = self.model.forward(obs)
        act=act_values[0]
        for i in range(1,12):
            act+=act_values[i]
        return torch.argmax(act)
    
    def get_ob(self):
        return [self.ob_generator[0].generate(), np.array(self.ob_generator[1].generate())]

    def sample(self):
        return self.action_space.sample()

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
        minibatch = random.sample(self.memory, self.batch_size)
        
        obs, actions, rewards, next_obs = self._encode_sample(minibatch)
        out = self.target_model.forward(next_obs, train=False)
        target=list()
        for i in range(12):
            target.append(rewards[:,i] + self.gamma * torch.max(out[i], dim=1)[0])
        target_f=list()
        #target=self.dense(torch.cat(target,dim=-1))
        out2 = self.model.forward(obs, train=False)
        for i in range(12):
            target_f.append(out2[i])
        for it in range(12):
            for i, action in enumerate(actions):
                target_f[it][i][action] = target[it][i]
        out3=self.model.forward(obs, train=True)
        #target_p=list()
        loss= self.criterion(out3[0],target_f[0])
        for i in range(1,12):
             loss+=self.criterion(out3[i], target_f[i])
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
