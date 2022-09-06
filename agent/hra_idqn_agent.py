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
        self.dense_2 = nn.Linear(20, 80)
        self.dense_3 = nn.Linear(20, size_out)
        self.softmax = nn.Softmax(-1)

    def _forward(self,x):
        x=F.relu(self.dense_1(x))
        x=F.relu(self.dense_2(x)).chunk(4,-1)
        x1=self.dense_3(x[0])
        x2 = self.dense_3(x[1])
        x3 = self.dense_3(x[2])
        x4 = self.dense_3(x[3])
        return x1,x2,x3,x4

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class IDQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, iid, idx):
        super().__init__(action_space, ob_generator, reward_generator)

        self.iid = iid
        self.idx = idx
        self.ob_generator = ob_generator
        ob_length = [self.ob_generator[0].ob_length, self.action_space.n]
        self.ob_length = sum(ob_length)

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

    def choose(self, ob):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        ob_oh = one_hot(ob[1], self.action_space.n)
        ob = torch.tensor(np.concatenate((ob[0], ob_oh))).float()
        act_values_1,act_values_2,act_values_3,act_values_4 = self.model.forward(ob)
        act_values=act_values_1+act_values_2+act_values_3+act_values_4
        return torch.argmax(act_values)

    def get_action(self, ob, phase, relation=None):
        # get all observation now
        ob_oh = one_hot(phase[self.idx], self.action_space.n)
        obs = torch.tensor(np.concatenate((ob[self.idx], ob_oh))).float()
        act_values_1, act_values_2, act_values_3, act_values_4 = self.model.forward(obs)
        act_values = act_values_1 + act_values_2 + act_values_3 + act_values_4
        return torch.argmax(act_values)
    
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
        #print(rewards[:,0:3])
        out_1,out_2,out_3,out_4 = self.target_model.forward(next_obs, train=False)
        #print(out_1)
        target_1 = torch.mean(rewards[:,0:3],dim=1) + self.gamma * torch.max(out_1, dim=1)[0]
        target_2 = torch.mean(rewards[:,3:6],dim=1) + self.gamma * torch.max(out_2, dim=1)[0]
        target_3 = torch.mean(rewards[:,6:9],dim=1) + self.gamma * torch.max(out_3, dim=1)[0]
        target_4 = torch.mean(rewards[:,9:12],dim=1) + self.gamma * torch.max(out_4, dim=1)[0]
        target=torch.stack([target_1,target_2,target_3,target_4],dim=0)
        target_f = torch.stack(self.model.forward(obs, train=False),dim=0)
        for i, action in enumerate(actions):
            target_f[0][i][action] = target[0][i]
            target_f[1][i][action] = target[1][i]
            target_f[2][i][action] = target[2][i]
            target_f[3][i][action] = target[3][i]
        loss = self.criterion(torch.stack(self.model.forward(obs, train=True),dim=0), target_f)
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
    
