import numpy as np
import torch
from torch import nn, no_grad
import torch.nn.functional as F
from  torch import optim
from torch.utils.data import DataLoader, Dataset
import os

class NN_dataset(Dataset):
    def __init__(self, feature, target):
        self.len = len(feature)
        self.features = torch.from_numpy(feature).float()
        self.target = torch.from_numpy(target).float()

    def __getitem__(self, idx):
        return self.features[idx, :], self.target[idx]

    def __len__(self):
        return self.len


class NN_predictor(object):
    def __init__(self, in_dim, out_dim, DEVICE, model_dir, reward_type):
        super(NN_predictor, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model =None
        self.make_model()
        self.DEVICE = DEVICE
        self.model.to(DEVICE).float()
        self.reward_type = reward_type
        self.criterion = nn.MSELoss()
        self.learning_rate = 0.001
        self.batch_size = 64
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.online_optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate * 0.1, momentum=0.9)
        self.model_dir = model_dir

    def predict(self, x):
        with no_grad():
            result = self.model.forward(x)
        return result

    def train(self, x_train, y_train, x_test, y_test, epochs):
        train_loss = 0.0
        test_loss = 0.0
        best_loss = np.inf
        train_dataset = NN_dataset(x_train, y_train)
        test_dataset = NN_dataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)
        for e in range(epochs):
            for i, data in enumerate(train_loader):
                x, y_true = data
                self.optimizer.zero_grad()
                x.to(self.DEVICE)
                y_true.to(self.DEVICE)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, torch.unsqueeze(y_true, dim=-1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            print(f'epoch{e}: train average loss {train_loss/ i}.')
            with no_grad():
                for i, data in enumerate(test_loader):
                    x, y_true = data
                    x.to(self.DEVICE)
                    y_true.to(self.DEVICE)
                    y_pred = self.model(x)
                    loss = self.criterion(y_pred, torch.unsqueeze(y_true, dim=-1))
                    test_loss += loss.item()
            if best_loss > test_loss:
                best_loss = test_loss
                self.save_model()
            print(f'epoch{e}: test average loss {test_loss / i}.')
            train_loss = 0.0
            test_loss = 0.0
        self.model = self.model.float()
        return self.model
    
    def train_while_control(self, x, target):
        for idx in range(x.shape[0]):
            x_train, y_true = x[idx,:], target[idx,:]
            
            self.online_optimizer.zero_grad()
            x_train.to(self.DEVICE)
            y_true.to(self.DEVICE)
            y_pred = self.model(x_train)
            loss = self.criterion(y_pred, y_true)
            loss.backward()
            self.optimizer.step()

    def make_model(self):
        self.model = N_net(self.in_dim, self.out_dim).float()

    def load_model(self):
        name = f"NN_inference_{self.reward_type}.pt"
        model_name = os.path.join(self.model_dir, name)
        self.model = N_net(self.in_dim, self.out_dim)
        self.model.load_state_dict(torch.load(model_name))
        self.model = self.model.float()


    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        name = f"NN_inference_{self.reward_type}.pt"
        model_name = os.path.join(self.model_dir, name)
        torch.save(self.model.state_dict(), model_name)
    
    def is_mode(self):
        name = f"NN_inference_{self.reward_type}.pt"
        model_name = os.path.join(self.model_dir, name)
        return os.path.isfile(model_name)


class N_net(nn.Module):
    def __init__(self, size_in, size_out):
        super(N_net, self).__init__()
        self.dense_1 = nn.Linear(size_in, 64)
        self.dense_2 = nn.Linear(64, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 20)
        self.dense_5 = nn.Linear(20, size_out)

    def forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x))
        x = self.dense_5(x)
        return x