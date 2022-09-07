import numpy as np
import torch
import pickle as pkl
from torch import nn
import torch.nn.functional as F
from  torch import optim
from torch.utils.data import DataLoader, Dataset
import os


class NN_predictor(object):
    def __init__(self, in_dim, out_dim, DEVICE, model_dir):
        super(NN_predictor, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model =None
        self.make_model()
        self.criterion = nn.MSELoss()
        self.learning_rate = 0.001
        self.batch_size = 64
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.DEVICE = DEVICE
        self.model.double().to(DEVICE)
        self.model_dir = model_dir

    def predict(self, x):
        return self.model.forward(x)
    
    def train(self, x_train, y_train, x_test, y_test, epochs=100):
        train_loss = 0.0
        test_loss = 0.0
        best_loss = np.inf
        train_loader = DataLoader(infer_dataset(x_train, y_train), batch_size=64)
        test_loader = DataLoader(infer_dataset(x_test, y_test), batch_size=64)
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
            with torch.no_grad():
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
        self.load_model()
        self.model.double()
        return self.model

    def make_model(self):
        self.model = N_net(self.in_dim, self.out_dim).double()

    def load_model(self):
        name = "NN_inference.pt"
        model_name = os.path.join(self.model_dir, name)
        self.model = N_net(self.in_dim, self.out_dim).double()
        self.model.load_state_dict(torch.load(model_name))


    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        name = "NN_inference.pt"
        model_name = os.path.join(self.model_dir, name)
        torch.save(self.model.state_dict(), model_name)


class N_net(nn.Module):
    def __init__(self, size_in, size_out):
        super(N_net, self).__init__()
        self.dense_1 = nn.Linear(size_in, 64)
        self.dense_2 = nn.Linear(62, 128)
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


class infer_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)