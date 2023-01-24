from torch import nn, no_grad
import torch
import numpy as np
from utils.model_utils import asym_adj
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim 
from pathlib import Path
from utils.preparation import inter2edge_slice, mask_op_with_truth, reconstruct_data_slice
import time
from tqdm import tqdm
# implementation based on https://github.com/nnzhan/Graph-WaveNet with some modification
# dataset [B, T, N, F], input [B, F, N ,T]


class GraphWN_dataset(Dataset):
    def __init__(self, feature, target):
        self.len = len(feature)
        self.features = feature.float()
        self.target = target.float()

    def __getitem__(self, idx):
        return self.features[idx, :], self.target[idx]

    def __len__(self):
        return self.len

def masked_mae(preds, labels, mask):
    loss = torch.abs(preds-labels)
    loss = loss * mask
    return torch.mean(loss)


class GraphWN_predictor(object):
    def __init__(self, N, node_update, adj_matrix, stats, in_dim, out_dim, DEVICE, model_dir):
        super(GraphWN_predictor, self).__init__()
        # 1 for unmasked 2 for masked
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nodes = N
        self.adj_matrix = adj_matrix
        self.DEVICE = DEVICE
        self.model = self.make_model().float()
        self._mean = torch.from_numpy(stats['_mean'].transpose(0,2,1,3)).float().to(self.DEVICE)
        self._std = torch.from_numpy(stats['_std'].transpose(0,2,1,3)).float().to(self.DEVICE)
        self.mask = torch.from_numpy(np.where(node_update == 1,1,0)[:, np.newaxis].repeat(3, axis=1)).T[np.newaxis,:,:,np.newaxis].float().to(self.DEVICE)
        # to evaluate model on 
        self.impute_mask = np.where(node_update == 2,1,0)[:, np.newaxis].repeat(3, axis=1)
        self.reserve_mask = np.where(node_update != 2,1,0)[:, np.newaxis].repeat(3, axis=1)
        self.eval_mask = torch.from_numpy(np.where(node_update == 2,1,0)[:, np.newaxis].repeat(3, axis=1)).T[np.newaxis,:,:,np.newaxis].float().to(self.DEVICE)
        self.criterion = masked_mae
        self.learning_rate = 0.01
        self.batch_size = 64
        self.decay = 0.0001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.decay)
        self.online_optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate * 0.1, weight_decay=self.decay)
        self.model_dir = Path(model_dir)
        self.name = self.__class__.__name__

    def inverse_scale(self, out):
        result = out * self._std + self._mean
        return result

    def scale(self, out):
        result = out
        result[:, 0:3, :, :] = (out[:, 0:3, :, :] -self._mean) / self._std
        return result

    def predict(self,buffer, states, phases, relation, mask_pos, mask_matrix, adj_matrix, mode='select'):
        # TODO: check order and input should be [1, 80, 11, 12]
        # use past input to impute current stat and phase then recover unmasked position
        self.model.eval()
        x = self.scale(torch.from_numpy(buffer.get().transpose(0,2,1,3)).float().to(self.DEVICE))
        result = states.copy()
        y_true_numpy = inter2edge_slice(relation, result, phases, mask_pos)
        y_eval = inter2edge_slice(relation, result, phases, []).T[np.newaxis,:3,:,np.newaxis]
        y_eval = torch.from_numpy(y_eval).float().to(self.DEVICE)
        with no_grad():
            h = self.model(x)
            edge_feature = self.inverse_scale(h)
            # prediction value used later in GraphWN
            loss = self.criterion(edge_feature, y_eval, self.eval_mask)
            # 1,3,80,1
        impute = edge_feature.to("cpu").numpy().transpose(0,2,1,3).squeeze().squeeze()
        prediction = reconstruct_data_slice(impute, None, relation) # current state
        infer = mask_op_with_truth(y_true_numpy, mask_matrix, adj_matrix, mode)

        infer[:, :3] = impute * self.impute_mask + infer[:, :3] * self.reserve_mask
        new_buffer = infer
        buffer.add(new_buffer[np.newaxis, :, :, np.newaxis])
        for i in mask_pos:
            # mask with inference value        
            result[i, :] = prediction[i, :]
        return result, loss

    def pred(self, states, traj, relation, mask_pos):
        self.model.eval()
        x = self.scale(torch.from_numpy(traj.transpose(0,2,1,3)).float().to(self.DEVICE))
        result = states
        with no_grad():
            h = self.model(x)
            edge_feature = self.inverse_scale(h)
            # prediction value used later in GraphWN
        impute = edge_feature.to("cpu").numpy().transpose(0,2,1,3).squeeze().squeeze()
        prediction = torch.from_numpy(reconstruct_data_slice(impute, None, relation)) # current state
        for i in mask_pos:
            # mask with inference value        
            result[i, :] = prediction[i, :]
        return result


    def train_while_control(self, buffer, states, phases, relation, mask_pos):
        self.model.train()
        self.optimizer.zero_grad()
        x = self.scale(torch.from_numpy(buffer.get(-1).transpose(0,2,1,3)).float().to(self.DEVICE))
        result = states.copy()
        y_true_numpy = inter2edge_slice(relation, result, phases, mask_pos).T[np.newaxis,:3,:,np.newaxis]
        y_eval = torch.from_numpy(y_true_numpy).float().to(self.DEVICE)
        h = self.model(x)
        edge_feature = self.inverse_scale(h)
        loss = self.criterion(edge_feature, y_eval, self.mask)
        loss.backward()
        self.optimizer.step()

    def train(self, x_train, y_train, x_val, y_val, epochs):
        self.model.train()
        train_loss = 0.0
        val_loss = 0.0
        best_loss = np.inf
        train_dataset = GraphWN_dataset(torch.from_numpy(x_train.transpose(0, 2, 1, 3)), torch.from_numpy(y_train.transpose(0, 2, 1, 3)))
        val_dataset = GraphWN_dataset(torch.from_numpy(x_val.transpose(0, 2, 1, 3)), torch.from_numpy(y_val.transpose(0, 2, 1, 3)))
        train_loader = DataLoader(train_dataset, batch_size=128)
        val_loader = DataLoader(val_dataset, batch_size=128)
        begin = time.time()
        with no_grad():
            for i, data in enumerate(val_loader):
                x, y_true = data
                x = x.to(self.DEVICE)
                y_true = y_true.to(self.DEVICE)
                y_pred = self.model(x)
                loss = self.criterion(self.inverse_scale(y_pred), y_true, self.mask)
                val_loss += loss.item()
        print(f'before training val average loss {val_loss / i}.')
        val_loss = 0.0
        for e in range(epochs):
            for i, data in enumerate(tqdm(train_loader)):
                x, y_true = data
                self.optimizer.zero_grad()
                x = x.to(self.DEVICE)
                y_true = y_true.to(self.DEVICE)
                y_pred = self.model(x)
                loss = self.criterion(self.inverse_scale(y_pred), y_true,  self.mask)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            print(f'epoch{e}: train average loss {train_loss/ i}.')
            with no_grad():
                for i, data in enumerate(val_loader):
                    x, y_true = data
                    x = x.to(self.DEVICE)
                    y_true = y_true.to(self.DEVICE)
                    y_pred = self.model(x)
                    loss = self.criterion(self.inverse_scale(y_pred), y_true, self.mask)
                    val_loss += loss.item()
            if best_loss > val_loss:
                best_loss = val_loss
                self.save_model()
            print(f'epoch{e}: val average loss {val_loss / i}.')
            train_loss = 0.0
            val_loss = 0.0
        end = time.time()
        print(end - begin)
        return self.model

    def eval(self, x_test, y_test):
        self.load_model()
        self.model.eval()
        test_dataset = GraphWN_dataset(torch.from_numpy(x_test.transpose(0, 2, 1, 3)), torch.from_numpy(y_test.transpose(0, 2, 1, 3)))
        test_loader = DataLoader(test_dataset, batch_size=128)
        test_loss = 0.0
        with no_grad():
            for i, data in enumerate(test_loader):
                x, y_true = data
                x.to(self.DEVICE)
                y_true.to(self.DEVICE)
                y_pred = self.model(x)
                loss = self.criterion(self.inverse_scale(y_pred), y_true, self.eval_mask)
                test_loss += loss.item()
        print(f'test average loss {test_loss / i}.')

    def make_model(self):
        model = GraphW_net(N=self.nodes, adj_matrix=self.adj_matrix, in_dim=self.in_dim, out_dim=self.out_dim, device=self.DEVICE).to(self.DEVICE).float()
        return model

    def load_model(self):
        name = f"GraphWN_inference.pt"
        model_name = Path.joinpath(self.model_dir, name)
        self.model = self.make_model()
        self.model.load_state_dict(torch.load(model_name))

    def save_model(self):
        if not Path.exists(self.model_dir): 
            Path.mkdir(self.model_dir, parents=True)
        name = f"GraphWN_inference.pt"
        model_name = Path.joinpath(self.model_dir, name)
        torch.save(self.model.state_dict(), model_name)
        print('update model')
    
    def is_model(self):
        name = f"GraphWN_inference.pt"
        model_name = Path.joinpath(self.model_dir, name)
        return Path.exists(model_name)


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout, order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*3+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

# TODO: should not give phase a normalization
class GraphW_net(nn.Module):
    def __init__(self, N, adj_matrix, dropout=0.3, channels=32, in_dim=11, out_dim=3, device='cpu'):
        super(GraphW_net, self).__init__()
        self.dropout = dropout
        self.blocks = 4
        self.kernel = 2
        self.layers = 2
        self.device = device
        # Fix base
        self.support = [torch.tensor(i).to(self.device) for i in [asym_adj(adj_matrix), asym_adj(np.transpose(adj_matrix))]]

     
        self.residual_channels = channels
        self.dilation_channels = channels
        self.skip_channels = channels * 8
        self.end_channels = channels * 16

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        # start encode 
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.residual_channels, kernel_size=(1,1))

        receptive_field = 1
        self.nodevec1 = nn.Parameter(torch.randn(N, 10), requires_grad=True).to(self.device)
        self.nodevec2 = nn.Parameter(torch.randn(10, N), requires_grad=True).to(self.device)
        for _ in range(self.blocks):
            additional_scope = self.kernel - 1
            new_dilation = 1
            for _ in range(self.layers):
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels, out_channels=self.dilation_channels, kernel_size=(1, self.kernel), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels, out_channels=self.dilation_channels, kernel_size=(1, self.kernel), dilation=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.dilation_channels, out_channels=self.residual_channels, kernel_size=(1,1)))
                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_channels, out_channels=self.skip_channels, kernel_size=(1,1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                
                self.gconv.append(gcn(self.dilation_channels, self.residual_channels, self.dropout))
        
        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels, out_channels=self.end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels, out_channels=out_dim, kernel_size=(1,1))

        self.receptive_field = receptive_field

    def forward(self, input):
        # pad (1,0,0,0) to keep T same dimenson before reaching gcn_adptive layer
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field-in_len, 0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        adp_adj_matrix = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        new_supports = self.support + [adp_adj_matrix]

        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)

            x = x + residual[:,:,:,-x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
