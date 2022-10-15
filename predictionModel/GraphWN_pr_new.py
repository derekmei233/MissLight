import torch.nn as nn
import torch
import numpy as np
from utils.model_utils import asym_adj
import torch.nn.functional as F
# implementation based on https://github.com/nnzhan/Graph-WaveNet




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

class Phase_Act_layer(nn.Module):
    def __init__(self, adj_mx, adj_phase):
        super(Phase_Act_layer, self).__init__()
        self.adj_mx = adj_mx
        # (N,N)
        self.adj_phase = adj_phase

    def forward(self, x, _mean, _std):
        '''
        :param x:(b,N,F,T)
        :param _mean:(1,1,F(11),1)
        :param _std:(1,1,F(11),1)
        '''
        _,N,_,_ = x.shape
        # TODO: no re-Normalization now
        x_renor = x[:,:,3:]
        x_renor_ = np.where((x==1.), 0, 1)
        # (B,T,N,2)
        onehot2phase = onehot_to_phase(x_renor_)
        # (B,T,N,N)
        # compute phase_act matrix of each time according to adj_phase,x_next_phase
        phase_matrix_list = []
        if len(self.adj_mx.shape) == 3:
            for l in range(self.adj_mx.shape[0]):
                phase_matrix_list.append(generate_actphase(onehot2phase, self.adj_mx[l, :, :], self.adj_phase))
        else:
            phase_matrix_list.append(generate_actphase(onehot2phase, self.adj_mx[:, :], self.adj_phase))
        # add self_loop
        phase_matrix = np.stack(phase_matrix_list, axis=0)  # (K, B, T, N, N)
        phase_matrix = phase_matrix + np.eye(N)
        return 


class gcn_adaptive(nn.Modules):
    def __init__(self,c_in,c_out, adj_phase, dropout, order=2):
        super(gcn_adaptive,self).__init__()
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
class GraphWN(nn.Module):
    def __init__(self, N, adj_matrix, adj_phase, dropout=0.3, channels=32, in_dim=11, out_dim=3, device='cpu'):
        super(GraphWN, self).__init__()
        self.dropout = dropout
        self.blocks = 5
        self.kernel = 2
        self.layers = 2
        # Fix base
        self.support = [torch.tensor(i).to(device) for i in [asym_adj(adj_matrix), asym_adj(np.transpose(adj_matrix))]]

     
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
        self.nodevec1 = nn.Parameter(torch.randn(N, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, N).to(device), requires_grad=True).to(device)

        for b in self.blocks:
            additional_scope = self.kernel - 1
            new_dilation = 1
            for l in self.layers:
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels, out_channels=self.dilation_channels, kernel_size=(1, self.kernel), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels, out_channels=self.dilation_channels, kernel_size=(1, self.kernel), dilation=new_dilation))
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels, out_channels=self.residual_channels, kernel_size=(1,1)))
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels, out_channels=self.skip_channels, kernel_size=(1,1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))

                new_dilation += 2
                receptive_field += additional_scope
                additional_scope *= 2
            if b == 1 and l == 1:
                self.gconv.append(gcn_adaptive(self.dilation_channels, self.residual_channels, adj_phase))
            else:
                self.gconv.append(gcn(self.dilation_channels, self.residual_channels))
        
        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels, out_channels=self.end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels, out_channels=out_dim, kernel_size=(1,1))

        self.receptive_field = receptive_field

    def forwrd(self, input):
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
