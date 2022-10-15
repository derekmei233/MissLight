import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from utils.model_utils import re_normalization, onehot_to_phase, generate_actphase, revise_unknown


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.order = order
        self.dropout = dropout

        self.linear = nn.Linear(c_in * (order * support_len + 1), c_out)

    def forward(self, x, support):
        # x is (B, F, N, T)
        support = [torch.tensor(i).to(x.device) for i in support[:-1]] + [support[-1]]
        out = [x]
        for basis in support:
            x1 = torch.einsum('bfnt,nm->bfmt', (x, basis))
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('bfnt,nm->bfmt', (x1, basis))
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)  # (B, F * (order * sup_len + 1), N, T)
        h = h.permute(0, 2, 3, 1)  # (B, N, T, F * (order * sup_len + 1))
        h = self.linear(h)
        h = F.dropout(h, self.dropout).permute(0, 3, 1, 2)
        return h  # (B, N, F, T)


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
        x_renor = re_normalization(x.cpu().numpy(), _mean, _std)[:, :, 3:]
        x_renor_ = np.where((np.abs(x_renor-1.) >= 1e-6), 0, 1)
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
        return phase_matrix


class GCN_act(nn.Module):
    def __init__(self, c_in, c_out, dropout, adj_phase, refill, mask_matrix, supports, support_len=3, order=2):
        super(GCN_act, self).__init__()
        self.order = order
        self.dropout = dropout
        self.supports = supports
        self.linear = nn.Linear(c_in * (order * support_len + 1), c_out)
        self.adj_phase = adj_phase
        self.act_phase_list = [Phase_Act_layer(np.eye(adj_phase.shape[0]), self.adj_phase)]
        self.act_phase_list += [Phase_Act_layer(i, self.adj_phase) for i in self.supports[:-1]]
        self.refill_kernel = Phase_Act_layer(self.supports[0], self.adj_phase)
        self.refill = refill
        self.mask_matrix = mask_matrix

    def forward(self, x, x_phase, _mean, _std, supports):
        # x is (B, F, N, T) -> (B, N, F, T)
        x = x.permute(0, 2, 1, 3)
        adp = supports[-1]
        if self.refill == 'refill':
            x_pass = self.refill_kernel(x_phase, _mean, _std)  # ()
            x_pass = torch.from_numpy(
                x_pass).type(torch.FloatTensor).to(x.device)  # (1,B,T,N,N)
            x = x.permute(0, 3, 1, 2)
            x_refill = torch.matmul(x_pass, x).squeeze().permute(0, 2, 3, 1)  # (1, B, T, N, F) -> (B, N, F,T)
            # TODO: test :-1
            x_refill = revise_unknown(x.permute(0, 2, 3, 1), x_refill[:, :, :, 1:], self.mask_matrix).to(x.device)
            #print(torch.all(x_refill.permute(0, 3, 1, 2) == x))
            x_refill = x_refill.permute(0, 3, 1, 2)  # (B, T, N, F)
        else:
            x_refill = x.permute(0, 3, 1, 2)

        tmp = self.act_phase_list[0](x_phase, _mean, _std)
        tmp = torch.from_numpy(tmp).type(torch.FloatTensor).to(x_refill.device)  # (1,B,T,N,N)
        out = [torch.matmul(tmp, x_refill).squeeze().permute(0, 3, 2, 1)]  # (B, T, N, F)
        for idx, basis in enumerate(self.supports):
            active_matrix = self.act_phase_list[idx]
            tmp = active_matrix(x_phase, _mean, _std)
            tmp = torch.from_numpy(tmp).type(torch.FloatTensor).to(x_refill.device)
            x1 = torch.matmul(tmp, x_refill).squeeze().permute(0, 3, 2, 1)
            out.append(x1)
            for k in range(2, self.order + 1):
                tmp = active_matrix(x_phase, _mean, _std)
                tmp = torch.from_numpy(tmp).type(torch.FloatTensor).to(x_refill.device)
                x2 = torch.matmul(tmp, x1.permute(0, 3, 2, 1)).squeeze().permute(0, 3, 2, 1)
                out.append(x2)
                x1 = x2
        # learnable parameters
        active_matrix = self.act_phase_list[0]
        tmp = active_matrix(x_phase, _mean, _std)
        tmp = torch.from_numpy(tmp).type(torch.FloatTensor).to(x_refill.device)
        adp = adp.repeat(x_refill.shape[0], x_refill.shape[1], 1, 1)  # (B, T, N, N)
        adp = adp * tmp  # TODO: or ,mm
        x1 = torch.matmul(adp, x_refill).squeeze().permute(0, 3, 2, 1)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = torch.matmul(adp, x1.permute(0, 3, 2, 1)).squeeze().permute(0, 3, 2, 1)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)  # (B, F * (order * sup_len + 1), N, T)
        h = h.permute(0, 2, 3, 1)  # (B, N, T, F * (order * sup_len + 1))
        h = self.linear(h)
        h = F.dropout(h, self.dropout).permute(0, 3, 1, 2)
        return h  # (B, N, F, T)


class GraphWNblocks_act(nn.Module):
    def __init__(self, residual_channel, dilation_channel, receptive_field, additional_scope,
                 kernel_size, skip_channel, dropout, adj_phase,
                 new_dilation, layers, supports, refill, mask_matrix):
        super(GraphWNblocks_act, self).__init__()
        self.additional_scope = kernel_size - 1
        self.receptive_field = receptive_field
        self.new_dilation = new_dilation
        self.additional_scope = additional_scope
        self.dropout = dropout
        self.adj_pahse = adj_phase
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.supports = supports
        self.layers = layers
        self.supports_len = len(supports) + 1
        self.adj_phase = adj_phase
        self.refill = refill
        self.mask_matrix = mask_matrix
        self.bn = nn.ModuleList()
        for i in range(self.layers):
            self.filter_convs.append(nn.Conv2d(residual_channel, dilation_channel, kernel_size=(1, kernel_size),
                                               dilation=self.new_dilation))
            # conv1 in original paper
            self.gate_convs.append(nn.Conv2d(residual_channel, dilation_channel, kernel_size=(1, kernel_size),
                                             dilation=self.new_dilation))
            self.residual_convs.append(nn.Conv2d(dilation_channel, residual_channel, kernel_size=(1, 1)))
            self.skip_convs.append(nn.Conv2d(dilation_channel, skip_channel, kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channel))
            self.new_dilation = self.new_dilation * 2
            self.receptive_field += self.additional_scope
            self.additional_scope *= 2
            if i == 0:
                self.gconv.append(GCN_act(dilation_channel, residual_channel, self.dropout, self.adj_phase,
                                          refill, mask_matrix, self.supports, self.supports_len))
            else:
                self.gconv.append(
                    GCN(dilation_channel, residual_channel, self.dropout, self.supports_len))
    def get_dilation(self):
        return self.new_dilation

    def get_recept(self):
        return self.receptive_field

    def get_scope(self):
        return self.additional_scope

    def forward(self, x, x_phase, skip, adp, _mean, _std):
        new_supports = self.supports + [adp]
        for i in range(self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3)::]
            except(Exception):
                skip = 0
            skip = s + skip
            x = self.gconv[i](x, x_phase, _mean, _std, new_supports)
            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)
            return x, skip


class GraphWNblocks(nn.Module):
    def __init__(self, residual_channel, dilation_channel, receptive_field, additional_scope,
                 kernel_size, skip_channel, dropout,
                 new_dilation, layers, supports):
        super(GraphWNblocks, self).__init__()
        self.additional_scope = kernel_size - 1
        self.receptive_field = receptive_field
        self.new_dilation = new_dilation
        self.additional_scope = additional_scope
        self.dropout = dropout
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.supports = supports
        self.layers = layers
        self.supports_len = len(supports) + 1
        self.bn = nn.ModuleList()
        for _ in range(self.layers):
            self.filter_convs.append(nn.Conv2d(residual_channel, dilation_channel, kernel_size=(1, kernel_size),
                                               dilation=self.new_dilation))
            # conv1 in original paper
            self.gate_convs.append(nn.Conv2d(residual_channel, dilation_channel, kernel_size=(1, kernel_size),
                                             dilation=self.new_dilation))
            self.residual_convs.append(nn.Conv2d(dilation_channel, residual_channel, kernel_size=(1, 1)))
            self.skip_convs.append(nn.Conv2d(dilation_channel, skip_channel, kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channel))
            self.new_dilation *= 2
            self.receptive_field += self.additional_scope
            self.additional_scope *= 2
            self.gconv.append(GCN(dilation_channel, residual_channel, self.dropout, self.supports_len))

    def get_dilation(self):
        return self.new_dilation

    def get_recept(self):
        return self.receptive_field

    def get_scope(self):
        return self.additional_scope

    def forward(self, x, x_phase, skip, adp, _mean, _std):
        new_supports = self.supports + [adp]
        for i in range(self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3)::]
            except(Exception):
                skip = 0
            skip = s + skip
            x = self.gconv[i](x, new_supports)
            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)
            return x, skip


class GraphWN(nn.Module):
    def __init__(self, N, T, supports, dropout, adj_phase, channels, DEVICE, in_dim, output_window, kernel,
                 blocks, layers, refill, mask_matrix):
        super(GraphWN, self).__init__()
        self.device = DEVICE
        self.dropout = dropout
        self.blocks = blocks
        self.supports = supports
        self.residual_channel = channels[0]
        self.dilation_channel = channels[1]
        self.kernel_size = kernel
        self.skip_channel = channels[2]
        self.end_channels = channels[3]

        # take random initialize E matrix
        self.layers = layers
        self.input_window = T
        self.output_dim = 1
        self.output_window = output_window
        self.receptive_field = self.output_dim
        self.new_dilation = 1
        self.additional_scope = 1
        self.adj_phase = adj_phase
        self.refill = refill
        self.mask_matrix = mask_matrix


        self.nodevec1 = nn.Parameter(torch.FloatTensor(N, 10).to(self.device))
        self.nodevec2 = nn.Parameter(torch.FloatTensor(10, N).to(self.device))
        if self.supports is None:
            self.supports = []
        self.support_len = len(supports) + 1
        self.start_conv = nn.Conv2d(in_dim, self.residual_channel, (1, 2))
        self.blocks = nn.ModuleList()
        for i in range(blocks):
            if i == 0:
                l_i = GraphWNblocks_act(self.residual_channel, self.dilation_channel, self.receptive_field,
                                        self.additional_scope,  self.kernel_size, self.skip_channel,
                                        self.dropout, self.adj_phase, self.new_dilation, self.layers, supports,
                                        self.refill, self.mask_matrix)
            elif i == blocks - 1:
                l_i = GraphWNblocks(self.residual_channel, self.dilation_channel, self.receptive_field,
                                    self.additional_scope, self.kernel_size, self.skip_channel,
                                    self.dropout, self.new_dilation - 1, self.layers, supports)
            else:
                l_i = GraphWNblocks(self.residual_channel, self.dilation_channel, self.receptive_field,
                                    self.additional_scope, self.kernel_size, self.skip_channel,
                                    self.dropout, self.new_dilation, self.layers, supports)
            self.blocks.append(l_i)
            self.new_dilation = l_i.get_dilation()
            self.receptive_field = l_i.get_recept()
            self.additional_scope = l_i.get_scope()
        self.end_conv_1 = nn.Conv2d(self.skip_channel, self.end_channels, (1, 1), bias=True)

        self.end_conv_2 = nn.Conv2d(self.end_channels, output_window, (1, 1), bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.nodevec1)
        init.xavier_uniform_(self.nodevec2)

    def forward(self, x, _mean, _std):
        # (B, N, F, T)
        x_phase = x
        x = x.permute(0, 2, 1, 3)  # (B, F, N, T)
        x = nn.functional.pad(x, (1, 0, 0, 0))
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = x
        x = self.start_conv(x)
        skip = 0
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        for block in self.blocks:
            x, skip = block(x, x_phase, skip, adp, _mean, _std)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x.permute(0, 2, 1, 3)
        return x


def get_model(N, T, supports, drop, adj_phase, channels, DEVICE, in_dim, out_dim, kernel, blocks,
              layers, refill, mask_matrix):
    model = GraphWN(N, T, supports, drop, adj_phase, channels, DEVICE, in_dim, out_dim, kernel, blocks,
                    layers, refill, mask_matrix)
    return model
