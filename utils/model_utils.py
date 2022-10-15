import numpy as np
import scipy.sparse as sp
import torch


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32)

def onehot_to_phase(phase):
    # phase_dic = {
    #     0: [[0,1],[2,1]],#['WS', 'ES'],
    #     1: [[3,1],[1,1]],#['NS', 'SS'],
    #     2: [[0,0],[2,0]],#['WL', 'EL'],
    #     3: [[3,0],[1,0]],#['NL', 'SL'],
    #     4: [[0,1],[0,0]],#['WS', 'WL'],
    #     5: [[2,1],[2,0]],#['ES', 'EL'],
    #     6: [[3,1],[3,0]],#['NS', 'NL'],
    #     7: [[1,1],[1,0]],#['SS', 'SL']
    # }
    phase_dic = {
        0: ['WS', 'ES'],
        1: ['NS', 'SS'],
        2: ['WL', 'EL'],
        3: ['NL', 'SL'],
        4: ['WS', 'WL'],
        5: ['ES', 'EL'],
        6: ['NS', 'NL'],
        7: ['SS', 'SL']
    }
    # phase:[B,N,8,T]->[B,T,N,8]
    phase = np.transpose(phase, (0, 3, 1, 2))
    batch_size, num_of_timesteps, num_of_vertices, _ = phase.shape
    phase_more = np.full((batch_size, num_of_timesteps,
                         num_of_vertices, 2, 2), ['XX', 'XX'])
    # idx must euqals to B*T*N
    idx = np.argwhere(phase == 1)
    assert len(idx) == batch_size*num_of_timesteps*num_of_vertices
    for x in idx:
        phase_more[x[0], x[1], x[2]] = np.array(phase_dic[x[3]])
    return phase_more

def generate_actphase(phase, adj_mx, adj_phase):
    batch_size, num_of_timesteps, num_of_vertices, phase_row, phase_col = phase.shape
    # self.phase_act:record adj matrix of every time(after activation)
    phase_act = np.zeros((batch_size, num_of_timesteps,
                         num_of_vertices, num_of_vertices))
    phase_act = phase_act.reshape(-1, num_of_vertices, num_of_vertices)
    for idx, adj_x in enumerate(adj_mx.flat):
        if adj_x == 1.:
            if idx >= num_of_vertices:
                source = int(idx/num_of_vertices)
                target = idx-source*num_of_vertices
            else:
                source = 0
                target = idx
            phase_node = phase[:, :, source].reshape(-1, phase_row, phase_col)

            for phase_idx, x in enumerate(phase_node):
                if adj_phase[source][target] in x or adj_phase[source][target][1] == 'R':
                    phase_act[phase_idx][source][target] = 1.
    phase_act = phase_act.reshape(
        batch_size, num_of_timesteps, num_of_vertices, num_of_vertices)
    return phase_act

def revise_unknown(origin_data, predict_data, mask_matrix):
    '''
    :param origin_data:(b,N,F,T)
    :param predict_data:(b,N,F,T-1)
    '''
    revise_data = torch.zeros_like(origin_data,dtype=torch.float)
    for node_idx, node in enumerate(mask_matrix):
        if node != 1:
            revise_data[:, node_idx, :, 0] = origin_data[:, node_idx, :, 0]
            revise_data[:, node_idx, :, 1:] = predict_data[:, node_idx, :, :]
        else:
            revise_data[:, node_idx] = origin_data[:, node_idx]
    return revise_data

def re_normalization(x, mean, std):
    x = x * std + mean
    return x