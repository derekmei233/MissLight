import numpy as np
import scipy.sparse as sp


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
    # phase:[B,8,N,T]->[B,T,N,8]
    phase = np.transpose(phase, (0, 3, 2, 1))
    batch_size, num_of_timesteps, num_of_vertices, _ = phase.shape
    phase_more = np.full((batch_size, num_of_timesteps,
                         num_of_vertices, 2, 2), ['XX', 'XX'])
    # idx must euqals to B*T*N
    idx = np.argwhere(phase == 1)
    # assert len(idx) == batch_size*num_of_timesteps*num_of_vertices
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

def re_normalization(x, mean, std):
    x = x * std + mean
    return x

def get_road_adj_phase(relation):
    road_info = relation['road_links']
    road_dict_road2id = relation['road_dict_road2id']
    num_roads = len(relation['road_dict_id2road'])
    adjacency_matrix = np.zeros(
        (int(num_roads), int(num_roads)), dtype=np.float32)
    '''
    "PHASE": [
        'WSES':0
        'NSSS':1
        'WLEL':2
        'NLSL':3
        'WSWL':4
        'ESEL':5
        'NSNL':6
        'SSSL':7
    ],
    '''

    adj_phase = np.full(
        (int(num_roads), int(num_roads)), 'XX')

    for inter_dic in road_info:
        for link_dic in road_info[inter_dic]:
            source = link_dic[0]
            target = link_dic[1]
            type_p = link_dic[2]
            direction = link_dic[3]

            if type_p == 'go_straight':
                if direction == 0:
                    # adj_phase.append([0,1])
                    # adj_phase.append('WS')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'WS'
                elif direction == 1:
                    # adj_phase.append([1,1])
                    # adj_phase.append('SS')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'SS'
                elif direction == 2:
                    # adj_phase.append([2,1])
                    # adj_phase.append('ES')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'ES'
                else:
                    # adj_phase.append([3,1])
                    # adj_phase.append('NS')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'NS'

            elif type_p == 'turn_left':
                if direction == 0:
                    # adj_phase.append([0,0])
                    # adj_phase.append('WL')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'WL'
                elif direction == 1:
                    # adj_phase.append([1,0])
                    # adj_phase.append('SL')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'SL'
                elif direction == 2:
                    # adj_phase.append([2,0])
                    # adj_phase.append('EL')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'EL'
                else:
                    # adj_phase.append([3,0])
                    # adj_phase.append('NL')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'NL'
            else:
                if direction == 0:
                    # adj_phase.append([0,2])
                    # adj_phase.append('WR')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'WR'
                elif direction == 1:
                    # adj_phase.append([1,2])
                    # adj_phase.append('SR')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'SR'
                elif direction == 2:
                    # adj_phase.append([2,2])
                    # adj_phase.append('ER')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'ER'
                else:
                    # adj_phase.append([3,2])
                    # adj_phase.append('NR')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'NR'

            adjacency_matrix[road_dict_road2id[source]
                             ][road_dict_road2id[target]] = 1

    return adjacency_matrix, adj_phase



    