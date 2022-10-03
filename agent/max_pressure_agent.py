from re import L
from . import BaseAgent
from prepareData import inter2state
import numpy as np


class MaxPressureAgent(BaseAgent):
    """
    Agent using Max-Pressure method to control traffic light
    """
    def __init__(self, action_space, ob_generator, reward_generator, iid, idx):
        super().__init__(action_space)
        self.iid = iid
        self.idx = idx
        self.sub_agents = 1
        self.ob_generator = ob_generator
        self.reward_generator = reward_generator
        self.I = self.ob_generator[0].world.id2intersection[self.iid]
        self.name = self.__class__.__name__
        
        # the minimum duration of time of one phase
        self.t_min = 10
        self.learning_start = np.inf

    def get_ob(self):
        # return true value but not use it later if idx in mask_pos
        return [self.ob_generator[0].generate(), np.array(self.ob_generator[1].generate())]

    def get_phase(self):
        return self.ob_generator[1].generate()

    def get_reward(self):
        '''take position at np.mean()'''
        return self.reward_generator.generate()

    # inference version
    
    def get_action(self, obs, phase, relation):
        # get lane pressure
        #lvc_1 = self.world.get_info("lane_count")
        inter_dict_id2inter = relation['inter_dict_id2inter']
        inter_in_roads = relation['inter_in_roads']
        road_dict_road2id = relation['road_dict_road2id']
        num_roads = len(road_dict_road2id)
        lvc = np.zeros((num_roads, 3), dtype=np.float32)
        # TODO : double check its reversed or not. yes its reversed
        for id_node, ob_length in enumerate(obs):
            direction = []
            direction.append(ob_length[0:3])
            direction.append(ob_length[3:6])
            direction.append(ob_length[6:9])
            direction.append(ob_length[9:])
            inter = inter_dict_id2inter[id_node]
            in_roads = inter_in_roads[inter]
            for id_road, road in enumerate(in_roads):
                # TODO check order here
                road_id = road_dict_road2id[road]
                lvc[road_id] = direction[id_road]

        if self.I.current_phase_time < self.t_min:
            return self.I.current_phase

        max_pressure = None
        action = -1
        for phase_id in range(len(self.I.phases)):
            # TODO: test below
            pres = []
            for a in self.I.phase_available_lanelinks[phase_id]:

                start = a[0][:-2]
                end = a[1][:-2]
                sub_start = int(a[0][-1])
                sub_end = int(a[1][-1])
                road_start = road_dict_road2id[start]
                road_end = road_dict_road2id[end]
                val_start = lvc[road_start][2 - sub_start]
                val_end = lvc[road_end][2 - sub_end]
                pres.append(val_start - val_end)
            pressure = sum(pres)
            """
            pressure = sum([lvc[road_dict_road2id[start[0][:-2]]][2 - int(start[0][-1])] -
                            lvc[road_dict_road2id[end[1][:-2]]][2 - int(end[0][-1])]
                            for start, end in self.I.phase_available_lanelinks[phase_id]])
            """
            if max_pressure is None or pressure > max_pressure:
                action = phase_id
                max_pressure = pressure

        return action
    
    def update_target_network(self):
        pass

    def remember(self, ob, action, reward, next_ob):
        pass

    def load_model(self):
        pass

    def save_model(self, model_dir):
        pass

    def test(self):
        print(f'{self.__repr__}: {self.cur_t, self.phase}')
