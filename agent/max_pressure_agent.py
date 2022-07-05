from . import BaseAgent
from prepareData import inter2state
import numpy as np


class MaxPressureAgent(BaseAgent):
    """
    Agent using Max-Pressure method to control traffic light
    """
    def __init__(self, action_space, I, world, ob_generator=None):
        super().__init__(action_space)
        self.I = I
        self.world = world
        self.world.subscribe("lane_count")
        #self.world.subscribe("phase")
        self.ob_generator = ob_generator
        
        # the minimum duration of time of one phase
        self.t_min = 10

    def get_ob(self):
        if self.ob_generator is not None:
            obs_lane = self.ob_generator[0].generate()
            return obs_lane
        else:
            return None

    def get_phase(self):
        if self.ob_generator is not None:
            cur_phase = self.ob_generator[1].generate()
            return cur_phase
        else:
            return None

    # max_pressure version
    """
    def get_action(self, ob):
        # get lane pressure
        lvc = self.world.get_info("lane_count")

        if self.I.current_phase_time < self.t_min:
            return self.I.current_phase

        max_pressure = None
        action = -1
        for phase_id in range(len(self.I.phases)):
            pressure = sum([lvc[start] - lvc[end] for start, end in self.I.phase_available_lanelinks[phase_id]])
            if max_pressure is None or pressure > max_pressure:
                action = phase_id
                max_pressure = pressure

        return action
    """
    # inference version
    def get_action(self, obs, relation, in_channels):
        # get lane pressure
        #lvc_1 = self.world.get_info("lane_count")
        inter_dict_id2inter = relation['inter_dict_id2inter']
        inter_in_roads = relation['inter_in_roads']
        road_dict_road2id = relation['road_dict_road2id']
        num_roads = len(road_dict_road2id)
        lvc = np.zeros((num_roads, in_channels - 8), dtype=np.float32)
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

    # original version
    def get_action_org(self, obs):
        # get lane pressure
        lvc = self.world.get_info("lane_count")
        if self.I.current_phase_time < self.t_min:
            return self.I.current_phase

        max_pressure = None
        action = -1
        for phase_id in range(len(self.I.phases)):
            pressure = sum([lvc[start] - lvc[end] for start, end in self.I.phase_available_lanelinks[phase_id]])
            if max_pressure is None or pressure > max_pressure:
                action = phase_id
                max_pressure = pressure

        return action

    def get_reward(self):
        return None
