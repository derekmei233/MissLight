from . import BaseMetric
import numpy as np

class TravelTimeMetric(BaseMetric):
    """
    Calculate average travel time of all vehicles.
    For each vehicle, travel time measures time between it entering and leaving the roadnet.
    """
    def __init__(self, world):
        self.world = world
        self.world.subscribe(["vehicles", "time"])
        self.vehicle_enter_time = {}
        self.travel_times = []

    def update(self, done=False):
        vehicles = self.world.get_info("vehicles")
        current_time = self.world.get_info("time")

        for vehicle in vehicles:
            if not vehicle in self.vehicle_enter_time:
                self.vehicle_enter_time[vehicle] = current_time

        for vehicle in list(self.vehicle_enter_time):
            if done or not vehicle in vehicles:
                self.travel_times.append(current_time - self.vehicle_enter_time[vehicle])
                del self.vehicle_enter_time[vehicle]

        return np.mean(self.travel_times) if len(self.travel_times) else 0
        
class MSEMetric(BaseMetric):
    def __init__(self, name, mask_pos):
        self.name = name
        self.buffer = list()
        self.record = list()
        self.mask_pos = mask_pos

    def add(self, pred, true):
        # should be [N_intersection, N_features]
        err = []
        # for i in self.mask_pos:
        #     err.append(np.mean(np.square(pred[i] - true[i])))
        for i in range(len(pred)):
            err.append(np.mean(np.square(pred[i] - true[i])))
        diff = np.array(err)
        self.buffer.append(diff)
    
    def update(self):
        # should be [N_intersection, N_features]
        assert len(self.buffer) != 0, 'no record inside buffer'
        epo_avg =sum(self.buffer) / len(self.buffer)
        self.record.append(epo_avg)
        self.buffer = list()

    def reset(self):
        self.buffer = list()
        self.record = list()
    
    def get_cur_result(self):
        return sum(self.buffer) / len(self.buffer)

    def get_result(self):
        return sum(self.record) / len(self.record)
