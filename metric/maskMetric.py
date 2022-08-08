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
    def __init__(self, name, world, mask_pos):
        'Relative Percent Difference: 2 * (x - y) / (abs(x) + abs(y)'
        self.name = name
        self.record = list()
        self.mask_pos = mask_pos
    
    def update(self, pred, true):
        # should be [N_intersection, N_features]
        diff = pred[self.mask_pos] - true[self.mask_pos]
        info = np.square(np.mean(pred))
        self.record.append(diff)

    def get_result(self):
        return sum(self.record)/ len(self.record)
        