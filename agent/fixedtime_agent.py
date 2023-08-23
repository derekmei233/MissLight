import gym

from agent.base import BaseAgent
import numpy as np

class FixedTimeAgent(BaseAgent):
    def __init__(self, action_space, ob_generator, reward_generator, \
        iid, idx, time=30, step=3600):
        super().__init__(action_space)

        self.iid = iid
        self.sub_agents = 1
        self.idx = idx
        self.name = self.__class__.__name__

        self.ob_generator = ob_generator
        ob_length = [self.ob_generator[0].ob_length, self.action_space.n]
        self.ob_length = sum(ob_length)
        self.reward_generator = reward_generator

        self.learning_start = np.inf

        self.time = time
        self.step = step
        self.phase = 0
    
    def get_phase(self):
        return self.ob_generator[1].generate()

    def get_current_time(self):
        return self.ob_generator[0].world.eng.get_current_time()
    
    def get_ob(self):
        return [np.zeros((12)), np.array([0])]

    def get_delay(self):
        return np.mean(self.ob_generator[2].generate())
    
    def get_reward(self):
        '''take position at np.mean()'''
        return self.reward_generator.generate()

    def choose(self, obs=None, phases=None):
        cur_time = self.get_current_time()
        if cur_time == 0:
            self.phase = 0
        elif cur_time % self.time == 0:
            self.phase += 1
            self.phase = self.phase % self.action_space.n
        return self.phase

    def get_action(self, obs, phase, relation=None):
        return self.choose()

    def sample(self):
        return self.choose()
    
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
    
    