import gym

from agent.base import BaseAgent
import numpy as np

class FixedTimeAgent(BaseAgent):
    def __init__(self, action_space, ob_generator, reward_generator, \
        iid, idx, time=30, step=3600):
        super().__init__(action_space)

        self.iid = iid
        self.idx = idx
        self.name = self.__class__.__name__

        self.ob_generator = ob_generator
        ob_length = [self.ob_generator[0].ob_length, self.action_space.n]
        self.ob_length = sum(ob_length)
        self.reward_generator = reward_generator

        self.learning_start = np.inf

        self.time = time
        self.step = step
    
    def get_phase(self):
        return self.ob_generator[1].generate()
    
    def get_ob(self):
        return [self.ob_generator[0].generate(), np.array(self.ob_generator[1].generate())]
    
    def get_reward(self):
        '''take position at np.mean()'''
        return self.reward_generator.generate()

    def choose(self, obs=None):
        if self.ob_generator.world. % self.time == (self.time - 1):
            self.phase += 1
            self.phase = self.phase % self.action_space.n
        self.cur_t += 1

        if self.cur_t == self.step
        return self.phase

    def get_action(self, obs, phase, relation=None):
        return self.choose(self, obs)

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
    
    