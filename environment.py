import gym
import numpy as np
import cityflow


class TSCEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.

    Parameters
    ----------
    world: World object
    agents: list of agent, corresponding to each intersection in world.intersections
    metric: Metric object, used to calculate evaluation metric
    """

    def __init__(self, world, agents, metric):
        self.world = world

        self.eng = self.world.eng
        self.n_agents = len(agents)
        self.n = self.n_agents

        assert len(agents) == self.n_agents

        self.agents = agents
        action_dims = [agent.action_space.n for agent in agents]
        self.action_space = gym.spaces.MultiDiscrete(action_dims)

        self.metric = metric

    def change_world(self, world):
        self.world.reset()
        # self.world = world
        self.eng = world.eng
        self.world.reset()

    def step(self, actions):
        assert len(actions) == self.n_agents
        self.world.step(actions)
        obs = [agent.get_ob() for agent in self.agents]
        rewards = [agent.get_reward() for agent in self.agents]
        dones = [False] * self.n_agents
        # infos = {"metric": self.metric.update()}
        infos = {}

        return obs, rewards, dones, infos

    def reset(self):
        self.world.reset()
        obs = [agent.get_ob() for agent in self.agents]
        return obs

    # def mask_init(self,mask_intersections):
    #     '''generate masked intersections and its id'''

    #     mask_id = np.zeros(self.shape,dtype=int)
    #     nb_num = mask_intersections['nb_num']
    #     mask_num = mask_intersections['mask_num']
    #     if nb_num == 4:
    #         mask_id[1][1] = np.random.randint(0,2)
    #         for i in range(1,self.shape[0]-1):
    #             if i>=2:
    #                 mask_id[i][1] = 0 if mask_id[i-1][1]==1 else 1
    #             for j in range(2,self.shape[1]-1):
    #                 mask_id[i][j] = 0 if mask_id[i][j-1]==1 else 1

    #     elif nb_num == 3:
    #         mask_id[0][1] = np.random.randint(0,2) # 第一行
    #         mask_id[self.shape[0]-1][1] = np.random.randint(0,2) # 最后一行
    #         for i in range(2,self.shape[1]-1):
    #             mask_id[0][i] = 0 if mask_id[0][i-1]==1 else 1
    #             mask_id[self.shape[0]-1][i] = 0 if mask_id[self.shape[0]-1][i-1]==1 else 1

    #         mask_id[1][0] = np.random.randint(0,2) # 第一列
    #         mask_id[1][self.shape[1]-1] = np.random.randint(0,2) # 最后一行
    #         for i in range(2,self.shape[0]-1):
    #             mask_id[i][0] = 0 if mask_id[i-1][0]==1 else 1
    #             mask_id[i][self.shape[1]-1] = 0 if mask_id[i-1][self.shape[1]-1]==1 else 1

    #     elif nb_num == 2:
    #         # 4个角
    #         num = 0
    #         while num < mask_num:
    #             mask_id[0][0] = np.random.randint(0,2)
    #             mask_id[self.shape[0]-1][0] = np.random.randint(0,2)
    #             mask_id[0][self.shape[1]-1] = np.random.randint(0,2)
    #             mask_id[self.shape[0]-1][self.shape[1]-1] = np.random.randint(0,2)

    #     return mask_id.reshape(-1)
        
    
    # def mask(self,obs):
    #     '''根据mask路口id,mask掉状态'''
    #     feature_num = len(obs[0])
    #     for i in range(self.n):
    #         if self.mask_id[i] == 1:
    #                 obs[i] = np.full((1,feature_num),-1.0)
    #     return obs
