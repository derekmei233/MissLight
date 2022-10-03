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


        self.agents = agents
        self.n_agents = sum(ag.sub_agents for ag in self.agents)
        self.n = sum([ag.sub_agents for ag in self.agents])
        action_dims = [agent.action_space.n for agent in self.agents]
        self.action_space = [agent.action_space for agent in self.agents]

        self.metric = metric

    def change_world(self, world):
        self.world.reset()
        # self.world = world
        self.eng = world.eng
        self.world.reset()

    def step(self, actions):
        """
        call world.step() and make simulator move to next step
        :param world: world class containing simulator

        :returns: obs, rewards, dones, infos
        :rtype: (list of np.array, list of np.array, list of bool, list of dict)

        """
        assert len(actions) == self.n_agents
        self.world.step(actions)
        obs = []
        [obs.extend(ag.get_ob()) if ag.sub_agents != 1 else obs.append(ag.get_ob()) for ag in self.agents]
        reward = []
        # TODO: rank it (currently there is no need)
        [reward.extend(ag.get_reward()) if ag.sub_agents != 1 else reward.append(ag.get_reward()) for ag in self.agents]
        dones = [False] * self.n
        # infos = {"metric": self.metric.update()}
        infos = {}

        return obs, reward, dones, infos

    def reset(self):
        """
        reset environment and return its initial state and phase through agent.get_ob() functions

        :returns: list of observations in each agent
        :rtype: list [N_agents, N_state + N_phases]
        """
        self.world.reset()
        obs = []
        [obs.extend(ag.get_ob()) if ag.sub_agents != 1 else obs.append(ag.get_ob()) for ag in self.agents]
        return obs
