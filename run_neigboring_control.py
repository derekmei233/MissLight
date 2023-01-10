from world import World
import gym
from agent.idqn_agent import IDQNAgent
from agent.idqn_neighbor_agent import NIDQNAgent
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
import numpy as np
from utils.agent_preparation import create_env, create_world



def create_neighboring_agents(world, mask_pos, device):
    agents = []
    for idx, i in enumerate(world.intersections):
        action_space = gym.spaces.Discrete(len(i.phases))
        if idx not in mask_pos:
            agents.append(IDQNAgent(
                action_space,
                [
                    LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
                    IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
                    LaneVehicleGenerator(world, i, ["lane_delay"], in_only=True, average='all')
                ],
                LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average='all', negative=True),
                i.id, idx, device
            ))
        else:
            agents.append(NIDQNAgent(
                action_space, 
                [
                    [LaneVehicleGenerator(world, world.intersections[j], ["lane_count"], in_only=True, average=None) for j in [idx - 4, idx + 1, idx + 4, idx - 1]],
                    IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
                    LaneVehicleGenerator(world, i, ["lane_delay"], in_only=True, average='all')
                ],
                [LaneVehicleGenerator( world, world.intersections[j], ["lane_waiting_count"], in_only=True, average='all', negative=True) for j in [idx - 4, idx + 1, idx + 4, idx - 1]],
                i.id, idx, device
            ))
    return agents


def naive_train(env, agents, episodes, action_interval):
    # take in environment and generate data for inference net
    # save t, phase_t, rewards_tp, state_tp, phase_tp(action_t) into dictionary
    information = []
    total_decision_num = 0
    best_att = np.inf
    for e in range(episodes):
        # collect [state_t, phase_t, reward_t, state_tp, phase_tp(action_t)] pair at observable intersections
        last_obs = env.reset()
        last_states,last_phases = list(zip(*last_obs))

        # last_states = np.array(last_states, dtype=np.float32)
        # last_phases = np.array(last_phases, dtype=np.int8)

        episodes_rewards = [0 for _ in agents]
        episodes_decision_num = 0
        i = 0
        while i < 3600:
            if i % action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.learning_start:
                        actions.append(agent.choose(last_states, last_phases)) # okay, since idqn only apply to observable intersections
                    else:
                        actions.append(agent.sample())
                rewards_list = []
                for _ in range(action_interval):
                    obs, rewards, _, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards_train = np.mean(rewards_list, axis=0)
                rewards = rewards_train
                cur_states, cur_phases = list(zip(*obs))

                for agent_id, agent in enumerate(agents):
                    if agent.name == 'FixedTimeAgent':
                        pass
                    else:
                        # no imputation, use obs directly
                        agent.remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id]) # okay
                        episodes_rewards[agent_id] += rewards[agent_id]
                        episodes_decision_num += 1
                total_decision_num += 1
                last_obs = obs
                last_states, last_phases = cur_states, cur_phases

            for agent_id, agent in enumerate(agents):
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()
        print("episodes:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        best_att = naive_execute(env, agents, e, best_att, action_interval)
    print(f'naive average travel time result: {best_att}')
    print('-' * 50)
    return information

def naive_execute(env, agents, e, best_att, action_interval):
    i = 0
    last_obs = env.reset()
    last_states, last_phases = list(zip(*last_obs))
    # last_states = np.array(last_states, dtype=np.float32)
    # last_phases = np.array(last_phases, dtype=np.int8)

    episodes_rewards = [0 for _ in agents]
    delay_list = np.zeros([1, len(agents)])
    count = 0
    while i < 3600:
        if i % action_interval == 0:
            actions = []
            delays = []
            for agent_id, agent in enumerate(agents):
                actions.append(agent.get_action(last_states, last_phases))
                delays.append(agent.get_delay())
            delay_list += np.array(delays)
            count += 1
            rewards_list = []


            for _ in range(action_interval):
                obs, rewards, _, _ = env.step(actions)
                i += 1
                rewards_list.append(rewards)
            rewards = np.mean(rewards_list, axis=0)
            for agent_id, agent in enumerate(agents):

                    episodes_rewards[agent_id] += rewards[agent_id]
            last_obs = obs
            last_states, last_phases = list(zip(*last_obs))
    att = env.eng.get_average_travel_time()
    if att < best_att:
        best_att = att
    print("episode:{}, Test:{}".format(e, att))
    print(f'delay: {np.array(delay_list) / count}')
    return best_att


if __name__ == "__main__":
    file = 'cityflow_hz4x4.cfg'
    world = create_world(file)
    agents = create_neighboring_agents(world, [5], 'cpu')
    env = create_env(world, agents)
    naive_train(env, agents, 100, 10)