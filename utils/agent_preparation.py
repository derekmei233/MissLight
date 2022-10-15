from world import World
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
from environment import TSCEnv

from agent.max_pressure_agent import MaxPressureAgent
from agent.sdqn_agent import SDQNAgent, build_shared_model
from agent.idqn_agent import IDQNAgent
from agent.fixedtime_agent import FixedTimeAgent

import gym
import torch.optim as optim


def create_world(config_file):
    return World(config_file, thread_num=8)

def create_env(world, agents):
    return TSCEnv(world, agents, None)

def create_fixedtime_agents(world, time=30):
    agents = []
    for idx, i in enumerate(world.intersections):
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(FixedTimeAgent(
            action_space, 
            [
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None), 
            IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
            ],
            LaneVehicleGenerator( world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
            i.id, idx, time=time
        ))
    return agents

def create_preparation_agents(world, mask_pos, time):
    agents = []
    for idx, i in enumerate(world.intersections):
        action_space = gym.spaces.Discrete(len(i.phases))
        if idx not in mask_pos:
            agents.append(IDQNAgent(
                action_space,
                [
                    LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
                    IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
                ],
                LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
                i.id, idx
            ))
        else:
            agents.append(FixedTimeAgent(
                action_space, 
                [
                LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None), 
                IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
                ],
                LaneVehicleGenerator( world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
                i.id, idx,time=time
            ))
    return agents

def create_maxp_agents(world):
    agents = []
    for idx, i in enumerate(world.intersections):
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(MaxPressureAgent(
            action_space,
            [
                LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
                IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
            ],
            LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
            i.id, idx
        ))
    return agents

def create_app1maxp_agents(world, mask_pos):
    agents = []
    for idx, i in enumerate(world.intersections):
        action_space = gym.spaces.Discrete(len(i.phases))
        if idx not in mask_pos:
            agents.append(IDQNAgent(
                action_space,
                [
                    LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
                    IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
                ],
                LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
                i.id, idx
            ))
        else:
            agents.append(MaxPressureAgent(
                action_space, 
                [
                LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None), 
                IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
                ],
                LaneVehicleGenerator( world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
                i.id, idx
            ))
    return agents
    
def create_idqn_agents(world):
    agents = []
    for idx, i in enumerate(world.intersections):
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(IDQNAgent(
            action_space,
            [
                LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
                IntersectionPhaseGenerator(world, i, ["phase"], targets=["cur_phase"], negative=False),
            ],
            LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average=None, negative=True),
            i.id, idx
        ))
    return agents

def create_sdqn_agents(world, mask_pos):
    agents = []
    obs_pos = list(set(range(len(world.intersections))) - set(mask_pos))
    iid = []
    ob_generator = []
    reward_generator = []
    for idx, inter in enumerate(world.intersections):
        ob_generator.append(
            [
                LaneVehicleGenerator(world, inter, ['lane_count'], in_only=True, average=None),
                IntersectionPhaseGenerator(world, inter, ["phase"], targets=['cur_phase'], negative=False)
            ])
        reward_generator.append(LaneVehicleGenerator(world, inter, ['lane_waiting_count'], in_only=True, average=None, negative=True))
        iid.append(inter.id)
    action_space = gym.spaces.Discrete(len(world.intersections[-1].phases))
    ob_length = ob_generator[0][0].ob_length + action_space.n
    q_model = build_shared_model(ob_length, action_space)
    target_q_model = build_shared_model(ob_length, action_space)
    optimizer = optim.RMSprop(q_model.parameters(), lr=0.001, alpha=0.9, centered=False, eps=1e-7)
    agents.append(SDQNAgent(action_space, ob_generator, reward_generator, iid, obs_pos, q_model, target_q_model, optimizer))
    return agents

def create_model_based_agents(world, mask_pos):
    # this should be the same as approach 1.2 S-S-O control
    agents = []
    obs_pos = list(set(range(len(world.intersections))) - set(mask_pos))
    iid = []
    ob_generator = []
    reward_generator = []
    for idx, inter in enumerate(world.intersections):
        ob_generator.append(
            [
                LaneVehicleGenerator(world, inter, ['lane_count'], in_only=True, average=None),
                IntersectionPhaseGenerator(world, inter, ["phase"], targets=['cur_phase'], negative=False)
            ])
        reward_generator.append(LaneVehicleGenerator(world, inter, ['lane_waiting_count'], in_only=True, average=None, negative=True))
        iid.append(inter.id)
    action_space = gym.spaces.Discrete(len(world.intersections[-1].phases))
    ob_length = ob_generator[0][0].ob_length + action_space.n
    q_model = build_shared_model(ob_length, action_space)
    target_q_model = build_shared_model(ob_length, action_space)
    optimizer = optim.RMSprop(q_model.parameters(), lr=0.001, alpha=0.9, centered=False, eps=1e-7)
    agents.append(SDQNAgent(action_space, ob_generator, reward_generator, iid, obs_pos, q_model, target_q_model, optimizer))
    return agents 