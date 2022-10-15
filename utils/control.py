from predictionModel.NN import NN_predictor
from predictionModel.SFM import SFM_predictor
from metric.maskMetric import MSEMetric

from utils.preparation import one_hot
from utils.data_generation import store_reshaped_data

import os
from tqdm import tqdm
import numpy as np
import torch


# protocol: last_obs is returned from env, (last_stats, last_phases) is returned from imputation

# F-F
def fixedtime_execute(logger, env, agents, action_interval, relation):
    env.eng.set_save_replay(True)
    name = logger.handlers[0].baseFilename
    save_dir = name[name.index('/output_data'): name.index('/logging')]
    env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_0.txt"))
    logger.info(f"FixedTime - FixedTime control")
    i = 0
    obs = env.reset()
    states, phases = list(zip(*obs))
    states = np.array(states, dtype=np.float32)
    phases = np.array(phases, dtype=np.int8)
    for i in range(3600):
        if i % action_interval == 0:
            actions = []
            for ag in agents:
                action = ag.get_action(states, phases, relation)
                actions.append(action)
            for _ in range(action_interval):
                env.step(actions)
                i += 1
    att = env.eng.get_average_travel_time()
    logger.info(f'FixedTime time interval: {agents[0].time}')
    logger.info(f'FixedTime average travel time result: {att}')
    logger.info('-' * 50)
    return None

# : I-F
def naive_train(logger, env, agents, episodes, action_interval, save_rate):
    # take in environment and generate data for inference net
    # save t, phase_t, rewards_tp, state_tp, phase_tp(action_t) into dictionary
    # collect data for training state inference model GraphWaveNet (obs for train and miss for eval)
    state_raw_data = []
    logger.info(f" IDQN -  FixedTime control")
    information = []
    total_decision_num = 0
    best_att = np.inf
    for e in tqdm(range(episodes)):
        # collect [state_t, phase_t, reward_t, state_tp, phase_tp(action_t)] pair at observable intersections
        save_state = []
        last_obs = env.reset()
        save_state.append(last_obs)
        last_states, last_phases = list(zip(*last_obs))
        last_states = np.array(last_states, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        episodes_rewards = [0 for _ in agents]
        episodes_decision_num = 0
        last_observable = []
        observable = []
        rewards_observable = []
        actions_observable = []
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
                save_state.append(obs)
                rewards_train = np.mean(rewards_list, axis=0)
                rewards = np.mean(rewards_train, axis=1)
                for agent_id, agent in enumerate(agents):
                    if agent.name == 'FixedTimeAgent':
                        pass
                    else:
                        # no imputation, use obs directly
                        agent.remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id]) # okay
                        episodes_rewards[agent_id] += rewards[agent_id]
                        episodes_decision_num += 1
                        last_observable.append(last_obs[agent_id])
                        observable.append(obs[agent_id])
                        # S_{t+1}-> R_t
                        actions_observable.append(actions[agent_id])
                        rewards_observable.append(rewards_train[agent_id])
                total_decision_num += 1
                last_obs = obs
                last_states, last_phases = list(zip(*last_obs))
                last_states = np.array(last_states, dtype=np.float32)
                last_phases = np.array(last_phases, dtype=np.int8)

            for agent_id, agent in enumerate(agents):
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()
        store_reshaped_data(information, [last_observable, rewards_observable, actions_observable, observable])
        state_raw_data.append(save_state)
        logger.info("episodes:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        best_att = naive_execute(logger, env, agents, e, best_att, information, state_raw_data, action_interval, save_rate)
    logger.info(f'naive average travel time result: {best_att}')
    logger.info('-' * 50)
    return information, state_raw_data

def naive_execute(logger, env, agents, e, best_att, information, state_raw_data, action_interval, save_rate):
    if e % save_rate == save_rate - 1:
        env.eng.set_save_replay(True)
        name = logger.handlers[0].baseFilename
        save_dir = name[name.index('/output_data'): name.index('/logging')]
        env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_%s.txt" % e))
    else:
        env.eng.set_save_replay(False)
    i = 0
    save_state = []
    last_obs = env.reset()
    save_state.append(last_obs)
    last_states, last_phases = list(zip(*last_obs))
    last_states = np.array(last_states, dtype=np.float32)
    last_phases = np.array(last_phases, dtype=np.int8)
    episodes_rewards = [0 for i in agents]
    last_observable = []
    observable = []
    rewards_observable = []
    actions_observable = []
    while i < 3600:
        if i % action_interval == 0:
            actions = []
            for agent_id, agent in enumerate(agents):
                actions.append(agent.get_action(last_states, last_phases))
            rewards_list = []
            for _ in range(action_interval):
                obs, rewards, _, _ = env.step(actions)
                i += 1
                rewards_list.append(rewards)
            rewards_train = np.mean(rewards_list, axis=0)
            rewards =  np.mean(rewards_train, axis=1)
            save_state.append(obs)
            for agent_id, agent in enumerate(agents):
                if agent.name == 'FixedTimeAgent':
                    pass
                else:
                    episodes_rewards[agent_id] += rewards[agent_id]
                    last_observable.append(last_obs[agent_id])
                    observable.append(obs[agent_id])
                    # S_{t+1}-> R_t
                    actions_observable.append(actions[agent_id])
                    rewards_observable.append(rewards_train[agent_id])
            last_obs = obs
            last_states, last_phases = list(zip(*last_obs))
            last_states = np.array(last_states, dtype=np.float32)
            last_phases = np.array(last_phases, dtype=np.int8)
    store_reshaped_data(information, [last_observable, rewards_observable, actions_observable, observable])
    state_raw_data.append(save_state)
    att = env.eng.get_average_travel_time()
    if att < best_att:
        best_att = att
    logger.info("episode:{}, Test:{}".format(e, att))
    return best_att

# M-M
def maxp_execute(logger, env, agents, action_interval, inference_net, mask_pos, relation, mask_matrix, adj_matrix):
    env.eng.set_save_replay(True)
    name = logger.handlers[0].baseFilename
    save_dir = name[name.index('/output_data'): name.index('/logging')]
    env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_0.txt"))
    logger.info(f"Max Pressure - Max Pressure control")
    i = 0
    record = MSEMetric('state mse', mask_pos)
    obs = env.reset()
    states, phases = list(zip(*obs))
    states = np.array(states, dtype=np.float32)
    phases = np.array(phases, dtype=np.int8)
    recovered = inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
    record.add(phases, recovered)
    for i in range(3600):
        if i % action_interval == 0:
            # TODO: implement other State inference model later
            # SFM inference states
            actions = []
            for ag in agents:
                action = ag.get_action(recovered, phases, relation)
                actions.append(action)
            for _ in range(action_interval):
                obs, _, _, _ = env.step(actions)
                i += 1
            states, phases = list(zip(*obs))
            states = np.array(states, dtype=np.float32)
            phases = np.array(phases, dtype=np.int8)
            recovered = inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
            record.add(phases, recovered)
    att = env.eng.get_average_travel_time()
    mse = record.get_cur_result()
    record.update()

    logger.info(f'MaxPressure average travel time result: {att}')
    logger.info(f'Maxpressure MSE: {mse}')
    logger.info('-' * 50)
    return record

# I-M 
def app1maxp_train(logger, env, agents, episode, action_interval, inference_net, mask_pos, relation, mask_matrix, adj_matrix, save_rate):
    logger.info(f" IDQN - Max Pressure control")
    total_decision_num = 0
    best_att = np.inf
    record = MSEMetric('state mse', mask_pos)
    for e in range(episode):
        # collect [state_t, phase_t, reward_t, state_tp, phase_tp(action_t)] pair at observable intersections
        last_obs = env.reset()
        last_states, last_phases = list(zip(*last_obs))
        last_states = np.array(last_states, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        last_recovered = inference_net.predict(last_states, last_phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
        record.add(last_states, last_recovered)
        episodes_rewards = [0 for _ in agents]
        i = 0
        episodes_decision_num = 0
        while i < 3600:
            if i % action_interval == 0:
                # TODO: implement other State inference model later
                # SFM inference states
                actions = []
                for agent_id, agent in enumerate(agents):
                    if agent.name == 'MaxPressureAgent':
                        action = agent.get_action(last_recovered, last_phases, relation)
                        actions.append(action)
                    else:
                        if total_decision_num > agent.learning_start:
                            actions.append(agent.choose(last_recovered, last_phases, relation))
                        else:
                            actions.append(agent.sample())
                rewards_list = []
                for _ in range(action_interval):
                    obs, rewards, _, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)
                rewards = np.mean(rewards, axis=1)
                for agent_id, agent in enumerate(agents):
                    if agent.name == 'MaxPressureAgent':
                        pass
                    else:
                        agent.remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id]) # no need to change obs
                        episodes_rewards[agent_id] += rewards[agent_id]
                        episodes_decision_num += 1
                total_decision_num += 1
                last_obs = obs
                last_states, last_phases = list(zip(*last_obs))
                last_states = np.array(last_states, dtype=np.float32)
                last_phases = np.array(last_phases, dtype=np.int32)
                last_recovered = inference_net.predict(last_states, last_phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
            for agent_id, agent in enumerate(agents):
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()
        cur_mse = record.get_cur_result()
        record.update()
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        logger.info("episode:{}, MSETrain:{}".format(e, cur_mse))
        best_att = app1maxp_execute(logger, env, agents, e, best_att, record, inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix, save_rate)
    avg_mse = record.get_result()

    logger.info(f'approach 1: maxpressure average travel time result: {best_att}')
    logger.info(f'final mse is: {avg_mse}')
    logger.info('-' * 50)
    return record

def app1maxp_execute(logger, env, agents, e, best_att, record, inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix, save_rate):
    if e % save_rate == save_rate - 1:
        env.eng.set_save_replay(True)
        name = logger.handlers[0].baseFilename
        save_dir = name[name.index('/output_data'): name.index('/logging')]
        env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_%s.txt" % e))
    else:
        env.eng.set_save_replay(False)
    i = 0
    obs = env.reset()
    states, phases = list(zip(*obs))
    states = np.array(states, dtype=np.float32)
    phases = np.array(phases, dtype=np.int8)
    recovered = inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
    record.add(states, recovered)
    while i < 3600:
        if i % action_interval == 0:
            # TODO: implement other State inference model later
            # SFM inference states
            actions = []
            for ag in agents:
                if ag.name == 'MaxPressureAgent':
                    action = ag.get_action(recovered, phases, relation)
                elif ag.name == 'IDQNAgent':
                    action = ag.get_action(recovered, phases)
                actions.append(action)
            for _ in range(action_interval):
                obs, _, _, _ = env.step(actions)
                i += 1
            
            states, phases = list(zip(*obs))
            states = np.array(states, dtype=np.float32)
            phases = np.array(phases, dtype=np.int8)
            recovered = inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')

    cur_mse = record.get_cur_result()
    record.update()
    att = env.eng.get_average_travel_time()
    if att < best_att:
        best_att = att
    logger.info("episode:{}, Test:{}".format(e, att))
    logger.info("episode:{}, MSETest:{}".format(e, cur_mse))
    return best_att

# S-S-O
def app1_trans_train(logger, env, agents, episode, action_interval, inference_net, mask_pos, relation, mask_matrix, adj_matrix, save_rate):
    # this method is used in approach 1 ,transfer and approach 2, shared-parameter. 
    logger.info(f"SDQN - SDQN - O control") 
    total_decision_num = 0
    best_att = np.inf
    record = MSEMetric('state mse', mask_pos)
    for e in range(episode):
        last_obs = env.reset()
        last_states, last_phases = list(zip(*last_obs))
        last_states = np.array(last_states, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        # only recover state_t since we need this var to determine action t
        last_recovered = inference_net.predict(last_states, last_phases, relation, mask_pos, mask_matrix, adj_matrix)
        record.add(last_states, last_recovered)
        episodes_decision_num = 0
        episodes_rewards = [0 for _ in range(agents[0].sub_agents)]
        i = 0
        while i < 3600:
            if i % action_interval == 0:
                for ag in agents:
                    if total_decision_num > ag.learning_start:
                        actions = ag.choose(last_recovered, last_phases, relation)
                    else:
                        actions = ag.sample()
                rewards_list = []
                for _ in range(action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards_train = np.mean(rewards_list, axis=0)
                rewards = np.mean(rewards_train, axis=1)

                cur_states, cur_phases = list(zip(*obs))
                cur_states = np.array(cur_states, dtype=np.float32)
                cur_phases = np.array(cur_phases, dtype=np.int8)
                cur_recovered = inference_net.predict(cur_states, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
                record.add(cur_states, cur_recovered)
                for ag in (agents):
                    for idx in ag.idx:
                        ag.remember(
                            (last_recovered[idx], last_phases[idx]), actions[idx], rewards[idx], (cur_recovered[idx], cur_phases[idx]), idx)
                        episodes_rewards[idx] += rewards[idx]
                        episodes_decision_num += 1
                total_decision_num += 1
                last_obs = obs
                last_states = cur_states
                last_phases = cur_phases
                last_recovered = cur_recovered
            for ag in agents:
                # only use experiences at observable intersections
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_model_freq == ag.update_model_freq - 1:
                    ag.replay()
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_target_model_freq == ag.update_target_model_freq - 1:
                    ag.update_target_network()
        cur_mse = record.get_cur_result()
        record.update()
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        logger.info("episode:{}, MSETrain:{}".format(e, cur_mse))
        best_att = app1_trans_execute(logger, env, agents, e, best_att, record, inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix, save_rate)
    avg_mse = record.get_result()
    logger.info(f'approach 1: transfer dqn average travel time result: {best_att}')
    logger.info(f'final mse is: {avg_mse}')
    logger.info('-' * 50)
    return record

def app1_trans_execute(logger, env, agents, e, best_att, record, inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix, save_rate):
    if e % save_rate == save_rate - 1:
        env.eng.set_save_replay(True)
        name = logger.handlers[0].baseFilename
        save_dir = name[name.index('/output_data'): name.index('/logging')]
        env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_%s.txt" % e))
    else:
        env.eng.set_save_replay(False)
    i = 0
    obs = env.reset()
    states, phases = list(zip(*env.reset()))
    states = np.array(states, dtype = np.float32)
    phases = np.array(phases, dtype = np.int8)
    recovered = inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
    record.add(states, recovered)
    while i < 3600:
        if i % action_interval == 0:
            # TODO: implement other State inference model later
            # SFM inference states
            for ag in agents:
                actions = ag.get_action(recovered, phases, relation)
            for _ in range(action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
            states, phases = list(zip(*obs))
            states = np.array(states, dtype = np.float32)
            phases = np.array(phases, dtype = np.int8)
            recovered = inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
            record.add(states, recovered)
    att = env.eng.get_average_travel_time()
    cur_mse = record.get_cur_result()
    record.update()
    if att < best_att:
        best_att = att
    logger.info("episode:{}, Test:{}".format(e, att))
    logger.info("episode:{}, MSETest:{}".format(e, cur_mse))
    return best_att

# I-I
def app2_conc_train(logger, env, agents, episode, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, model_dir, reward_type, save_rate):
    logger.info(f"IDQN - IDQN control")
    logger.info(f"reward inference model: {reward_type}")
    if reward_type == 'SFM':
        reward_inference_net = SFM_predictor()
    elif reward_type == 'NN_st' or reward_type == 'NN_stp':
        reward_inference_net = NN_predictor(agents[0].ob_length, 1, 'cpu', model_dir)
        reward_inference_net.load_model()
    else:
        raise RuntimeError('not implemented yet')
    total_decision_num = 0
    best_att = np.inf
    record = MSEMetric('state mse', mask_pos)
    reward_record = MSEMetric('reward mse', mask_pos)
    for e in range(episode):
        last_obs = env.reset()
        last_states, last_phases = list(zip(*last_obs))
        last_states = np.array(last_states, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        last_recovered = state_inference_net.predict(last_states, last_phases, relation, mask_pos, mask_matrix, adj_matrix)
        record.add(last_states, last_recovered)
        episodes_decision_num = 0
        episodes_rewards = [0 for _ in agents]
        i = 0
        while i < 3600:
            if i % action_interval == 0:
                actions = []
                for ag in agents:
                    if total_decision_num > ag.learning_start:
                        action = ag.choose(last_recovered, last_phases)
                        actions.append(action)
                    else:
                        action = ag.sample()
                        actions.append(action)
                rewards_list = []
                for _ in range(action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards_train = np.mean(rewards_list, axis=0)
                rewards = np.mean(rewards_train, axis=1)
                if reward_type == 'SFM':
                    # TODO: check later 
                    rewards_predicted = reward_inference_net.predict(rewards_train, None, relation, mask_pos, mask_matrix, adj_matrix)
                    rewards_recovered = np.mean(rewards_predicted, axis=1)
                    reward_record.add(rewards, rewards_recovered)
                elif reward_type == 'NN_st':
                    rewards_recovered = rewards.copy()
                    for pos in mask_pos:
                        tmp = reward_inference_net.predict(torch.from_numpy(np.concatenate((last_recovered[pos], one_hot(last_phases[pos], agents[pos].action_space.n)))))
                        rewards_recovered[pos] = tmp
                    reward_record.add(rewards, rewards_recovered)
                elif reward_type == 'NN_stp':
                    # TODO: implement later
                    raise RuntimeError("not implemented")
                else:
                    raise RuntimeError("not implemented")
                cur_states, cur_phases = list(zip(*obs))
                cur_states = np.array(cur_states, dtype=np.float32)
                cur_phases = np.array(cur_phases, dtype=np.int8)
                cur_recovered = state_inference_net.predict(cur_states, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
                record.add(cur_states, cur_recovered)
                for agent_id, agent in enumerate(agents):
                    agent.remember(
                        (last_recovered[agent_id], last_phases[agent_id]), actions[agent_id], rewards_recovered[agent_id], (cur_recovered[agent_id], cur_phases[agent_id]))
                    episodes_rewards[agent_id] += rewards_recovered[agent_id]
                    episodes_decision_num += 1
                total_decision_num += 1
                last_obs = obs
                last_states = cur_states
                last_phases = cur_phases
                last_recovered = cur_recovered
            for agent_id, ag in enumerate(agents):
                # only use experiences at observable intersections
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_model_freq == ag.update_model_freq - 1:
                    ag.replay()
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_target_model_freq == ag.update_target_model_freq - 1:
                    ag.update_target_network()
        cur_mse = record.get_cur_result()
        reward_cur_mse = reward_record.get_cur_result()
        record.update()
        reward_record.update()
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        logger.info("episode:{}, MSETrain:{}".format(e, cur_mse))
        logger.info("episode:{}, Reward_MSETrain:{}".format(e, reward_cur_mse))        

        best_att = app2_conc_execute(logger, env, agents, e, best_att, record, state_inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix, save_rate)
    avg_mse = record.get_result()
    reward_avg_mse = record.get_result()
    logger.info(f'approach 2: concurrent idqn average travel time result: {best_att}')
    logger.info(f'final mse is: {avg_mse}')
    logger.info(f'final reward mse is: {reward_avg_mse}')
    logger.info('-' * 50)
    return record

# S-S-A
def app2_conc_execute(logger, env, agents, e, best_att, record, state_inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix, save_rate):
    if e % save_rate == save_rate - 1:
        env.eng.set_save_replay(True)
        name = logger.handlers[0].baseFilename
        save_dir = name[name.index('/output_data'): name.index('/logging')]
        env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_%s.txt" % e))
    else:
        env.eng.set_save_replay(False)
    i = 0
    obs = env.reset()
    states, phases = list(zip(*obs))
    states = np.array(states, dtype = np.float32)
    phases = np.array(phases, dtype = np.int8)
    recovered = state_inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
    record.add(states, recovered)
    while i < 3600:
        if i % action_interval == 0:
            # TODO: implement other State inference model later
            # SFM inference states
            actions = []
            for ag in agents:
                action = ag.get_action(recovered, phases, relation)
                actions.append(action)
            for _ in range(action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
            states, phases = list(zip(*obs))
            states = np.array(states, dtype = np.float32)
            phases = np.array(phases, dtype = np.int8)
            recovered = state_inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
            record.add(states, recovered)
    att = env.eng.get_average_travel_time()
    cur_mse = record.get_cur_result()
    record.update()
    if att < best_att:
        best_att = att
    logger.info("episode:{}, Test:{}".format(e, att))
    logger.info("episode:{}, MSETest:{}".format(e, cur_mse))
    return best_att

def app2_shared_train(logger, env, agents, episode, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, model_dir, reward_type, save_rate):
    # this method is used in approach 1 ,transfer and approach 2, shared-parameter. 
    logger.info(f"SDQN - SDQN - A control")
    logger.info(f"reward inference model: {reward_type}")
    if reward_type == 'SFM':
        reward_inference_net = SFM_predictor()
    elif reward_type == 'NN_st' or reward_type == 'NN_stp':
        reward_inference_net = NN_predictor(agents[0].ob_length, 1, 'cpu', model_dir, reward_type)
        reward_inference_net.load_model()
    else:
        raise RuntimeError('not implemented yet')

    total_decision_num = 0
    best_att = np.inf
    record = MSEMetric('state mse', mask_pos)
    reward_record = MSEMetric('reward mse', mask_pos)
    for e in range(episode):
        last_obs = env.reset()
        last_states, last_phases = list(zip(*last_obs))
        last_states = np.array(last_states, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        # only recover state_t since we need this var to determine action t
        last_recovered = state_inference_net.predict(last_states, last_phases, relation, mask_pos, mask_matrix, adj_matrix)
        record.add(last_states, last_recovered)
        episodes_decision_num = 0
        episodes_rewards = [0 for _ in range(agents[0].sub_agents)]
        i = 0
        while i < 3600:
            if i % action_interval == 0:
                for ag in agents:
                    if total_decision_num > ag.learning_start:
                        actions = ag.choose(last_recovered, last_phases, relation)
                    else:
                        actions = ag.sample()
                rewards_list = []
                for _ in range(action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards_train = np.mean(rewards_list, axis=0)
                rewards = np.mean(rewards_train, axis=1)
                if reward_type == 'SFM':
                    rewards_predicted = reward_inference_net.predict(rewards_train, None, relation, mask_pos, mask_matrix, adj_matrix)
                    rewards_recovered = np.mean(rewards_predicted, axis=1)
                    reward_record.add(rewards, rewards_recovered)
                elif reward_type == 'NN_st':
                    rewards_recovered = rewards.copy()
                    for pos in mask_pos:
                        tmp = reward_inference_net.predict(torch.from_numpy(np.concatenate((last_recovered[pos], one_hot(last_phases[pos], agents[0].action_space.n)))))
                        rewards_recovered[pos] = tmp
                    reward_record.add(rewards, rewards_recovered)
                elif reward_type == 'NN_stp':
                    # TODO: implement later
                    a = None
                else:
                    raise RuntimeError("not implemented")
                cur_states, cur_phases = list(zip(*obs))
                cur_states = np.array(cur_states, dtype=np.float32)
                cur_phases = np.array(cur_phases, dtype=np.int8)
                cur_recovered = state_inference_net.predict(cur_states, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
                record.add(cur_states, cur_recovered)
                for ag in agents:
                    for idx in ag.idx:
                        ag.remember(
                            (last_recovered[idx], last_phases[idx]), actions[idx], rewards_recovered[idx], (cur_recovered[idx], cur_phases[idx]), idx)
                        episodes_rewards[idx] += rewards[idx]
                        episodes_decision_num += 1
                total_decision_num += 1
                last_obs = obs
                last_states = cur_states
                last_phases = cur_phases
                last_recovered = cur_recovered
            for ag in agents:
                # only use experiences at observable intersections
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_model_freq == ag.update_model_freq - 1:
                    ag.replay()
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_target_model_freq == ag.update_target_model_freq - 1:
                    ag.update_target_network()
        cur_mse = record.get_cur_result()
        reward_cur_mse = reward_record.get_cur_result()
        record.update()
        reward_record.update()
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        logger.info("episode:{}, MSETrain:{}".format(e, cur_mse))
        logger.info("episode:{}, Reward_MSETrain:{}".format(e, reward_cur_mse))
        best_att = app2_shared_execute(logger, env, agents, e, best_att, record, state_inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix, save_rate)
    avg_mse = record.get_result()
    logger.info(f'approach 2: shared dqn average travel time result: {best_att}')
    logger.info(f'final mse is: {avg_mse}')
    logger.info('-' * 50)
    return record

def app2_shared_execute(logger, env, agents, e, best_att, record, state_inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix, save_rate):
    if e % save_rate == save_rate - 1:
        env.eng.set_save_replay(True)
        name = logger.handlers[0].baseFilename
        save_dir = name[name.index('/output_data'): name.index('/logging')]
        env.eng.set_replay_file(os.path.join(save_dir, 'replay', "replay_%s.txt" % e))
    else:
        env.eng.set_save_replay(False)
    i = 0
    obs = env.reset()
    states, phases = list(zip(*env.reset()))
    states = np.array(states, dtype = np.float32)
    phases = np.array(phases, dtype = np.int8)
    recovered = state_inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
    record.add(states, recovered)
    while i < 3600:
        if i % action_interval == 0:
            # TODO: implement other State inference model later
            # SFM inference states
            actions = []
            for ag in agents:
                actions = ag.get_action(recovered, phases, relation)
            for _ in range(action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
            states, phases = list(zip(*obs))
            states = np.array(states, dtype = np.float32)
            phases = np.array(phases, dtype = np.int8)
            recovered = state_inference_net.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix, 'select')
            record.add(states, recovered)
    att = env.eng.get_average_travel_time()
    cur_mse = record.get_cur_result()
    record.update()
    if att < best_att:
        best_att = att
    logger.info("episode:{}, Test:{}".format(e, att))
    logger.info("episode:{}, MSETest:{}".format(e, cur_mse))
    return best_att

def model_based_shared_train(logger, env, agents, episode, action_interval, state_inference_net, mask_pos, relation, mask_matrix, adj_matrix, model_dir, reward_type, save_rate, update_times=10):
    # model based approach with sdqn 
    logger.info(f"SDQN - SDQN -A control with model based learning")
    logger.info(f"reward inference model: NN_st")

    reward_inference_net = NN_predictor(agents[0].ob_length, 1, 'cpu', model_dir, reward_type)
    # inference model is initailized with trained parameters
    reward_inference_net.load_model()
    total_decision_num = 0
    best_att = np.inf
    record = MSEMetric('state mse', mask_pos)
    reward_record = MSEMetric('reward mse', mask_pos)
    for e in range(episode):
        last_obs = env.reset()
        last_states, last_phases = list(zip(*last_obs))
        last_states = np.array(last_states, dtype=np.float32)
        last_phases = np.array(last_phases, dtype=np.int8)
        # only recover state_t since we need this var to determine action t
        last_recovered = state_inference_net.predict(last_states, last_phases, relation, mask_pos, mask_matrix, adj_matrix)
        record.add(last_states, last_recovered)
        episodes_decision_num = 0
        episodes_rewards = [0 for _ in range(agents[0].sub_agents)]
        i = 0
        while i < 3600:
            if i % action_interval == 0:
                for ag in agents:
                    if total_decision_num > ag.learning_start:
                        actions = ag.choose(last_recovered, last_phases, relation)
                    else:
                        actions = ag.sample()
                rewards_list = []
                for _ in range(action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards_train = np.mean(rewards_list, axis=0)
                rewards = np.mean(rewards_train, axis=1)

                rewards_recovered = rewards.copy()
                for pos in mask_pos:
                    if reward_type == 'NN_st':
                        tmp = reward_inference_net.predict(torch.from_numpy(np.concatenate((last_recovered[pos], one_hot(last_phases[pos], agents[0].action_space.n)))).float())
                        rewards_recovered[pos] = tmp
                    elif reward_type == 'NN_sta':
                        tmp = reward_inference_net.predict(torch.from_numpy(np.concatenate((last_recovered[pos], one_hot(actions[pos], agents[0].action_space.n)))).float())
                        rewards_recovered[pos] = tmp                        
                reward_record.add(rewards, rewards_recovered)

                cur_states, cur_phases = list(zip(*obs))
                cur_states = np.array(cur_states, dtype=np.float32)
                cur_phases = np.array(cur_phases, dtype=np.int8)
                cur_recovered = state_inference_net.predict(cur_states, cur_phases, relation, mask_pos, mask_matrix, adj_matrix)
                record.add(cur_states, cur_recovered)
                for ag in agents:
                    for idx in ag.idx:
                        ag.remember(
                            (last_recovered[idx], last_phases[idx]), actions[idx], rewards_recovered[idx], (cur_recovered[idx], cur_phases[idx]), idx)
                        episodes_rewards[idx] += rewards[idx]
                        episodes_decision_num += 1
                total_decision_num += 1
                last_obs = obs
                last_states = cur_states
                last_phases = cur_phases
                last_recovered = cur_recovered
            for ag in agents:
                # only use experiences at observable intersections
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_model_freq == ag.update_model_freq - 1:
                    ag.replay()
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_target_model_freq == ag.update_target_model_freq - 1:
                    ag.update_target_network()
            # transition model (reward model in our case) training starts here
            x, target = agents[0].get_latest_sample()
            reward_inference_net.train_while_control(x, target)
            # TODO: try action later
            for ag in agents:
                # only use experiences at observable intersections
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_model_freq == ag.update_model_freq - 1:
                    ag.replay_img(reward_inference_net, update_times, reward_type)
                if total_decision_num > ag.learning_start and total_decision_num % ag.update_target_model_freq == ag.update_target_model_freq - 1:
                    ag.update_target_network()
        cur_mse = record.get_cur_result()
        reward_cur_mse = reward_record.get_cur_result()
        record.update()
        reward_record.update()
        logger.info("episode:{}, Train:{}".format(e, env.eng.get_average_travel_time()))
        logger.info("episode:{}, MSETrain:{}".format(e, cur_mse))
        logger.info("episode:{}, Reward_MSETrain:{}".format(e, reward_cur_mse))
        best_att = app2_shared_execute(logger, env, agents, e, best_att, record, state_inference_net, action_interval, mask_pos, relation, mask_matrix, adj_matrix, save_rate)
    avg_mse = record.get_result()
    logger.info(f'approach 2: shared dqn average travel time result: {best_att}')
    logger.info(f'final mse is: {avg_mse}')
    logger.info('-' * 50)
    return record
