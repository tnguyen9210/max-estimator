
# TODOs
import re
import numpy as np
import numpy.matlib as matlib

import copy
from tqdm import tqdm
from collections import defaultdict


inf1 = 0.0
inf2 = 1000

def convert_tables(Q_table, Q_nvisits, num_actions, num_depths):
    # print(Q_table)
    # print(Q_nvisits)
    # print(num_actions)
    # print(num_depths)
    # stop
    Q_table_ary = inf1*np.ones((num_depths, num_actions, num_actions))
    Q_nvisits_ary = np.zeros((num_depths, num_actions, num_actions))
    for state, action_muhats in Q_table.items():
        elems = re.findall(r'\d+', state)
        pos0 = int(elems[0])
        pos1 = int(elems[1])
        # action_max_idx = np.argmax(action_muhats)
        # V_table_ary[pos0, pos1] = action_muhats[action_max_idx]
        # V_nvisits_ary[pos0, pos1] = Q_nvisits[state][action_max_idx]
        Q_table_ary[pos0, pos1, :] = action_muhats
        Q_nvisits_ary[pos0, pos1, :] = Q_nvisits[state] 

    return Q_table_ary, Q_nvisits_ary

def create_eps_sched_fn(sched_type, min_eps=None, max_eps=None, decay_rate=None):
    if sched_type == "poly":
        def eps_sched_fn(nvisits):
            return 1.0/np.sqrt(nvisits)
    elif sched_type == "exp":
        def eps_sched_fn(i_eps):
            eps = min_eps + (max_eps - min_eps)*np.exp(-decay_rate*i_eps)
            return eps
    elif sched_type == "fixed":
        def eps_sched_fn(i_eps):
            # eps = min_eps + (max_eps - min_eps)*np.exp(-decay_rate*i_eps)
            eps = min_eps
            return eps
    
    return eps_sched_fn

def create_lr_sched_fn(sched_type, lr=None):
    if sched_type == "linear":
        def lr_sched_fn(nvisits):
            return 1.0/nvisits
    elif sched_type == "poly":
        def lr_sched_fn(nvisits):
            return 1.0/(nvisits**0.8)
    elif sched_type == "fixed":
        def lr_sched_fn(nvisits):
            return lr
    else:
        stop
        
    return lr_sched_fn


def greedy_policy(action_muhats):
    action_max_muhat = np.max(action_muhats)
    action_max_idxes = np.where(action_muhats == action_max_muhat)[0]
    # action = np.argmax(action_means)
    action = np.random.choice(action_max_idxes)
    return action


def eps_greedy_policy(action_muhats, action_nvisits, state, num_depths, eps): 
    num_actions = len(action_muhats)
    num_actions_half = int(num_actions/2)
    elems = re.findall(r'\d+', state)
    state_depth = int(elems[0])
    state_width = int(elems[1])
    if state_depth in [0, num_depths-1]:
    #     eps_greedy_action = 0
        
    # elif state_depth == 0:
        action_muhats = copy.deepcopy(action_muhats)
        action_muhats[action_nvisits == 0] = inf2
        greedy_action = greedy_policy(action_muhats)

        action_probs = eps*np.ones(num_actions)/num_actions
        action_probs[greedy_action] += 1 - eps
        eps_greedy_action = np.random.choice(num_actions, 1, p=action_probs)[0]
        
    elif state_width < num_actions_half:        
        action_muhats_half = copy.deepcopy(action_muhats[:num_actions_half])
        action_muhats_half[action_nvisits[:num_actions_half] == 0] = inf2
        greedy_action = greedy_policy(action_muhats_half)

        action_probs_half = eps*np.ones(num_actions_half)/num_actions_half
        action_probs_half[greedy_action] += 1 - eps
        eps_greedy_action = np.random.choice(
            num_actions_half, 1, p=action_probs_half)[0]
        
    elif state_width >= num_actions_half:
        action_muhats_half = copy.deepcopy(action_muhats[num_actions_half:])
        action_muhats_half[action_nvisits[num_actions_half:] == 0] = inf2
        greedy_action = greedy_policy(action_muhats_half)

        action_probs_half = eps*np.ones(num_actions_half)/num_actions_half
        action_probs_half[greedy_action] += 1 - eps
        eps_greedy_action = np.random.choice(
            num_actions_half, 1, p=action_probs_half)[0] + num_actions_half
        
    return eps_greedy_action


def q_learning(env, num_actions, num_steps_train,
               gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

    num_depths = args["num_depths"]
    num_actions_half = int(num_actions/2)
    # keep track of useful statistics
    stats = []
    Q_table = defaultdict(lambda: inf1*np.ones(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))
    
    cur_state, info = env.reset()
    cur_state = f"{cur_state}"
    start_state = copy.deepcopy(cur_state)
    
    for i_step in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):

        # print(f"\n-> i_step = {i_step}")
        # print(f"cur_state = {cur_state}")
        # choose the action a_t using epsilon greedy policy
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = eps_greedy_policy(
            Q_table[cur_state], Q_nvisits[cur_state], cur_state, num_depths, eps)
        # if cur_state == "[1 2]":
        #     print(Q_table[cur_state])

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        # print(f"action = {action}")
        # print(f"reward = {reward:0.2f}")
        # print(f"new_state = {new_state}")

        if not terminated:
            elems = re.findall(r'\d+', new_state)
            new_state_depth = int(elems[0])
            new_state_width = int(elems[1])
            # print(new_state_depth)
            if new_state_depth is [0, num_depths-1]:
                print(new_state_depth)
                stop
            # action_muhats = copy.deepcopy(Q_table[new_state])
            # action_nvisits = Q_nvisits[new_state]
            # action_muhats[action_nvisits == 0] = -np.inf
            # if new_state_width < num_actions_half:
            #     Q_est = np.max(action_muhats[:num_actions_half])
            # elif new_state_width >= num_actions_half:
            #     Q_est = np.max(action_muhats[num_actions_half:])
            # Q_est = Q_est if np.sum(action_nvisits) != 0 else 0.0
            if new_state_width < num_actions_half:
                Q_est = np.max(Q_table[new_state][:num_actions_half])
            elif new_state_width >= num_actions_half:
                Q_est = np.max(Q_table[new_state][num_actions_half:])
            
            td_error = reward + gamma*Q_est - Q_table[cur_state][action]
        else:
            td_error = reward - Q_table[cur_state][action]

        # print(f"before update = {Q_table[cur_state][action]:0.2f}")
        Q_nvisits[cur_state][action] += 1
        lr = lr_sched_fn(Q_nvisits[cur_state][action]) 
        Q_table[cur_state][action] += lr*td_error
        # print(f"lr = {lr:0.2f}")
        # print(f"td_error = {td_error:0.2f}")
        # print(f"after update = {Q_table[cur_state][action]:0.2f}")
        
        if terminated:
            # print("terminated")
            cur_state, info = env.reset()
            cur_state = f"{cur_state}"
            # print(f"reset_state = {cur_state}")
        else:
            cur_state = new_state

        # action_muhats = copy.deepcopy(Q_table[start_state])
        # action_nvisits = Q_nvisits[start_state]
        # action_muhats[action_nvisits == 0] = -np.inf
        Q_start_est = np.max(Q_table[start_state])
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, Q_start_est))
    
    # print(Q_table[start_state])
    # print(np.max(Q_table[start_state]))
    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.num_depths)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats


def double_q_learning(env, num_actions, num_steps_train,
               gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):
        
    # keep track of useful statistics
    stats = []
    Q_table = defaultdict(lambda: inf1*np.ones(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))
    DQ_table = defaultdict(lambda: inf1*np.ones((num_actions, 2)))
    DQ_nvisits = defaultdict(lambda: np.zeros((num_actions, 2)))
    
    cur_state, info = env.reset()
    cur_state = f"{cur_state}"
    start_state = copy.deepcopy(cur_state)
    
    for i_step in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):

        # print(f"\n-> i_step = {i_step}")
        # print(f"cur_state = {cur_state}")
        # choose the action a_t using epsilon greedy policy
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = eps_greedy_policy(Q_table[cur_state], Q_nvisits[cur_state], eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        # print(f"action = {action}")
        # print(f"reward = {reward}")
        # print(f"new_state = {new_state}")
        
        # next_action_best = np.argmax(DQ_table[new_state][:, Q_idx])
        # Q_est = DQ_table[new_state][next_action_best, 1-Q_idx]
        
        if not terminated:
            Q_idx = np.random.randint(0, 2)
            Q_est = double_estimator(DQ_table, new_state, Q_idx)
            td_error = reward + gamma*Q_est - DQ_table[cur_state][action, Q_idx]
        else:
            td_error = reward - DQ_table[cur_state][action, Q_idx]

        DQ_nvisits[cur_state][action, Q_idx] += 1
        lr = lr_sched_fn(DQ_nvisits[cur_state][action, Q_idx])
        DQ_table[cur_state][action, Q_idx] += lr*td_error

        Q_table[cur_state][action] = np.mean(DQ_table[cur_state][action,:])
        Q_nvisits[cur_state][action] = np.sum(DQ_nvisits[cur_state][action,:])
        
        if terminated:
            # print("terminated")
            cur_state, info = env.reset()
            cur_state = f"{cur_state}"
            # print(f"reset_state = {cur_state}")
        else:
            cur_state = new_state
            
        # stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))
        Q_idx = np.random.randint(0, 2)
        Q_start_est = double_estimator(DQ_table, start_state, Q_idx)
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, Q_start_est))

    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.num_depths)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats

def double_estimator(DQ_table, new_state, Q_idx):
    next_action_best = np.argmax(DQ_table[new_state][:, Q_idx])
    Q_est = DQ_table[new_state][next_action_best, 1-Q_idx]
    return Q_est

def weightedms_q_learning(
        env, num_actions, num_steps_train,
        gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

    num_depths = args["num_depths"]
    num_actions_half = int(num_actions/2)
    
    # keep track of useful statistics
    stats = []
    num_data = args["weightedms_num_data"]
    num_actions = env.action_space.n
    Q_table = defaultdict(lambda: inf1*np.ones(num_actions))
    Q2_table = defaultdict(lambda: inf1*np.ones(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))
    # Q_nupdates = defaultdict(lambda: np.zeros(num_actions))
    Q_sigmahats = defaultdict(lambda: np.ones(num_actions)*1e10)
    weights_var = defaultdict(lambda: np.zeros(num_actions))
    probs = np.ones(num_actions)/num_actions

    cur_state, info = env.reset()
    cur_state = f"{cur_state}"
    start_state = copy.deepcopy(cur_state)
    
    for i_step in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):
        
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = eps_greedy_policy(
            Q_table[cur_state], Q_nvisits[cur_state], cur_state, num_depths, eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"

        if not terminated:
            elems = re.findall(r'\d+', new_state)
            new_state_depth = int(elems[0])
            new_state_width = int(elems[1])

            if new_state_width < num_actions_half:
                action_muhats = copy.deepcopy(Q_table[new_state][:num_actions_half])
                action_sigmahats = Q_sigmahats[new_state][:num_actions_half]
                action_sigmahats[action_sigmahats < 1e-5] = 1e-5
                
                Q_est = weightedms_estimator(
                    action_muhats, action_sigmahats, num_actions_half, num_data)

            elif new_state_width >= num_actions_half:
                action_muhats = copy.deepcopy(Q_table[new_state][num_actions_half:])
                action_sigmahats = Q_sigmahats[new_state][num_actions_half:]
                action_sigmahats[action_sigmahats < 1e-5] = 1e-5
                
                Q_est = weightedms_estimator(
                    action_muhats, action_sigmahats, num_actions_half, num_data)            
            
            # compute td_target
            td_target = reward + gamma*Q_est
        else:
            td_target = reward

        Q_nvisits[cur_state][action] += 1
        lr = lr_sched_fn(Q_nvisits[cur_state][action])

        Q_table[cur_state][action] = (1-lr)*Q_table[cur_state][action] + lr*td_target
        Q2_table[cur_state][action] = (1-lr)*Q2_table[cur_state][action] + lr*td_target**2

        if Q_nvisits[cur_state][action] >= 1:
            weights_var[cur_state][action] = \
                (1-lr)**2*weights_var[cur_state][action] + lr**2
            n = 1.0/weights_var[cur_state][action]
            diff = Q2_table[cur_state][action] - Q_table[cur_state][action]**2
            if diff < 0:
                diff = 0
            Q_sigmahats[cur_state][action] = np.sqrt(diff/n)

        if terminated:
            cur_state, info = env.reset()
            cur_state = f"{cur_state}"
        else:
            cur_state = new_state

        action_muhats = copy.deepcopy(Q_table[start_state])
        action_sigmahats = Q_sigmahats[start_state]
        action_sigmahats[action_sigmahats < 1e-5] = 1e-5
        
        Q_start_est = weightedms_estimator(
            action_muhats, action_sigmahats, num_actions, num_data)
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, Q_start_est))

    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.num_depths)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats

def weightedms_estimator(
        action_muhats, action_sigmahats, num_actions, num_data):
    # cur_muhats = Q_table[est_state]
    # cur_sigmahats = Q_sigmahats[est_state]
    # cur_sigmahats[cur_sigmahats < 1e-4] = 1e-4

    # sample 
    action_muhats_mat = matlib.repmat(action_muhats, num_data, 1)
    action_sigmahats_mat = matlib.repmat(action_sigmahats, num_data, 1)
    eps_mat = np.random.randn(num_data, num_actions)

    samples = action_muhats_mat + action_sigmahats_mat*eps_mat
    samples_max_idxes = np.argmax(samples, 1)

    # compute probs 
    probs = np.zeros(num_actions)
    idxes, cnts = np.unique(samples_max_idxes, return_counts=True)
    probs[idxes[cnts > 0]] = cnts[cnts > 0]
    probs = probs/num_data

    # compute Q_est 
    Q_est = np.dot(probs, action_muhats)
    # print(Q_est)
    # stop
    return Q_est

def haver2_q_learning(env, num_actions, num_steps_train,
               gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

    num_depths = args["num_depths"]
    num_actions_half = int(num_actions/2)
    
    # get params
    # action_sigma = args["action_sigma"]
    haver_alpha = args["haver_alpha"]
    haver_delta = args["haver_delta"]
    haver_const = args["haver_const"]
    
    # keep track of useful statistics
    stats = []
    Q_table = defaultdict(lambda: inf1*np.ones(num_actions))
    Q2_table = defaultdict(lambda: inf1*np.ones(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))

    Q_sigmahats = defaultdict(lambda: np.ones(num_actions)*1e10)
    weights_var = defaultdict(lambda: np.zeros(num_actions))
    
    cur_state, info = env.reset()
    cur_state = f"{cur_state}"
    start_state = copy.deepcopy(cur_state)
    
    for i_step in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):

        # print(f"\n-> i_step = {i_step}")
        # print(f"cur_state = {cur_state}")
        # choose the action a_t using epsilon greedy policy
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = eps_greedy_policy(
            Q_table[cur_state], Q_nvisits[cur_state], cur_state, num_depths, eps)
        
        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        # print(f"nvisits = {nvisits}")
        # print(f"eps = {eps}")
        # print(f"action = {action}")
        # print(f"reward = {reward:0.2f}")
        # print(f"new_state = {new_state}")

        if not terminated:
            elems = re.findall(r'\d+', new_state)
            new_state_depth = int(elems[0])
            new_state_width = int(elems[1])

            if new_state_width < num_actions_half:
                action_muhats = copy.deepcopy(Q_table[new_state][:num_actions_half])
                action_sigmahats = Q_sigmahats[new_state][:num_actions_half]
                action_sigmahats[action_sigmahats < 1e-5] = 1e-5
                action_nvisits = Q_nvisits[new_state][:num_actions_half]
                action_weights_var = weights_var[new_state][:num_actions_half]
                
                Q_est = haver2_estimator(
                    action_muhats, action_sigmahats, action_nvisits, action_weights_var,
                    num_actions_half, haver_alpha, haver_delta, haver_const, lr_sched_fn)
                
            elif new_state_width >= num_actions_half:
                action_muhats = copy.deepcopy(Q_table[new_state][num_actions_half:])
                action_sigmahats = Q_sigmahats[new_state][num_actions_half:]
                action_sigmahats[action_sigmahats < 1e-5] = 1e-5
                action_nvisits = Q_nvisits[new_state][num_actions_half:]
                action_weights_var = weights_var[new_state][num_actions_half:]
                
                Q_est = haver2_estimator(
                    action_muhats, action_sigmahats, action_nvisits, action_weights_var,
                    num_actions_half, haver_alpha, haver_delta, haver_const, lr_sched_fn)
            
            # td_error = reward + gamma*Q_est - Q_table[cur_state][action]
            td_target = reward + gamma*Q_est
        else:
            # td_error = reward - Q_table[cur_state][action]
            td_target = reward

        Q_nvisits[cur_state][action] += 1
        lr = lr_sched_fn(Q_nvisits[cur_state][action])

        Q_table[cur_state][action] = \
            (1-lr)*Q_table[cur_state][action] + lr*td_target
        Q2_table[cur_state][action] = \
            (1-lr)*Q2_table[cur_state][action] + lr*td_target**2

        if Q_nvisits[cur_state][action] >=  1:
            weights_var[cur_state][action] = \
                (1-lr)**2*weights_var[cur_state][action] + lr**2
            n = 1.0/weights_var[cur_state][action]
            diff = Q2_table[cur_state][action] - Q_table[cur_state][action]**2
            if diff < 0:
                diff = 0
            Q_sigmahats[cur_state][action] = np.sqrt(diff)
        
        if terminated:
            # print("terminated")
            cur_state, info = env.reset()
            cur_state = f"{cur_state}"
            # print(f"reset_state = {cur_state}")
        else:
            cur_state = new_state
            
        action_muhats = copy.deepcopy(Q_table[start_state])
        action_sigmahats = Q_sigmahats[start_state]
        action_sigmahats[action_sigmahats < 1e-5] = 1e-5
        action_nvisits = Q_nvisits[start_state]
        action_weights_var = weights_var[start_state]
                
        Q_start_est = haver2_estimator(
                    action_muhats, action_sigmahats, action_nvisits, action_weights_var,
                    num_actions, haver_alpha, haver_delta, haver_const, lr_sched_fn)
        # Q_start_est = np.max(Q_table[start_state])
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, Q_start_est))

    # print(Q_nvisits[start_state])
    # print(Q_table[start_state])
    # print(np.max(Q_table[start_state]))
    # Q_start_est = haver2_estimator(
    #     Q_table, Q_sigmahats, Q_nvisits, start_state, num_actions,
    #     haver_alpha, haver_delta, haver_const, lr_sched_fn)
    # print(Q_start_est)
    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.num_depths)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats


def haver2_estimator(
        action_muhats, action_sigmahats, action_nvisits, action_weights_var,
        num_actions, haver_alpha, haver_delta, haver_const, lr_sched_fn):

    # print(num_actions)
    # print(len(action_muhats))
    action_muhats[action_nvisits == 0] = -np.inf
    action_max_idx = np.argmax(action_muhats)
    action_max_muhat = action_muhats[action_max_idx]
    action_max_sigmahat = action_sigmahats[action_max_idx]
    action_max_nvisits = action_nvisits[action_max_idx]
    action_max_weights_var = action_weights_var[action_max_idx]
    # print(action_muhats)
    # print(action_sigmahats)
    # print(action_nvisits)
    # print(action_weights_var)


    Q_est_sum = 0
    Q_est_cnt = 0
    for i in range(num_actions):
        if action_nvisits[i] != 0:
            # avg = action_max_sigmahat**2/action_max_nvisits \
            #     + action_sigmahats[i]**2/action_nvisits[i]
            avg = action_max_sigmahat**2*action_max_weights_var \
                + action_sigmahats[i]**2*action_weights_var[i]
            # avg = action_max_sigmahat**2 + action_sigmahats[i]**2
            thres = haver_const*np.sqrt(avg*np.log(num_actions**haver_alpha/haver_delta))
            # print(i)
            # print(thres)
            # print(action_max_muhat)
            # print(action_muhats[i])
            if action_max_muhat - action_muhats[i] <= thres:
                # Q_est_sum += action_muhats[i]*action_nvisits[i]
                # Q_est_cnt += action_nvisits[i]
                Q_est_sum += action_muhats[i]/action_weights_var[i]
                Q_est_cnt += 1.0/action_weights_var[i]
            
    Q_est = Q_est_sum/Q_est_cnt if Q_est_cnt != 0 else 0.0
    # print(f"Q_est_sum = {Q_est_sum}")
    # print(f"Q_est_cnt = {Q_est_cnt}")
    # print(f"Q_est = {Q_est}")
    return Q_est



def create_q_algo(algo_name, **args):
    if algo_name == "max" or algo_name == "q_learning":
        q_algo = q_learning
    elif algo_name == "avg" or algo_name == "avg_q_learning":
        q_algo = avg_q_learning
    elif algo_name == "double" or algo_name == "double_q_learning":
        q_algo = double_q_learning
    elif algo_name == "haver" or algo_name == "haver_q_learning":
        q_algo = haver_q_learning
    elif algo_name == "haver2" or algo_name == "haver2_q_learning":
        q_algo = haver2_q_learning
    elif algo_name == "weightedms" or algo_name == "weightedms_q_learning":
        q_algo = weightedms_q_learning

    return q_algo

def create_q_estim(algo_estim, **args):
    if algo_estim == "max" or algo_estim == "q_learning":
        q_algo = q_learning
    elif algo_estim == "avg" or algo_estim == "avg_q_learning":
        q_algo = avg_q_learning
    elif algo_estim == "double" or algo_estim == "double_q_learning":
        q_algo = double_q_learning
    elif algo_estim == "haver" or algo_estim == "haver_q_learning":
        q_algo = haver_q_learning
    elif algo_estim == "haver2" or algo_estim == "haver2_q_learning":
        q_algo = haver2_estimator
    elif algo_estim == "weightedms" or algo_name == "weightedms_q_learning":
        q_algo = weightedms_estimator

    return q_algo
