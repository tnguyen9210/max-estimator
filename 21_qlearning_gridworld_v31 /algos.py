
# TODOs
import re
import numpy as np
import numpy.matlib as matlib

import copy
from tqdm import tqdm
from collections import defaultdict

inf_Q = 0.0
inf_mu = np.inf


def convert_tables(Q_table, Q_nvisits, num_actions, gw_size):
    Q_table_ary = np.zeros((gw_size, gw_size, num_actions))
    Q_nvisits_ary = np.zeros((gw_size, gw_size, num_actions))
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


def eps_greedy_policy(action_muhats, action_nvisits, eps): 
    num_actions = len(action_muhats)
    action_muhats = copy.deepcopy(action_muhats)
    action_muhats[action_nvisits == 0] = inf_mu
    greedy_action = greedy_policy(action_muhats)
    
    action_probs = eps*np.ones(num_actions)/num_actions
    action_probs[greedy_action] += 1 - eps
    eps_greedy_action = np.random.choice(num_actions, 1, p=action_probs)[0]
    
    return eps_greedy_action


def greedy_policy2(action_muhats, action_sigmahats, action_nvisits, action_ess):
    num_actions = len(action_muhats)
    total_nvisits = np.sum(action_nvisits)
    if total_nvisits == 0:
        action_max_muhat = np.max(action_muhats)
        action_max_idxes = np.where(action_muhats == action_max_muhat)[0]
        # action = np.argmax(action_means)
        action = np.random.choice(action_max_idxes)
        return action
    
    total_ess = np.sum(action_ess)
    action_maxlcb_idx = None
    action_maxlcb = -np.inf
    for i in range(num_actions):
        num = 2*action_sigmahats[i]**2/(action_ess[i]+1)
        log = np.log(2*num_actions*total_ess**2/0.05)
        action_i_bonus = np.sqrt(num*log)
        action_i_lcb = action_muhats[i] - action_i_bonus
        # print(i)
        # print(action_i_lcb)
        if action_i_lcb > action_maxlcb:
            action_maxlcb_idx = i
            action_maxlcb = action_i_lcb
            
    
    return action_maxlcb_idx


def eps_greedy_policy2(action_muhats, action_sigmahats, action_nvisits, action_ess, eps): 
    num_actions = len(action_muhats)
    action_muhats = copy.deepcopy(action_muhats)
    action_muhats[action_nvisits == 0] = inf_mu
    greedy_action = greedy_policy2(action_muhats, action_sigmahats, action_nvisits, action_ess)
    
    action_probs = eps*np.ones(num_actions)/num_actions
    action_probs[greedy_action] += 1 - eps
    eps_greedy_action = np.random.choice(num_actions, 1, p=action_probs)[0]
    
    return eps_greedy_action




def q_learning(env, num_actions, num_steps_train,
               gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):
        
    # keep track of useful statistics
    stats = []
    Q_table = defaultdict(lambda: np.zeros(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))
    
    cur_state, info = env.reset()
    cur_state = f"{cur_state}"
    start_state = copy.deepcopy(cur_state)
    
    for i_step in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):

        # choose the action a_t using epsilon greedy policy
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = eps_greedy_policy(
            Q_table[cur_state], Q_nvisits[cur_state], eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        # print(f"\n-> i_step = {i_step}")
        # print(f"cur_state = {cur_state}")
        # print(f"action = {action}")
        # print(f"reward = {reward:0.2f}")
        # print(f"new_state = {new_state}")
        # if terminated:
        #     print("terminated")

        if not terminated:
            action_muhats = copy.deepcopy(Q_table[new_state])
            action_nvisits = Q_nvisits[new_state]
            action_muhats[action_nvisits == 0] = -inf_mu
            Q_est = np.max(action_muhats) if np.sum(action_nvisits) != 0 else 0
            
            td_error = reward + gamma*Q_est - Q_table[cur_state][action]
        else:
            td_error = reward - Q_table[cur_state][action]

        Q_nvisits[cur_state][action] += 1
        lr = lr_sched_fn(Q_nvisits[cur_state][action]) 
        Q_table[cur_state][action] += lr*td_error
        
        if terminated:
            # print("terminated")
            cur_state, info = env.reset()
            cur_state = f"{cur_state}"
            # print(f"reset_state = {cur_state}")
        else:
            cur_state = new_state

        action_muhats = copy.deepcopy(Q_table[start_state])
        action_nvisits = Q_nvisits[start_state]
        action_muhats[action_nvisits == 0] = -inf_mu
        Q_start_est = np.max(action_muhats) if np.sum(action_nvisits) != 0 else 0
        
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, Q_start_est))
    
    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.size)

    return Q_table_ary, Q_nvisits_ary, stats


def weightedms_q_learning(
        env, num_actions, num_steps_train,
        gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):
        
    # keep track of useful statistics
    stats = []
    num_data = args["weightedms_num_data"]
    num_actions = env.action_space.n
    Q_table = defaultdict(lambda: inf_Q*np.ones(num_actions))
    Q2_table = defaultdict(lambda: inf_Q*np.ones(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))
    
    Q_sigmahats = defaultdict(lambda: np.ones(num_actions)*0)
    Q_ess = defaultdict(lambda: 0*np.ones(num_actions))
    Q_ess_weights = defaultdict(lambda: 0*np.ones(num_actions))
    Q2_ess_weights = defaultdict(lambda: 0*np.ones(num_actions))
    
    cur_state, info = env.reset()
    cur_state = f"{cur_state}"
    start_state = copy.deepcopy(cur_state)
    
    for i_step in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):
        
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = eps_greedy_policy2(
            Q_table[cur_state], Q_sigmahats[cur_state],
            Q_nvisits[cur_state], Q_ess[cur_state], eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"

        if not terminated:
            # compute Q_est with weighted estimator
            # get 
            action_muhats = copy.deepcopy(Q_table[new_state])
            action_nvisits = Q_nvisits[new_state]
            action_muhats[action_nvisits == 0] = -inf_mu

            action_sigmahats = Q_sigmahats[new_state]
            action_sigmahats[action_sigmahats < 1e-5] = 1e-5
            
            Q_est = weightedms_estimator(
                    action_muhats, action_sigmahats, action_nvisits, 
                    num_actions, num_data)

            # compute td_target
            td_target = reward + gamma*Q_est
        else:
            td_target = reward

        Q_nvisits[cur_state][action] += 1
        lr = lr_sched_fn(Q_nvisits[cur_state][action])

        Q_table[cur_state][action] = \
            (1-lr)*Q_table[cur_state][action] + lr*td_target
        Q2_table[cur_state][action] = \
            (1-lr)*Q2_table[cur_state][action] + lr*td_target**2

        if Q_nvisits[cur_state][action] >= 1:
            Q_ess_weights[cur_state][action] = \
                (1-lr)*Q_ess_weights[cur_state][action] + lr
            Q2_ess_weights[cur_state][action] = \
                (1-lr)**2*Q2_ess_weights[cur_state][action] + lr**2
            Q_ess[cur_state][action] = \
                1./Q2_ess_weights[cur_state][action]
            
        if Q_nvisits[cur_state][action] >= 1:
            diff = Q2_table[cur_state][action] - Q_table[cur_state][action]**2
            # diff = action_sigma**2
            if diff < 0:
                diff = 0
            Q_sigmahats[cur_state][action] = np.sqrt(diff/Q_ess[cur_state][action])

        if terminated:
            cur_state, info = env.reset()
            cur_state = f"{cur_state}"
        else:
            cur_state = new_state

        action_muhats = copy.deepcopy(Q_table[start_state])
        action_nvisits = Q_nvisits[start_state]
        action_muhats[action_nvisits == 0] = -inf_mu
        
        action_sigmahats = Q_sigmahats[start_state]
        action_sigmahats[action_sigmahats < 1e-5] = 1e-5

        Q_start_est = weightedms_estimator(
            action_muhats, action_sigmahats, action_nvisits, num_actions, num_data)
        
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, Q_start_est))

    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.size)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats

def weightedms_estimator(
        action_muhats, action_sigmahats, action_nvisits, num_actions, num_data):

    if np.sum(action_nvisits) == 0:
        return 0
    
    # sample 
    # action_muhats_mat = matlib.repmat(action_muhats, num_data, 1)
    # action_sigmahats_mat = matlib.repmat(action_sigmahats, num_data, 1)
    # eps_mat = np.random.randn(num_data, num_actions)
    # samples = action_muhats_mat + action_sigmahats_mat*eps_mat
    samples = np.random.normal(
        action_muhats, action_sigmahats, size=(num_data, num_actions))
    
    # samples[action_nvisits == 0,:] = -inf_mu
    samples_max_idxes = np.argmax(samples, 1)

    # compute probs 
    probs = np.zeros(num_actions)
    idxes, cnts = np.unique(samples_max_idxes, return_counts=True)
    probs[idxes[cnts > 0]] = cnts[cnts > 0]
    probs = probs/num_data

    # print(action_sigmahats)
    # print(action_muhats)
    # print(probs)
    
    # compute Q_est 
    Q_est = np.dot(
        probs[action_nvisits != 0], action_muhats[action_nvisits != 0])
    
    # Q_est = Q_est if np.sum(action_nvisits) != 0 else 0.0
    # print(f"Q_est_ = {Q_est:0.2f}")
    # stop
    return Q_est


def haver_q_learning(env, num_actions, num_steps_train,
               gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

    # get params
    # action_sigma = args["action_sigma"]
    haver_alpha = args["haver_alpha"]
    haver_delta = args["haver_delta"]
    haver_const = args["haver_const"]
    
    # keep track of useful statistics
    stats = []
    Q_table = defaultdict(lambda: inf_Q*np.ones(num_actions))
    Q2_table = defaultdict(lambda: inf_Q*np.ones(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))

    Q_sigmahats = defaultdict(lambda: np.ones(num_actions)*1e10)
    Q_ess = defaultdict(lambda: 0*np.ones(num_actions))
    Q_ess_weights = defaultdict(lambda: 0*np.ones(num_actions))
    Q2_ess_weights = defaultdict(lambda: 0*np.ones(num_actions))
    
    cur_state, info = env.reset()
    cur_state = f"{cur_state}"
    start_state = copy.deepcopy(cur_state)
    
    for i_step in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):

        # choose the action a_t using epsilon greedy policy
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = eps_greedy_policy(
            Q_table[cur_state], Q_nvisits[cur_state], eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        # print(f"action = {action}")
        # print(f"reward = {reward}")
        # print(f"new_state = {new_state}")

        if not terminated:
            action_muhats = copy.deepcopy(Q_table[new_state])
            action_nvisits = Q_nvisits[new_state]
            action_muhats[action_nvisits == 0] = -inf_mu
                
            action_sigmahats = Q_sigmahats[new_state]
            action_sigmahats[action_sigmahats < 1e-5] = 1e-5
                
            action_ess = Q_ess[new_state]
            
            Q_est = haver_estimator(
                    action_muhats, action_sigmahats, action_nvisits, action_ess,
                    num_actions, haver_alpha, haver_delta, haver_const,
                    lr_sched_fn, args, debug=False)
            
            td_target = reward + gamma*Q_est
        else:
            td_target = reward

        Q_nvisits[cur_state][action] += 1
        lr = lr_sched_fn(Q_nvisits[cur_state][action])

        Q_table[cur_state][action] = \
            (1-lr)*Q_table[cur_state][action] + lr*td_target
        Q2_table[cur_state][action] = \
            (1-lr)*Q2_table[cur_state][action] + lr*td_target**2

        if Q_nvisits[cur_state][action] >= 1:
            Q_ess_weights[cur_state][action] = \
                (1-lr)*Q_ess_weights[cur_state][action] + lr
            Q2_ess_weights[cur_state][action] = \
                (1-lr)**2*Q2_ess_weights[cur_state][action] + lr**2
            Q_ess[cur_state][action] = \
                1./Q2_ess_weights[cur_state][action]
            
        if Q_nvisits[cur_state][action] > 1:
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
        action_nvisits = Q_nvisits[start_state]
        action_muhats[action_nvisits == 0] = -inf_mu
        
        action_sigmahats = Q_sigmahats[start_state]
        action_sigmahats[action_sigmahats < 1e-5] = 1e-5
        
        action_ess = Q_ess[start_state]
                
        Q_start_est = haver_estimator(
            action_muhats, action_sigmahats, action_nvisits, action_ess,
            num_actions, haver_alpha, haver_delta, haver_const, lr_sched_fn,
            args, debug=False)

        # Q_start_est = np.max(Q_table[start_state])
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, Q_start_est))

    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.size)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats

def haver_estimator(
        action_muhats, action_sigmahats, action_nvisits, action_ess,
        num_actions, haver_alpha, haver_delta, haver_const, lr_sched_fn,
        args, debug):

    if debug:
        print(action_nvisits)
        print(action_ess)
        print(action_muhats)
        print(action_sigmahats)

    total_ess = np.sum(action_ess)
    total_nvisits = np.sum(action_nvisits)
    if total_nvisits == 0:
        return 0
        
    action_max_idx = np.argmax(action_muhats)
    action_max_muhat = action_muhats[action_max_idx]
    action_max_sigmahat = action_sigmahats[action_max_idx]
    action_max_nvisits = action_nvisits[action_max_idx]
    action_max_ess = action_ess[action_max_idx]
    
    # Q_est_sum = 0
    # Q_est_cnt = 0
    Bset_muhats = []
    Bset_nvisits = []
    for i in range(num_actions):
        if action_nvisits[i] != 0:
            # avg = action_sigma**2/action_max_ess \
            #     + action_sigma**2/action_ess[i]
            avg = action_max_sigmahat**2/action_max_ess \
                + action_sigmahats[i]**2/action_ess[i]
            # avg = action_sigma**2/action_max_nvisits \
            #     + action_sigma**2/action_nvisits[i]
            # avg = action_max_sigmahat**2/action_max_nvisits \
            #     + action_sigmahats[i]**2/action_nvisits[i]
            # avg = action_max_sigmahat**2 + action_sigmahats[i]**2

            # print(action_sigmahats[i])
            log = np.log(num_actions**haver_alpha/haver_delta)
            thres = haver_const*np.sqrt(avg*log)
            
            # print(f"avg = {avg:0.2f}")
            # print(f"thres = {thres:0.2f}")
            # print(action_max_muhat)
            # print(action_muhats[i])
            if action_max_muhat - action_muhats[i] <= thres:
                # Q_est_sum += action_muhats[i]*action_nvisits[i]
                # Q_est_cnt += 1.0*action_nvisits[i]
                Bset_muhats.append(action_muhats[i])
                Bset_nvisits.append(action_ess[i])

    Bset_muhats = np.array(Bset_muhats)
    Bset_nvisits = np.array(Bset_nvisits)
    Q_est = np.dot(Bset_muhats, Bset_nvisits)/np.sum(Bset_nvisits)
    
    if debug:
        print(f"action_max_idx = {action_max_idx}")
        print(f"action_max_muhat = {action_max_muhat:0.2f}")

        print(f"Bset_nvisits = {Bset_nvisits}")
        print(f"Bset_muhats = {Bset_muhats}")
        # print(tmp)
        print(f"Q_est = {Q_est:.2f}")
    # stop
    # print(f"Q_est_sum = {Q_est_sum}")
    # print(f"Q_est_cnt = {Q_est_cnt}")
    
    return Q_est

def haver3_q_learning(env, num_actions, num_steps_train,
               gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

    # get params
    # action_sigma = args["action_sigma"]
    haver_alpha = args["haver_alpha"]
    haver_delta = args["haver_delta"]
    haver_const = args["haver_const"]
    
    # keep track of useful statistics
    stats = []
    Q_table = defaultdict(lambda: inf_Q*np.ones(num_actions))
    Q2_table = defaultdict(lambda: inf_Q*np.ones(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))

    Q_sigmahats = defaultdict(lambda: np.ones(num_actions)*0)
    Q_ess = defaultdict(lambda: 0*np.ones(num_actions))
    Q_ess_weights = defaultdict(lambda: 0*np.ones(num_actions))
    Q2_ess_weights = defaultdict(lambda: 0*np.ones(num_actions))
    
    cur_state, info = env.reset()
    cur_state = f"{cur_state}"
    start_state = copy.deepcopy(cur_state)
    
    for i_step in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):

        # choose the action a_t using epsilon greedy policy
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = eps_greedy_policy2(
            Q_table[cur_state], Q_sigmahats[cur_state],
            Q_nvisits[cur_state], Q_ess[cur_state], eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        # print(f"action = {action}")
        # print(f"reward = {reward}")
        # print(f"new_state = {new_state}")

        if not terminated:
            action_muhats = copy.deepcopy(Q_table[new_state])
            action_nvisits = Q_nvisits[new_state]
            action_muhats[action_nvisits == 0] = -inf_mu
                
            action_sigmahats = Q_sigmahats[new_state]
            action_sigmahats[action_sigmahats < 1e-5] = 1e-5
                
            action_ess = Q_ess[new_state]
            
            Q_est = haver3_estimator(
                    action_muhats, action_sigmahats, action_nvisits, action_ess,
                    num_actions, haver_alpha, haver_delta, haver_const,
                    lr_sched_fn, args, debug=False)
            
            td_target = reward + gamma*Q_est
        else:
            td_target = reward

        Q_nvisits[cur_state][action] += 1
        lr = lr_sched_fn(Q_nvisits[cur_state][action])

        Q_table[cur_state][action] = \
            (1-lr)*Q_table[cur_state][action] + lr*td_target
        Q2_table[cur_state][action] = \
            (1-lr)*Q2_table[cur_state][action] + lr*td_target**2

        if Q_nvisits[cur_state][action] >= 1:
            Q_ess_weights[cur_state][action] = \
                (1-lr)*Q_ess_weights[cur_state][action] + lr
            Q2_ess_weights[cur_state][action] = \
                (1-lr)**2*Q2_ess_weights[cur_state][action] + lr**2
            Q_ess[cur_state][action] = \
                1./Q2_ess_weights[cur_state][action]
            
        if Q_nvisits[cur_state][action] >= 1:
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
        action_nvisits = Q_nvisits[start_state]
        action_muhats[action_nvisits == 0] = -inf_mu
        
        action_sigmahats = Q_sigmahats[start_state]
        action_sigmahats[action_sigmahats < 1e-5] = 1e-5
        
        action_ess = Q_ess[start_state]
                
        Q_start_est = haver3_estimator(
            action_muhats, action_sigmahats, action_nvisits, action_ess,
            num_actions, haver_alpha, haver_delta, haver_const, lr_sched_fn,
            args, debug=False)

        # Q_start_est = np.max(Q_table[start_state])
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, Q_start_est))

    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.size)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats

def haver3_estimator(
        action_muhats, action_sigmahats, action_nvisits, action_ess,
        num_actions, haver_alpha, haver_delta, haver_const, lr_sched_fn,
        args, debug):

    if debug:
        print(action_nvisits)
        print(action_ess)
        print(action_muhats)
        print(action_sigmahats)

    total_ess = np.sum(action_ess)
    total_nvisits = np.sum(action_nvisits)
    if total_nvisits == 0:
        return 0
    
    action_maxlcb_idx = None
    action_maxlcb = -np.inf
    for i in range(num_actions):
        num = 2*action_sigmahats[i]**2/(action_ess[i]+1)
        log = np.log(2*num_actions*total_ess**2/haver_delta)
        action_i_bonus = np.sqrt(num*log)
        action_i_lcb = action_muhats[i] - action_i_bonus
        # print(i)
        # print(action_i_lcb)
        if action_i_lcb > action_maxlcb:
            action_maxlcb_idx = i
            action_maxlcb = action_i_lcb
    
    # action_max_idx = np.argmax(action_muhats)
    action_maxlcb_muhat = action_muhats[action_maxlcb_idx]
    action_maxlcb_sigmahat = action_sigmahats[action_maxlcb_idx]
    action_maxlcb_nvisits = action_nvisits[action_maxlcb_idx]
    action_maxlcb_ess = action_ess[action_maxlcb_idx]
    
    # Q_est_sum = 0
    # Q_est_cnt = 0
    Bset_muhats = []
    Bset_nvisits = []
    for i in range(num_actions):
        if action_nvisits[i] != 0:
            # avg = action_sigma**2/action_maxlcb_ess \
            #     + action_sigma**2/action_ess[i]
            avg = action_maxlcb_sigmahat**2/action_maxlcb_ess \
                + action_sigmahats[i]**2/action_ess[i]
            # avg = action_sigma**2/action_maxlcb_nvisits \
            #     + action_sigma**2/action_nvisits[i]
            # avg = action_maxlcb_sigmahat**2/action_maxlcb_nvisits \
            #     + action_sigmahats[i]**2/action_nvisits[i]
            # avg = action_maxlcb_sigmahat**2 + action_sigmahats[i]**2
            log = np.log(num_actions**haver_alpha/haver_delta)
            thres = haver_const*np.sqrt(avg*log)
            
            # print(f"avg = {avg:0.2f}")
            # print(f"thres = {thres:0.2f}")
            # print(action_maxlcb_muhat)
            # print(action_muhats[i])
            if action_maxlcb_muhat - action_muhats[i] <= thres:
                # Q_est_sum += action_muhats[i]*action_nvisits[i]
                # Q_est_cnt += 1.0*action_nvisits[i]
                Bset_muhats.append(action_muhats[i])
                Bset_nvisits.append(action_ess[i])

    Bset_muhats = np.array(Bset_muhats)
    Bset_nvisits = np.array(Bset_nvisits)
    Q_est = np.dot(Bset_muhats, Bset_nvisits)/np.sum(Bset_nvisits)
    
    if debug:
        print(f"action_maxlcb_idx = {action_maxlcb_idx}")
        print(f"action_maxlcb_muhat = {action_maxlcb_muhat:0.2f}")

        print(f"Bset_nvisits = {Bset_nvisits}")
        print(f"Bset_muhats = {Bset_muhats}")
        # print(tmp)
        print(f"Q_est = {Q_est:.2f}")
    # stop
    # print(f"Q_est_sum = {Q_est_sum}")
    # print(f"Q_est_cnt = {Q_est_cnt}")
    
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
    elif algo_name == "haver3" or algo_name == "haver3_q_learning":
        q_algo = haver3_q_learning
    elif algo_name == "weightedms" or algo_name == "weightedms_q_learning":
        q_algo = weightedms_q_learning

    return q_algo
