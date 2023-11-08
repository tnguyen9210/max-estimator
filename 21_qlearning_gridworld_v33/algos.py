
# TODOs
import re
import numpy as np
import numpy.matlib as matlib

import copy
from tqdm import tqdm
from collections import defaultdict

inf_Q = 0.0
inf_mu = np.inf
default_sigma = 10e2
i_step_range = np.arange(0, 50)
debug = False

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

def create_q_policy_fn(q_policy_type):
    if q_policy_type == "greedy":
        q_policy_fn = eps_greedy_policy
    elif q_policy_type == "maxucb":
        q_policy_fn = maxucb_policy
    elif q_policy_type == "maxlcb":
        q_policy_fn = maxlcb_policy
    else:
        stop
        
    return q_policy_fn


def greedy_policy(action_muhats):
    action_max_muhat = np.max(action_muhats)
    action_max_idxes = np.where(action_muhats == action_max_muhat)[0]
    # action = np.argmax(action_means)
    action = np.random.choice(action_max_idxes)
    return action


def eps_greedy_policy(
        action_muhats, action_sigmahats, action_nvisits, action_ess,
        eps, i_step):
    
    num_actions = len(action_muhats)
    action_muhats = copy.deepcopy(action_muhats)
    action_muhats[action_nvisits == 0] = inf_mu
    greedy_action = greedy_policy(action_muhats)
    
    action_probs = eps*np.ones(num_actions)/num_actions
    action_probs[greedy_action] += 1 - eps
    eps_greedy_action = np.random.choice(num_actions, 1, p=action_probs)[0]
    
    return eps_greedy_action


def maxucb_policy(
        action_muhats, action_sigmahats, action_nvisits, action_ess,
        eps, i_step, ucb_delta=0.05):

    if debug and i_step in i_step_range:
        print(f"maxucb_policy")
    num_actions = len(action_muhats)
    total_ess = np.sum(action_ess)
    
    action_maxucb_idx = None
    action_maxucb_ucb = -np.inf
    for i in range(num_actions):
        if debug and i_step in i_step_range:
            print(f"i = {i}")
        if action_nvisits[i] == 0:
            action_maxucb_idx = i
            action_maxucb_muhat = np.inf
            if debug and i_step in i_step_range:
                print(f"break, action_maxucb_idx = {action_maxucb_idx}")
            break

        sigma_tmp = action_sigmahats[i]
        if sigma_tmp == 0:
            sigma_tmp = default_sigma
        
        num = 2*sigma_tmp**2/(action_ess[i])
        log = np.log(2*num_actions*total_ess**2/ucb_delta)
        action_i_bonus = np.sqrt(num*log)
        action_i_ucb = action_muhats[i] + action_i_bonus
        if action_i_ucb > action_maxucb_ucb:
            action_maxucb_idx = i
            action_maxucb_ucb = action_i_ucb

        if debug and i_step in i_step_range:
            print(f"action_i_muhat = {action_muhats[i]:0.2f}")
            print(f"action_i_bonus = {action_i_bonus:0.2f}")
            print(f"sigma_tmp = {sigma_tmp:.2f}")
            print(f"action_i_ucb = {action_i_ucb:0.2f}")
            print(f"action_maxucb_idx = {action_maxucb_idx}")

    if debug and i_step in i_step_range:
        print(f"action_nvisits = {action_nvisits}")
        print(f"action_ess = {action_ess}")
        print(f"action_muhats = {action_muhats}")
        print(f"action_sigmahats = {action_sigmahats}")
        print(f"action_maxucb_idx = {action_maxucb_idx}")
        print(f"action_maxucb_ucb = {action_maxucb_ucb:0.2f}")

    return action_maxucb_idx

def maxlcb_policy(
        action_muhats, action_sigmahats, action_nvisits, action_ess,
        eps, i_step, ucb_delta=0.05):

    if debug and i_step in i_step_range:
        print(f"maxlcb_policy")
    num_actions = len(action_muhats)
    total_ess = np.sum(action_ess)
    
    action_maxlcb_idx = None
    action_maxlcb_ucb = -np.inf
    for i in range(num_actions):
        if debug and i_step in i_step_range:
            print(f"i = {i}")
        if action_nvisits[i] == 0:
            action_maxlcb_idx = i
            action_maxlcb_muhat = np.inf
            if debug and i_step in i_step_range:
                print(f"break, action_maxlcb_idx = {action_maxlcb_idx}")
            break

        sigma_tmp = action_sigmahats[i]
        if sigma_tmp == 0:
            sigma_tmp = default_sigma
        
        num = 2*sigma_tmp**2/(action_ess[i])
        log = np.log(2*num_actions*total_ess**2/ucb_delta)
        action_i_bonus = np.sqrt(num*log)
        action_i_lcb = action_muhats[i] - action_i_bonus
        if action_i_lcb > action_maxlcb_ucb:
            action_maxlcb_idx = i
            action_maxlcb_ucb = action_i_lcb

        if debug and i_step in i_step_range:
            print(f"action_i_muhat = {action_muhats[i]:0.2f}")
            print(f"action_i_bonus = {action_i_bonus:0.2f}")
            print(f"sigma_tmp = {sigma_tmp:.2f}")
            print(f"action_i_lcb = {action_i_lcb:0.2f}")
            print(f"action_maxlcb_idx = {action_maxlcb_idx}")

    if debug and i_step in i_step_range:
        print(f"action_nvisits = {action_nvisits}")
        print(f"action_ess = {action_ess}")
        print(f"action_muhats = {action_muhats}")
        print(f"action_sigmahats = {action_sigmahats}")
        print(f"action_maxlcb_idx = {action_maxlcb_idx}")
        print(f"action_maxlcb_ucb = {action_maxlcb_ucb:0.2f}")

    return action_maxlcb_idx


def q_learning(
        env, num_actions, num_steps_train,
        gamma, lr_sched_fn, q_policy_fn, eps_sched_fn, tdqm_disable, args=None):
        
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
        if debug and i_step in i_step_range:
            print(f"\n-> i_step = {i_step}")
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = q_policy_fn(
            Q_table[cur_state], Q_sigmahats[cur_state],
            Q_nvisits[cur_state], Q_ess[cur_state], eps, i_step)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        if debug and i_step in i_step_range:
            print(f"cur_state = {cur_state}")
            print(f"action = {action}")
            print(f"reward = {reward:0.2f}")
            print(f"new_state = {new_state}")
            if terminated:
                print("terminated")

        if not terminated:
            action_muhats = copy.deepcopy(Q_table[new_state])
            action_nvisits = Q_nvisits[new_state]
            action_muhats[action_nvisits == 0] = -inf_mu
            Q_est = np.max(action_muhats) if np.sum(action_nvisits) != 0 else 0

            # compute td_target
            td_target = reward + gamma*Q_est 
        else:
            td_target = reward

        Q_nvisits[cur_state][action] += 1
        lr = lr_sched_fn(Q_nvisits[cur_state][action])

        if debug and i_step in i_step_range:
            print(f"lr = {lr:0.2f}")
            print(f"Q_nvisits[cur_state][action] = {Q_nvisits[cur_state][action]}")
            print(f"Q_table[cur_state][action], before = {Q_table[cur_state][action]:0.2f}")
            
        Q_table[cur_state][action] = \
            (1-lr)*Q_table[cur_state][action] + lr*td_target
        Q2_table[cur_state][action] = \
            (1-lr)*Q2_table[cur_state][action] + lr*td_target**2

        if debug and i_step in i_step_range:
            print(f"Q_table[cur_state][action], after = {Q_table[cur_state][action]:0.2f}")

        if debug and i_step in i_step_range:
            print(f"Q_sigmahats[cur_state][action], before = {Q_sigmahats[cur_state][action]:0.2f}")

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
            Q_sigmahats[cur_state][action] = np.sqrt(diff)

        if debug and i_step in i_step_range:
            print(f"Q_sigmahats[cur_state][action], after = {Q_sigmahats[cur_state][action]:0.2f}")
            
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
        gamma, lr_sched_fn, q_policy_fn, eps_sched_fn, tdqm_disable, args=None):
        
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

        # choose the action a_t using epsilon greedy policy
        if debug and i_step in i_step_range:
            print(f"\n-> i_step = {i_step}")
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = q_policy_fn(
            Q_table[cur_state], Q_sigmahats[cur_state],
            Q_nvisits[cur_state], Q_ess[cur_state], eps, i_step)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        if debug and i_step in i_step_range:
            print(f"cur_state = {cur_state}")
            print(f"action = {action}")
            print(f"reward = {reward:0.2f}")
            print(f"new_state = {new_state}")
            if terminated:
                print("terminated")

        if not terminated:
            # compute Q_est with weighted estimator
            action_nvisits = Q_nvisits[new_state]
            action_ess = Q_ess[new_state]
            action_muhats = copy.deepcopy(Q_table[new_state])
            action_muhats[action_nvisits == 0] = -inf_mu

            action_sigmahats = copy.deepcopy(Q_sigmahats[new_state])
            # action_sigmahats[action_sigmahats < 1e-5] = 1e-5
            # action_sigmahats[action_nvisits == 0] = default_sigma

            Q_est = weightedms_estimator(
                action_muhats, action_sigmahats, action_nvisits, action_ess,
                num_actions, num_data, i_step)

            # compute td_target
            td_target = reward + gamma*Q_est
        else:
            td_target = reward

        Q_nvisits[cur_state][action] += 1
        lr = lr_sched_fn(Q_nvisits[cur_state][action])

        if debug and i_step in i_step_range:
            print(f"lr = {lr:0.2f}")
            print(f"Q_nvisits[cur_state][action] = {Q_nvisits[cur_state][action]}")
            print(f"Q_table[cur_state][action], before = {Q_table[cur_state][action]:0.2f}")
            
        Q_table[cur_state][action] = \
            (1-lr)*Q_table[cur_state][action] + lr*td_target
        Q2_table[cur_state][action] = \
            (1-lr)*Q2_table[cur_state][action] + lr*td_target**2

        if debug and i_step in i_step_range:
            print(f"Q_table[cur_state][action], after = {Q_table[cur_state][action]:0.2f}")

        if debug and i_step in i_step_range:
            print(f"Q_sigmahats[cur_state][action], before = {Q_sigmahats[cur_state][action]:0.2f}")

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
            Q_sigmahats[cur_state][action] = np.sqrt(diff)

        if debug and i_step in i_step_range:
            print(f"Q_sigmahats[cur_state][action], after = {Q_sigmahats[cur_state][action]:0.2f}")
        

        if terminated:
            cur_state, info = env.reset()
            cur_state = f"{cur_state}"
        else:
            cur_state = new_state

        action_nvisits = Q_nvisits[start_state]
        action_ess = Q_ess[start_state]
        action_muhats = copy.deepcopy(Q_table[start_state])
        action_muhats[action_nvisits == 0] = -inf_mu
        
        action_sigmahats = copy.deepcopy(Q_sigmahats[start_state])
        # action_sigmahats[action_sigmahats < 1e-5] = 1e-5
        # action_sigmahats[action_nvisits == 0] = default_sigma

        Q_start_est = weightedms_estimator(
            action_muhats, action_sigmahats, action_nvisits, action_ess,
            num_actions, num_data, i_step)
        
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, Q_start_est))

    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.size)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats

def weightedms_estimator(
        action_muhats, action_sigmahats, action_nvisits, action_ess,
        num_actions, num_data, i_step):

    if debug and i_step in i_step_range:
        print(f"weightedms_estimator")
        print(f"action_nvisits = {action_nvisits}")
        print(f"action_ess = {action_ess}")
        print(f"action_muhats = {action_muhats}")
        print(f"action_sigmahats = {action_sigmahats}")
            
    if np.sum(action_nvisits) == 0:
        return 0
    
    # sample 
    # action_muhats_mat = matlib.repmat(action_muhats, num_data, 1)
    # action_sigmahats_mat = matlib.repmat(action_sigmahats, num_data, 1)
    # eps_mat = np.random.randn(num_data, num_actions)
    # samples = action_muhats_mat + action_sigmahats_mat*eps_mat
    action_sigmahats_adjusted = copy.deepcopy(action_sigmahats)
    for i in range(num_actions):
        if action_sigmahats_adjusted[i] == 0:
            action_sigmahats_adjusted[i] = default_sigma
        else:
            action_sigmahats_adjusted[i] /= action_ess[i]
    if debug and i_step in i_step_range:
        print(f"action_sigmahats = {action_sigmahats}")
        print(f"action_sigmahats_adjusted = {action_sigmahats_adjusted}")
        
    samples = np.random.normal(
        action_muhats, action_sigmahats_adjusted, size=(num_data, num_actions))
    
    # samples[action_nvisits == 0,:] = -inf_mu
    samples_max_idxes = np.argmax(samples, 1)

    # compute probs 
    Bset_probs = np.zeros(num_actions)
    Bset_idxes, Bset_cnts = np.unique(samples_max_idxes, return_counts=True)
    Bset_probs[Bset_idxes[Bset_cnts > 0]] = Bset_cnts[Bset_cnts > 0]
    Bset_probs = Bset_probs/num_data

    # print(action_sigmahats)
    # print(action_muhats)
    # print(probs)
    
    # compute Q_est 
    Q_est = np.dot(
        Bset_probs[action_nvisits != 0], action_muhats[action_nvisits != 0])
    
    # Q_est = Q_est if np.sum(action_nvisits) != 0 else 0.0
    # print(f"Q_est_ = {Q_est:0.2f}")
    # stop
    if debug and i_step in i_step_range:
        print(f"Bset_idxes = {Bset_idxes}")
        print(f"Bset_probs = {Bset_probs}")
        # print(tmp)
        print(f"Q_est = {Q_est:.2f}")
    
    return Q_est


def haver3_q_learning(
        env, num_actions, num_steps_train,
        gamma, lr_sched_fn, q_policy_fn, eps_sched_fn, tdqm_disable, args=None):

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
        if debug and i_step in i_step_range:
            print(f"\n-> i_step = {i_step}")
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = q_policy_fn(
            Q_table[cur_state], Q_sigmahats[cur_state],
            Q_nvisits[cur_state], Q_ess[cur_state], eps, i_step)
        
        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        if debug and i_step in i_step_range:
            print(f"cur_state = {cur_state}")
            print(f"action = {action}")
            print(f"reward = {reward:0.2f}")
            print(f"new_state = {new_state}")
            if terminated:
                print("terminated")

        if not terminated:
            action_nvisits = Q_nvisits[new_state]
            action_ess = Q_ess[new_state]
            action_muhats = copy.deepcopy(Q_table[new_state])
            action_muhats[action_nvisits == 0] = -inf_mu
                
            action_sigmahats = Q_sigmahats[new_state]
            # action_sigmahats[action_sigmahats < 1e-5] = 1e-5
            
            Q_est = haver3_estimator(
                action_muhats, action_sigmahats, action_nvisits, action_ess,
                num_actions, haver_alpha, haver_delta, haver_const, lr_sched_fn,
                i_step, args)

            # compute td_target
            td_target = reward + gamma*Q_est
        else:
            td_target = reward

        Q_nvisits[cur_state][action] += 1
        lr = lr_sched_fn(Q_nvisits[cur_state][action])

        if debug and i_step in i_step_range:
            print(f"lr = {lr:0.2f}")
            print(f"Q_nvisits[cur_state][action] = {Q_nvisits[cur_state][action]}")
            print(f"Q_table[cur_state][action], before = {Q_table[cur_state][action]:0.2f}")
        
        Q_table[cur_state][action] = \
            (1-lr)*Q_table[cur_state][action] + lr*td_target
        Q2_table[cur_state][action] = \
            (1-lr)*Q2_table[cur_state][action] + lr*td_target**2

        if debug and i_step in i_step_range:
            print(f"Q_table[cur_state][action], after = {Q_table[cur_state][action]:0.2f}")

        if debug and i_step in i_step_range:
            print(f"Q_sigmahats[cur_state][action], before = {Q_sigmahats[cur_state][action]:0.2f}")

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

        if debug and i_step in i_step_range:
            print(f"Q_sigmahats[cur_state][action], after = {Q_sigmahats[cur_state][action]:0.2f}")
        
        if terminated:
            # print("terminated")
            cur_state, info = env.reset()
            cur_state = f"{cur_state}"
            # print(f"reset_state = {cur_state}")
        else:
            cur_state = new_state

        action_nvisits = Q_nvisits[start_state]
        action_ess = Q_ess[start_state]
        action_muhats = copy.deepcopy(Q_table[start_state])
        action_muhats[action_nvisits == 0] = -inf_mu
        
        action_sigmahats = Q_sigmahats[start_state]
        # action_sigmahats[action_sigmahats < 1e-5] = 1e-5
        
        Q_start_est = haver3_estimator(
            action_muhats, action_sigmahats, action_nvisits, action_ess,
            num_actions, haver_alpha, haver_delta, haver_const, lr_sched_fn,
            i_step, args)

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
        i_step, args):

    if debug and i_step in i_step_range:
        print(f"haver3_estimator")
        print(f"action_nvisits = {action_nvisits}")
        print(f"action_ess = {action_ess}")
        print(f"action_muhats = {action_muhats}")
        print(f"action_sigmahats = {action_sigmahats}")

    total_ess = np.sum(action_ess)
    total_nvisits = np.sum(action_nvisits)
    if total_nvisits == 0:
        return 0
    
    action_maxlcb_idx = None
    action_maxlcb_sigmahat = None
    action_maxlcb_muhat = -np.inf
    for i in range(num_actions):
        sigma_tmp = action_sigmahats[i]
        if sigma_tmp == 0:
            sigma_tmp = default_sigma
            
        num = 2*sigma_tmp**2/(action_ess[i]+1)
        log = np.log(2*num_actions*total_ess**2/haver_delta)
        action_i_bonus = np.sqrt(num*log)
        action_i_lcb = action_muhats[i] - action_i_bonus
        # print(i)
        # print(action_i_lcb)
        if action_i_lcb > action_maxlcb_muhat:
            action_maxlcb_idx = i
            action_maxlcb_muhat = action_i_lcb
            action_maxlcb_sigmahat = sigma_tmp
    
    # action_max_idx = np.argmax(action_muhats)
    # action_maxlcb_muhat = action_muhats[action_maxlcb_idx]
    # action_maxlcb_sigmahat = action_sigmahats[action_maxlcb_idx]
    action_maxlcb_nvisits = action_nvisits[action_maxlcb_idx]
    action_maxlcb_ess = action_ess[action_maxlcb_idx]
    
    # Q_est_sum = 0
    # Q_est_cnt = 0
    Bset_idxes = []
    Bset_muhats = np.zeros(num_actions)
    Bset_nvisits = np.zeros(num_actions)
    for i in range(num_actions):
        if action_nvisits[i] != 0:

            sigma_tmp = action_sigmahats[i]
            if sigma_tmp == 0:
                sigma_tmp = default_sigma
            
            # avg = action_sigma**2/action_maxlcb_ess \
            #     + action_sigma**2/action_ess[i]
            avg = action_maxlcb_sigmahat**2/action_maxlcb_ess \
                + sigma_tmp**2/action_ess[i]
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
                Bset_muhats[i] = action_muhats[i]
                Bset_nvisits[i] = action_ess[i]
                Bset_idxes.append(i)

    # Bset_muhats = np.array(Bset_muhats)
    # Bset_nvisits = np.array(Bset_nvisits)
    Bset_probs = Bset_nvisits/np.sum(Bset_nvisits)
    Q_est = np.dot(Bset_muhats, Bset_probs)
    
    if debug and i_step in i_step_range:
        print(f"action_maxlcb_idx = {action_maxlcb_idx}")
        print(f"action_maxlcb_muhat = {action_maxlcb_muhat:0.2f}")

        print(f"Bset_idxes = {Bset_idxes}")
        # print(f"Bset_muhats = {Bset_muhats}")
        print(f"Bset_probs = {Bset_probs}")
        print(f"Bset_nvisits = {Bset_nvisits}")
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
