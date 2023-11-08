
# TODOs
# 1. create proper eps_sched_fn, we are using fixed 1.0/np.sqrt(nvisits) for now

import numpy as np
import numpy.matlib as matlib

import copy
from tqdm import tqdm
from collections import defaultdict

    
def create_eps_sched_fn(sched_type, min_eps=None, max_eps=None, decay_rate=None):
    if sched_type == "poly":
        def eps_sched_fn(nvisits):
            print("yes", nvisits)
            stop
            return 1.0/np.sqrt(nvisits)
    elif sched_type == "exp":
        def eps_sched_fn(i_eps):
            eps = min_eps + (max_eps - min_eps)*np.exp(-decay_rate*i_eps)
            return eps
    
    return eps_sched_fn

def create_lr_sched_fn(sched_type):
    if sched_type == "linear":
        def lr_sched_fn(nvisits):
            return 1.0/nvisits
    elif sched_type == "poly":
        def lr_sched_fn(nvisits):
            return 1.0/(nvisits**0.8)
    else:
        stop
        
    return lr_sched_fn


def greedy_policy(emp_vals):
    action = np.argmax(emp_vals)
    return action


def eps_greedy_policy(emp_vals, eps): 
    num_actions = len(emp_vals)
    greedy_action = greedy_policy(emp_vals)
    
    action_probs = eps*np.ones(num_actions)/num_actions
    action_probs[greedy_action] += 1 - eps
    eps_greedy_action = np.random.choice(num_actions, 1, p=action_probs)[0]
    
    return eps_greedy_action


def evaluate(env, Q_table, num_episodes_eval, max_steps, seed_ary=None):
    
    if not seed_ary:
        seed_ary = np.arange(num_episodes_eval, dtype=np.int32) + 100
        
    episode_reward_ary = []
    for i_eps in tqdm(range(num_episodes_eval)):
        
        state, info = env.reset(seed=int(seed_ary[i_eps]))
        state = f"{state}"
            
        episode_reward = 0
        for step in range(max_steps):
            action = greedy_policy(Q_table, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            new_state = f"{new_state}"
            
            episode_reward += reward

            if terminated or truncated:
                break

            state = new_state

        episode_reward_ary.append(episode_reward)

    return episode_reward_ary    


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

        # print(f"\n-> i_step = {i_step}")
        # print(f"state = {state}")
        # choose the action a_t using epsilon greedy policy
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        eps = 1.0/np.sqrt(nvisits)
        action = eps_greedy_policy(Q_table[cur_state], eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        # print(f"action = {action}")
        # print(f"reward = {reward}")
        # print(f"new_state = {new_state}")

        if not terminated:
            Q_est = np.max(Q_table[new_state])
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
            
        # stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))
        stats.append((i_step, reward, np.max(Q_table[start_state])))

    return Q_table, stats

def avg_q_learning(env, Q_table, Q_nvisits, num_steps_train,
                   gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):
        
    # keep track of useful statistics
    stats = []
    
    for i_eps in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):
        state, info = env.reset()
        state = f"{state}"
        start_state = copy.deepcopy(state)
        
        episode_reward = 0.0
        for i_step in range(max_steps):
            # choose the action a_t using epsilon greedy policy
            nvisits = np.sum(Q_nvisits[state]) + 1
            eps = 1.0/np.sqrt(nvisits)
            action = eps_greedy_policy(Q_table[state], eps)

            new_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            new_state = f"{new_state}"

            Q_est = np.mean(Q_table[new_state])
            td_target = reward + gamma*Q_est
            td_error = td_target - Q_table[state][action]

            Q_nvisits[state][action] += 1
            lr = lr_sched_fn(Q_nvisits[state][action]) 
            Q_table[state][action] += lr*td_error
            

            if terminated or truncated:
                break

            state = new_state

        stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))

    return Q_table, stats 


def double_q_learning(env, num_actions, num_steps_train,
               gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):
        
    # keep track of useful statistics
    stats = []
    Q_table = defaultdict(lambda: np.zeros(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))
    DQ_table = defaultdict(lambda: np.zeros((num_actions, 2)))
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
        eps = 1.0/np.sqrt(nvisits)
        action = eps_greedy_policy(Q_table[cur_state], eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        # print(f"action = {action}")
        # print(f"reward = {reward}")
        # print(f"new_state = {new_state}")
        
        Q_idx = np.random.randint(0, 2)
        next_action_best = np.argmax(DQ_table[new_state][:, Q_idx])
        Q_est = DQ_table[new_state][next_action_best, 1-Q_idx]
        
        if not terminated:
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
        stats.append((i_step, reward, np.max(Q_table[start_state])))

    return Q_table, stats


def weightedms_q_learning(
        env, num_actions, num_steps_train,
        gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):
        
    # keep track of useful statistics
    stats = []
    num_samples = args["num_samples"]
    num_actions = env.action_space.n
    Q_table = defaultdict(lambda: np.zeros(num_actions))
    Q2_table = defaultdict(lambda: np.zeros(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))
    # Q_nupdates = defaultdict(lambda: np.zeros(num_actions))
    Q_sigma = defaultdict(lambda: np.ones(num_actions)*1e10)
    weights_var = defaultdict(lambda: np.zeros(num_actions))
    probs = np.ones(num_actions)/num_actions

    cur_state, info = env.reset()
    cur_state = f"{cur_state}"
    start_state = copy.deepcopy(cur_state)
    
    for i_step in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):
        
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        eps = 1.0/np.sqrt(nvisits)
        action = eps_greedy_policy(Q_table[cur_state], eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"

        # apply weighted estim for Q_table
        emp_vals = Q_table[new_state]
        cur_sigma = Q_sigma[new_state]
        cur_sigma[cur_sigma < 1e-4] = 1e-4

        emp_vals_mat = matlib.repmat(emp_vals, num_samples, 1)
        cur_sigma_mat = matlib.repmat(cur_sigma, num_samples, 1)
        eps_mat = np.random.randn(num_samples, num_actions)

        # sample and compute es
        samples = emp_vals_mat + cur_sigma_mat*eps_mat
        samples_max_idxes = np.argmax(samples, 1)

        probs = np.zeros(num_actions)
        idxes, cnts = np.unique(samples_max_idxes, return_counts=True)
        probs[idxes[cnts > 0]] = cnts[cnts > 0]
        probs = probs/num_samples
        Q_est = np.dot(probs, emp_vals)
        # print(Q_est)
        # stop

        if not terminated:
            td_target = reward + gamma*Q_est
        else:
            td_target = reward

        Q_nvisits[cur_state][action] += 1
        lr = lr_sched_fn(Q_nvisits[cur_state][action])

        Q_table[cur_state][action] = (1-lr)*Q_table[cur_state][action] + lr*td_target
        Q2_table[cur_state][action] = (1-lr)*Q2_table[cur_state][action] + lr*td_target**2

        if Q_nvisits[cur_state][action] > 1:
            weights_var[cur_state][action] = \
                (1-lr)**2*weights_var[cur_state][action] + lr**2
            n = 1.0/weights_var[cur_state][action]
            diff = Q2_table[cur_state][action] - Q_table[cur_state][action]**2
            if diff < 0:
                diff = 0
            Q_sigma[cur_state][action] = np.sqrt(diff/n)

        if terminated:
            cur_state, info = env.reset()
            cur_state = f"{cur_state}"
        else:
            cur_state = new_state

        # stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))
        stats.append((i_step, reward, np.max(Q_table[start_state])))
        
    return Q_table, stats 


def haver_q_learning(env, num_actions, num_steps_train,
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

        # print(f"\n-> i_step = {i_step}")
        # print(f"state = {state}")
        # choose the action a_t using epsilon greedy policy
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        eps = 1.0/np.sqrt(nvisits)
        action = eps_greedy_policy(Q_table[cur_state], eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        # print(f"action = {action}")
        # print(f"reward = {reward}")
        # print(f"new_state = {new_state}")

        if not terminated:
            Q_est = np.max(Q_table[new_state])
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
            
        # stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))
        stats.append((i_step, reward, np.max(Q_table[start_state])))

    return Q_table, stats


# def haver_q_learning(
#         env, Q_table, Q_nvisits, num_steps_train,
#         gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

#     # keep track of useful statistics
#     stats = []
    
#     for i_eps in tqdm(
#             range(num_steps_train), desc="train q_learning", disable=tdqm_disable):
#         state, info = env.reset()
#         state = f"{state}"
#         start_state = copy.deepcopy(state)
        
#         episode_reward = 0.0
#         for i_step in range(max_steps):
#             # print(f"i_step: {i_step}")
#             # choose the action a_t using epsilon greedy policy
#             nvisits = np.sum(Q_nvisits[state]) + 1
#             eps = 1.0/np.sqrt(nvisits)
#             action = eps_greedy_policy(Q_table[state], eps)

#             new_state, reward, terminated, truncated, info = env.step(action)
#             episode_reward += reward
#             new_state = f"{new_state}"

#             emp_vals = copy.deepcopy(Q_table[new_state])
#             nvisits = copy.deepcopy(Q_nvisits[new_state])
#             Q_est = haver(emp_vals, nvisits, args["haver_const"], lr_sched_fn)
#             td_target = reward + gamma*Q_est
#             td_error = td_target - Q_table[state][action]
#             # print(f"Q_est = {Q_est}")
#             # print(f"td_target = {td_target}")
#             # print(f"td_error = {td_error}")
#             # print(f"Q_table[state][action] = {Q_table[state][action]}")

#             Q_nvisits[state][action] += 1
#             lr = lr_sched_fn(Q_nvisits[state][action])
#             Q_table[state][action] += lr*td_error
            

#             if terminated or truncated:
#                 break

#             state = new_state

#         # stats.append((
#         #     i_eps, episode_reward, i_step+1, np.max(Q_table[state_deepcopy])))
#         stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))

#     return Q_table, stats 


# def haver(emp_vals, nvisits, haver_const, lr_sched_fn):
#     K = len(emp_vals)
#     B_vals = []
#     B_nvisits = []
#     emp_vals[emp_vals == 0] = -np.inf
#     emp_max_idx = np.argmax(emp_vals)
#     emp_max_val = emp_vals[emp_max_idx]
#     nvisits_max = nvisits[emp_max_idx]

#     est_sum = 0
#     est_cnt = 0
#     for i in range(K):
#         if nvisits[i] != 0:
#             if emp_max_val - emp_vals[i] <= haver_const*np.sqrt(
#                     (lr_sched_fn(nvisits_max) + lr_sched_fn(nvisits[i]))*np.log(K)):
#                 # B_vals.append(emp_vals[i])
#                 # B_nvisits.append(nvisits[i])
#                 est_sum += emp_vals[i]
#                 est_cnt += 1
            
#     # est = np.mean(B_vals)
#     est = est_sum/est_cnt if est_cnt != 0 else 0.0
#     # print(f"est = {est}")
#     return est


def haver2_q_learning(
        env, Q_table, Q_nvisits, num_steps_train,
        gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

    # keep track of useful statistics
    stats = []
    
    for i_eps in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):
        state, info = env.reset()
        state = f"{state}"
        start_state = copy.deepcopy(state)
        
        episode_reward = 0.0
        for i_step in range(max_steps):
            # print(f"i_step: {i_step}")
            # choose the action a_t using epsilon greedy policy
            nvisits = np.sum(Q_nvisits[state]) + 1
            eps = 1.0/np.sqrt(nvisits)
            action = eps_greedy_policy(Q_table[state], eps)

            new_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            new_state = f"{new_state}"

            emp_vals = copy.deepcopy(Q_table[new_state])
            nvisits = copy.deepcopy(Q_nvisits[new_state])
            Q_est = haver(emp_vals, nvisits, args["haver_const"], lr_sched_fn)
            td_target = reward + gamma*Q_est
            td_error = td_target - Q_table[state][action]
            # print(f"Q_est = {Q_est}")
            # print(f"td_target = {td_target}")
            # print(f"td_error = {td_error}")
            # print(f"Q_table[state][action] = {Q_table[state][action]}")

            Q_nvisits[state][action] += 1
            lr = lr_sched_fn(Q_nvisits[state][action])
            Q_table[state][action] += lr*td_error
            

            if terminated or truncated:
                break

            state = new_state

        # stats.append((
        #     i_eps, episode_reward, i_step+1, np.max(Q_table[state_deepcopy])))
        stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))

    return Q_table, stats 


def haver2(emp_vals, nvisits, haver_const, lr_sched_fn):
    K = len(emp_vals)
    B_vals = []
    B_nvisits = []
    emp_vals[emp_vals == 0] = -np.inf
    emp_max_idx = np.argmax(emp_vals)
    emp_max_val = emp_vals[emp_max_idx]
    nvisits_max = nvisits[emp_max_idx]

    est_sum = 0
    est_cnt = 0
    for i in range(K):
        if nvisits[i] != 0:
            if emp_max_val - emp_vals[i] <= haver_const*np.sqrt(
                    (lr_sched_fn(nvisits_max) + lr_sched_fn(nvisits[i]))*np.log(K)):
                # B_vals.append(emp_vals[i])
                # B_nvisits.append(nvisits[i])
                est_sum += nvisits[i]*emp_vals[i]
                est_cnt += nvisits[i]
            
    # est = np.mean(B_vals)
    est = est_sum/est_cnt if est_cnt != 0 else 0.0
    # print(f"est = {est}")
    return est
        
        
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
