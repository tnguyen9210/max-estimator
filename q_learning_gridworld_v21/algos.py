
# TODOs
# 1. create proper eps_sched_fn, we are using fixed 1.0/np.sqrt(nvisits) for now

import copy
import numpy as np 
from tqdm import tqdm


    
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


def greedy_policy(Q_table, state):
    action = np.argmax(Q_table[state][:])
    return action


def eps_greedy_policy(Q_table, state, eps):
    # 
    num_actions = len(Q_table[state])
    greedy_action = greedy_policy(Q_table, state)
    
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


def q_learning(env, Q_table, Q_nvisits, num_steps_train, max_steps,
               gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):
        
    # keep track of useful statistics
    stats = []
    state, info = env.reset()
    state = f"{state}"
    start_state = copy.deepcopy(state)
    Qest = []
    
    episode_reward = 0.0
    for i_step in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):
            
        # choose the action a_t using epsilon greedy policy
        nvisits = np.sum(Q_nvisits[state]) + 1
        eps = 1.0/np.sqrt(nvisits)
        action = eps_greedy_policy(Q_table, state, eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        new_state = f"{new_state}"

        if not terminated:
            Q_est = np.max(Q_table[new_state])
            td_target = reward + gamma*Q_est
            td_error = td_target - Q_table[state][action]
        else:
            td_error = reward - Q_table[state][action]

        Q_nvisits[state][action] += 1
        lr = lr_sched_fn(Q_nvisits[state][action]) 
        Q_table[state][action] += lr*td_error
        
        state = new_state

        # stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))
        stats.append((i_step, reward, np.max(Q_table[start_state])))

    return Q_table, stats

def avg_q_learning(env, Q_table, Q_nvisits, num_episodes_train, max_steps,
                   gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):
        
    # keep track of useful statistics
    stats = []
    
    for i_eps in tqdm(
            range(num_episodes_train), desc="train q_learning", disable=tdqm_disable):
        state, info = env.reset()
        state = f"{state}"
        start_state = copy.deepcopy(state)
        
        episode_reward = 0.0
        for i_step in range(max_steps):
            # choose the action a_t using epsilon greedy policy
            nvisits = np.sum(Q_nvisits[state]) + 1
            eps = 1.0/np.sqrt(nvisits)
            action = eps_greedy_policy(Q_table, state, eps)

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

        # stats.append((
        #     i_eps, episode_reward, i_step+1, np.max(Q_table[state_deepcopy])))
        stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))

    return Q_table, stats 


def double_q_learning(env, Q_table, Q_nvisits, num_episodes_train, max_steps,
                      gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

    Q_table1 = copy.deepcopy(Q_table)
    Q_table2 = copy.deepcopy(Q_table)
    Q_nvisits1 = copy.deepcopy(Q_nvisits)
    Q_nvisits2 = copy.deepcopy(Q_nvisits)

    stats = []
    for i_eps in tqdm(
            range(num_episodes_train), "train double_q_learning", disable=tdqm_disable):
        state, info = env.reset()
        state = f"{state}"
        start_state = copy.deepcopy(state)

        episode_reward = 0.0
        for i_step in range(max_steps):
            # choose the action a_t using epsilon greedy policy
            nvisits = np.sum(Q_nvisits[state]) + 1
            eps = 1.0/np.sqrt(nvisits)
            action = eps_greedy_policy(Q_table, state, eps)

            new_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            new_state = f"{new_state}"

            if np.random.rand() < 0.5:
                next_action_best = np.argmax(Q_table1[new_state]) 
                td_target = reward + gamma*Q_table2[new_state][next_action_best]
                td_error = td_target - Q_table1[state][action]

                Q_nvisits1[state][action] += 1
                lr = lr_sched_fn(Q_nvisits1[state][action]) 
                Q_table1[state][action] += lr*td_error
                
                
            else:
                next_action_best = np.argmax(Q_table2[new_state]) 
                td_target = reward + gamma*Q_table1[new_state][next_action_best]
                td_error = td_target - Q_table2[state][action]

                Q_nvisits2[state][action] += 1
                lr = lr_sched_fn(Q_nvisits2[state][action]) 
                Q_table2[state][action] += lr*td_error
            
            # next_action_best = np.argmax(Q_table2[new_state]) 
            # td_target = reward + gamma*Q_table2[new_state][next_action_best]
            # td_error = td_target - Q_table2[state][action]

            # Q_nvisits2[state][action] += 1
            # lr = lr_sched_fn(Q_nvisits2[state][action]) 
            # Q_table2[state][action] += lr*td_error
        
            Q_table[state][action] = (Q_table1[state][action] + Q_table2[state][action])/2
            Q_nvisits[state][action] = Q_nvisits1[state][action] + Q_nvisits2[state][action]
            
            if terminated or truncated:
                break

            state = new_state

        stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))

    return Q_table, stats


def weightedms_q_learning(env, Q_table, Q_nvisits, num_episodes_train, max_steps,
                   gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):
        
    # keep track of useful statistics
    stats = []
    
    for i_eps in tqdm(
            range(num_episodes_train), desc="train q_learning", disable=tdqm_disable):
        state, info = env.reset()
        state = f"{state}"
        start_state = copy.deepcopy(state)
        
        episode_reward = 0.0
        for i_step in range(max_steps):
            # choose the action a_t using epsilon greedy policy
            nvisits = np.sum(Q_nvisits[state]) + 1
            eps = 1.0/np.sqrt(nvisits)
            action = eps_greedy_policy(Q_table, state, eps)

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

        # stats.append((
        #     i_eps, episode_reward, i_step+1, np.max(Q_table[state_deepcopy])))
        stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))

    return Q_table, stats 


def weightedms(emp_vals, nvisits, sigma, booster=1):
    posterior_var = 1 / nvisits

    emp_vals[emp_vals == 0] = -np.inf
    emp_max_idx = np.argmax(emp_vals)
    emp_max_val = emp_vals[emp_max_idx]
    emp_delta = emp_max_val - emp_vals
    probs = np.exp(- 1/(2*sigma**2) * nvisits * emp_delta**2)
    probs[emp_max_idx] = booster
    probs = probs/np.sum(probs)
    est = np.dot(probs, emp_vals)
    


def haver_q_learning(
        env, Q_table, Q_nvisits, num_episodes_train, max_steps,
        gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

    # keep track of useful statistics
    stats = []
    
    for i_eps in tqdm(
            range(num_episodes_train), desc="train q_learning", disable=tdqm_disable):
        state, info = env.reset()
        state = f"{state}"
        start_state = copy.deepcopy(state)
        
        episode_reward = 0.0
        for i_step in range(max_steps):
            # print(f"i_step: {i_step}")
            # choose the action a_t using epsilon greedy policy
            nvisits = np.sum(Q_nvisits[state]) + 1
            eps = 1.0/np.sqrt(nvisits)
            action = eps_greedy_policy(Q_table, state, eps)

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


def haver(emp_vals, nvisits, haver_const, lr_sched_fn):
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
                est_sum += emp_vals[i]
                est_cnt += 1
            
    # est = np.mean(B_vals)
    est = est_sum/est_cnt if est_cnt != 0 else 0.0
    # print(f"est = {est}")
    return est


def haver2_q_learning(
        env, Q_table, Q_nvisits, num_episodes_train, max_steps,
        gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

    # keep track of useful statistics
    stats = []
    
    for i_eps in tqdm(
            range(num_episodes_train), desc="train q_learning", disable=tdqm_disable):
        state, info = env.reset()
        state = f"{state}"
        start_state = copy.deepcopy(state)
        
        episode_reward = 0.0
        for i_step in range(max_steps):
            # print(f"i_step: {i_step}")
            # choose the action a_t using epsilon greedy policy
            nvisits = np.sum(Q_nvisits[state]) + 1
            eps = 1.0/np.sqrt(nvisits)
            action = eps_greedy_policy(Q_table, state, eps)

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
        q_algo = haver2_q_learning

    return q_algo
