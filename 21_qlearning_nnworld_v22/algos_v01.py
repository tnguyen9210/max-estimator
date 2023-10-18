
# TODOs
import re
import numpy as np
import numpy.matlib as matlib

import copy
from tqdm import tqdm
from collections import defaultdict


def convert_tables(Q_table, Q_nvisits, num_actions, num_depths):
    Q_table_ary = np.zeros((num_depths, num_actions, num_actions))
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


def greedy_policy(action_means):
    action = np.argmax(action_means)
    return action


def eps_greedy_policy(action_means, eps): 
    num_actions = len(action_means)
    action_means = copy.deepcopy(action_means)
    action_means[action_means == 0] = -np.inf
    greedy_action = greedy_policy(action_means)
    
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
        # print(f"state = {cur_state}")
        # choose the action a_t using epsilon greedy policy
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
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
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, np.max(Q_table[start_state])))
    
    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.num_depths)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats

    
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
            # eps = 1.0/np.sqrt(nvisits)
            eps = eps_sched_fn(nvisits)
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

        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, np.max(Q_table[start_state])))

    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.num_depths)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats


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
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
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
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, np.max(Q_table[start_state])))

    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.num_depths)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats


def weightedms_q_learning(
        env, num_actions, num_steps_train,
        gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):
        
    # keep track of useful statistics
    stats = []
    num_data = args["weightedms_num_data"]
    num_actions = env.action_space.n
    Q_table = defaultdict(lambda: np.zeros(num_actions))
    Q2_table = defaultdict(lambda: np.zeros(num_actions))
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
        action = eps_greedy_policy(Q_table[cur_state], eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"

        if not terminated:
            # compute Q_est with weighted estimator
            # get 
            cur_muhats = Q_table[new_state]
            cur_sigmahats = Q_sigmahats[new_state]
            cur_sigmahats[cur_sigmahats < 1e-4] = 1e-4

            # sample 
            cur_muhats_mat = matlib.repmat(cur_muhats, num_data, 1)
            cur_sigmahats_mat = matlib.repmat(cur_sigmahats, num_data, 1)
            eps_mat = np.random.randn(num_data, num_actions)

            samples = cur_muhats_mat + cur_sigmahats_mat*eps_mat
            samples_max_idxes = np.argmax(samples, 1)

            # compute probs 
            probs = np.zeros(num_actions)
            idxes, cnts = np.unique(samples_max_idxes, return_counts=True)
            probs[idxes[cnts > 0]] = cnts[cnts > 0]
            probs = probs/num_data

            # compute Q_est 
            Q_est = np.dot(probs, cur_muhats)
            # print(Q_est)
            # stop

            # compute td_target
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
            Q_sigmahats[cur_state][action] = np.sqrt(diff/n)

        if terminated:
            cur_state, info = env.reset()
            cur_state = f"{cur_state}"
        else:
            cur_state = new_state

        # stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, np.max(Q_table[start_state])))

    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.num_depths)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats


def haver_q_learning(env, num_actions, num_steps_train,
               gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

    # get params
    action_sigma = args["action_sigma"]
    haver_delta = args["haver_delta"]
    haver_const = args["haver_const"]
    
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
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = eps_greedy_policy(Q_table[cur_state], eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        # print(f"action = {action}")
        # print(f"reward = {reward}")
        # print(f"new_state = {new_state}")

        if not terminated:
            cur_means = copy.deepcopy(Q_table[new_state])
            nvisits = copy.deepcopy(Q_nvisits[new_state])
            Q_est = haver(
                cur_means, nvisits, num_actions,
                action_sigma, haver_delta, haver_const, lr_sched_fn)
            
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
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, np.max(Q_table[start_state])))

    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.num_depths)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats



def haver(action_means, nvisits, num_actions,
          action_sigma, haver_delta, haver_const, lr_sched_fn):
    
    action_means[action_means == 0] = -np.inf
    amax_idx = np.argmax(action_means)
    amax_val = action_means[amax_idx]
    amax_nvisits = nvisits[amax_idx]

    est_sum = 0
    est_cnt = 0
    for i in range(num_actions):
        if nvisits[i] != 0:
            avg = action_sigma**2*(
                lr_sched_fn(amax_nvisits) + lr_sched_fn(nvisits[i]))
            thres = haver_const*np.sqrt(avg*np.log(num_actions**2/haver_delta))
            if amax_val - action_means[i] <= thres:
                est_sum += action_means[i]
                est_cnt += 1
            
    est = est_sum/est_cnt if est_cnt != 0 else 0.0
    # print(f"est = {est}")
    return est


def haver2_q_learning(env, num_actions, num_steps_train,
               gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

    # get params
    # action_sigma = args["action_sigma"]
    haver_alpha = args["haver_alpha"]
    haver_delta = args["haver_delta"]
    haver_const = args["haver_const"]
    
    # keep track of useful statistics
    stats = []
    Q_table = defaultdict(lambda: np.zeros(num_actions))
    Q2_table = defaultdict(lambda: np.zeros(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))

    Q_sigmahats = defaultdict(lambda: np.ones(num_actions)*1e10)
    weights_var = defaultdict(lambda: np.zeros(num_actions))
    
    cur_state, info = env.reset()
    cur_state = f"{cur_state}"
    start_state = copy.deepcopy(cur_state)
    
    for i_step in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):

        # print(f"\n-> i_step = {i_step}")
        # print(f"state = {state}")
        # choose the action a_t using epsilon greedy policy
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = eps_greedy_policy(Q_table[cur_state], eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        # print(f"action = {action}")
        # print(f"reward = {reward}")
        # print(f"new_state = {new_state}")

        if not terminated:
            cur_muhats = copy.deepcopy(Q_table[new_state])
            cur_sigmahats = Q_sigmahats[new_state]
            cur_sigmahats[cur_sigmahats < 1e-4] = 1e-4
            cur_nvisits  = Q_nvisits[new_state]
            Q_est = haver2(
                cur_muhats, cur_sigmahats, cur_nvisits, num_actions,
                haver_alpha, haver_delta, haver_const, lr_sched_fn)
            
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

        if Q_nvisits[cur_state][action] >  1:
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
            
        # stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, np.max(Q_table[start_state])))

    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.num_depths)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats


def haver2(action_muhats, action_sigmahats, nvisits, num_actions,
           haver_alpha, haver_delta, haver_const, lr_sched_fn):
    action_muhats = copy.deepcopy(action_muhats)
    action_muhats[action_muhats == 0] = -np.inf
    action_max_idx = np.argmax(action_muhats)
    action_max_muhat = action_muhats[action_max_idx]
    action_max_sigmahat = action_sigmahats[action_max_idx]
    action_max_nvisits = nvisits[action_max_idx]

    mu_est_sum = 0
    mu_est_cnt = 0
    for i in range(num_actions):
        if nvisits[i] != 0:
            avg = action_max_sigmahat**2/action_max_nvisits \
                + action_sigmahats[i]**2/nvisits[i]
            thres = haver_const*np.sqrt(avg*np.log(num_actions**haver_alpha/haver_delta))
            if action_max_muhat - action_muhats[i] <= thres:
                mu_est_sum += action_muhats[i]
                mu_est_cnt += 1
            
    mu_est = mu_est_sum/mu_est_cnt if mu_est_cnt != 0 else 0.0
    # print(f"est = {est}")
    return mu_est


// Implied Volatility Suite (TG Fork)
//
// Description: Displays the Implied Volatility, which is usually calculated from options, but here is calculated indirectly from spot price directly, either using a model or model-free using the VIXfix.
// The model-free VIXfix based approach can detect times of high volatility, which usually coincides with panic and hence lowest prices. Inversely, the model-based approach can detect times of highest greed.
//
// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © SegaRKO
// Forked and updated by Tartigradia to fix some issues in the calculations, convert to pinescript v5 and reverse engineered to reproduce the "Implied Volatility Rank & Model Free IVR" indicator by the same author (but closed source) and allow to plot both model-based and model-free implied volatilities simultaneously.
// If you like this indicator, please show the original author SegaRKO some love: https://www.tradingview.com/script/E8zeTYik-Implied-Volatility-Suite/

//@version=5
indicator('Implied Volatility Suite (TG Fork)', shorttitle='IV-TG', precision=3, timeframe="", timeframe_gaps=false)


//Implied Volatility Type
VolCalc = input.string(title='Calculation of Implied Volatility', defval='Both', options=['Model Implied Volatility', 'VixFix', 'Both'])
VolatilityChoice = input.string(title='Choose Volatility Data', defval='IV Rank', options=['Implied Volatility', 'IV Rank', 'IV Percentile', 'Volatility Skew Index'])
//Model Expiry and Length Parameters
spot = input(close, title='Source')
Minutes = input.int(0, title='Minutes until expiry (Model-Only)', minval=0)
Hours = input.int(365, title='Hours until expiry (Model-Only)', minval=0)
Days = input.int(0, title='Days until expiry, including weekends and holidays (Model-Only)', minval=0)
Expiry = (Hours + Minutes / 60) / 24 + Days
modelIV_ranklen = input.int(365, title='Model IV Rank/Percentile Length (Model-Only)', minval=1)
Interval_Width = input.float(1, title='Interval Width (Model-Only)')
Old_IV = input(true, title='Use old method to calculate Model-Based Implied Volatility?')
//VixFix Length Parameters
VIXFixLength = input.int(22, title='VixFix Length (recommended values: 9, 22, 66, 132, or 252)', minval=1)
VIXlength2 = input.int(252, title='VixFix Rank/Percentile Length', minval=1)

// Old Model-Based Volatility Smile Functions
Model_Upside(i) =>
    UP = ta.stdev(spot, modelIV_ranklen) * i + spot
    LogReturnUP = math.log(UP[1] / UP[2])
    AverageUP = ta.sma(LogReturnUP, modelIV_ranklen)
    STDEVUP = ta.stdev(LogReturnUP, modelIV_ranklen)
    Expiry_AverageUP = Expiry * AverageUP
    Expiry_STDEVUP = STDEVUP * math.sqrt(Expiry)
    upperUP = UP * math.exp(Expiry_AverageUP + Interval_Width * Expiry_STDEVUP)
    lowerUP = UP * math.exp(Expiry_AverageUP - Interval_Width * Expiry_STDEVUP)
    widthUP = upperUP - lowerUP
    standard_devUP = widthUP / 4
    IV_upside = standard_devUP / UP / math.sqrt(Expiry / 365) * 100
    IV_upside


Model_Downside(i) =>
    DN = spot - ta.stdev(spot, modelIV_ranklen) * i
    LogReturnDN = math.log(DN[1] / DN[2])
    AverageDN = ta.sma(LogReturnDN, modelIV_ranklen)
    STDEVDN = ta.stdev(LogReturnDN, modelIV_ranklen)
    Expiry_AverageDN = Expiry * AverageDN
    Expiry_STDEVDN = STDEVDN * math.sqrt(Expiry)
    upperDN = DN * math.exp(Expiry_AverageDN + Interval_Width * Expiry_STDEVDN)
    lowerDN = DN * math.exp(Expiry_AverageDN - Interval_Width * Expiry_STDEVDN)
    widthDN = upperDN - lowerDN
    standard_devDN = widthDN / 4
    IV_downside = standard_devDN / DN / math.sqrt(Expiry / 365) * 100
    IV_downside

Model_Based_Skew = math.avg(Model_Upside(1), Model_Upside(2), Model_Upside(3)) - math.avg(Model_Downside(1), Model_Downside(2), Model_Downside(3))

// New Model-Based IV calculations
Model_New() =>
    LogReturn = math.log(spot[1] / spot[2])
    Average = ta.sma(LogReturn, Expiry)
    STDEV = ta.stdev(LogReturn, Expiry)
    Time_Average = Expiry * Average
    Time_STDEV = STDEV * math.sqrt(Expiry)
    upper = spot * math.exp(Time_Average + Interval_Width * Time_STDEV)
    lower = spot * math.exp(Time_Average - Interval_Width * Time_STDEV)
    width = upper - lower
    standard_dev = width / 2
    //Optimizer
    sigma = standard_dev / (spot * math.sqrt(Expiry / 365))
    sigma

Model_Oldest() =>
    LogReturn = math.log(close[1] / close[2])
    Average = ta.sma(LogReturn, modelIV_ranklen)
    STDEV = ta.stdev(LogReturn, modelIV_ranklen)
    Expiry_Average = Expiry * Average
    Expiry_STDEV = STDEV * math.sqrt(Expiry)

    upper = close * math.exp(Expiry_Average + Interval_Width * Expiry_STDEV)
    lower = close * math.exp(Expiry_Average - Interval_Width * Expiry_STDEV)

    width = upper - lower
    standard_dev = width / 2
    IV = standard_dev / (close * math.sqrt(Expiry / 365)) * 100
    IV

// Model-Based Implied Volatility
IV = if Old_IV
    //(Model_Upside(0) + Model_Downside(0)) / 2  // Since we use i=0, it's equivalent to the Model_Oldest() results (the first model IV calculation the original author wrote) - Update: in fact no, this sometimes return na, such as on a weekly timeframe, for some strange reasons, hence it's not totally equivalent
    Model_Oldest()
else
    sigma = Model_New()
    if sigma > 1
        sigma / 2
    else
        sigma

//VixFix Volatility Smile Functions
VIX_Upside(i) =>
    VIXUPspot = ta.stdev(spot, VIXFixLength) * i + spot
    VIXUPLow = ta.stdev(low, VIXFixLength) * i + low
    VixUP = (ta.highest(VIXUPspot, VIXFixLength) - VIXUPLow) / ta.highest(VIXUPspot, VIXFixLength) * 100
    VixUP

VIX_Downside(i) =>
    VIXDNspot = spot - ta.stdev(spot, VIXFixLength) * i
    VIXDNLow = low - ta.stdev(low, VIXFixLength) * i
    VixDN = (ta.highest(VIXDNspot, VIXFixLength) - VIXDNLow) / ta.highest(VIXDNspot, VIXFixLength) * 100
    VixDN

VixFix_Skew = math.avg(VIX_Upside(1), VIX_Upside(2), VIX_Upside(3)) - math.avg(VIX_Downside(1), VIX_Downside(2), VIX_Downside(3))

//VixFix Implied Volatility
VixFix = (VIX_Upside(0) + VIX_Downside(0)) / 2  // since i=0, this is equivalent to the standard VIXfix calculation: (highest(close, VIXFixLength) - low) / highest(close, VIXFixLength) * 100

//Logic Statements
//Implied Volatility Logic
//Volatility = if VolCalc == 'VixFix'
//    VixFix
//else
//    IV
Volatility_VF = VixFix
Volatility_Model = IV

//Length for rank and percentile logic
//Len = if VolCalc == 'VixFix'
//    VIXlength2
//else
//    modelIV_ranklen
Len_VF = VIXlength2
Len_Model = modelIV_ranklen

//Implied Volatility Rank
f_IVR(Volatility, Len) =>
    (Volatility - ta.lowest(Volatility, Len)) / (ta.highest(Volatility, Len) - ta.lowest(Volatility, Len)) * 100

IVR_VF = f_IVR(Volatility_VF, Len_VF)
IVR_Model = f_IVR(Volatility_Model, Len_Model)

//Implied Volatility Percentile
pctileRank_VF = ta.percentrank(Volatility_VF, Len_VF)
pctileRank_Model = ta.percentrank(Volatility_Model, Len_Model)

//Skew Index Logic
Skew_Index = array.from(VixFix_Skew, Model_Based_Skew)

//Volatiliy Data Logic
float VolatilityData_VF = na
float VolatilityData_Model = na
if VolatilityChoice == 'IV Percentile'
    VolatilityData_VF := pctileRank_VF
    VolatilityData_Model := pctileRank_Model
else if VolatilityChoice == 'IV Rank'
    VolatilityData_VF := IVR_VF
    VolatilityData_Model := IVR_Model
else if VolatilityChoice == 'Volatility Skew Index'
    VolatilityData_VF := VixFix_Skew
    VolatilityData_Model := Model_Based_Skew
else if VolatilityChoice == 'Implied Volatility'
    VolatilityData_VF := Volatility_VF * 100
    VolatilityData_Model := Volatility_Model * 100

//Color Logic
color pcolor_VF = color.aqua
color pcolor_Model = color.yellow
if VolatilityChoice == 'IV Percentile' or VolatilityChoice == 'IV Rank'
    pcolor_VF := VolatilityData_VF < 50 ? color.red : VolatilityData_VF >= 50 ? color.aqua : color.aqua
    pcolor_Model := VolatilityData_Model < 50 ? color.blue : VolatilityData_Model >= 50 ? color.yellow : color.yellow
else if VolatilityChoice == 'Volatility Skew Index'
    pcolor_VF := color.aqua
    pcolor_Model := color.yellow
else if VolatilityChoice == 'Implied Volatility'
    pcolor_VF := color.aqua
    pcolor_Model := color.yellow

//Plot Statements
plot(VolatilityData_Model, color=pcolor_Model,
     title='Model-Based Volatility Data',
     display=VolCalc == 'Model Implied Volatility' or VolCalc == 'Both' ? display.all : display.none)
plot(VolatilityData_VF, color=pcolor_VF, title='Model-Free (VIXfix) Volatility Data', display=VolCalc == 'VixFix' or VolCalc == 'Both' ? display.all : display.none)



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
