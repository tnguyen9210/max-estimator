
import copy
import numpy as np 
from tqdm import tqdm


def create_eps_decay_fn(min_eps, max_eps, decay_rate):
    def eps_decay_fn(i_eps):
        eps = min_eps + (max_eps - min_eps)*np.exp(-decay_rate*i_eps)
        return eps
    
    return eps_decay_fn

def create_lr_fn(lr_type):
    if lr_type == "linear":
        def lr_fn(nvisits):
            return 1/nvisits
    elif lr_type == "poly":
        def lr_fn(nvisits):
            return 1/(nvisits**0.8)
    else:
        stop
        
    return lr_fn


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


def q_learning(env, Q_table, Q_nvisits, num_episodes_train, max_steps,
               gamma, lr_fn, eps_decay_fn, seed_ary=None):
        
    # keep track of useful statistics
    stats = []
    
    for i_eps in tqdm(
            range(num_episodes_train), desc="train q_learning", disable=False):
        eps = eps_decay_fn(i_eps)
        state, info = env.reset()
        state = f"{state}"
        start_state = copy.deepcopy(state)
        
        episode_reward = 0.0
        for i_step in range(max_steps):
            # choose the action a_t using epsilon greedy policy
            nvisits = np.sum(Q_nvisits[state]) + 1
            eps = 1/np.sqrt(nvisits)
            action = eps_greedy_policy(Q_table, state, eps)

            new_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            new_state = f"{new_state}"

            next_action_best = np.argmax(Q_table[new_state])
            td_target = reward + gamma*Q_table[new_state][next_action_best]
            td_error = td_target - Q_table[state][action]

            Q_nvisits[state][action] += 1
            lr = lr_fn(Q_nvisits[state][action]) 
            Q_table[state][action] += lr*td_error
            

            if terminated or truncated:
                break

            state = new_state

        # stats.append((
        #     i_eps, episode_reward, i_step+1, np.max(Q_table[state_deepcopy])))
        stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))

    return Q_table, stats 

def double_q_learning(env, Q_table, Q_nvisits, num_episodes_train, max_steps,
                      gamma, lr_fn, eps_decay_fn, seed_ary=None):

    Q_table1 = copy.deepcopy(Q_table)
    Q_table2 = copy.deepcopy(Q_table)
    Q_nvisits1 = copy.deepcopy(Q_nvisits)
    Q_nvisits2 = copy.deepcopy(Q_nvisits)

    stats = []
    for i_eps in tqdm(
            range(num_episodes_train), "train double_q_learning", disable=False):
        eps = eps_decay_fn(i_eps)
        state, info = env.reset()
        state = f"{state}"
        start_state = copy.deepcopy(state)

        episode_reward = 0.0
        for i_step in range(max_steps):
            # choose the action a_t using epsilon greedy policy
            nvisits = np.sum(Q_nvisits[state]) + 1
            eps = 1/np.sqrt(nvisits)
            action = eps_greedy_policy(Q_table, state, eps)

            new_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            new_state = f"{new_state}"

            if np.random.rand() < 0.5:
                next_action_best = np.argmax(Q_table1[new_state]) 
                td_target = reward + gamma*Q_table2[new_state][next_action_best]
                td_error = td_target - Q_table1[state][action]

                Q_nvisits1[state][action] += 1
                lr = lr_fn(Q_nvisits1[state][action]) 
                Q_table1[state][action] += lr*td_error
                
                
            else:
                next_action_best = np.argmax(Q_table2[new_state]) 
                td_target = reward + gamma*Q_table1[new_state][next_action_best]
                td_error = td_target - Q_table2[state][action]

                Q_nvisits2[state][action] += 1
                lr = lr_fn(Q_nvisits2[state][action]) 
                Q_table2[state][action] += lr*td_error
            
            # next_action_best = np.argmax(Q_table2[new_state]) 
            # td_target = reward + gamma*Q_table2[new_state][next_action_best]
            # td_error = td_target - Q_table2[state][action]

            # Q_nvisits2[state][action] += 1
            # lr = lr_fn(Q_nvisits2[state][action]) 
            # Q_table2[state][action] += lr*td_error
        
            Q_table[state][action] = (Q_table1[state][action] + Q_table2[state][action])/2
            Q_nvisits[state][action] = Q_nvisits1[state][action] + Q_nvisits2[state][action]
            
            if terminated or truncated:
                break

            state = new_state

        stats.append((i_step + 1, episode_reward/(i_step+1), np.max(Q_table[start_state])))

    return Q_table, stats
    

def running_avg(vec, window_size):
    r_avg = np.zeros(len(vec))
    for i in range(window_size):
        r_avg[i] = sum(vec[:i+1])/(i+1)
    for i in range(window_size, len(vec)):
        r_avg[i] = sum(vec[i+1-window_size:i+1])/(window_size)
    return r_avg
