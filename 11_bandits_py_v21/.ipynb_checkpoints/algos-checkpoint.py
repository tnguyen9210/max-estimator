
import numpy as np

def max_estimator(action_rewards_list, num_actions, num_samples_list, args=None):
    action_muhats_list = []
    for action_idx in range(num_actions):
        action_idx_muhat = np.mean(action_rewards_list[action_idx])
        action_muhats_list.append(action_idx_muhat)
    
    mu_est = np.max(action_muhats_list)
    return mu_est

def avg_estimator(action_rewards_list, num_actions, num_samples_list, args=None):
    action_muhats_list = []
    for action_idx in range(num_actions):
        action_idx_muhat = np.mean(action_rewards_list[action_idx])
        action_muhats_list.append(action_idx_muhat)
        
    mu_est = np.mean(action_muhats_list)
    return mu_est

def double_estimator(action_rewards_list, num_actions, num_samples_list, args=None):
    action_muhats_list1 = []
    action_muhats_list2 = []
    for action_idx in range(num_actions):
        num_samples_half = int(num_samples_list[action_idx]/2)
        action_idx_muhat1 = np.mean(action_rewards_list[action_idx][:num_samples_half])
        action_idx_muhat2 = np.mean(action_rewards_list[action_idx][num_samples_half:])
        
        action_muhats_list1.append(action_idx_muhat1)
        action_muhats_list2.append(action_idx_muhat2)

    action_muhats_1 = np.hstack(action_muhats_list1)
    action_muhats_2 = np.hstack(action_muhats_list2)
    
    action_max_muhat_1 = np.max(action_muhats_1)
    action_max_muhat_2 = np.max(action_muhats_2)

    mu_est_1 = np.mean(action_muhats_2[action_muhats_1 == action_max_muhat_1])
    mu_est_2 = np.mean(action_muhats_1[action_muhats_2 == action_max_muhat_2])
    
    mu_est = (mu_est_1 + mu_est_2)/2

    return mu_est


def weightedms_estimator(action_rewards_list, num_actions, num_samples_list, args=None):
    # sample data based on empirical muhats and sigmahats
    num_data = args["weightedms_num_data"]
    action_muhats_list = []
    action_sigmahats_list = []
    for action_idx in range(num_actions):
        num_samples = num_samples_list[action_idx]
        action_idx_muhat = np.mean(action_rewards_list[action_idx])
        action_muhats_list.append(action_idx_muhat)
        
        action_idx_sigmahat = np.std(action_rewards_list[action_idx], ddof=1)/np.sqrt(num_samples)
        action_sigmahats_list.append(action_idx_sigmahat)

    action_muhats = np.hstack(action_muhats_list)
    action_sigmahats = np.hstack(action_sigmahats_list)
    
    action_sigmahats[action_sigmahats < 1e-4] = 1e-4
    data = np.random.normal(
        action_muhats, action_sigmahats, size=(num_data, num_actions))
    # print(action_muhats)
    # print(action_sigmahats)
    # print(data)

    # compute action weights
    action_idxes = np.argmax(data, 1)
    action_idxes, action_cnts = np.unique(action_idxes, return_counts=True)
    action_weights = np.zeros(num_actions)
    action_weights[action_idxes[action_cnts > 0]] = action_cnts[action_cnts > 0]
    action_weights /= num_data

    # weightedms estimator
    mu_est = np.dot(action_weights, action_muhats)
    
    return mu_est

def haver_estimator(action_rewards_list, num_actions, num_samples_list, args=None):
    haver_alpha = args["haver_alpha"]
    haver_delta = args["haver_delta"]
    haver_const = args["haver_const"]
    action_muhats_list = []
    action_sigmahats_list = []
    for action_idx in range(num_actions):
        num_samples = num_samples_list[action_idx]
        action_idx_muhat = np.mean(action_rewards_list[action_idx])
        action_muhats_list.append(action_idx_muhat)
        
        action_idx_sigmahat = np.std(action_rewards_list[action_idx], ddof=1)/np.sqrt(num_samples)
        action_sigmahats_list.append(action_idx_sigmahat)

    action_muhats = np.hstack(action_muhats_list)
    action_sigmahats = np.hstack(action_sigmahats_list)
    
    # action_muhats = np.mean(action_rewards, axis=0)
    # action_sigmahats = np.std(action_rewards, axis=0, ddof=1) / np.sqrt(num_samples)
    # action_sigmahats[action_sigmahats < 1e-4] = 1e-4

    action_max_idx = np.argmax(action_muhats)
    action_max_muhat = action_muhats[action_max_idx]
    action_max_sigmahat = action_sigmahats[action_max_idx]

    mu_est_sum = 0
    mu_est_cnt = 0
    for i in range(num_actions):
        avg = action_max_sigmahat**2 + action_sigmahats[i]**2
        thres = haver_const*np.sqrt(avg*np.log(num_actions**haver_alpha/haver_delta))
        if action_max_muhat - action_muhats[i] <= thres:
            mu_est_sum += action_muhats[i]*num_samples_list[i]
            mu_est_cnt += num_samples_list[i]

    mu_est = mu_est_sum/mu_est_cnt
    
    return mu_est

def haver2_estimator(action_rewards_list, num_actions, num_samples_list, args=None):
    haver_alpha = args["haver_alpha"]
    haver_delta = args["haver_delta"]
    haver_const = args["haver_const"]
    action_muhats_list = []
    action_sigmahats_list = []
    for action_idx in range(num_actions):
        num_samples = num_samples_list[action_idx]
        action_idx_muhat = np.mean(action_rewards_list[action_idx])
        action_muhats_list.append(action_idx_muhat)
        
        action_idx_sigmahat = np.std(action_rewards_list[action_idx], ddof=1)/np.sqrt(num_samples)
        action_sigmahats_list.append(action_idx_sigmahat)

    action_muhats = np.hstack(action_muhats_list)
    action_sigmahats = np.hstack(action_sigmahats_list)
    
    # action_muhats = np.mean(action_rewards, axis=0)
    # action_sigmahats = np.std(action_rewards, axis=0, ddof=1) / np.sqrt(num_samples)
    # action_sigmahats[action_sigmahats < 1e-4] = 1e-4

    action_max_idx = np.argmax(action_muhats)
    action_max_muhat = action_muhats[action_max_idx]
    action_max_sigmahat = action_sigmahats[action_max_idx]

    mu_est_sum = 0
    mu_est_cnt = 0
    for i in range(num_actions):
        avg = action_max_sigmahat**2 + action_sigmahats[i]**2
        thres = haver_const*np.sqrt(avg*np.log(num_actions**haver_alpha/haver_delta))
        if action_max_muhat - action_muhats[i] <= thres:
            mu_est_sum += action_muhats[i]
            mu_est_cnt += 1.0

    # print(mu_est_cnt)
    mu_est = mu_est_sum/mu_est_cnt
    
    return mu_est


def create_estimator(est_name):
    if est_name == "max":
        estimator = max_estimator
    elif est_name == "avg":
        estimator = avg_estimator
    elif est_name == "double":
        estimator = double_estimator
    elif est_name == "weightedms":
        estimator = weightedms_estimator
    elif est_name == "haver":
        estimator = haver_estimator
    elif est_name == "haver2":
        estimator = haver2_estimator
    

    return estimator
