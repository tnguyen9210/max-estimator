
import numpy as np

def max_estimator(action_rewards, num_actions, num_samples, args=None):
    action_muhats = np.mean(action_rewards, axis=0)
    mu_est = np.max(action_muhats)
    return mu_est

def avg_estimator(action_rewards, num_actions, num_samples, args=None):
    action_muhats = np.mean(action_rewards, axis=0)
    mu_est = np.mean(action_muhats)
    return mu_est

def double_estimator(action_rewards, num_actions, num_samples, args=None):
    num_samples_half = int(num_samples/2)
    action_rewards_1 = action_rewards[:num_samples_half,:]
    action_rewards_2 = action_rewards[num_samples_half:,:]

    action_muhats_1 = np.mean(action_rewards_1, axis=0)
    action_muhats_2 = np.mean(action_rewards_2, axis=0)

    action_muhat_max_1 = np.max(action_muhats_1)
    action_muhat_max_2 = np.max(action_muhats_2)

    mu_est_1 = np.mean(action_muhats_2[action_muhats_1 == action_muhat_max_1])
    mu_est_2 = np.mean(action_muhats_1[action_muhats_2 == action_muhat_max_2])
    
    mu_est = (mu_est_1 + mu_est_2)/2

    return mu_est


def weightedms_estimator(action_rewards, num_actions, num_samples, args=None):
    # sample data based on empirical muhats and sigmahats
    num_data = args["weightedms_num_data"]
    action_muhats = np.mean(action_rewards, axis=0)
    action_sigmahats = np.std(action_rewards, axis=0, ddof=1) / np.sqrt(num_samples)
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

def haver_estimator(action_rewards, num_actions, num_samples, args=None):
    haver_alpha = args["haver_alpha"]
    haver_delta = args["haver_delta"]
    haver_const = args["haver_const"]
    action_muhats = np.mean(action_rewards, axis=0)
    action_sigmahats = np.std(action_rewards, axis=0, ddof=1) / np.sqrt(num_samples)
    action_sigmahats[action_sigmahats < 1e-4] = 1e-4

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
            mu_est_cnt += 1

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
    

    return estimator
