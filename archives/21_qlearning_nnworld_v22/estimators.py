
import numpy as np


def max_estimator(action_muhats):
    mu_est = np.max(action_muhats)
    return mu_est

def haver_estimator(action_muhats, action_sigmahats, num_actions, action_nvisits, args=None):
    haver_alpha = args["haver_alpha"]
    haver_delta = args["haver_delta"]
    haver_const = args["haver_const"]

    action_max_idx = np.argmax(action_muhats)
    action_max_muhat = action_muhats[action_max_idx]
    action_max_sigmahat = action_sigmahats[action_max_idx]
    action_max_nvisits = action_nvisits[action_max_idx]

    mu_bset = []
    mu_est_sum = 0
    mu_est_cnt = 0
    for i in range(num_actions):
        avg = action_max_sigmahat**2/action_max_nvisits \
            + action_sigmahats[i]**2/action_nvisits[i]
        thres = haver_const*np.sqrt(avg*np.log(num_actions**haver_alpha/haver_delta))
        if action_max_muhat - action_muhats[i] <= thres:
            mu_est_sum += action_muhats[i]
            mu_est_cnt += 1
            mu_bset.append(i)

    mu_est = mu_est_sum/mu_est_cnt
    
    return mu_est, mu_bset
