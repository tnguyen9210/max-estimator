
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt 

import time 
import multiprocessing

from algos import *

manager = multiprocessing.Manager()
error_list = manager.list()

def run_trial(
        i_trial, estimator, num_actions, num_samples, action_mus_true=None):
    
    random.seed(10000 + i_trial)
    np.random.seed(10000 + i_trial)
    
    # # set action_means
    if not action_mus_true:
        action_mus_true = 0.02 * (0.05 - 0.02)*np.random.rand(num_actions)
        
    mu_max = np.max(action_mus_true)
    
    # generate rewards
    action_rewards = np.random.binomial(
        1, action_mus_true, (num_samples, num_actions))

    # action_mu_hat = np.mean(action_rewards)
    # action_var_hat = (action_mu_hat - action_mu_hat**2)/num_updates
    # action_var_hat[action_var_hat < 0] = 0.0
    # action_sigma_hat = np.sqrt(action_var_hat)
    # action_sigma_hat[action_sigma_hat < 1e-4] = 1e-4

    mu_est = estimator(action_rewards, num_actions, num_samples)

    error_list.append(mu_max - mu_est)    

# params
num_actions = 30
num_samples = 10
num_trials = 1000

pool = multiprocessing.Pool()

est_bias_dict = defaultdict(list)
est_var_dict = defaultdict(list)
est_mse_dict = defaultdict(list)

est_name_ary = ["max"]
num_samples_ary = np.arange(1000, 11000, 1000)
for est_name in est_name_ary:
    print(f"\n-> est_name = {est_name}")
    estimator = create_estimator(est_name)

    for num_samples in num_samples_ary:
        trial_args = [
            (i, estimator, num_actions, num_samples) for i in range(num_trials)]
        pool.starmap(run_trial, trial_args)

        error_ary = np.hstack(error_list)
        est_bias = np.mean(error_ary)
        est_var = np.var(error_ary, ddof=1)
        est_mse = est_bias**2 + est_var
        
        est_bias_dict[est_name].append(est_bias)
        est_var_dict[est_name].append(est_var)
        est_mse_dict[est_name].append(est_mse)


fig, axes = fig, axes = plt.subplots(
        nrows=2, ncols=2, sharex=True, sharey=False, figsize=(8,8))
# axes = [axes]
axes = axes.ravel()

est_name_ary = ["haver", "weightedms", "double"]
est_name_ary = ["max"]
x_ary = num_samples_ary
for est_name in est_name_ary:
    axes[0].plot(x_ary, est_bias_dict[est_name], label=est_name)
    axes[1].plot(x_ary, np.abs(est_bias_dict[est_name]), label=est_name)
    axes[2].plot(x_ary, est_var_dict[est_name], label=est_name)
    axes[3].plot(x_ary, est_mse_dict[est_name], label=est_name)

axes[0].set_title("mse")
axes[1].set_title("bias")
axes[2].set_title("var")
axes[0].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
axes[1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
axes[2].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()
    
    
        
    
    
    
    
