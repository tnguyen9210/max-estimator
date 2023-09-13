
import random 
import numpy as np 

import multiprocessing
from algos import *

# manager = multiprocessing.Manager()
# error_list = manager.list()

def run_trial(
        i_trial, reward_fn, estimator, num_actions, num_samples, args):
    
    random.seed(10000 + i_trial)
    np.random.seed(10000 + i_trial)
    
    # set action_mus and action_sigmas
    if not args["action_mus"]:
        action_mus = 0.02 * (0.05 - 0.02)*np.random.rand(num_actions)
    else:
        action_mus = args["action_mus"]
        
    action_sigmas = args["action_sigmas"]
        
    # print(action_mus)
    # print(action_sigmas)
    # generate rewards
    action_rewards = reward_fn(
        action_mus, action_sigmas, num_actions, num_samples)
    
    # action_mus_hat = np.mean(action_rewards, axis=0)
    # action_sigmas_hat = np.std(action_rewards, axis=0, ddof=1)
    # print(action_mus_hat)
    # print(action_sigmas_hat)
    
    # apply estimator
    mu_est = estimator(action_rewards, num_actions, num_samples, args)    
    # print(mu_est)

    mu_max = np.max(action_mus)
    # print(mu_max)
    error_list.append(mu_est - mu_max)