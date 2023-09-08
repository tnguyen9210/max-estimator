from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from tqdm import tqdm
import time 

# import gymnasium as gym
import gym
import gym_examples
from gym.wrappers import FlattenObservation

from algos import *
from utils import *

start_time = time.time()
random.seed(123)
np.random.seed(123)
tdqm_disable = True

# params
env_id = "gym_examples/GridWorld-v1"
gridworld_size = 3
max_steps = 99
gamma = 0.95

num_trials = 1
num_episodes_train = 10000
num_episodes_eval = 100

lr_sched_type = "linear"
lr_sched_fn = create_lr_sched_fn(lr_sched_type)

max_eps = 1.0
min_eps = 0.05
decay_rate = 0.0005
eps_sched_type = "poly"
eps_sched_fn = create_eps_sched_fn(eps_sched_type, min_eps, max_eps, decay_rate)

est_name = "double" 
q_algo_name = "double_q_learning"
q_algo = create_q_algo(q_algo_name)

# create gym env
env = gym.make(env_id, size=gridworld_size)
env_wrapped = FlattenObservation(env)
num_actions = env_wrapped.action_space.n
# print(f"num_actions = {num_actions}")
# print(env_wrapped.reset())
# stop

# episodes_lengths_ary = np.zeros((num_trials, num_episodes_train))
# episodes_rewards_ary = np.zeros((num_trials, num_episodes_train))
# episodes_start_vals_ary = np.zeros((num_trials, num_episodes_train))
manager = multiprocessing.Manager()
episodes_lengths_ary = manager.list()
episodes_rewards_ary = manager.list()
episodes_start_vals_ary = manager.list()

def run_trial(i_trial):
    
    random.seed(10000+i_trial)
    np.random.seed(10000+i_trial)

    env = gym.make(env_id, size=gridworld_size)
    env_wrapped = FlattenObservation(env)
    # env_wrapped.reset(seed=10000+i_trial)
    
    lr_sched_fn = create_lr_sched_fn(lr_sched_type)
    eps_sched_fn = create_eps_sched_fn(eps_sched_type, min_eps, max_eps, decay_rate)
    q_algo = create_q_algo(q_algo_name)
    
    # init Q_table, Q_nvisits
    Q_table = defaultdict(lambda: np.zeros(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))
    Q_table, stats = q_algo(
        env_wrapped, Q_table, Q_nvisits, num_episodes_train, max_steps,
        gamma, lr_sched_fn, eps_sched_fn, tdqm_disable)
    
    episode_lengths, episode_rewards, episode_start_vals= zip(*stats)
    episodes_lengths_ary.append(episode_lengths)
    episodes_rewards_ary.append(episode_rewards)
    episodes_start_vals_ary.append(episode_start_vals)


pool = multiprocessing.Pool()
pool.map(run_trial, range(num_trials))

episodes_lengths_ary = np.hstack([episodes_lengths_ary])
episodes_rewards_ary = np.hstack([episodes_rewards_ary])
episodes_start_vals_ary = np.hstack([episodes_start_vals_ary])

episode_lengths_mean = np.mean(episodes_lengths_ary, 0)
episode_rewards_mean = np.mean(episodes_rewards_ary, 0)
episode_start_vals_mean = np.mean(episodes_start_vals_ary, 0)
print(f"last_episode_length = {episode_lengths_mean[-1]}")
print(f"last_episode_reward_per_step = {episode_rewards_mean[-1]:.4f}")
print(f"last_episode_estim_start_val = {episode_start_vals_mean[-1]:.4f}")

end_time = time.time()
print(f"it takes {end_time-start_time}")
