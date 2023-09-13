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

est_name = "max" 
q_algo_name = "q_learning"
q_algo = create_q_algo(q_algo_name)

# create gym env
env = gym.make(env_id, size=gridworld_size)
env_wrapped = FlattenObservation(env)
num_actions = env_wrapped.action_space.n
# print(f"num_actions = {num_actions}")
# print(env_wrapped.reset())
# stop

manager = multiprocessing.Manager()
episode_lengths_ary = manager.list()
episode_rewards_ary = manager.list()
episode_start_vals_ary = manager.list()
Q_table_ary = manager.list()

def run_trial(i_trial):
    
    random.seed(10000+i_trial)
    np.random.seed(10000+i_trial)

    env = gym.make(env_id, size=gridworld_size)
    env_wrapped = FlattenObservation(env)
    # env_wrapped.reset(seed=10000+i_trial) # we didn't use env.np_random
    
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
    episode_lengths_ary.append(episode_lengths)
    episode_rewards_ary.append(episode_rewards)
    episode_start_vals_ary.append(episode_start_vals)
    Q_table_ary.append(Q_table)

pool = multiprocessing.Pool()
pool.map(run_trial, range(num_trials))

episode_lengths_ary = np.hstack([episode_lengths_ary])
episode_rewards_ary = np.hstack([episode_rewards_ary])
episode_start_vals_ary = np.hstack([episode_start_vals_ary])

episode_lengths_mean = np.mean(episode_lengths_ary, 0)
episode_rewards_mean = np.mean(episode_rewards_ary, 0)
episode_start_vals_mean = np.mean(episode_start_vals_ary, 0)
print(f"last_episode_length = {episode_lengths_mean[-1]}")
print(f"last_episode_reward_per_step = {episode_rewards_mean[-1]:.4f}")
print(f"last_episode_estim_start_val = {episode_start_vals_mean[-1]:.4f}")

end_time = time.time()
print(f"it takes {end_time-start_time}")

# evaluate
# episode_reward_ary = evaluate(env_wrapped, Q_table, num_episodes_eval, max_steps)
# reward_mean = np.mean(episode_reward_ary)
# reward_std = np.std(episode_reward_ary)
# print(f"reward = {reward_mean:.2f} +/- {reward_std:.2f}")

env = gym.make(
    'gym_examples/GridWorld-v1', size=gridworld_size, render_mode="human")
env_wrapped = FlattenObservation(env)
episode_reward_ary = evaluate(env_wrapped, Q_table, 3, 20)
