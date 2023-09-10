from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm 

# import gymnasium as gym
import gym
import gym_examples
from gym.wrappers import FlattenObservation

from algos import *
from utils import *

def main():
    random.seed(123)
    np.random.seed(123)
    tdqm_disable = False

    # params
    env_id = "gym_examples/GridWorld-v1"
    gridworld_size = 3
    max_steps = 99
    gamma = 0.95

    num_trials = 1
    num_episodes_train = 1000
    num_episodes_eval = 100

    lr_sched_type = "linear"
    lr_sched_fn = create_lr_sched_fn(lr_sched_type)

    max_eps = 1.0
    min_eps = 0.05
    decay_rate = 0.0005
    eps_sched_type = "poly"
    eps_sched_fn = create_eps_sched_fn(eps_sched_type, min_eps, max_eps, decay_rate)

    haver_const_ary = 2
    est_name = "haver" 
    q_algo_name = "haver_q_learning"

    # create gym env
    env = gym.make(env_id, size=gridworld_size)
    env_wrapped = FlattenObservation(env)
    num_actions = env_wrapped.action_space.n
    # print(f"num_actions = {num_actions}")
    # print(env_wrapped.reset())
    # stop
    
    haver_const_ary = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    for haver_const in haver_const_ary:
        print(f"haver_const = {haver_const}")
        q_algo = create_q_algo(q_algo_name, haver_const=haver_const)
        episodes_lengths_ary = np.zeros((num_trials, num_episodes_train))
        episodes_rewards_ary = np.zeros((num_trials, num_episodes_train))
        episodes_start_vals_ary = np.zeros((num_trials, num_episodes_train))
        for i_trial in tqdm(range(num_trials), desc="run trials", disable=True):
            # set trial seed
            random.seed(10000+i_trial)
            np.random.seed(10000+i_trial)

            # init Q_table, Q_nvisits
            Q_table = defaultdict(lambda: np.zeros(num_actions))
            Q_nvisits = defaultdict(lambda: np.zeros(num_actions))
            Q_table, stats = q_algo(
                env_wrapped, Q_table, Q_nvisits, num_episodes_train, max_steps,
                gamma, lr_sched_fn, eps_sched_fn, haver_const, tdqm_disable)

            episode_lengths, episode_rewards, episode_start_vals= zip(*stats)
            episodes_lengths_ary[i_trial,:] = episode_lengths
            episodes_rewards_ary[i_trial,:] = episode_rewards
            episodes_start_vals_ary[i_trial,:] = episode_start_vals

        episode_lengths_mean = np.mean(episodes_lengths_ary, 0)
        episode_rewards_mean = np.mean(episodes_rewards_ary, 0)
        episode_start_vals_mean = np.mean(episodes_start_vals_ary, 0)
        print(f"last_episode_length = {episode_lengths_mean[-1]}")
        print(f"last_episode_reward_per_step = {episode_rewards_mean[-1]:.4f}")
        print(f"last_episode_estim_start_val = {episode_start_vals_mean[-1]:.4f}")

        fig, axes = fig, axes = plt.subplots(
            nrows=3, ncols=1, sharex=True, sharey=False, figsize=(8,8))
        # axes = [axes]

        x_ary = np.linspace(0, num_episodes_train-1, num=100, dtype=np.int32)
        axes[0].plot(x_ary, running_avg(episode_rewards_mean[x_ary], 10), label=q_algo_name)
        axes[0].set_title("reward_per_step")
        axes[0].plot(x_ary, episode_rewards_mean[x_ary], label=q_algo_name)
        axes[0].set_title("reward_per_step")
        axes[0].legend()
        axes[1].plot(x_ary, episode_start_vals_mean[x_ary], label=q_algo_name)
        axes[1].set_title("estim_qval_start_state")
        axes[1].legend()
        axes[2].plot(x_ary, episode_lengths_mean[x_ary], label=q_algo_name)
        axes[2].set_title("episode_length")
        axes[2].legend()
        plt.show()

if __name__ == '__main__':
    main()
