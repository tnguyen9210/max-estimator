
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm 
import time

# import gymnasium as gym
import gym
import gym_examples
from gym.wrappers import FlattenObservation

from algos import *
# from gridworld import GridworldEnv

def main():
    start_time = time.time()
    random.seed(123)
    np.random.seed(123)
    
    # params
    env_id = "GridWorld-v0"
    gridworld_size = 3
    max_steps = 99
    gamma = 0.95
    num_episodes_train = 10000
    num_episodes_eval = 100
    
    max_eps = 1.0
    min_eps = 0.05
    decay_rate = 0.0005
    eps_decay_fn = create_eps_decay_fn(min_eps, max_eps, decay_rate)
    lr_fn = create_lr_fn(lr_type="linear")
    
    # create gym env
    # env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
    env = gym.make(
        'gym_examples/GridWorld-v1', size=gridworld_size)
    # env.reset(seed=1)
    env_wrapped = FlattenObservation(env)
    # print(env_wrapped.reset())
    # stop
    
    num_actions = env.action_space.n
    print(f"num_actions = {num_actions}")
    
    # init Q_table, lr_table
    Q_table = defaultdict(lambda: np.zeros(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))
    
    env_wrapped.reset(seed=2)
    Q_table, stats = q_learning(	
        env_wrapped, Q_table, Q_nvisits, num_episodes_train, max_steps,
        gamma, lr_fn, eps_decay_fn)

    episode_lengths, episode_rewards, episode_start_vals= zip(*stats)
    episode_lengths = np.array(episode_lengths)
    episode_rewards = np.array(episode_rewards)
    episode_start_vals = np.array(episode_start_vals)
    print(f"last_episode_length = {episode_lengths[-1]}")
    print(f"last_episode_reward_per_step = {episode_rewards[-1]:.4f}")
    print(f"last_episode_estim_start_val = {episode_start_vals[-1]:.4f}")

    end_time = time.time()
    print(f"it takes {end_time - start_time}")

    fig, axes = fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True, sharey=False, figsize=(10,5))
    # axes = [axes]
    
    x_ary = np.linspace(0, num_episodes_train-1, num=100, dtype=np.int32)
    axes[0].plot(x_ary, running_avg(episode_rewards[x_ary], 10), label='Q-learning')
    axes[0].set_title("reward_per_step")
    axes[0].legend()
    axes[1].plot(x_ary, episode_start_vals[x_ary], label='Q-learning')
    axes[1].set_title("estim_qval_start_state")
    axes[1].legend()
    axes[2].plot(x_ary, episode_lengths[x_ary], label='Q-learning')
    axes[2].set_title("episode_length")
    axes[2].legend()
    plt.show()
    
    # # evaluate
    # episode_reward_ary = evaluate(env_wrapped, Q_table, num_episodes_eval, max_steps)
    # reward_mean = np.mean(episode_reward_ary)
    # reward_std = np.std(episode_reward_ary)
    # print(f"reward = {reward_mean:.2f} +/- {reward_std:.2f}")

    # env = gym.make(
    #     'gym_examples/GridWorld-v1', size=gridworld_size, render_mode="human")
    # env_wrapped = FlattenObservation(env)
    # episode_reward_ary = evaluate(env_wrapped, Q_table, 3, 20)

if __name__ == '__main__':
    main()
