
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
# from gridworld import GridworldEnv

def main():
    random.seed(123)
    np.random.seed(123)
    
    # params
    env_id = "GridWorld-v0"
    gridworld_size = 3
    lr = 0.7
    max_steps = 99
    gamma = 0.95
    num_episodes_train = 1000
    num_episodes_eval = 100
    
    max_eps = 1.0
    min_eps = 0.05
    decay_rate = 0.0005
    eps_decay_fn = create_eps_decay_fn(min_eps, max_eps, decay_rate)
    
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
    
    # init Qtable
    # Qtable = np.zeros((num_states, num_actions))
    Qtable = defaultdict(lambda: np.zeros(num_actions))
    
    # run q_learning
    num_runs = 50
    episode_rewards_ary = np.zeros((num_runs, num_episodes_train))
    episode_lengths_ary = np.zeros((num_runs, num_episodes_train))
    for i_run in tqdm(range(num_runs), desc="run trials"):
        env_wrapped.reset(seed=i_run)
        
        Qtable = defaultdict(lambda: np.zeros(num_actions))
        Qtable, stats = q_learning(
            env_wrapped, Qtable, num_episodes_train, max_steps,
            lr, gamma, eps_decay_fn)

        episode_lengths, episode_rewards = zip(*stats)
        # print(episode_rewards[:5])
        # print(episode_lengths[:5])
        episode_rewards_ary[i_run,:] = episode_rewards
        episode_lengths_ary[i_run,:] = episode_lengths

    fig, axes = fig, axes = plt.subplots(
        nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10,5))
    axes = [axes]
    axes[0].plot(running_avg(np.mean(episode_rewards_ary, 0), 10), label='Q-learning')
    plt.legend()
    plt.show()
    # # evaluate
    # episode_reward_ary = evaluate(env_wrapped, Qtable, num_episodes_eval, max_steps)
    # reward_mean = np.mean(episode_reward_ary)
    # reward_std = np.std(episode_reward_ary)
    # print(f"reward = {reward_mean:.2f} +/- {reward_std:.2f}")

    # env = gym.make(
    #     'gym_examples/GridWorld-v1', size=gridworld_size, render_mode="human")
    # env_wrapped = FlattenObservation(env)
    # episode_reward_ary = evaluate(env_wrapped, Qtable, 3, 20)

if __name__ == '__main__':
    main()
