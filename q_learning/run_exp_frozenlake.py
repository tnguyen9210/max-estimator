
from collections import defaultdict
import numpy as np

import gym

from algos import *

def main():
    # params
    env_id = "FrozenLake-v1"
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
    env = gym.make(
        "FrozenLake-v1", map_name="4x4",
        is_slippery=False, render_mode="human")
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    print(f"num_states = {num_states}")
    print(f"sample_state = {env.observation_space.sample()}")
    print(f"num_actions = {num_actions}")
    print(f"sample_action = {env.action_space.sample()}")
    
    
    # init Qtable
    Qtable = defaultdict(lambda: np.zeros(num_actions))
    
    # run q_learning
    Qtable = q_learning(
        env, Qtable, num_episodes_train, max_steps, lr, gamma, eps_decay_fn)

    # evaluate
    episode_reward_ary = evaluate(env, Qtable, num_episodes_eval, max_steps)
    reward_mean = np.mean(episode_reward_ary)
    reward_std = np.std(episode_reward_ary)
    print(f"reward = {reward_mean:.2f} +/- {reward_std:.2f}")

if __name__ == '__main__':
    main()
