

from collections import defaultdict
import random
import numpy as np

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
    env_id = "CliffWalking-v0"
    lr = 0.7
    max_steps = 99
    gamma = 0.95
    num_episodes_train = 500
    num_episodes_eval = 100
    
    max_eps = 1.0
    min_eps = 0.05
    decay_rate = 0.0005
    eps_decay_fn = create_eps_decay_fn(min_eps, max_eps, decay_rate)
    
    # create gym env
    env = gym.make('CliffWalking-v0')
    env.reset(seed=1)
    env_wrapped = FlattenObservation(env)
    # print(env_wrapped.reset())
    # stop
    
    num_actions = env.action_space.n
    print(f"num_actions = {num_actions}")
    
    # init Qtable
    # Qtable = np.zeros((num_states, num_actions))
    Qtable = defaultdict(lambda: np.zeros(num_actions))
    
    # run q_learning
    Qtable = q_learning(
        env_wrapped, Qtable, num_episodes_train, max_steps, lr, gamma, eps_decay_fn)
    
    # evaluate
    episode_reward_ary = evaluate(env_wrapped, Qtable, num_episodes_eval, max_steps)
    reward_mean = np.mean(episode_reward_ary)
    reward_std = np.std(episode_reward_ary)
    print(f"reward = {reward_mean:.2f} +/- {reward_std:.2f}")

    # # visualize
    # env = gym.make('CliffWalking-v0',  render_mode="human")
    # env_wrapped = FlattenObservation(env)
    # episode_reward_ary = evaluate(env_wrapped, Qtable, 3, max_steps)

if __name__ == '__main__':
    main()
