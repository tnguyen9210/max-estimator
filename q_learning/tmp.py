
from collections import defaultdict
import numpy as np 



def main():
    # num_actions = 4
    # Q = defaultdict(lambda: np.zeros(num_actions))
    # print(Q)
    # state = tuple([0, 2])
    # print(Q[state])
    # Q[state] = 2
    # print(Q)
    seed_ary = np.arange(10, dtype=int) + 1
    for seed in seed_ary:
        print(type(seed))
        np.random.seed(seed)
    print(seed_ary)
    stop

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        # YOUR CODE HERE
        action_probs = np.ones(nA)*epsilon/nA
        action_values = Q[observation]
        greedy_action = np.argmax(action_values)
        
        action_probs[greedy_action] += 1-epsilon
        
        return action_probs
    return policy_fn


def cdf(weights):
    total = sum(weights)
    assert(total==1)
    percentile = []
    cumsum = 0
    for w in weights:
        cumsum += w
        percentile.append(cumsum / total)
    return percentile


def choose_action(percentiles):
    r = np.random.rand()
    for i in range(len(percentiles)):
        if r<=percentiles[i]:
            return i
    raise Error('no choice made')


if __name__ == '__main__':
    main()
