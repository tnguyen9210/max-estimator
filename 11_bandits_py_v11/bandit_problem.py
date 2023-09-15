
import numpy as np 

class BanditProblem:
    def __init__(self, problem_instance, reward_dist, num_actions,
                 action_mus=None, action_sigmas=None,
                 **args):
        
        self.num_actions = num_actions
        
        if problem_instance == "all_best":
            action_mus = np.zeros(num_actions)
        elif problem_instance == "multi_gap_nonlinear":
            # adjust gap_splits into correct format
            gap_deltas = args["gap_deltas"]
            gap_splits = args["gap_splits"]
            if isinstance(gap_splits, (float, int)):
                gap_splits = [gap_splits]
            if isinstance(gap_splits, (list, np.ndarray)):
                if isinstance(gap_splits[0], (float)):
                    gap_splits = \
                        [int(np.floor(gap_split*num_actions))
                         for gap_split in gap_splits]

            # add last index to gap_splits
            if gap_splits[-1] != num_actions:
                gap_splits.append(num_actions)
                    
            action_mus = np.zeros(num_actions)
            num_splits = len(gap_splits)
            for i_gap in range(1, num_splits):
                lower_idx = gap_splits[i_gap-1]
                upper_idx = gap_splits[i_gap]
                action_mus[lower_idx:upper_idx] = gap_deltas[i_gap-1]
                
        self.action_mus = action_mus
        
        if action_sigmas is not None:
            self.action_sigmas = action_sigmas
        else:
            self.action_sigmas = np.ones(num_actions)
            
        self.reward_fn = create_reward_fn(reward_dist)
            
    def get_rewards(self, num_samples):
        return self.reward_fn(
            self.action_mus, self.action_sigmas, self.num_actions, num_samples)
    
            
        
def create_reward_fn(reward_dist):
    if reward_dist == "bernoulli":
        reward_fn = reward_bernoulli
    elif reward_dist == "normal":
        reward_fn = reward_normal
    
    return reward_fn


def reward_bernoulli(mus, sigmas, num_actions, num_samples):
    rewards = np.random.binomial(1, mus, (num_samples, num_actions))
    return rewards


def reward_normal(mus, sigmas, num_actions, num_samples):
    rewards = np.random.normal(mus, sigmas, size=(num_samples, num_actions))
    return rewards
