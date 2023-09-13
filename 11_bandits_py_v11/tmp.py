
import numpy as np

# np.random.seed(123)

num_samples = 10000
num_actions = 2
action_mus_true = [0.02, 0.5]

action_rewards = np.random.binomial(
    1, action_mus_true, (num_samples, num_actions))

print(action_rewards)
print(action_rewards.shape)

action_mus_hat = np.mean(action_rewards, 0)

print(action_mus_hat)
print(action_mus_hat.shape)
