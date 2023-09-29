
import numpy as np


num_trials = 1
num_samples = 10000
num_arms = 10

delta_conf = 0.1

t_range = np.arange(1, num_samples+1)
dev_range = np.sqrt(2/t_range*np.log(1/delta_conf))

for i_trial in range(num_trials):
    samples_x = np.random.normal(size=(num_samples))
    samples_x_cumsum = np.cumsum(samples)
    samples_x_avg = samples_x_cumsum/t_range
    samples_y = np.zeros(num_samples)
    samples_y[-samples_x_avg >= dev_range] = 1
    
