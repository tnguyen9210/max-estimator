
import numpy as np


num_trials = 2
num_samples = 10000
num_arms = 10

delta_conf = 0.1

t_range = np.arange(1, num_samples+1)
fixed_dev_range = np.sqrt(2/t_range*np.log(1/delta_conf))
anytime_dev_range = np.sqrt(2/t_range*np.log(1/delta_conf))


samples_x = np.random.normal(size=(num_samples, num_trials))
samples_x_cumsum = np.cumsum(samples_x, axis=0)
samples_x_avg = samples_x_cumsum/t_range

# for i_trial in range(num_trials):
#     fixed_sample_x
#     an_samples_x_avg = samples_x_cumsum/t_range
#     trialsamples_y = np.zeros(num_samples)
#     samples_y[-samples_x_avg >= dev_range] = 1

print(samples_x.shape)
print(samples_x[:5,:5])

# any_samples_x = 

# for i_trial in range(num_trials):
anytime_t_range = np.arange(1, num_samples+1)
anytime_t2_range = anytime_t_range**2
anytime_thres = np.sqrt(2/anytime_t_range*np.log(
    4*anytime_t_range**2/delta_conf))
print(anytime_thres[:10])
anytime_t2_range = np.repeats(anytime_t_range, num_trials, axis=0)
print(anytime_t2_range)
anytime_samples_y = np.zeros(num_samples, num_trials)
anytime_samples_y[-samples_x_cumsum[:, i_trial] >= anytime_t2_range] = 1
anytime_idx_y = np.sum(anytime_samples_y) >= 1
    
