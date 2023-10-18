
import numpy as np


def running_avg(vec, window_size):
    r_avg = np.zeros(len(vec))
    for i in range(window_size):
        r_avg[i] = sum(vec[:i+1])/(i+1)
    for i in range(window_size, len(vec)):
        r_avg[i] = sum(vec[i+1-window_size:i+1])/(window_size)
    return r_avg
