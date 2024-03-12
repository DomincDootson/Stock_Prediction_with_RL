import os
import math

import numpy as np

def sigmoid(x):
    """Performs sigmoid operation"""
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))
    


def get_state(data, t, n_days):
    """The state for the agent will be a vector of changes in prices over the past n_days. 
    If we are too close to the beginning of the time period, we pad out. We finally pass through
    a sigmoid so that the values are between 0 and 1"""
    d = t - n_days + 1
    block = data[d: t + 1] if d >= 0 else abs(d) * [data[0]] + data[0: t + 1]  # pad with t0
    res = [sigmoid(b_p_1 - b) for b, b_p_1 in zip(block, block[1:])] 

    return np.array([res])
