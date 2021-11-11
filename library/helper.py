import numpy as np
import functools
import operator

def compute_gae(next_value, rewards, masks, values, gamma=0.99, lam=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def functools_reduce_iconcat(a):
    return np.array(functools.reduce(operator.iconcat, a, []),dtype="float32")

def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x