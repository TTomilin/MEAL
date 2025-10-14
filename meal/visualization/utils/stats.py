from typing import Iterable

import numpy as np


# MDP

def cumulative_rewards_from_rew_list(rews):
    return [sum(rews[:t]) for t in range(len(rews))]


# Gridworld

def manhattan_distance(pos1, pos2):
    """Returns manhattan distance between two points in (x, y) format"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def pos_distance(pos0, pos1):
    return tuple(np.array(pos0) - np.array(pos1))


# Randomness

def rnd_uniform(low, high):
    if low == high:
        return low
    return np.random.uniform(low, high)


def rnd_int_uniform(low, high):
    if low == high:
        return low
    return np.random.choice(range(low, high + 1))


# Statistics

def std_err(lst):
    """Computes the standard error"""
    sd = np.std(lst)
    n = len(lst)
    return sd / np.sqrt(n)


def mean_and_std_err(lst):
    "Mean and standard error of list"
    mu = np.mean(lst)
    return mu, std_err(lst)


def dict_mean_and_std_err(d):
    """
    Takes in a dictionary with lists as keys, and returns a dictionary
    with mean and standard error for each list as values
    """
    assert all(isinstance(v, Iterable) for v in d.values())
    result = {}
    for k, v in d.items():
        result[k] = mean_and_std_err(v)
    return result
