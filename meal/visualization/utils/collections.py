import cProfile
import io
import pstats
from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def append_dictionaries(dictionaries):
    """
    Append many dictionaries with numbers as values into one dictionary with lists as values.

    {a: 1, b: 2}, {a: 3, b: 0}  ->  {a: [1, 3], b: [2, 0]}
    """
    assert all(
        set(d.keys()) == set(dictionaries[0].keys()) for d in dictionaries
    ), "All key sets are the same across all dicts"
    final_dict = defaultdict(list)
    for d in dictionaries:
        for k, v in d.items():
            final_dict[k].append(v)
    return dict(final_dict)


def merge_dictionaries(dictionaries):
    """
    Merge many dictionaries by extending them to one another.
    {a: [1, 7], b: [2, 5]}, {a: [3], b: [0]}  ->  {a: [1, 7, 3], b: [2, 5, 0]}
    """
    assert all(
        set(d.keys()) == set(dictionaries[0].keys()) for d in dictionaries
    ), "All key sets are the same across all dicts"
    final_dict = defaultdict(list)
    for d in dictionaries:
        for k, v in d.items():
            final_dict[k].extend(v)
    return dict(final_dict)


def rm_idx_from_dict(d, idx):
    """
    Takes in a dictionary with lists as values, and returns
    a dictionary with lists as values, but containing
    only the desired index

    NOTE: this is a MUTATING METHOD, returns the POPPED IDX
    """
    assert all(isinstance(v, Iterable) for v in d.values())
    new_d = {}
    for k, v in d.items():
        new_d[k] = [d[k].pop(idx)]
    return new_d


def take_indexes_from_dict(d, indices, keys_to_ignore=[]):
    """
    Takes in a dictionary with lists as values, and returns
    a dictionary with lists as values, but with subsampled indices
    based on the `indices` input
    """
    assert all(isinstance(v, Iterable) for v in d.values())
    new_d = {}
    for k, v in d.items():
        if k in keys_to_ignore:
            continue
        new_d[k] = np.take(d[k], indices)
    return new_d


def profile(fnc):
    """A decorator that uses cProfile to profile a function (from https://osf.io/upav8/)"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class OvercookedException(Exception):
    pass


def is_iterable(obj):
    return isinstance(obj, Iterable)
