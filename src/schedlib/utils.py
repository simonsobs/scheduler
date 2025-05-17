from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from functools import reduce
from contextlib import contextmanager
from scipy import interpolate
from collections.abc import Iterable
from jax.tree_util import SequenceKey, DictKey
import fnmatch

second = 1
minute = 60 # second
hour = 60 * minute
day = 24 * hour
sidereal_day = 0.997269566 * day
deg = np.pi / 180

def str2ctime(time_str):
    ctime = (pd.Timestamp(time_str) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    return ctime

def str2datetime(time_str):
    ctime = str2ctime(time_str)
    return datetime.fromtimestamp(int(ctime), tz=timezone.utc)

def datetime2str(dtime):
    return dtime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

def ct2dt(ctime):
    if isinstance(ctime, list):
        return [datetime.fromtimestamp(t).astimezone(timezone.utc) for t in ctime]
    else:
        try:
            return datetime.fromtimestamp(ctime).astimezone(timezone.utc)
        except TypeError:
            raise ValueError(f"ctime should be int, float or iterable, not {type(ctime)}")

def dt2ct(dtime):
    if isinstance(dtime, Iterable):
        return np.array([float(d.timestamp()) for d in dtime])
    else:
        try:
            return float(dtime.timestamp())
        except TypeError:
            raise ValueError(f"dtime should be datetime or iterable, not {type(dtime)}")

def mask2ranges(mask):
    """
    Convert a boolean mask to a set of ranges.

    This function takes a boolean mask and returns an array of start and end indices
    that define the true regions of the mask. Note that the end index is exclusive.

    Parameters
    ----------
    mask : np.ndarray
        A 1-D boolean mask.

    Returns
    -------
    ranges : np.ndarray
        2-D array of start and end indices for the true regions of the mask.

    Examples
    --------
    >>> mask = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0])
    >>> mask2ranges(mask)
    array([[2, 5],
           [7, 11]])
    """
    # handle a bunch of special cases first
    if len(mask) == 0: return np.empty((0, 2), dtype=int)
    if np.all(mask): return np.array([[0, len(mask)]], dtype=int)
    if not np.any(mask): return np.empty((0, 2), dtype=int)
    # general case
    bounds = np.where(np.diff(mask) != 0)[0] + 1  # 1 to n-1
    bounds = np.concatenate(([0], bounds, [len(mask)]))  # 0 to n
    bounds = np.vstack((bounds[:-1], bounds[1:])).T # [[(0 to n-1), (1 to n)], ...]
    return bounds[mask[bounds[:, 0]] == 1]

def ranges2mask(ranges, imax):
    mask = np.zeros(imax, dtype=bool)
    for i_left, i_right in ranges:
        mask[i_left:i_right] = True
    return mask

def ranges_pad(ranges, pad, imax):
    """pad each range and merge overlapping ranges"""
    ranges = ranges + np.array([-pad, pad])
    ranges = np.clip(ranges, 0, imax)
    if len(ranges) < 2: return ranges
    # merge overlapping ranges
    return np.array(reduce(lambda l, r: l[:-1] + [[l[-1][0], r[1]]] if l[-1][-1] >= r[0] else l + [r],
                           ranges[1:], [ranges[0]]))

def ranges_complement(ranges, imax):
    """return the complement ranges"""
    return mask2ranges(~ranges2mask(ranges, imax))

def ranges_contain(ranges, x):
    """is x within the ranges"""
    return np.any((x >= ranges[:, 0]) * (x <= ranges[:, 1]))

def ranges_intersect(ranges, r):
    """find intersection between one range versus a set of ranges"""
    return np.any((r[0] <= ranges[:, 1]) * (r[1] >= ranges[:, 0]))

# convenience wrapper for interpolation: numpy-like scipy interpolate
def interp_extra(x_new, x, y, fill_value='extrapolate'):
    """interpolate with extrapolation"""
    return interpolate.interp1d(x, y, fill_value=fill_value, bounds_error=False, kind='cubic', assume_sorted=False)(x_new)

def interp_bounded(x_new, x, y):
    """interpolate with bounded extrapolation"""
    return interpolate.interp1d(x, y, fill_value=(y[0], y[-1]), bounds_error=False, kind='cubic', assume_sorted=False)(x_new)

def within_bound(x, bounds):
    """return a boolean mask indicating whether x is within the bound"""
    return (x >= bounds[0]) * (x <= bounds[1])

# ====================
# Random utilities
# ====================
# In order to produce reproducible schedules, we need to be careful
# about random number generator state, especially if the service runs
# in a server setting, because the state of our random number will
# then depends on the order of requests received which is not desirable
# for reproducibility. We use the following pattern to ensure
# reproducibility: always use these wrapper functions to generate
# random numbers, and always pass a PRNGKey object to them.
# It should be obvious where the inspirataion comes from.

class PRNGKey:
    """motivation: reduce side effects: same key should always produce same
    result and not affect other keys"""
    def __init__(self, key):
        self.key = key  # need to be hashable
        self.state = None

    @contextmanager
    def set_state(self):
        old_state = np.random.get_state()
        try:
            if self.state is None:
                seed = hash(self.key) % (2**32 - 1)
                np.random.seed(seed)
                self.state = np.random.get_state()
            else:
                np.random.set_state(self.state)
            yield
        finally:
            np.random.set_state(old_state)

    def split(self, n=2):
        """split the key into n keys. Used tuple to avoid collisions"""
        return [PRNGKey((self.key, i)) for i in range(n)]

# now we can make some wrappers for common numpy.random functions
def uniform(key: PRNGKey, low=0.0, high=1.0, size=None):
    with key.set_state():
        return np.random.uniform(low, high, size)

def daily_static_key(t: datetime):
    return PRNGKey((t.year, t.month, t.day))

def pprint(seq, **kwargs):
    """pretty print"""
    print(pformat(seq, **kwargs))

def pformat(seq, **kwargs):
    """pretty format"""
    from unittest.mock import patch
    from dataclasses import is_dataclass
    from equinox import tree_pformat
    from schedlib.core import Block
    # force is_dataclass to return False for Block
    def _new_isdataclass(fun):
        def wrapper(obj):
            if isinstance(obj, Block): return False
            else: return fun(obj)
        return wrapper
    with patch("dataclasses.is_dataclass", _new_isdataclass(is_dataclass)):
        return tree_pformat(seq, **kwargs)

# ====================
# path related
# ====================

def path2key(path, ignore_seqkey=False):
    """convert a path (used in tree_util.tree_map_with_path) to a dot-separated key
    
    Parameters
    ----------
    path: a list of SequenceKey or DictKey
    ignore_array: if True, ignore the SequencyKey index in the path 

    Returns
    -------
    key: a string of dot-separated keys

    """
    keys = []
    for p in path:
        if isinstance(p, SequenceKey):
            if not ignore_seqkey:
                keys.append(p.idx)
        elif isinstance(p, DictKey):
            keys.append(p.key)
        else:
            raise ValueError(f"unknown path type {type(p)}")
    return ".".join([str(k) for k in keys])

def match_query(path, query):
    """in order for a query to match with a path, it can
    satisfy the following: 
    1. the query is a substring of the path
    2. the query is a glob pattern that matches the path
    3. if the query is a comma-separated list of multiple queries, 
    any of them meeting comdition 1 and 2 will return True
    """
    key = path2key(path)
    # first match the constraint to key
    queires = query.split(",")
    for q in queires:
        if q in key: 
            return True
        if fnmatch.fnmatch(key, q): 
            return True
    return False

def nested_update(dictionary, update_dict, new_keys_allowed=True):
    """update a nested dictionary recursively but
    never add new keys"""
    if update_dict is None: return dictionary
    for key, value in update_dict.items():
        if key in dictionary and isinstance(dictionary[key], dict) and isinstance(value, dict):
            nested_update(dictionary[key], value)
        else:
            # optional to prevent new keys to be added with new_keys_allowed=False
            if key in dictionary or new_keys_allowed:
                dictionary[key] = value
    return dictionary

def round_robin(lists):
    """
    Iterate over several lists in a round-robin fashion.

    This generator function takes a list of lists as input and yields elements from each list
    in a round-robin order, cycling through the lists until all elements are exhausted.

    Parameters
    ----------
    lists : list of lists
        A list of lists from which elements will be yielded in a round-robin manner.

    Yields
    ------
    element :
        The next element in the round-robin sequence. The type of `element` depends on the
        content of the input `lists`.

    Examples
    --------
    >>> lists = [[1, 2, 3], ['a', 'b'], [10.1, 10.2]]
    >>> for element in round_robin(lists):
    ...     print(element)
    ...
    1
    a
    10.1
    2
    b
    10.2
    3

    Notes
    -----
    - The function cycles through each list, yielding one element at a time from each list, before
      moving to the next list in the sequence.
    - If the lists are of unequal length, the function will continue cycling through the shorter lists
      until all lists are fully exhausted.
    - The function internally manages the index of each list to keep track of the next element to yield.
      Once all elements from all lists have been yielded, the generator stops.

    """
    n = len(lists)  # Number of lists
    idxs = [0] * n  # Index tracker for each list
    while True:
        for i in range(n):
            if idxs[i] < len(lists[i]):
                yield lists[i][idxs[i]]  # Yield the next element from the current list
                idxs[i] += 1
            elif all(idxs[i] >= len(lists[i]) for i in range(n)):
                return
            else:
                continue  # Move to the next list if the current one is exhausted


# ------------------
# logging utils
# ------------------

def init_logger(name):
    import logging, sys
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s ')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def set_logging_level(level=2):
    import logging
    try:
        level = {
            1: logging.DEBUG,
            2: logging.INFO,
            3: logging.WARNING,
            4: logging.ERROR,
            5: logging.CRITICAL,
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }[level]
    except KeyError: pass

    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith("schedlib"):
            logging.getLogger(logger_name).setLevel(level)

def set_verbosity(verbosity=2):
    import logging

    verbosity = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }[verbosity]

    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith("schedlib"):
            logging.getLogger(logger_name).setLevel(verbosity)
