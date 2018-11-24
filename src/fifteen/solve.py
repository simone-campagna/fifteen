import bz2
import collections
import functools
import itertools
import json
import pickle
import os

from .config import (
    config_getkey,
)
from .disjoint_patterns import (
    dpdb_make_heuristic,
    dpdb_iter_infos,
    dpdb_list,
    dpdb_get_db,
)
from .driver import make_driver
from .search_algorithm import (
    get_algorithms,
    get_algorithm,
)
from .utils import UNDEFINED


__all__ = [
    'get_solve_defaults',
    'get_algorithms',
    'get_algorithm',
    'get_heuristics',
    'get_heuristic',
]


HEURISTICS = [
    'hamming',
    'manhattan',
    'linear_conflicts',
]


def get_heuristic(heuristic, size, default=UNDEFINED):
    if isinstance(heuristic, list):
        heuristic = tuple(heuristic)
    puzzle_type = config_getkey('puzzle_type')
    reflection = config_getkey('dpdb.reflection')
    memdb = config_getkey('dpdb.memdb')
    if memdb is None:
        memdb = size <= 4
    return _impl_get_heuristic(heuristic, size, puzzle_type, reflection, memdb, default)


@functools.lru_cache(maxsize=1024)
def _impl_get_heuristic(heuristic, size, puzzle_type, reflection, memdb, default):
    heuristics = list(_iter_heuristics(heuristic, size, puzzle_type, reflection, memdb, default=default))
    if len(heuristics) == 1:
        return heuristics[0]
    elif len(heuristics) == 0:
        raise ValueError("heuristic {} not found".format(heuristic))
    else:
        return lambda puzzle: max(hfun(puzzle) for hfun in heuristics)


def _iter_heuristics(heuristic, size, puzzle_type, reflection, memdb, default=UNDEFINED):
    driver = make_driver(size)
    if callable(heuristic):
        yield heuristic
    elif (not isinstance(heuristic, str)) and isinstance(heuristic, collections.Sequence):
        for h in heuristic:
            for hfun in _iter_heuristics(h, size, puzzle_type, reflection, memdb, default=UNDEFINED):
                if hfun is not None:
                    yield hfun
    elif '+' in heuristic:
        yield from _iter_heuristics(tuple(heuristic.split('+')), size, puzzle_type, reflection, memdb, default=default)
    elif heuristic in {'h', 'hamming'}:
        yield driver.make_hamming_distance()
    elif heuristic in {'m', 'mh', 'manhattan'}:
        yield driver.make_manhattan_distance()
    elif heuristic in {'lc', 'linear_conflicts'}:
        yield driver.make_linear_conflicts_distance()
    elif heuristic in {'dp', 'disjoint_patterns'}:
        yield dpdb_make_heuristic(size, puzzle_type=puzzle_type, reflection=reflection, memdb=memdb)
    elif heuristic in {'dp-*', 'disjoint_patterns-*'}:
        yield from _iter_heuristics('dp-{size}:*', size, puzzle_type, reflection, memdb, default=default)
    elif heuristic.startswith('dp-{size}:'.format(size=size)) or heuristic.startswith('disjoint_patterns-{size}:'.format(size=size)):
        label = heuristic.split(':', 1)[1]
        if label == '*':
            label = None
        for dpdb_info in dpdb_iter_infos(size=size, label=label):
            yield dpdb_make_heuristic(dpdb_info.size, label=dpdb_info.label, puzzle_type=puzzle_type, reflection=reflection, memdb=memdb)
    else:
        if default is UNDEFINED:
            raise KeyError(heuristic)
        else:
            yield default


def get_heuristics(size=None):
    yield from HEURISTICS
    num_disjoint_patterns = 0
    for dpdb_info in dpdb_list():
        if size is None or size == dpdb_info.size:
            dpdb = dpdb_get_db(dpdb_info)
            if dpdb.cache_exists():
                yield "disjoint_patterns-{size}:{label}".format(size=dpdb_info.size, label=dpdb_info.label)
                num_disjoint_patterns += 1
    if num_disjoint_patterns:
        yield "disjoint_patterns"
 

def get_solve_defaults(puzzle, *, algorithm=None, heuristic=None):
    if algorithm is None:
        if puzzle.size < 4:
            algorithm = 'a_star'
        else:
            algorithm = 'ida_star'
    if not heuristic:
        size = puzzle.size
        hfun = dpdb_make_heuristic(size)
        if hfun is None:
            heuristic = 'lc'
        else:
            heuristic = 'dp'
    return algorithm, heuristic


def solve(puzzle, *, algorithm=None, heuristic=None, tracker=None):
    algorithm, heuristic = get_solve_defaults(puzzle, algorithm=algorithm, heuristic=heuristic)
    if callable(algorithm):
        solver_function = algorithm
    else:
        solver_function = get_algorithm(algorithm)
    if callable(heuristic):
        heuristic_cost = heuristic
    else:
        heuristic_cost = get_heuristic(heuristic, puzzle.size, default=None)
        if heuristic_cost is None:
            raise ValueError("unknown heuristic {}".format(heuristic))
    return puzzle.solve(search_algorithm=solver_function, heuristic_cost=heuristic_cost, tracker=tracker)
