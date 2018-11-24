#!/usr/bin/env python3

import itertools
import re


__all__ = [
    'Puzzle3',
    'Puzzle8',
    'Puzzle15',
    'puzzle_class',
    'make_puzzle',
    'read_puzzle',
    'write_puzzle',
]

from .config import config_getkey
from .driver import make_driver, Node
from .utils import (
    flatten,
)


class PuzzleMeta(type):
    def __new__(mcls, class_name, class_bases, class_dict):
        cls = super().__new__(mcls, class_name, class_bases, class_dict)
        size = cls.__size__
        cls.__driver__ = make_driver(size)
        digits = max(1, len(str(size * size - 1)))
        cls.__digits__ = digits
        tile_fmt = "{{:>{digits}s}}".format(digits=digits)
        cls.__sfmt__ = '\n'.join(" ".join(tile_fmt for _ in range(size)) for _ in range(size))
        tile_fmt = "{{:>{digits}d}}".format(digits=digits)
        cls.__dfmt__ = '\n'.join(" ".join(tile_fmt for _ in range(size)) for _ in range(size))
        return cls


class PuzzleBase(object):
    def __init__(self, init=None):
        if isinstance(init, Node):
            self.__node = init
        elif init is None:
            self.__node = self.__driver__.goal
        else:
            driver = self.__driver__
            tiles = flatten(init)
            size = self.__size__
            if len(tiles) != size * size:
                raise ValueError("tiles number mismatch: {} != {}".format(len(tiles), size * size))
            if set(tiles) == set(driver.goal.tiles):
                cursor = tiles.index(0)
            else:
                cursor = None
            self.__node = self._make_node(tiles, cursor)

    def __iter__(self):
        yield from self.__node.tiles

    def __len__(self):
        return len(self.__node.tiles)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx = self.rc2k[idx]
        return self.__node.tiles[idx]

    @property
    def node(self):
        return self.__node

    @property
    def tiles(self):
        return self.__node.tiles

    @property
    def cursor(self):
        return self.__node.cursor

    @property
    def cursor_rc(self):
        return self.k2rc[self.__node.cursor]

    @property
    def goal_br(self):
        return self.__driver__.goal_br

    @property
    def goal_ul(self):
        return self.__driver__.goal_ul

    def goal(self):
        return self.__driver__.goal

    @property
    def driver(self):
        return self.__driver__

    @property
    def size(self):
        return self.__size__

    @property
    def rc2k(self):
        return self.__driver__.rc2k

    @property
    def k2rc(self):
        return self.__driver__.k2rc

    @classmethod
    def get_dual_move(cls, move):
        return cls.driver.dual_move(move)

    def __repr__(self):
        return "{}({})".format(
            type(self).__name__, super().__repr__())

    def __str__(self):
        map_tile = self._map_tile
        return self.__sfmt__.format(*[map_tile(t) for t in self.__node.tiles])

    def as_string(self):
        return self.__dfmt__.format(*self.__node.tiles)

    @classmethod
    def _map_tile(cls, tile):
        if tile < 0:
            return '*'
        elif tile == 0:
            return '_' * cls.__digits__
        else:
            return str(tile)

    def get_moves(self):
        return self.__driver__.get_moves(self.__node)

    def is_solvable(self):
        return self.__driver__.is_solvable(self.__node)

    def is_solved(self):
        return self.__driver__.is_solved(self.__node)

    def move(self, moves):
        if isinstance(moves, int):
            moves = [moves]
        driver = self.__driver__
        return self.__class__(driver.move(self.__node, driver.iter_int_moves(moves)))
        
    def _make_node(self, tiles, cursor):
        return self.__driver__.make_node(tiles, cursor)

    def get_move_to(self, puzzle, verify=True):
        node_from = self.node
        node_to = puzzle.node
        return self.__driver__.get_move(node_from, node_to, verify=verify)
        
    def solve(self, *, search_algorithm, heuristic_cost, tracker=None):
        return self.__driver__.solve(
            self.node,
            search_algorithm=search_algorithm,
            heuristic_cost=heuristic_cost,
            tracker=tracker)

    def str_moves(self, moves):
        return ''.join(self.__driver__.iter_str_moves(moves))


_CLASS_CACHE = {}

def puzzle_class(size):
    if size in _CLASS_CACHE:
        return _CLASS_CACHE[size]
    else:
        label = size * size - 1
        cls = PuzzleMeta(
            "Puzzle" + str(label),
            (PuzzleBase,),
            {'__size__': size})
        _CLASS_CACHE[size] = cls
        return cls



Puzzle3 = puzzle_class(2)
Puzzle8 = puzzle_class(3)
Puzzle15 = puzzle_class(4)
            

_RE_SPLIT = re.compile(r"\s*,\s*|\s+")
def make_puzzle(tiles):
    if isinstance(tiles, str):
        tokens = _RE_SPLIT.split(tiles.strip().strip(','))
        tiles = []
        for token in tokens:
            if set(token) == {'_'}:
                tiles.append(0)
            elif token == '*':
                tiles.append(-1)
            else:
                tiles.append(int(token))
    else:
        tiles = flatten(tiles)
    if not tiles:
        raise ValueError("empty puzzle not allowed")
    size = int(len(tiles) ** 0.5)
    if size * size != len(tiles):
        raise ValueError("bad tiles number {}: not a perfect square".format(len(tiles)))
    cls = puzzle_class(size)
    return cls(tiles)


def write_puzzle(puzzle, file):
    if isinstance(file, str):
        with open(file, "w") as f_handle:
            return write_puzzle(puzzle.as_string(), f_handle)
    file.write(str(puzzle) + '\n')


def read_puzzle(file):
    if isinstance(file, str):
        with open(file, "r") as f_handle:
            return read_puzzle(f_handle)
    return make_puzzle(file.read())
