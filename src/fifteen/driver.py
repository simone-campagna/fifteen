import collections
import enum
import itertools
import math
from weakref import WeakSet

# import pylru

__all__ = [
    "Node",
    "Driver",
    "make_driver",
    "setup_driver",
]


from .config import config_getkey
from .puzzle_type import PuzzleType
from . import utils


Node = collections.namedtuple("Node", "tiles cursor manhattan idx")


class Driver(object):
    REGISTRY = WeakSet()
    MOVES = {
        'u': 0,
        'd': 1,
        'l': 2,
        'r': 3,
    }
    INV_MOVES = {i: m for m, i in MOVES.items()}

    def __init__(self, size):
        k_indices = tuple(range(size * size))

        num_tiles = size * size
        max_tile = num_tiles - 1
        max_tile_pwr10 = 1
        while max_tile_pwr10 < max_tile:
            max_tile_pwr10 *= 10
        self.hash_factors = tuple(max_tile_pwr10 ** i for i in reversed(range(num_tiles)))

        move_up = self.MOVES['u']
        move_down = self.MOVES['d']
        move_left = self.MOVES['l']
        move_right = self.MOVES['r']

        goal_ul = self.make_node(list(range(size * size)), 0, manhattan=0)
        goal_ul_tiles = goal_ul.tiles
        goal_br = self.make_node(list(range(1, size * size)) + [0], size * size - 1, manhattan=0)
        goal_br_tiles = goal_br.tiles
        
        #  UL       BR
        # 
        # 0 1 2    8 7 6
        # 3 4 5 -> 5 4 3
        # 6 7 8    2 1 0
        #
        #
        # 1 2 3    0 8 7
        # 4 5 6 -> 6 5 4
        # 7 8 0    3 2 1
        #
        
        rc2k = {}
        k2rc = []
        refk = []
        irows = [[] for _ in range(size)]
        icols = [[] for _ in range(size)]
        rot180_k = []
        rot180_v = []
        for r in range(size):
            irow = irows[r]
            for c in range(size):
                k = r * size + c
                irow.append(k)
                icols[c].append(k)
                rc2k[(r, c)] = k
                k2rc.append((r, c))
                refk.append(c * size + r)
                r180_k = (size - 1 - r) * size + size - 1 - c
                rot180_k.append(r180_k)
                rot180_v.append(goal_br_tiles[k])
        k_indices = list(range(size * size))
        tiles = k_indices
                
        nghd = [None for _ in k_indices]
        moves = [None for _ in k_indices]
        moves_d = [None for _ in k_indices]
        for r in range(size):
            for c in range(size):
                k = rc2k[(r, c)]
                lst = []
                if r > 0:
                    lst.append((move_up, rc2k[(r - 1, c)]))
                if r < size - 1:
                    lst.append((move_down, rc2k[(r + 1, c)]))
                if c > 0:
                    lst.append((move_left, rc2k[(r, c - 1)]))
                if c < size - 1:
                    lst.append((move_right, rc2k[(r, c + 1)]))
                moves[k] = tuple(lst)
                moves_d[k] = {m: k for m, k in lst if k >= 0}
                nghd[k] = [k for dummy_move, k in lst]
        g_pos_k_ul = {tile: k for k, tile in enumerate(goal_ul_tiles)}
        g_pos_k_br = {tile: k for k, tile in enumerate(goal_br_tiles)}
        g_pos_rc_ul = {tile: k2rc[k] for k, tile in enumerate(goal_ul_tiles)}
        g_pos_rc_br = {tile: k2rc[k] for k, tile in enumerate(goal_br_tiles)}
        
        manhattan_matrix_ul = [[None for _ in tiles] for _ in k_indices]
        manhattan_matrix_br = [[None for _ in tiles] for _ in k_indices]
        for k in k_indices:
            manhattan_matrix_ul[goal_ul_tiles[k]][k] = 0
            manhattan_matrix_br[goal_br_tiles[k]][k] = 0
        for k0, k1 in itertools.combinations(k_indices, 2):
            r0, c0 = k2rc[k0]
            r1, c1 = k2rc[k1]
            distance = abs(r1 - r0) + abs(c1 - c0)
            manhattan_matrix_ul[goal_ul_tiles[k1]][k0] = distance
            manhattan_matrix_ul[goal_ul_tiles[k0]][k1] = distance
            manhattan_matrix_br[goal_br_tiles[k1]][k0] = distance
            manhattan_matrix_br[goal_br_tiles[k0]][k1] = distance

        self.size = size
        self.k_indices = tuple(k_indices)
        self.tiles = tuple(tiles)
        self.rc2k = rc2k
        self.k2rc = k2rc
        self.refk = refk
        self.irows = irows
        self.icols = icols
        self.nghd = tuple(tuple(x) for x in nghd)
        self.moves = tuple(moves)
        self.moves_d = tuple(moves_d)
        self.g_pos_k_ul = g_pos_k_ul
        self.g_pos_k_br = g_pos_k_br
        self.g_pos_k_d = {
            PuzzleType.UL: g_pos_k_ul,
            PuzzleType.BR: g_pos_k_br,
        }
        self.g_pos_rc_ul = g_pos_rc_ul
        self.g_pos_rc_br = g_pos_rc_br
        self.g_pos_rc_d = {
            PuzzleType.UL: g_pos_rc_ul,
            PuzzleType.BR: g_pos_rc_br,
        }
        self.goal_ul = goal_ul
        self.goal_br = goal_br
        self.goal_d = {
            PuzzleType.UL: goal_ul,
            PuzzleType.BR: goal_br,
        }
        self.rot180_k = rot180_k
        self.rot180_v = rot180_v
        self.manhattan_matrix_ul = tuple(tuple(x) for x in manhattan_matrix_ul)
        self.manhattan_matrix_br = tuple(tuple(x) for x in manhattan_matrix_br)
        self.manhattan_matrix_d = {
            PuzzleType.UL: self.manhattan_matrix_ul,
            PuzzleType.BR: self.manhattan_matrix_br,
        }
        self.dual_moves = {
            move_up: move_down,
            move_down: move_up,
            move_left: move_right,
            move_right: move_left,
        }
        self.__get_neighbors = None
        self.goal = None
        self.manhattan_matrix = None
        self.g_pos_k = None
        self.g_pos_rc = None
        self._puzzle_type = None
        self.puzzle_type = config_getkey('puzzle_type')
        self.REGISTRY.add(self)
        self.get_node_idx = lambda node: node[-1]

    def rotate(self, ks):
        rot180_k = self.rot180_k
        return tuple(rot180_k[k] for k in ks)

    @classmethod
    def setup(cls, puzzle_type=None):
        if puzzle_type is None:
            puzzle_type = config_getkey('puzzle_type')
        for instance in cls.REGISTRY:
            instance.puzzle_type = puzzle_type

    @property
    def puzzle_type(self):
        return self._puzzle_type

    @puzzle_type.setter
    def puzzle_type(self, value):
        self.goal = self.goal_d[value]
        self.manhattan_matrix = self.manhattan_matrix_d[value]
        self.g_pos_k = self.g_pos_k_d[value]
        self.g_pos_rc = self.g_pos_rc_d[value]
        self._puzzle_type = value

    def make_node(self, tiles, cursor, manhattan=None, idx=None):
        if manhattan is None:
            manhattan_matrix = self.manhattan_matrix
            manhattan = 0
            for k, cell in enumerate(tiles):
                if cell > 0:
                    manhattan += manhattan_matrix[cell][k]
        if idx is None:
            idx = sum(pi * fi for pi, fi in zip(tiles, self.hash_factors))
        return Node(tiles=tiles, cursor=cursor, manhattan=manhattan, idx=idx)

    def reset_get_neighbors(self):
        if self.__get_neighbors is None:
            return self.make_get_neighbors()
        else:
            self.__get_neighbors.__forbidden_moves__.clear()
            return self.__get_neighbors

    def make_get_neighbors(self):
        forbidden_moves = collections.defaultdict(set)
        self.__get_neighbors = None
        moves = self.moves
        dual_moves = self.dual_moves
        manhattan_matrix = self.manhattan_matrix
        hash_factors = self.hash_factors
        # make_node = self.make_node
        def get_neighbors(node):
            nonlocal moves
            nonlocal dual_moves
            nonlocal manhattan_matrix
            nonlocal forbidden_moves
            # nonlocal make_node
            old_tiles, old_cursor, old_manhattan, old_idx = node
            node_forbidden_moves = forbidden_moves[old_idx]
            for move, new_cursor in moves[old_cursor]:
                if move in node_forbidden_moves:
                    continue
                tile = old_tiles[new_cursor]
                new_tiles = old_tiles[:]
                new_tiles[old_cursor] = tile
                new_tiles[new_cursor] = 0
                new_idx = old_idx - hash_factors[new_cursor] * tile + hash_factors[old_cursor] * tile
                if tile > 0:
                    mh_tile = manhattan_matrix[tile]
                    new_manhattan = old_manhattan - mh_tile[new_cursor] + mh_tile[old_cursor]
                else:
                    new_manhattan = old_manhattan
                neighbor = (new_tiles, new_cursor, new_manhattan, new_idx)
                forbidden_moves[new_idx].add(dual_moves[move])
                if tile < 0:
                    yield neighbor, move, 0
                else:
                    yield neighbor, move, 1
        self.__get_neighbors = get_neighbors
        get_neighbors.__forbidden_moves__ = forbidden_moves
        return get_neighbors

    @property
    def get_neighbors(self):
        if self.__get_neighbors is None:
            self.make_get_neighbors()
        return self.__get_neighbors

    def move(self, node, moves):
        tiles = list(node.tiles)
        old_cursor = node.cursor
        moves_d = self.moves_d
        for move in moves:
            if isinstance(move, str):
                move = self.MOVES[move]
            new_cursor = moves_d[old_cursor][move]
            tiles[old_cursor], tiles[new_cursor] = tiles[new_cursor], tiles[old_cursor]
            old_cursor = new_cursor
        return self.make_node(tiles, old_cursor)

    def get_moves(self, node):
        return tuple(self.moves_d[node.cursor])

    def get_move(self, node_from, node_to, verify=True):
        tiles_from, cursor_from, manhattan_from, idx_from = node_from
        tiles_to, cursor_to, manhattan_to, idx_to = node_to
        inv_moves = {k: m for m, k in self.moves[cursor_from]}
        if verify:
            for x, y in zip(tiles_from, tiles_to):
                if x > 0 and y > 0 and x != y:
                    raise ValueError("no move from {} to {}".format(node_from, node_to))
        return inv_moves[cursor_to]

    def dual_move(self, move):
        return self.dual_moves[move]

    def reflected_k(self, k):
        return self.refk[k]

    def _inversions(self, node):
        tiles = list(node.tiles)
        size = self.size
        num_inversions = 0
        for x, y in itertools.combinations(tiles, 2):
            if x and y and x > y:
                num_inversions += 1
        return num_inversions

    def is_solvable(self, node):
        num_inversions = self._inversions(node)
        k2rc = self.k2rc
        if self.size % 2 == 0:
            r, c = k2rc[node.cursor]
            e_r = r % 2 == 0
            e_num_inversions = num_inversions % 2 == 0
            return e_r != e_num_inversions
        else:
            return num_inversions % 2 == 0

    def is_solved(self, node):
        goal = self.goal
        return node == goal

    def make_hamming_distance(self):
        goal_tiles = self.goal.tiles
        def hamming(node):
            distance = 0
            for tile, g_tile in zip(node.tiles, goal_tiles):
                if tile and tile != g_tile:
                    distance += 1
            return distance
        return hamming
    
    def make_manhattan_distance(self):
        def manhattan(node):
            return node[2]
        return manhattan

    # REM def make_linear_conflicts_distance(self):
    # REM     g_pos_rc = self.g_pos_rc
    # REM     k2rc = self.k2rc
    # REM     rc2k = {v: k for k, v in k2rc}
    # REM     irows = self.irows
    # REM     icols = self.icols
    # REM     def linear_conflicts(node):
    # REM         tiles = node[0]
    # REM         distance = 0
    # REM         for row in irows:
    # REM             for k, idx0 in enumerate(row):
    # REM                 cell0 = tiles[idx0]
    # REM                 if cell0:
    # REM                     r0, c0 = k2rc[idx0]
    # REM                     g_r0, g_c0 = g_pos_rc[cell0]
    # REM                     distance += abs(g_r0 - r0) + abs(g_c0 - c0)
    # REM                     if g_r0 == r0:
    # REM                         for idx1 in row[k + 1:]:
    # REM                             cell1 = tiles[idx1]
    # REM                             if cell1:
    # REM                                 g_r1, g_c1 = g_pos_rc[cell1]
    # REM                                 if g_r1 == g_r0:
    # REM                                     r1, c1 = k2rc[idx1]
    # REM                                     if (c0 < c1 and g_c0 > g_c1) or (c0 > c1 and g_c0 < g_c1):
    # REM                                         # print("lc1: r: k={} {} {} ||| k1={} {} {}".format(
    # REM                                         #     idx0, (r0, c0), cell0,
    # REM                                         #     idx1, (r1, c1), cell1,
    # REM                                         # ))
    # REM                                         distance += 1
    # REM         for col in icols:
    # REM             for k, idx0 in enumerate(col):
    # REM                 cell0 = tiles[idx0]
    # REM                 if cell0:
    # REM                     r0, c0 = k2rc[idx0]
    # REM                     g_r0, g_c0 = g_pos_rc[cell0]
    # REM                     if g_c0 == c0:
    # REM                         for idx1 in col[k + 1:]:
    # REM                             cell1 = tiles[idx1]
    # REM                             if cell1:
    # REM                                 g_r1, g_c1 = g_pos_rc[cell1]
    # REM                                 if g_c1 == g_c0:
    # REM                                     r1, c1 = k2rc[idx1]
    # REM                                     if (r0 < r1 and g_r0 > g_r1) or (r0 > r1 and g_r0 < g_r1):
    # REM                                         # print("lc1: c: k={} {} {} ||| k1={} {} {}".format(
    # REM                                         #     idx0, (r0, c0), cell0,
    # REM                                         #     idx1, (r1, c1), cell1,
    # REM                                         # ))
    # REM                                         distance += 1
    # REM         return distance
    # REM     return linear_conflicts

    def make_linear_conflicts_distance(self):
        k_indices = self.k_indices
        g_pos_rc = self.g_pos_rc
        rc2k = self.rc2k
        k2rc = self.k2rc
        size = self.size
        goal_tiles = self.goal[0]
        lc_set = [None for _ in k_indices]
        for tile, (r, c) in g_pos_rc.items():
            tset = {t: [set(), set()] for t in self.tiles if t > 0}
            rk_indices = [rc2k[(r, c1)] for c1 in range(c + 1, size)]
            for c1 in range(size - 1):
                for c2 in range(c1 + 1, size):
                    t1 = goal_tiles[rc2k[(r, c1)]]
                    t2 = goal_tiles[rc2k[(r, c2)]]
                    if t1 > 0 and t2 > 0:
                        tset[t2][0].add(t1)
            ck_indices = [rc2k[(r1, c)] for r1 in range(r + 1, size)]
            for r1 in range(size - 1):
                for r2 in range(r1 + 1, size):
                    t1 = goal_tiles[rc2k[(r1, c)]]
                    t2 = goal_tiles[rc2k[(r2, c)]]
                    if t1 > 0 and t2 > 0:
                        tset[t2][1].add(t1)
            lc_set[rc2k[(r, c)]] = (rk_indices, ck_indices, tset)

        def linear_conflicts(node):
            tiles = node[0]
            distance = node[2]
            for k, tile in enumerate(tiles):
                if tile > 0:
                    r, c = k2rc[k]
                    rk_indices, ck_indices, tset = lc_set[k]
                    trset, tcset = tset[tile]
                    if trset:
                        for k1 in rk_indices:
                            tile1 = tiles[k1]
                            if tile1 in trset:
                                # print("lc2: r: k={} {} {} ||| k1={} {} {}".format(
                                #     k, self.k2rc[k], tile,
                                #     k1, self.k2rc[k1], tile1,
                                # ))
                                # print("  ", (r, c), k, rk_indices, rset)
                                distance += 1
                    if tcset:
                        for k1 in ck_indices:
                            tile1 = tiles[k1]
                            if tile1 in tcset:
                                # print("lc2: c: k={} {} {} ||| k1={} {} {}".format(
                                #     k, self.k2rc[k], tile,
                                #     k1, self.k2rc[k1], tile1,
                                # ))
                                # print("  ", (r, c), k, ck_indices, cset)
                                distance += 1
            return distance
        return linear_conflicts

    def solve(self, node, *, search_algorithm, heuristic_cost, tracker=None):
        get_neighbors = self.reset_get_neighbors()
        size = self.size
        start_node = node
        goal_node = self.goal
        path = search_algorithm(
            start_node, found=(lambda node: node == goal_node),
            get_neighbors=get_neighbors,
            heuristic_cost=heuristic_cost,
            get_idx=self.get_node_idx,
            tracker=tracker)
        if path is not None:
            moves = []
            get_move = self.get_move
            if path:
                node_from = path[0]
                for node_to in path[1:]:
                    moves.append(get_move(node_from, node_to))
                    node_from = node_to
            return moves

    def make_int_move(self, move):
        if not isinstance(move, int):
            move = self.MOVES[move]
        return move

    def make_str_move(self, move):
        if not isinstance(move, str):
            move = self.INV_MOVES[move]
        return move

    def iter_int_moves(self, moves):
        make_int_move = self.make_int_move
        for move in moves:
            yield make_int_move(move)

    def iter_str_moves(self, moves):
        make_str_move = self.make_str_move
        for move in moves:
            yield make_str_move(move)


_CACHE = {}

def make_driver(size):
    if size not in _CACHE:
        _CACHE[size] = Driver(size)
    return _CACHE[size]


def setup_driver(puzzle_type=None):
    Driver.setup(puzzle_type)
