import bz2 as zmodule
import collections
import datetime
import enum
import functools
import glob
# import gzip as zmodule
import itertools
import json
import logging
import os
import sys
import time

from concurrent.futures import ProcessPoolExecutor

try:
    import ujson
except ImportError:
    ujson = json

import sqlalchemy as sqla
from sqlalchemy.sql import func as sqla_func

from . import search_algorithm
from .driver import Driver, make_driver
from .puzzle_type import PuzzleType, rot_puzzle_type
from .config import config_getkey
from .utils import (
    UNDEFINED,
    grouper, ProgressBar, round_robin, timing,
)


__all__ = [
    'dpdb_get_dbdir',
    'dpdb_set_dbdir',
    'dpdb_list_meta_filenames',
    'dpdb_read_meta',
    'dpdb_write_meta',
    'dpdb_setup',
    'dpdb_get_dbs',
    'DpdbInfo',
    'dpdb_clear',
    'dpdb_register',
    'dpdb_get_info',
    'dpdb_get_db',
    'dpdb_get_memdb',
    'dpdb_list',
    'DpdbInitMode',
    'dpdb_create',
    'dpdb_make_heuristic',
]


LOG = logging.getLogger(__name__)
DEFAULT_DBDIR = os.path.join(os.path.dirname(__file__), "databases")
DBDIR = DEFAULT_DBDIR


def dpdb_get_dbdir():
    global DBDIR
    return DBDIR


def dpdb_set_dbdir(dbdir=None):
    global DBDIR
    global DEFAULT_DBDIR
    if dbdir is None:
        dbdir = DEFAULT_DBDIR
    DBDIR = os.path.normpath(os.path.abspath(dbdir))


def dpdb_list_meta_filenames(dbdir=None):
    if dbdir is None:
        dbdir = dpdb_get_dbdir()
    for meta_filename in glob.glob(os.path.join(dbdir, "*.meta")):
        yield meta_filename


def dpdb_read_meta(dpdb_meta_filename):
    with open(dpdb_meta_filename, "r") as fh:
        return DpdbInfo(**ujson.load(fh))


def dpdb_write_meta(dpdb_info, dpdb_meta_filename=None):
    dct = dpdb_info.as_dict()
    if dpdb_meta_filename is None:
        dbdir = dpdb_get_dbdir()
        dpdb_meta_filename = os.path.join(dbdir, "dp.{puzzle_type}.{size}:{label}.meta".format(**dct))
    with open(dpdb_meta_filename, "w") as fh:
        fh.write(ujson.dumps(dct, sort_keys=True, indent=4) + '\n')


def dpdb_setup(dbdir=UNDEFINED):
    if dbdir is UNDEFINED:
        dbdir = config_getkey('dpdb.dbdir')
    dpdb_clear()
    dpdb_set_dbdir(dbdir)
    for dpdb_meta_filename in dpdb_list_meta_filenames():
        dpdb_info = dpdb_read_meta(dpdb_meta_filename)
        dpdb_register(dpdb_info)


def dpdb_rotate_info(dpdb_info):
    dpdb = Dpdb(dpdb_info)
    rot_dpdb_info = dpdb.rotate_cache()
    dpdb_register(rot_dpdb_info)
    dpdb_write_meta(rot_dpdb_info)
        

def dpdb_get_dbs(dbkeys=None, puzzle_type=None):
    dbd = collections.OrderedDict()
    for dpdb_info in dpdb_list(puzzle_type=puzzle_type):
        dbkey = '{}:{}'.format(dpdb_info.size, dpdb_info.label)
        dbd[dbkey] = dpdb_info
    if dbkeys is not None:
        dbkeys = list(dbkeys)
    else:
        dbkeys = list(dbd)
    dbs = collections.OrderedDict()
    for dbkey in dbkeys:
        dbs[dbkey] = dbd[dbkey]
    return dbs


class DpdbInfo(collections.namedtuple('DpdbInfoBase', 'puzzle_type size label patterns filenames')):
    def __new__(cls, puzzle_type, size, label, patterns):
        if isinstance(puzzle_type, str):
            puzzle_type = getattr(PuzzleType, puzzle_type)
        if isinstance(patterns, str):
            data = [int(i) for i in patterns.split()]
            dct = collections.defaultdict(list)
            for i, v in enumerate(data):
                dct[v].append(i)
            items = sorted(dct.items(), key=lambda x: x[0])
            patterns = [item[1] for item in items[1:]]
        patterns = tuple(tuple(pattern) for pattern in patterns)
        goal_tr = list(range(size * size))
        vlist = []
        for pattern in patterns:
            for idx in pattern:
                vlist.append(goal_tr[idx])
        if len(vlist) != len(set(vlist)):
            raise ValueError("invalid patterns {} for size {}".format(patterns, size))
        fmt_filename = os.path.join(dpdb_get_dbdir(), 'dp.{puzzle_type.name}.{size}:{label}.cache.{pattern_id}')
        filenames = tuple(fmt_filename.format(puzzle_type=puzzle_type, size=size, label=label, pattern_id=pattern_id) for pattern_id, _ in enumerate(patterns))
        return super().__new__(
            cls,
            puzzle_type=puzzle_type,
            size=size,
            label=label,
            patterns=patterns,
            filenames=filenames)
 
    def fqname(self):
        return "{pt}-{size}:{label}".format(
            pt=self.puzzle_type.name,
            size=self.size,
            label=self.label)

    def rotated(self):
        driver = make_driver(self.size)
        rot_patterns = tuple(driver.rotate(pattern) for pattern in self.patterns)
        return self.__class__(
            puzzle_type=rot_puzzle_type(self.puzzle_type),
            size=self.size,
            label=self.label,
            patterns=rot_patterns)

    def exists(self):
        return all(os.path.exists(filename) for filename in self.filenames)

    def as_dict(self):
        return {
            "puzzle_type": self.puzzle_type.name,
            "size": self.size,
            "label": self.label,
            "patterns": self.patterns}


class Dpdb(object):
    def __init__(self, dpdb_info, *, max_buffer_size=500, key_type=None, db_filename=None):
        # TODO rimuovere key_type (sempre 'string')
        self._dpdb_info = dpdb_info
        driver = make_driver(dpdb_info.size)
        if db_filename is None:
            pt = driver.puzzle_type.name
            db_filename = os.path.join(dpdb_get_dbdir(), 'dp.{pt}.{di.size}:{di.label}.sqlite'.format(pt=pt, di=dpdb_info))
        else:
            db_filename = os.path.abspath(db_filename)
        self._db_filename = db_filename
        self._uri = "sqlite:///" + db_filename
        self._engine = sqla.create_engine(self._uri)
        metadata = sqla.MetaData()
        size = self._dpdb_info.size

        puzzle_type = self._dpdb_info.puzzle_type
        r_puzzle_type = rot_puzzle_type(puzzle_type)

        max_pattern_len = max(len(pattern) for pattern in dpdb_info.patterns)
        max_index = dpdb_info.size ** 2
        max_index_pwr10 = 1
        while max_index_pwr10 < max_index:
            max_index_pwr10 *= 10
        if key_type is None:
            if max_index_pwr10 ** max_pattern_len <= 2 ** 60:
                key_type = 'int'
            else:
                key_type = 'string'
        if key_type == 'int':
            f_pattern_start_id = [max_index_pwr10 ** i for i in range(max_pattern_len - 1, -1, -1)]

            def make_key(pattern_id, pattern_start_id):
                result = 0
                for pi, fi in zip(pattern_start_id, f_pattern_start_id):
                    result += pi * fi
                return result

            sqla_type = sqla.BigInteger
        else:
            def make_key(pattern_id, pattern_start_id):
                return ','.join(str(p) for p in pattern_start_id)

            sqla_type = sqla.String

        self.make_key = make_key

        self._cursor_columns = tuple(
            sqla.Column("cursor_{}".format(cursor), sqla.Integer, default=0, primary_key=False)
            for cursor in range(size * size))
        self._costs_table = sqla.Table(
            "costs_{size}_{label}".format(size=self._dpdb_info.size, label=self._dpdb_info.label),
            metadata,
            sqla.Column('pattern_id', sqla.Integer,
                        primary_key=True, autoincrement=False),
            sqla.Column('pattern_start_id', sqla_type,
                        primary_key=True, autoincrement=False),
            sqla.Index("p_id_index", "pattern_id", "pattern_start_id"),
            *self._cursor_columns,
        )
        self._buffer = []
        self._max_buffer_size = max_buffer_size
        self._key_type = key_type
        self._connection = None

    def rotated(self):
        return self.__class__(
            dpdb_info=self._dpdb_info.rotated(),
            max_buffer_size=self._max_buffer_size,
            key_type=self._key_type,
            db_filename=self._db_filename)

    def merge(self, *db_filenames):
        for db_filename in db_filenames:
            with timing("dpdb[{}] merge {} -> {}".format(self.dpdb_info.fqname(), db_filename, self._engine)):
                dpdb = self.__class__(self._dpdb_info, db_filename=db_filename)
                self.put_costs(dpdb.get_all_costs())

    def export_cache(self):
        if not self.db_exists():
            raise ValueError("missing db {}".format(self._uri))
        dpdb_info = self._dpdb_info
        with timing("dpdb[{}] export_cache".format(dpdb_info.fqname())):
            for pattern_id, dummy_pattern_filename in enumerate(dpdb_info.filenames):
                pattern_costs_data = []
                for dummy_pattern_id, pattern_start_id, *cost_data in self.get_pattern_costs(pattern_id):
                    pattern_costs_data.append((pattern_start_id, tuple(cost_data)))
                self._write_cache_file(dpdb_info, pattern_id, pattern_costs_data)

    @classmethod
    def _read_cache_file(cls, dpdb_info, pattern_id):
        pattern_filename = dpdb_info.filenames[pattern_id]
        with timing("dpdb[{}] reading cache file {}".format(dpdb_info.fqname(), pattern_filename)):
            with zmodule.open(pattern_filename, "rt") as fh:
                return ujson.load(fh)

    @classmethod
    def _write_cache_file(cls, dpdb_info, pattern_id, pattern_costs_data):
        pattern_filename = dpdb_info.filenames[pattern_id]
        with timing("dpdb[{}] writing cache file {}".format(dpdb_info.fqname(), pattern_filename)):
            with zmodule.open(pattern_filename, "wb") as fh:
                fh.write(bytes(ujson.dumps(pattern_costs_data, ensure_ascii=True), 'ascii'))

    def import_cache(self):
        dpdb_info = self._dpdb_info
        if not self.cache_exists():
            raise ValueError("missing cache {}".format(dpdb_info.filenames))
        with timing("dpdb[{}] import_cache".format(dpdb_info.fqname())):
            engine = self._engine
            table = self._costs_table
            iquery = table.insert()
            costs = []
            for pattern_id, pattern_filename in enumerate(dpdb_info.filenames):
                pattern_costs_data = self._read_cache_file(dpdb_info, pattern_id)
                with engine.begin() as connection:
                    for pattern_start_id, cost_data in pattern_cost_data:
                        data = (pattern_id, pattern_start_id) + tuple(cost_data)
                        connection.execute(iquery.values(data))
            
    def rotate_cache(self):
        rot_dpdb = self.rotated()
        dpdb_info = self._dpdb_info
        with timing("dpdb[{}] rotate_cache".format(dpdb_info.fqname())):
            rot_dpdb_info = rot_dpdb.dpdb_info
            make_key = self.make_key
            rot_make_key = rot_dpdb.make_key
            size = dpdb_info.size
            driver = make_driver(size)
            rot180_k = driver.rot180_k
            rot180_v = driver.rot180_v
            rot180_kv = [rot180_k[v] for v in rot180_v]
            rot_costs = []
            positions = list(range(size * size))
            for pattern_id, (pattern, rot_pattern) in enumerate(zip(dpdb_info.patterns, rot_dpdb_info.patterns)):
                pindex = list(range(len(pattern)))
                pindex_permutations = list(itertools.permutations(pindex, len(pattern)))
                pattern_costs = dict(self._read_cache_file(dpdb_info, pattern_id))
                rot_pattern_cost_data = []
                with timing("dpdb[{}] rotate pattern[{}]".format(dpdb_info.fqname(), pattern_id)):
                    for pattern_position in itertools.combinations(positions, len(pattern)):
                        for pindex_permutation in pindex_permutations:
                            pattern_start = tuple(pattern_position[pidx] for pidx in pindex_permutation)
                            pattern_start_id = make_key(pattern_id, pattern_start)
                            rot_pattern_start_id = rot_make_key(pattern_id, [rot180_k[x] for x in pattern_start])
                            cost_data = pattern_costs.pop(pattern_start_id)
                            rot_cost_data = [cost_data[rk] for rk in rot180_k]
                            rot_pattern_cost_data.append((rot_pattern_start_id, rot_cost_data))
                            # print("{} -> {}".format(pattern_start_id, rot_pattern_start_id))
                self._write_cache_file(rot_dpdb_info, pattern_id, rot_pattern_cost_data)
        return rot_dpdb_info

    @property
    def dpdb_info(self):
        return self._dpdb_info

    def cache_exists(self):
        return self._dpdb_info.exists()

    def load_cache(self):
        dpdb_info = self._dpdb_info
        if self.cache_exists():
            with timing("dpdb[{}] load_cache".format(dpdb_info.fqname())):
                costs = []
                for pattern_id, pattern_filename in enumerate(dpdb_info.filenames):
                    costs.append(dict(self._read_cache_file(dpdb_info, pattern_id)))
                return costs
        else:
            raise ValueError("missing cache {}".format(dpdb_info.filenames))

    def db_exists(self):
        for tablename in [self._costs_table.name]:
            if not self._engine.has_table(tablename):
                return False
        return True

    def drop(self):
        with self._engine.begin() as connection:
            self._costs_table.drop(bind=connection, checkfirst=True)

    def create(self):
        with self._engine.begin() as connection:
            self._costs_table.create(bind=connection, checkfirst=True)

    def insert_bulk(self, groups):
        table = self._costs_table
        iquery = table.insert()
        with self._engine.begin() as connection:
            for pattern_id, pattern_start_id, costs in groups:
                vdata = {'pattern_id': pattern_id, 'pattern_start_id': pattern_start_id}
                for cursor, cost in costs.items():
                    vdata[self._cursor_columns[cursor].name] = cost
                connection.execute(iquery.values(vdata))

    def buffer_costs(self, pattern_id, pattern_start_id, costs):
        self._buffer.append((pattern_id, pattern_start_id, costs))
        if len(self._buffer) >= self._max_buffer_size:
            self.buffer_flush()

    def buffer_flush(self):
        if self._buffer:
            self.insert_bulk(self._buffer)
            self._buffer.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.buffer_flush()
        return False
            
    def _setup_conn(self):
        self._connection = self._engine.begin()

    def get_cost(self, pattern_id, pattern_start_id, cursor):
        t = self._costs_table
        with self._engine.begin() as connection:
            col = self._cursor_columns[cursor]
            query = sqla.select([col]).where(sqla.and_(t.c.pattern_id == pattern_id, t.c.pattern_start_id == pattern_start_id))
            return connection.execute(query).fetchone()[0]

    def get_cost_bulk(self, cursor, p_ids):
        t = self._costs_table
        if p_ids:
            where_condition = sqla.or_(*[sqla.and_(t.c.pattern_id == p_id, t.c.pattern_start_id == ps_id) for p_id, ps_id in p_ids])
            col = self._cursor_columns[cursor]
            query = sqla.select([sqla_func.sum(col)]).where(where_condition)
            with self._engine.begin() as connection:
                return connection.execute(query).fetchone()[0]

    def put_costs(self, costs):
        engine = self._engine
        table = self._costs_table
        iquery = table.insert()
        with timing("dpdb[{}] put_costs".format(self.dpdb_info.fqname())):
            with engine.begin() as connection:
                for cost_data in costs:
                    connection.execute(iquery.values(cost_data))
            
    def get_pattern_costs(self, pattern_id):
        engine = self._engine
        table = self._costs_table
        costs = []
        with timing("dpdb[{}] get_pattern_costs[{}]".format(self.dpdb_info.fqname(), pattern_id)):
            with engine.begin() as connection:
                query = sqla.select([table]).where(table.c.pattern_id == pattern_id)
                for result in connection.execute(query):
                    costs.append(tuple(result))
        return costs

    def get_all_costs(self):
        engine = self._engine
        table = self._costs_table
        costs = []
        with timing("dpdb[{}] get_all_costs".format(self.dpdb_info.fqname())):
            with engine.begin() as connection:
                query = sqla.select([table])
                for result in connection.execute(query):
                    costs.append(tuple(result))
        return costs

    def get_records(self, pattern_id):
        t = self._costs_table
        with self._engine.begin() as connection:
            query = sqla.select([t.c.pattern_start_id]).where(t.c.pattern_id == pattern_id)
            for entry in connection.execute(query):
                yield entry[0]

    def get_costs(self, pattern_id, pattern_start_id):
        t = self._costs_table
        with self._engine.begin() as connection:
            query = sqla.select(self._cursor_columns).where(sqla.and_(t.c.pattern_id == pattern_id, t.c.pattern_start_id == pattern_start_id))
            result = connection.execute(query).fetchone()
            if result:
                costs_d = {}
                for cursor, column in enumerate(self._cursor_columns):
                    costs_d[cursor] = result[column.name]
                return costs_d
            else:
                return None
        

DISJOINT_PATTERNS_DB = collections.defaultdict(collections.OrderedDict)


def dpdb_clear():
    global DISJOINT_PATTERNS_DB
    DISJOINT_PATTERNS_DB.clear()

    
def dpdb_register(dpdb_info):
    global DISJOINT_PATTERNS_DB
    dkey = (dpdb_info.puzzle_type, dpdb_info.size)
    dct = DISJOINT_PATTERNS_DB[dkey]
    label = dpdb_info.label
    dct[label] = dpdb_info
    dct.move_to_end(label, last=False)
    return dpdb_info


def dpdb_iter_infos(size, label=None, puzzle_type=None):
    global DISJOINT_PATTERNS_DB
    if puzzle_type is None:
        puzzle_type = config_getkey("puzzle_type")
    dkey = (puzzle_type, size)
    if dkey not in DISJOINT_PATTERNS_DB:
        return
    dct = DISJOINT_PATTERNS_DB[dkey]
    if label is None:
        for label, dpdb_info in dct.items():
            if dpdb_info.exists():
                yield dpdb_info
    else:
        yield dct[label]


def dpdb_get_info(size, label=None, puzzle_type=None):
    global DISJOINT_PATTERNS_DB
    if puzzle_type is None:
        puzzle_type = config_getkey("puzzle_type")
    dkey = (puzzle_type, size)
    if dkey not in DISJOINT_PATTERNS_DB:
        raise KeyError(dkey)
    dct = DISJOINT_PATTERNS_DB[dkey]
    if label is None:
        for label, dpdb_info in dct.items():
            if dpdb_info.exists():
                return dpdb_info
        return None
    else:
        return dct[label]


def dpdb_get_db(dpdb_info, **kwargs):
    try:
        return Dpdb(dpdb_info, **kwargs)
    except:
        LOG.exception("cannot load db {}".format(dpdb_info))
        return None

     
def dpdb_list(puzzle_type=None):
    if puzzle_type is None:
        puzzle_type = config_getkey("puzzle_type")
    for dummy_dkey, dct in sorted(DISJOINT_PATTERNS_DB.items(), key=lambda x: x[0]):
        for key, dpdb_info in dct.items():
            if puzzle_type is dpdb_info.puzzle_type:
                # print(dummy_dkey, dpdb_info)
                yield dpdb_info


class DpdbInitMode(enum.Enum):
    restart = 0
    from_scratch = 1
    from_cache = 2


def dpdb_merge(dpdb_info,
               input_db_filenames,
               db_filename):
    dpdb = Dpdb(dpdb_info, db_filename=db_filename)
    if not dpdb.db_exists():
        dpdb.create()
    dpdb.merge(*input_db_filenames)


def dpdb_create(dpdb_info,
                init_mode=DpdbInitMode.restart,
                progress_bar=True,
                workers=1,
                algorithm=None,
                futures_queue_size=None,
                nodes_queue_size=None,
                db_filename=None,
                pattern_index_slice=slice(0, None, 1),
                iteration_slice=slice(0, None, 1),
                verbose=False):
    dpdb = Dpdb(dpdb_info, db_filename=db_filename)
    if init_mode is DpdbInitMode.from_scratch:
        dpdb.drop()
    elif init_mode is DpdbInitMode.from_cache:
        dpdb.drop()
        dpdb.create()
        dpdb.import_cache()
    kwargs = {
        'pattern_index_slice': pattern_index_slice,
        'iteration_slice': iteration_slice,
        'verbose': verbose,
    }
    if workers <= 0:
        _dpdb_create_db(dpdb, algorithm=algorithm, progress_bar=progress_bar,
                        pool=None,
                        **kwargs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            _dpdb_create_db(dpdb, algorithm=algorithm, progress_bar=progress_bar,
                            pool=pool, workers=workers,
                            futures_queue_size=futures_queue_size,
                            nodes_queue_size=nodes_queue_size,
                            **kwargs)
    return dpdb


def dpdb_init(dpdb_info):
    dpdb = Dpdb(dpdb_info)
    dpdb.export_cache()


class DpdbDriver(Driver):
    def __init__(self, size, solver_function):
        super().__init__(size)
        self.solver_function = solver_function

    def make_groups(self, mask, value=0):
        nghd = self.nghd
        zgroups = {}
        zmembers = {}
        g_zgroupid = 0
        for i, v in enumerate(mask):
            if v == value:
                zgroupids = set()
                for j in nghd[i]:
                    if j in zgroups:
                        zgroupids.add(zgroups[j])
                if zgroupids:
                    zgroupid = zgroupids.pop()
                    zgroup = zmembers[zgroupid]
                    zgroup.add(i)
                    for zi in zgroupids:
                        zgroup.update(zmembers.pop(zi))
                    for k in zgroup:
                        zgroups[k] = zgroupid
                else:
                    zgroupid = g_zgroupid
                    g_zgroupid += 1
                    zgroups[i] = zgroupid
                    zmembers[zgroupid] = {i}
        data = []
        for group in zmembers.values():
            mprio = []
            for member in group:
                neighs = 0
                for neigh in nghd[member]:
                    if mask[neigh] != value:
                        neighs += 1
                mprio.append((member, neighs))
            mprio.sort(key=lambda x: x[1])
            g_repr = mprio[-1][0]
            data.append((g_repr, tuple(group)))
        return data

    def make_pattern_start_goal(self, pattern_start, pattern_goal):
        goal_tiles = self.goal.tiles
        pattern_start_goal = [-1 for _ in goal_tiles]
        for i, g in zip(pattern_start, pattern_goal):
            pattern_start_goal[i] = g
        return pattern_start_goal

    def make_pattern_goal(self, pattern):
        goal_tiles = self.goal.tiles
        return [goal_tiles[i] for i in pattern]

    def solve_pattern(self, pattern, node, tracker=None):
        get_neighbors = self.reset_get_neighbors()
        goal_tiles = self.goal.tiles
        t0 = time.time()
        gpattern = tuple((k, goal_tiles[k]) for k in pattern)

        def found(node):
            nonlocal gpattern
            tiles = node[0]
            for k, goal_tile in gpattern:
                if tiles[k] != goal_tile:
                    return False
            return True
            #return all(tiles[k] == goal_tile for k, goal_tile in gpattern)

        manhattan_distance = self.make_manhattan_distance()
        solution = self.solver_function(
            node,
            found=found,
            get_neighbors=get_neighbors,
            heuristic_cost=manhattan_distance,
            get_idx=self.get_node_idx)
        cost = len(solution) - 1
        prev_tiles = solution[0][0]
        for node in solution[1:]:
            tiles, cursor, dummy_mh, dummy_idx = node
            if prev_tiles[cursor] == -1:
                cost -= 1
            prev_tiles = tiles
        t1 = time.time()
        if tracker:
            tracker.add_solve(t1 - t0)
        return cost

    def compute_costs(self, pattern, node_groups, tracker=None):
        costs = {}
        for node, group in node_groups:
            cost = self.solve_pattern(pattern, node, tracker=tracker)
            for member in group:
                costs[member] = cost
        return costs

    def solve_bulk(self, items, tracker):
        results = []
        for pattern_id, pattern_start_id, pattern, node_group in items:
            results.append((pattern_id, pattern_start_id, self.compute_costs(pattern, node_group, tracker=tracker)))
        return tracker, results


class DpdbTracker(object):
    def __init__(self, tot_iterations):
        self.tot_iterations = tot_iterations
        self.num_iterations = 0
        self.num_resumed_iterations = 0
        self.solve_elapsed = 0.0
        self.solve_num = 0

    def new(self):
        return self.__class__(self.tot_iterations)

    def add_iteration(self, num=1):
        self.num_iterations += num

    def add_resumed_iteration(self, num=1):
        self.num_iterations += num
        self.num_resumed_iterations += num

    def add_solve(self, elapsed, num=1):
        self.solve_num += num
        self.solve_elapsed += elapsed

    def merge(self, tracker):
        self.num_iterations += tracker.num_iterations
        self.num_resumed_iterations += tracker.num_resumed_iterations
        self.solve_num += tracker.solve_num
        self.solve_elapsed += tracker.solve_elapsed


class DpdbProgressBar(object):
    def __init__(self, pattern, every=1.0):
        self.progress_bar = ProgressBar()
        self.every = every
        self.line_fmt = "### computing pattern " + str(pattern) + " - {fraction:7.2%} [{solve_num:8d}/{solve_elapsed:.2f}/{ave_solve_elapsed:.2f}/{total_elapsed:.2f}] [{eta}]"
        self.tstart = None
        self.tlast = None
        self.start()

    def start(self):
        self.tstart = time.time()
        self.tlast = None

    def finish(self, msg=None):
        self.progress_bar.finish(msg)

    def show(self, tracker):
        dtnow = datetime.datetime.now()
        tnow = dtnow.timestamp()
        num_iterations = tracker.num_iterations
        num_resumed_iterations = tracker.num_resumed_iterations
        tot_iterations = tracker.tot_iterations
        solve_num = tracker.solve_num
        solve_elapsed = tracker.solve_elapsed
        if self.tlast is None or (tnow - self.tlast > self.every) or (num_iterations >= tot_iterations):
            total_elapsed = tnow - self.tstart
            if solve_num:
                ave_solve_elapsed = solve_elapsed / solve_num
            else:
                ave_solve_elapsed = 0.0
            eta = '--'
            if num_iterations:
                fraction = num_iterations / tot_iterations
                computed_fraction = (num_iterations - num_resumed_iterations) / tot_iterations
                if computed_fraction > 0.0:
                    missing_seconds = total_elapsed * (1.0 - computed_fraction) / computed_fraction
                    eta = "{:%Y%m%d %H:%M:%S}".format(dtnow + datetime.timedelta(seconds=missing_seconds))
            else:
                fraction = 0.0
            msg = self.line_fmt.format(
                fraction=fraction,
                solve_num=solve_num,
                solve_elapsed=solve_elapsed,
                ave_solve_elapsed=ave_solve_elapsed,
                total_elapsed=tnow - self.tstart,
                eta=eta)
            self.progress_bar.show(msg)
            self.tlast = tnow


def consume_futures(dpdb, futures, *,
                    tracker=None,
                    progress_bar=None,
                    max_queue_size=None, sleep_seconds=0.1):
    while futures:
        old_futures = tuple(futures)
        futures.clear()
        for future in old_futures:
            if future.done():
                try:
                    r_tracker, results = future.result()
                except KeyboardInterrupt:
                    continue
                n_iterations = 0
                for pattern_id, pattern_start_id, costs in results:
                    n_iterations += 1
                    dpdb.buffer_costs(pattern_id=pattern_id, pattern_start_id=pattern_start_id, costs=costs)
                if tracker:
                    tracker.add_iteration(n_iterations)
                    tracker.merge(r_tracker)
                    if progress_bar:
                        progress_bar.show(tracker)
            else:
                futures.append(future)
        if max_queue_size is None or len(futures) < max_queue_size:
            break
        else:
            time.sleep(sleep_seconds)


@functools.lru_cache(maxsize=10)
def _get_dpdb_driver(size, algorithm):
    solver_function = search_algorithm.get_algorithm(algorithm)
    return DpdbDriver(size, solver_function)


def _call_method(size, algorithm, method_name, *args, **kwargs):
    method = getattr(_get_dpdb_driver(size, algorithm), method_name)
    return method(*args, **kwargs)


def _dpdb_create_db(dpdb, *,
                    algorithm=None,
                    progress_bar=True,
                    pool=None,
                    workers=1,
                    futures_queue_size=None,
                    nodes_queue_size=None,
                    pattern_index_slice=slice(0, None, 1),
                    iteration_slice=slice(0, None, 1),
                    verbose=False):
    if pool:
        if futures_queue_size is None:
            futures_queue_size = 2 * workers
    if dpdb.db_exists():
        update = True
    else:
        dpdb.create()
        update = False
    dpdb_info = dpdb.dpdb_info
    size = dpdb_info.size
    if algorithm is None:
        algorithm = 'ida_star'
    solver_function = search_algorithm.get_algorithm(algorithm)
    dpdb_driver = DpdbDriver(size, solver_function)
    manhattan_matrix = dpdb_driver.manhattan_matrix
    make_node = dpdb_driver.make_node

    positions = list(range(size * size))
    make_key = dpdb.make_key

    patterns = dict(enumerate(dpdb_info.patterns))
    pattern_indices = set(range(*pattern_index_slice.indices(len(patterns))))

    it_start = iteration_slice.start
    if it_start is None:
        it_start = 0
    it_stop = iteration_slice.stop
    it_step = iteration_slice.step
    if it_step is None:
        it_step = 1

    def make_iter_select(it_start, it_stop, it_step, tot_iterations):
        if it_stop is not None:
            max_iterations = min(tot_iterations, it_stop)
        else:
            max_iterations = tot_iterations
        def iter_select(n_iter):
            return it_start <= n_iter < max_iterations and (n_iter - it_start) % it_step == 0
        return iter_select

    for pattern_id, pattern in patterns.items():
        if pattern_id not in pattern_indices:
            continue
        num_permutations = 1
        for i in range(len(positions) - len(pattern) + 1, len(positions) + 1):
            num_permutations *= i
        if it_stop is None:
            mx = num_permutations
        else:
            mx = max(num_permutations, it_stop)
        num_iterations = 1 + (mx - 1 - it_start) // it_step
        iter_select = make_iter_select(it_start, it_stop, it_step, num_permutations)

        if verbose:
            print("# progress_bar: {}".format(bool(progress_bar)))
            print("# pool: {}".format(bool(pool)))
        if pool:
            if nodes_queue_size is None:
                nodes_queue_size = max(1, min(20, num_iterations // 50))
            if verbose:
                print("#   + pool.type={}".format(type(pool).__name__))
                print("#   + pool.workers={}".format(workers))
                print("#   + pool.futures_queue_size={}".format(futures_queue_size))
                print("#   + pool.nodes_queue_size={}".format(nodes_queue_size))
        tracker = DpdbTracker(num_iterations)
        if progress_bar:
            pbar = DpdbProgressBar(pattern)
            pbar.show(tracker)
        else:
            pbar = None
        pattern_goal = dpdb_driver.make_pattern_goal(pattern)
        pindex = list(range(len(pattern)))
        pindex_permutations = list(itertools.permutations(pindex, len(pattern)))
        p_records = set()
        if update:
            p_records.update(dpdb.get_records(pattern_id=pattern_id))
        if pool:
            futures_queue = []
            nodes_queue = []
        with dpdb:
            try:
                dpdb.buffer_flush()
                n_iter = -1
                for pattern_position in itertools.combinations(positions, len(pattern)):
                    mask = [0 for _ in positions]
                    for i in pattern_position:
                        mask[i] = 1
                    groups_data = dpdb_driver.make_groups(mask, 0)
                    for pindex_permutation in pindex_permutations:
                        n_iter += 1
                        if not iter_select(n_iter):
                            continue
                        pattern_start = tuple(pattern_position[pidx] for pidx in pindex_permutation)
                        pattern_start_id = make_key(pattern_id, pattern_start)
                        if pattern_start_id in p_records:
                            tracker.add_resumed_iteration()
                            if pbar:
                                pbar.show(tracker)
                            continue
                        pattern_start_goal = dpdb_driver.make_pattern_start_goal(pattern_start, pattern_goal)
                        node_groups = []
                        for group_repr, group in groups_data:
                            pattern_start_goal_g = pattern_start_goal[:]
                            cursor = group_repr
                            pattern_start_goal_g[cursor] = 0
                            mh = 0
                            for mline, v in zip(manhattan_matrix, pattern_start_goal_g):
                                if v > 0:
                                    mh += mline[v]
                            # node = DNode(tuple(pattern_start_goal_g), cursor, mh)
                            node = make_node(pattern_start_goal_g, cursor, mh)
                            node_groups.append((node, group))
                        if pool:
                            consume_futures(dpdb, futures_queue, tracker=tracker, max_queue_size=futures_queue_size, progress_bar=pbar)
                            nodes_queue.append((pattern_id, pattern_start_id, pattern, node_groups))
                            if len(nodes_queue) >= nodes_queue_size:
                                futures_queue.append(pool.submit(_call_method, size, algorithm, "solve_bulk", nodes_queue, tracker.new()))
                                nodes_queue = []
                        else:
                            costs = dpdb_driver.compute_costs(pattern, node_groups, tracker=tracker)
                            dpdb.buffer_costs(pattern_id=pattern_id, pattern_start_id=pattern_start_id, costs=costs)
                            tracker.add_iteration()
                            if pbar:
                                pbar.show(tracker)
                if pool:
                    if nodes_queue:
                        for pool_nodes_chunk in round_robin(nodes_queue, workers):
                            if pool_nodes_chunk:
                                futures_queue.append(pool.submit(_call_method, size, algorithm, "solve_bulk", pool_nodes_chunk, tracker.new()))
                    consume_futures(dpdb, futures_queue, tracker=tracker, max_queue_size=0, progress_bar=pbar)
            except KeyboardInterrupt:
                if pbar:
                    pbar.finish()
                if pool:
                    if futures_queue:
                        print("collecting {} jobs [press ^C again to exit]...".format(len(futures_queue)))
                        try:
                            consume_futures(dpdb, futures_queue, tracker=tracker, max_queue_size=0, progress_bar=pbar)
                        except KeyboardInterrupt:
                            pass
                        finally:
                            if pbar:
                                pbar.finish()
                print("--- interrupted ---")
                break
        if pbar:
            pbar.finish()
    return dpdb


#  drr:
#
#  puzzle:
#
#  1 2 3     1 _ 4
#  _ 5 6     2 5 7
#  4 7 8     3 6 8
#
#  goal:
#
#  1 2 3     1 4 7
#  4 5 6     2 5 8
#  7 8 _     3 6 _
#
#       0 1 2
#       3 4 5
#       6 7 _
#
#  1 1 2     1 1 1
#  1 2 2     1 2 2
#  1 2 _     2 2 _
#
# 1:
#   d: pattern: (0, 1, 3, 6)
#   d: pgoal:   (1, 2, 4, 7)
#   d: pstart:  (0, 1, 6, 7)
#
#   r: pattern: (0, 3, 6, 1)
#   r: pgoal:   (1, 4, 7, 2)
#   r: pstart:  (0, 2, 5, 3)
#

@functools.lru_cache(maxsize=10)
def _impl_dpdb_get_memdb(dpdb_info):
    dpdb = dpdb_get_db(dpdb_info)
    if not dpdb.cache_exists():
        return
    return dpdb.load_cache()


@functools.lru_cache(maxsize=10)
def _impl_dpdb_make_heuristic(dpdb_info, puzzle_type, reflection, memdb):
    size = dpdb_info.size
    dpdb = dpdb_get_db(dpdb_info)
    if not dpdb.cache_exists():
        return
    if memdb:
        mdb = dpdb_get_memdb(dpdb_info.size, dpdb_info.label)
        def get_cost(pattern_id, pattern_start_id, cursor):
            return mdb[pattern_id][pattern_start_id][cursor]
    else:
        get_cost = dpdb.get_cost

    driver = Driver(size)
    refk = driver.refk
    goal = driver.goal.tiles
    reflected_goal = [goal[refk[p]] for p in range(len(goal))]
    g_values = [None for _ in goal]
    if reflection:
        g_reflected_values = [None for _ in reflected_goal]
    g_patterns = dpdb_info.patterns
    g_pattern_ids = []
    for pattern_id, pattern in enumerate(dpdb_info.patterns):
        g_pattern_ids.append(pattern_id)
        pvalues = [goal[p] for p in pattern]
        for p_pos, pos in enumerate(pattern):
            g_tile = goal[pos]
            g_values[g_tile] = (pattern_id, p_pos)
        if reflection:
            reflected_pattern = sorted(refk[k] for k in pattern)
            for p_pos, pos in enumerate(pattern):
                g_reflected_tile = reflected_goal[pos]
                g_reflected_values[g_reflected_tile] = (pattern_id, p_pos)
            reflected_pvalues = [reflected_goal[p] for p in pattern]
    #print("::: goal=", goal)
    #print("::: g_values=", g_values)
    #print("::: reflected_goal=", reflected_goal)
    #print("::: g_reflected_values=", g_reflected_values)

    make_key = dpdb.make_key

    if dpdb is not None:
        if reflection:
            def disjoint_patterns(node):
                nonlocal g_pattern_ids
                nonlocal g_values
                nonlocal g_reflected_values
                tiles, cursor, manhattan, idx = node
                cost = 0
                reflected_cost = 0
                pattern_starts = [[None for _ in pattern] for pattern in g_patterns]
                reflected_pattern_starts = [[None for _ in pattern] for pattern in g_patterns]
                reflected_cursor = refk[cursor]
                for pos, tile in enumerate(tiles):
                    if tile:
                        pattern_id, p_pos = g_values[tile]
                        pattern_starts[pattern_id][p_pos] = pos
                        pattern_id, p_pos = g_reflected_values[tile]
                        reflected_pattern_starts[pattern_id][p_pos] = refk[pos]
                for pattern_id in g_pattern_ids:
                    pattern_start = pattern_starts[pattern_id]
                    pattern_start_id = make_key(pattern_id, pattern_start)
                    cost += get_cost(pattern_id, pattern_start_id, cursor)
                    reflected_pattern_start = reflected_pattern_starts[pattern_id]
                    reflected_pattern_start_id = make_key(pattern_id, reflected_pattern_start)
                    reflected_cost += get_cost(pattern_id, reflected_pattern_start_id, reflected_cursor)
                return max(cost, reflected_cost)
        else:
            def disjoint_patterns(node):
                nonlocal g_pattern_ids
                nonlocal g_values
                tiles, cursor, manhattan, idx = node
                # tiles = conv_tiles(tiles)
                cost = 0
                pattern_starts = [[None for _ in pattern] for pattern in g_patterns]
                for pos, tile in enumerate(tiles):
                    if tile:
                        pattern_id, p_pos = g_values[tile]
                        pattern_starts[pattern_id][p_pos] = pos
                for pattern_id in g_pattern_ids:
                    pattern_start = pattern_starts[pattern_id]
                    pattern_start_id = make_key(pattern_id, pattern_start)
                    cost += get_cost(pattern_id, pattern_start_id, cursor)
                return cost
        return disjoint_patterns


def dpdb_get_memdb(size, label=None, puzzle_type=None):
    dpdb_info = dpdb_get_info(size, label, puzzle_type=puzzle_type)
    return _impl_dpdb_get_memdb(dpdb_info)


def dpdb_make_heuristic(size, label=None, puzzle_type=PuzzleType.BR, reflection=UNDEFINED, memdb=UNDEFINED):
    dpdb_info = dpdb_get_info(size, label)
    if not dpdb_info:
        raise ValueError("no such disjoint patterns database {}:{}".format(size, label))
    if memdb is UNDEFINED:
        memdb = config_getkey('dpdb.memdb')
    if memdb is None:
        memdb = size <= 4
    if reflection is UNDEFINED:
        reflection = config_getkey('dpdb.reflection')
    return _impl_dpdb_make_heuristic(dpdb_info, puzzle_type, reflection, memdb)
