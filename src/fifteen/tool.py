import argparse
import collections
import itertools
import json
import multiprocessing
import random
import sys
import time

from .config import (
    config_get,
    config_set,
)
from .disjoint_patterns import (
    dpdb_setup,
    dpdb_list,
    dpdb_get_db,
    dpdb_get_dbs,
    DpdbInitMode,
    dpdb_create,
    dpdb_merge,
    dpdb_init,
    dpdb_rotate_info,
)
from .driver import (
    setup_driver,
)
from .puzzle_type import (
    PuzzleType,
)
from .puzzle import (
    puzzle_class, write_puzzle, read_puzzle
)
from .solve import (
    solve,
    get_solve_defaults,
    get_heuristics,
    get_heuristic,
    get_algorithms,
)
from .utils import (
    Tracker,
    Tile,
    Timer,
    show_tiles,
    show_multiple_tiles,
)


def _create_puzzle(init, output=None):
    if isinstance(init, str):
        puzzle = read_puzzle(init)
        if output is None:
            output = init
    else:
        puzzle = puzzle_class(init)()
    return puzzle, output


def function_dpdb_show(dbkeys):
    def map_tile(tile):
        if tile == -1:
            return Tile('_', expand=True)
        elif tile < -1:
            return Tile('*')
        else:
            return Tile(tile)

    dbs = dpdb_get_dbs(dbkeys)
    for dbkey, dpdb_info in dbs.items():
        print("=== dpdb[{}:{}] ===".format(dpdb_info.size, dpdb_info.label))
        # for field in dpdb_info._fields:
        #     print("  {} = {}".format(field, getattr(dpdb_info, field)))
        pclass = puzzle_class(dpdb_info.size)
        driver = pclass.__driver__
        goal = driver.goal_br
        tiles = [-2 for _ in goal.tiles]
        tiles[goal.cursor] = -1
        for pattern_id, pattern in enumerate(dpdb_info.patterns):
            for i in pattern:
                tiles[i] = pattern_id
        show_tiles(tiles, map_function=map_tile)
        for pattern_id, pattern in enumerate(dpdb_info.patterns):
            print("\npattern[{}] {}".format(pattern_id, pattern))
            tiles = [-2 for _ in goal.tiles]
            tiles[goal.cursor] = -1
            for i in pattern:
                tiles[i] = pattern_id
            # reflected_tiles = [tiles[driver.refk[k]] for k in range(len(tiles))]
            # show_multiple_tiles([pclass(tiles), pclass(reflected_tiles)])
            show_tiles(tiles, map_function=map_tile)


def function_dpdb_list():
    def _exists(b):
        if b:
            return 'ok'
        else:
            return '--'
    dbs = dpdb_get_dbs()
    table = [("DBKEY", "TYPE", "SIZE", "LABEL", "DB", "CACHE")]
    for dbkey, dpdb_info in dbs.items():
        dpdb = dpdb_get_db(dpdb_info)
        table.append((dbkey, str(dpdb_info.puzzle_type.name), str(dpdb_info.size), str(dpdb_info.label), _exists(dpdb.db_exists()), _exists(dpdb.cache_exists())))
    lengths = {key: max(len(row[i]) for row in table) for i, key in enumerate(table[0])}
    fmt = "{{:{l[DBKEY]}s}} {{:{l[TYPE]}s}} {{:{l[SIZE]}s}} {{:<{l[LABEL]}s}} {{:{l[DB]}s}} {{:{l[CACHE]}s}}".format(l=lengths)
    for row in table:
        print(fmt.format(*row))
        #print("{:10s} {:2d} {:<20s} : {}".format(dbkey, dpdb_info.size, dpdb_info.label, status))


def function_dpdb_merge(dbkey, input_db_filenames, db_filename):
    dbs = dpdb_get_dbs([dbkey])
    if dbkey in dbs:
        dpdb_info = dbs[dbkey]
        dpdb_merge(dpdb_info, input_db_filenames, db_filename)
    else:
        print("!!! dpdb {!r} not found".format(dbkey))


def function_dpdb_create(dbkeys, init_mode, progress_bar,
                       workers, futures_queue_size, nodes_queue_size,
                       algorithm, db_filename,
                       pattern_index_slice, iteration_slice, verbose):
    dbs = dpdb_get_dbs(dbkeys)
    create_options = {
        'init_mode': init_mode,
        'progress_bar': progress_bar,
        'workers': workers,
        'algorithm': algorithm,
        'futures_queue_size': futures_queue_size,
        'nodes_queue_size': nodes_queue_size,
        'db_filename': db_filename,
        'pattern_index_slice': pattern_index_slice,
        'iteration_slice': iteration_slice,
        'verbose': verbose,
    }
    for dbkey in dbkeys:
        if dbkey not in dbs:
            print("!!! dpdb {!r} not found".format(dbkey))
            continue
        dpdb_info = dbs[dbkey]
        db = dpdb_create(dpdb_info, **create_options)


def function_dpdb_init(dbkeys):
    dbs = dpdb_get_dbs(dbkeys)
    for dbkey in dbkeys:
        if dbkey not in dbs:
            print("!!! dpdb {!r} not found".format(dbkey))
            continue
        dpdb_info = dbs[dbkey]
        db = dpdb_init(dpdb_info)

    
def function_dpdb_rotate(dbkeys, overwrite):
    dbs = dpdb_get_dbs(dbkeys)
    for dbkey in dbkeys:
        if dbkey not in dbs:
            print("!!! dpdb {!r} not found".format(dbkey))
            continue
        dpdb_info = dbs[dbkey]
        rot_dpdb_info = dpdb_info.rotated()
        if (not overwrite) and rot_dpdb_info.exists():
            print("dpdp[{}] already exists".format(rot_dpdb_info.fqname()))
        else:
            dpdb_rotate_info(dpdb_info)

    
def function_create(init, output, num_random_moves, try_solve, algorithm, heuristics):
    puzzle, output = _create_puzzle(init, output)
    seen = {puzzle}
    dual = puzzle.__driver__.dual_moves
    moves = []
    prev_solution = []
    algorithm, heuristic = get_solve_defaults(puzzle, algorithm=algorithm, heuristic=heuristics)
    if isinstance(heuristic, str):
        hmsg = heuristic
    else:
        hmsg = '+'.join(heuristic)
    while True:
        for i in range(num_random_moves - len(moves)):
            puzzle_moves = set(puzzle.get_moves())
            if moves:
                puzzle_moves.discard(dual[moves[-1]])
            if puzzle_moves:
                puzzle_moves = list(puzzle_moves)
                random.shuffle(puzzle_moves)
                for move in puzzle_moves:
                    pz = puzzle.move(move)
                    if pz not in seen:
                        puzzle = pz
                        moves.append(move)
                        seen.add(puzzle)
                        break
                else:
                    break
            else:
                break
        if try_solve:
            heuristic = get_heuristic(heuristic, size=puzzle.size)
            solution = solve(puzzle, algorithm=algorithm, heuristic=heuristic)
            print("solve {} -> {}".format(len(moves), len(solution)))
            moves = [dual[move] for move in reversed(solution)]
        else:
            solution = [dual[move] for move in reversed(moves)]
        if (not try_solve) or len(solution) >= num_random_moves:
            break
        if len(solution) == len(prev_solution):
            if moves:
                move = moves.pop(-1)
                #print("pop!", move)
                puzzle = puzzle.move(dual[move])
                prev_solution = []
            else:
                break
        else:
            prev_solution = solution
        
    print("moves:    {} #{}".format(puzzle.str_moves(moves), len(moves)))
    print("solution: {} #{}".format(puzzle.str_moves(solution), len(solution)))
    solution_len = len(solution)
    if output is None:
        output = "{size}x{size}.{solution_len}.{num_tiles}p"
    output = output.format(
        size=puzzle.size,
        num_tiles=puzzle.size ** 2 - 1,
        solution_len=solution_len)
    if output is not None:
        print("output file: {}".format(output))
        write_puzzle(puzzle, output)
    else:
        print(puzzle)


def function_move(init, output, moves):
    puzzle, output = _create_puzzle(init, output)
    print(puzzle)
    mlist = []
    for move in moves:
        mlist.extend(move)
    puzzle = puzzle.move(mlist)
    print(puzzle)
    if puzzle.is_solved():
        print("Solved!")
    if output is not None:
        write_puzzle(puzzle, output)


def function_play(init, output):
    puzzle, output = _create_puzzle(init, output)
    print(puzzle)
    print()
    moves = []
    while True:
        mqueue = list(input("move: [{}] ".format("|".join(itertools.chain(puzzle.str_moves(puzzle.get_moves()), 'q')))))
        while mqueue:
            m = mqueue.pop(0)
            if m == 'q':
                break
            try:
                puzzle = puzzle.move(m)
                moves.append(m)
            except KeyError:
                print("!!! invalid move {}".format(m))
                while mqueue:
                    print("--- move {} discarded".format(mqueue.pop(0)))
                continue
            print(puzzle)
            print()
            if puzzle.is_solved():
                print("You solved the puzzle!")
                while mqueue:
                    print("--- move {} discarded".format(mqueue.pop(0)))
                break
        else:
            continue
        break
    print("Your moves: {}".format("".join(moves)))
    if output:
        write_puzzle(puzzle, output)


def function_show(items):
    algorithms_shown = False
    heuristics_shown = set()
    for kind, value in items:
        if kind == 'algorithms' and not algorithms_shown:
            print("=== Algorithms ===")
            for algorithm in get_algorithms():
                print("    {}".format(algorithm))
            algorithms_shown = True
        elif kind == 'heuristics' and value not in heuristics_shown:
            print("=== Heuristics[size={}] ===".format(value))
            for heuristic in get_heuristics(size=value):
                print("    {}".format(heuristic))
            heuristics_shown.add(value)
        elif kind == 'puzzle':
            print("=== {} ===".format(value))
            puzzle, dummy_output = _create_puzzle(value)
            print(puzzle)


def function_solve(input_files, algorithm, heuristics, show_full_solution):
    for input_file in input_files:
        print("=== {} ===".format(input_file))
        puzzle, dummy_output = _create_puzzle(input_file)
        print(puzzle)
        print()
        algorithm, heuristic = get_solve_defaults(puzzle, algorithm=algorithm, heuristic=heuristics)
        if isinstance(heuristic, str):
            hmsg = heuristic
        else:
            hmsg = '+'.join(heuristic)
        print("using algorithm={}, heuristic={}".format(algorithm, hmsg))
        heuristic = get_heuristic(heuristic, size=puzzle.size)
        tracker = Tracker()
        with Timer() as timer:
            solution = solve(puzzle, algorithm=algorithm, heuristic=heuristic, tracker=tracker)
        print("solve: {} iterations in {:.1f} seconds".format(tracker.count, timer.t_elapsed))
        if solution is None:
            print("no solutions found")
        else:
            if show_full_solution:
                pz = puzzle
                full_solution = [pz]
                for move in solution:
                    pz = pz.move(move)
                    full_solution.append(pz)
                show_multiple_tiles(full_solution)
                final = full_solution[-1]
            else:
                final = puzzle.move(solution)
            if final.is_solved():
                print("found solution {} #{}".format(puzzle.str_moves(solution), len(solution)))
            else:
                print("found wrong solution {} #{}".format(puzzle.str_moves(solution), len(solution)))
                print("ERR: solve failed:")
                print(final)

     
def function_stats(init):
    puzzle, output = _create_puzzle(init)
    print(puzzle)
    print()
    print("is solvable: {}".format(puzzle.is_solvable()))
    print("is solved: {}".format(puzzle.is_solved()))
    print("distances:")
    table = []
    size = puzzle.size
    driver = puzzle.driver
    node = puzzle.node
    for heuristic in get_heuristics():
        hfunction = get_heuristic(heuristic, size, default=None)
        if hfunction is not None:
            table.append((heuristic, hfunction(node)))
    maxlen = max(len(x[0]) for x in table)
    for name, value in table:
        print("    {name:<{maxlen}s} : {value}".format(name=name, maxlen=maxlen, value=value))


def type_maybe_int(x):
    x = x.strip()
    if x:
        return int(x)
    else:
        return None


def type_slice(x):
    return slice(*[type_maybe_int(token) for token in x.split(":")])


def type_on_off(x):
    x = x.lower()
    if x in {'on', 'true'}:
        return True
    elif x in {'off', 'false'}:
        return False
    else:
        return bool(int(x))


def type_config_dbdir(x):
    return ('dpdb.dbdir', x)


def type_config_puzzle_type(x):
    puzzle_type = getattr(PuzzleType, x.upper())
    return ('puzzle_type', puzzle_type)


def type_config_memdb(x):
    return ('dpdb.memdb', type_on_off(x))


def type_config_reflection(x):
    return ('dpdb.reflection', type_on_off(x))


def main():
    top_level_parser = argparse.ArgumentParser()
    top_level_parser.set_defaults(
        function=None,
        function_args=[])

    default_config_kwargs = []
    top_level_parser.add_argument(
        "-d", "--dbdir",
        dest='config_kwargs',
        metavar="DIR",
        default=default_config_kwargs,
        action='append',
        type=type_config_dbdir, 
        help="set dpdb dir")
    top_level_parser.add_argument(
        "-t", "--puzzle-type",
        dest='config_kwargs',
        metavar="TYPE",
        default=default_config_kwargs,
        action='append',
        choices=tuple(type_config_puzzle_type(v) for v in PuzzleType.__members__.keys()),
        type=type_config_puzzle_type, 
        help="set puzzle type")
    top_level_parser.add_argument(
        "-m", "--memdb",
        dest='config_kwargs',
        metavar="M",
        default=default_config_kwargs,
        action='append',
        choices=('on', 'off'),
        type=type_config_memdb,
        help="set dpdb memdb")
    top_level_parser.add_argument(
        "-r", "--reflection",
        dest='config_kwargs',
        metavar="R",
        default=default_config_kwargs,
        action='append',
        choices=('on', 'off'),
        type=type_config_reflection,
        help="set dpdb reflection")
    top_level_parser.add_argument(
        "-v", "--verbose",
        default=False,
        action="store_true",
        help="verbose mode")


    subparsers = top_level_parser.add_subparsers()

    # dpdb:
    dpdb_parser = subparsers.add_parser(
        "dpdb",
        description="""\
Disjoint Patterns DB!
"""
    )
    dpdb_parser.set_defaults(
        function=function_dpdb_list,
        function_args=[])

    dpdb_subparsers = dpdb_parser.add_subparsers()

    # dpdb.list:
    dpdb_list_parser = dpdb_subparsers.add_parser(
        "list",
        description="""\
List available DBs
""")
    dpdb_list_parser.set_defaults(
        function=function_dpdb_list,
        function_args=[])

    # dpdb.show:
    dpdb_show_parser = dpdb_subparsers.add_parser(
        "show",
        description="""\
Show DBs
""")
    dpdb_show_parser.set_defaults(
        function=function_dpdb_show,
        function_args=["dbkeys"])

    # dpdb.create:
    dpdb_create_parser = dpdb_subparsers.add_parser(
        "create",
        description="""\
Create DBs
""")
    dpdb_create_parser.set_defaults(
        function=function_dpdb_create,
        function_args=["dbkeys", "init_mode", "progress_bar",
                       "workers", "futures_queue_size", "nodes_queue_size",
                       "algorithm", "db_filename",
                       "pattern_index_slice", "iteration_slice", "verbose"])

    # dpdb.merge:
    dpdb_merge_parser = dpdb_subparsers.add_parser(
        "merge",
        description="""\
Merge DBs
""")
    dpdb_merge_parser.set_defaults(
        function=function_dpdb_merge,
        function_args=["dbkey", "db_filename", "input_db_filenames"])

    # dpdb.init:
    dpdb_init_parser = dpdb_subparsers.add_parser(
        "init",
        description="""\
Init DBs
""")
    dpdb_init_parser.set_defaults(
        function=function_dpdb_init,
        function_args=["dbkeys"])

    # dpdb.rotate:
    dpdb_rotate_parser = dpdb_subparsers.add_parser(
        "rotate",
        description="""\
Init DBs
""")
    dpdb_rotate_parser.set_defaults(
        function=function_dpdb_rotate,
        function_args=["dbkeys", "overwrite"])

    # create:
    create_parser = subparsers.add_parser(
        "create",
        description="""\
Create!
"""
    )
    create_parser.set_defaults(
        function=function_create,
        function_args=["init", "output", "num_random_moves", "try_solve", "algorithm", "heuristics"])

    # show:
    show_parser = subparsers.add_parser(
        "show",
        description="""\
Show!
"""
    )
    show_parser.set_defaults(
        function=function_show,
        function_args=["items"])

    default_items = []
    show_parser.add_argument(
        "-A", "--algorithms",
        dest="items",
        action="append_const", const=("algorithms", None),
        default=default_items,
        help="show algorithms")
        
    show_parser.add_argument(
        "-H", "--heuristics",
        metavar='H',
        dest="items",
        nargs='?', const=('heuristics', 4),
        action='append', type=lambda x: ('heuristics', int(x)),
        default=default_items,
        help="show heuristics for given size")
        
    show_parser.add_argument(
        dest="items",
        nargs='*', type=lambda x: ('puzzle', x),
        #action="append", type=lambda x: ('puzzle', x),
        default=default_items,
        help="show puzzle")
        

    # play:
    play_parser = subparsers.add_parser(
        "play",
        description="""\
Play!
"""
    )
    play_parser.set_defaults(
        function=function_play,
        function_args=["init", "output"])

    # stats:
    stats_parser = subparsers.add_parser(
        "stats",
        description="""\
Stats!
"""
    )
    stats_parser.set_defaults(
        function=function_stats,
        function_args=["init"])

    # move:
    move_parser = subparsers.add_parser(
        "move",
        description="""\
Move!
"""
    )
    move_parser.set_defaults(
        function=function_move,
        function_args=["init", "output", "moves"])

    # solve:
    solve_parser = subparsers.add_parser(
        "solve",
        description="""\
Solve!
"""
    )
    solve_parser.set_defaults(
        function=function_solve,
        function_args=["input_files", "algorithm", "heuristics", "show_full_solution"])

    for parser in create_parser, stats_parser, play_parser, move_parser:
        default_init = 4
        igroup = parser.add_mutually_exclusive_group()
        igroup.add_argument(
            "-s", "--size",
            dest="init",
            type=int,
            default=default_init,
            help="start a new empty board with the given size")
        igroup.add_argument(
            "-i", "--input",
            dest="init",
            type=str,
            default=default_init,
            help="open input file")

    solve_parser.add_argument(
            "input_files",
            type=str,
            default=[],
            nargs="+",
            help="input files")

    for parser in create_parser, play_parser, move_parser:
        parser.add_argument(
            "-o", "--output",
            default=None,
            help="output filename")

    create_parser.add_argument(
        "-m", "--num-random-moves",
        default=5,
        type=int,
        help="number of random moves")

    create_parser.add_argument(
        "-t", "--try-solve",
        default=False,
        action="store_true",
        help="try to solve")

    move_parser.add_argument(
        "moves",
        type=str,
        nargs='+',
        help="moves")

    for parser in dpdb_create_parser, create_parser, solve_parser:
        parser.add_argument(
            "-A", "--algorithm",
            choices=tuple(get_algorithms()),
            default=None,
            help="solver algorithm")

    for parser in create_parser, solve_parser:
        parser.add_argument(
            "-H", "--heuristic",
            metavar='H',
            dest="heuristics",
            type=str,
            action="append",
            default=[],
            help="add heuristic")

    solve_parser.add_argument(
        "-s", "--show-full-solution",
        default=False,
        action="store_true",
        help="show full solution")

    for parser in dpdb_create_parser, dpdb_init_parser, dpdb_rotate_parser, dpdb_show_parser:
        dbkeys_group = parser.add_mutually_exclusive_group(required=True)
        default_dbkeys = []
        dbkeys_group.add_argument(
            "-d", "--db",
            dest="dbkeys",
            action="append",
            default=default_dbkeys,
            type=str,
            help="select db")

        dbkeys_group.add_argument(
            "-a", "--all",
            dest="dbkeys",
            default=default_dbkeys,
            action="store_const", const=None,
            help="select all dbs")

    dpdb_merge_parser.add_argument(
        "-d", "--db",
        dest="dbkey",
        type=str,
        required=True,
        help="output db")

    dpdb_merge_parser.add_argument(
        "-i", "--input",
        dest="input_db_filenames",
        type=str,
        nargs='+',
        help="input db filenames")

    dpdb_init_mode_group = dpdb_create_parser.add_mutually_exclusive_group()
    dpdb_init_mode_default = DpdbInitMode.restart
    dpdb_init_mode_group.add_argument(
        "-f", "--from-scratch",
        dest="init_mode",
        default=dpdb_init_mode_default,
        action="store_const", const=DpdbInitMode.from_scratch,
        help="start new DBs from scratch")
    dpdb_init_mode_group.add_argument(
        "-c", "--from-cache",
        dest="init_mode",
        default=dpdb_init_mode_default,
        action="store_const", const=DpdbInitMode.from_cache,
        help="start new DBs from cache file")
    dpdb_init_mode_group.add_argument(
        "-r", "--restart",
        dest="init_mode",
        default=dpdb_init_mode_default,
        action="store_const", const=DpdbInitMode.restart,
        help="restart DBs")

    dpdb_rotate_parser.add_argument(
        "-f", "--force-overwrite",
        dest="overwrite",
        default=False,
        action="store_true",
        help="overwrite existing files")
    
    dpdb_create_select_group = dpdb_create_parser.add_argument_group("iteration selection")
    dpdb_create_select_group.add_argument(
        "-x", "--pattern-index-select",
        dest="pattern_index_slice",
        default="::1",
        type=type_slice,
        help="pattern select")
    dpdb_create_select_group.add_argument(
        "-i", "--iteration-select",
        dest="iteration_slice",
        default="::1",
        type=type_slice,
        help="iteration select")

    ncpus = multiprocessing.cpu_count()
    workers_group = dpdb_create_parser.add_mutually_exclusive_group()
    workers_group.add_argument(
        "-j", "--workers",
        dest="workers",
        metavar='W',
        default=ncpus,
        type=int,
        nargs='?', const=ncpus,
        help="number of pool workers")
    workers_group.add_argument(
        "-J", "--no-workers",
        dest="workers",
        default=ncpus,
        action="store_const", const=0,
        help="no pool")
    dpdb_create_parser.add_argument(
        "-q", "--futures-queue-size",
        metavar='Q',
        default=None,
        type=int,
        help="max number of items in futures queue")
    dpdb_create_parser.add_argument(
        "-n", "--nodes-queue-size",
        metavar='N',
        default=None,
        type=int,
        help="max number of items in nodes queue")

    for parser in dpdb_create_parser, dpdb_merge_parser:
        parser.add_argument(
            "-o", "--output",
            dest="db_filename",
            default=None,
            type=str,
            help="set db filename")

    progress_bar_group = dpdb_create_parser.add_mutually_exclusive_group()
    progress_bar_group.add_argument(
        "-p", "--progress-bar",
        dest="progress_bar", default=True,
        action="store_true",
        help="show progress bar")
    progress_bar_group.add_argument(
        "-P", "--no-progress-bar",
        dest="progress_bar", default=True,
        action="store_false",
        help="do not show progress bar")

    namespace = top_level_parser.parse_args()
    if namespace.function is None:
        top_level_parser.print_help()

    config_set(**dict(namespace.config_kwargs))
    setup_driver()
    if namespace.verbose:
        for key, value in config_get().items():
            print("# config[{!r}] = {!r}".format(key, value))

    dpdb_setup()

    function_kwargs = {arg: getattr(namespace, arg) for arg in namespace.function_args}
    function = namespace.function
    if function is None:
        top_level_parser.print_help()
    else:
        function(**function_kwargs)
