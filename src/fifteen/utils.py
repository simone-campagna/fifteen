import bisect
import collections
import contextlib
import functools
import itertools
import re
import sys
import time


__all__ = [
    # REM 'LRCache',
    'map_tile',
    'Tile',
    'format_tiles',
    'format_multiple_tiles',
    'show_tiles',
    'grouper',
    'Tracker',
]


UNDEFINED = object()

# REM not working
# REM class LRCache(object):
# REM     def __init__(self, maxsize=1024, default_factory=None):
# REM         self._maxsize = max(1, maxsize)
# REM         self._ordered_fqkeys = []
# REM         self._count = itertools.count()
# REM         self._data = {}
# REM         self._default_factory = default_factory
# REM 
# REM     @property
# REM     def maxsize(self):
# REM         return self._maxsize
# REM 
# REM     def __iter__(self):
# REM         for prio, key in self._ordered_fqkeys:
# REM             yield key
# REM 
# REM     def items(self):
# REM         for dummy_prio, key in self._ordered_fqkeys:
# REM             yield key, self._data[key][1]
# REM 
# REM     def keys(self):
# REM         for dummy_prio, key in self._ordered_fqkeys:
# REM             yield key
# REM 
# REM     def values(self):
# REM         for dummy_prio, key in self._ordered_fqkeys:
# REM             yield self._data[key][1]
# REM 
# REM     def __len__(self):
# REM         return len(self._data)
# REM 
# REM     def __delitem__(self, key):
# REM         okeys = self._ordered_fqkeys
# REM         fqkey, dummy_value = self._data.pop(key)
# REM         idx = bisect.bisect_left(okeys, fqkey)
# REM         del okeys[idx]
# REM 
# REM     def __contains__(self, key):
# REM         return key in self._data
# REM     
# REM     def _clean(self):
# REM         maxsize = self._maxsize - 1
# REM         okeys = self._ordered_fqkeys
# REM         if len(self._data) >= maxsize:
# REM             for dummy_prio, d_key in okeys[maxsize:]:
# REM                 # print("   !!! deleting {} = {}".format(d_key, self._data[d_key]))
# REM                 del self._data[d_key]
# REM             del okeys[maxsize:]
# REM 
# REM     def __setitem__(self, key, value):
# REM         okeys = self._ordered_fqkeys
# REM         entry = self._data.pop(key, None)
# REM         if entry is not None:
# REM             idx = bisect.bisect_left(okeys, entry[0])
# REM             del okeys[idx]
# REM         self._clean()
# REM         fqkey = (-next(self._count), key)
# REM         # print("   !!! add {} = {}".format(key, (fqkey, value)))
# REM         self._data[key] = (fqkey, value)
# REM         bisect.insort_left(okeys, fqkey)
# REM 
# REM     def __getitem__(self, key):
# REM         okeys = self._ordered_fqkeys
# REM         if key in self._data:
# REM             fqkey, value = self._data[key]
# REM             idx = bisect.bisect_left(okeys, fqkey)
# REM             del okeys[idx]
# REM         else:
# REM             if self._default_factory is None:
# REM                 raise KeyError(key)
# REM             else:
# REM                 value = self._default_factory()
# REM                 self._clean()
# REM         fqkey = (-next(self._count), key)
# REM         self._data[key] = (fqkey, value)
# REM         bisect.insort_left(okeys, fqkey)
# REM         return value


class Tile(str):
    __re_fmt__ = re.compile(r'^[^\d]*(\d*).*$')

    def __new__(cls, value, expand=False):
        instance = super().__new__(cls, value)
        instance._expand = expand
        return instance

    @functools.lru_cache(maxsize=128)
    def _get_fmt_size(self, fmt):
        m = self.__re_fmt__.match(fmt)
        if m:
            return int(m.groups()[0])
        else:
            return None

    def __format__(self, fmt):
        fmt_size = self._get_fmt_size(fmt)
        if fmt_size is None:
            return super().__format__(fmt)
        else:
            rep = (fmt_size + len(self) - 1) // max(1, len(self))
            return (self * rep)[:fmt_size]


def map_tile(tile):
    if tile == 0:
        return Tile('_', expand=True)
    elif tile < 0:
        return Tile('*')
    else:
        return Tile(tile)


def format_tiles_lines(tiles, *, map_function=map_tile, size=None, prefix=''):
    if size is None:
        size = int(len(tiles) ** 0.5)
    if size * size != len(tiles):
        raise ValueError("size {} does not fit #{} tiles".format(size, len(tiles)))

    def _map_tile(t):
        if t < 0:
            return '*'
        else:
            return str(t)

    stiles = [map_function(t) for t in tiles]
    maxlen = max(len(tile) for tile in stiles)
    ifmt = "{{:>{ml}s}}".format(ml=maxlen)
    lfmt = prefix + " ".join(ifmt for _ in range(size))
    lines = []
    for sline in grouper(stiles, size):
        lines.append(lfmt.format(*sline))
    return lines

def format_tiles(tiles, *, map_function=map_tile, size=None, prefix=''):
    return '\n'.join(format_tiles_lines(tiles, map_function=map_function, size=size, prefix=prefix))


def show_tiles(tiles, *, map_function=map_tile, size=None, prefix='', file=sys.stdout):
    print(format_tiles(tiles, map_function=map_function, size=size, prefix=prefix), file=sys.stdout)

def format_multiple_tiles_lines(tiles_list, *, map_function=map_tile, size=None, prefix='', max_line_length=80, separator=' | ', hseparators=None):
    separator_length = len(separator)
    max_line_length -= len(prefix)
    if hseparators is None:
        hseparators = [
            # ' ' * max_line_length,
            '-' * max_line_length,
            # ' ' * max_line_length,
        ]
    group_lines = []
    group_line = []
    group_line_length = 0
    for tiles in tiles_list:
        tiles_lines = format_tiles_lines(tiles, map_function=map_function, size=size, prefix='')
        lengths = {len(line) for line in tiles_lines}
        if len(lengths) == 1:
            line_length = lengths.pop()
        else:
            line_length = max(lengths)
            tiles_lines = [line + (' ' * (line_length - len(line))) for line in tiles_lines]
        if group_line:
            if group_line_length + separator_length + line_length <= max_line_length:
                group_line.append((line_length, tiles_lines))
                group_line_length += separator_length + line_length
            else:
                group_lines.append((group_line_length, group_line))
                group_line_length = line_length
                group_line = [(line_length, tiles_lines)]
        else:
            group_line_length = line_length
            group_line = [(line_length, tiles_lines)]
    if group_line:
        group_lines.append((group_line_length, group_line))
    output_lines = []
    for ig, (group_line_length, group_line) in enumerate(group_lines):
        # print("===", ig, group_line_length, len(group_line))
        max_lines = max(len(lines) for lines in group_line)
        glines = []
        for ic, (line_length, lines) in enumerate(group_line):
            # print("  -", ic, line_length, len(lines))
            if len(lines) < max_lines:
                empty_line = ' ' * line_length
                lines += [empty_line for _ in range(max_lines - len(lines))]
            glines.append(lines)
        # print("///", len(glines), glines)
        if ig > 0:
            output_lines.extend(prefix + hseparator for hseparator in hseparators)
        for rlines in zip(*glines):
            # print(":::", rlines)
            output_lines.append(prefix + separator.join(rlines))
    return output_lines


def format_multiple_tiles(tiles_list, size=None, prefix='', max_line_length=80, separator=' | ', hseparators=None):
    return '\n'.join(format_multiple_tiles_lines(tiles_list, size=size, prefix=prefix,
                     max_line_length=max_line_length, separator=separator, hseparators=hseparators))


def show_multiple_tiles(tiles_list, size=None, prefix='', max_line_length=80, separator=' | ', hseparators=None, file=sys.stdout):
    print(format_multiple_tiles(tiles_list, size=size, prefix=prefix,
                                max_line_length=max_line_length, separator=separator, hseparators=hseparators),
          file=sys.stdout)


def grouper(iterable, size):
    bfr = []
    for item in iterable:
        if len(bfr) >= max(1, size):
            yield tuple(bfr)
            bfr = []
        bfr.append(item)
    if bfr:
        yield tuple(bfr)


class ProgressBar(object):
    def __init__(self, file=sys.stdout):
        self.msg = None
        self.file = file

    def show(self, msg):
        if self.msg is not None:
            reset = '\r' + (' ' * len(self.msg))
        else:
            reset = ''
        print(reset + '\r' + msg, file=self.file, end='')
        self.file.flush()
        self.msg = msg

    def finish(self, msg=None):
        if msg is None:
            msg = self.msg
            if msg is None:
                msg = ''
        self.show(msg + '\n')
        self.msg = None


def round_robin(items, num_chunks):
    chunks = [[] for _ in range(num_chunks)]
    ichunk = iter(itertools.cycle(chunks))
    for item, chunk in zip(items, itertools.cycle(chunks)):
        chunk.append(item)
    return chunks


def flatten(lst):
    out = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


class Tracker(object):
    def __init__(self, count=0):
        self._count = count

    def add(self, value=1):
        self._count += value

    @property
    def count(self):
        return self._count

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self._count)


class Timer(object):
    def __init__(self):
        self._t_start = None
        self._t_stop = None

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    @property
    def t_elapsed(self):
        if self._t_start is not None and self._t_stop is not None:
            return self._t_stop - self._t_start

    def start(self):
        self._t_start = time.time()

    def stop(self):
        self._t_stop = time.time()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


@contextlib.contextmanager
def timing(message, file=sys.stderr):
    print("# {message}...".format(message=message), file=file)
    file.flush()
    try:
        with Timer() as timer:
            yield timer
    finally:
        print("# {message} done [{elapsed:.2f}s]".format(message=message, elapsed=timer.t_elapsed), file=file)
        file.flush()
