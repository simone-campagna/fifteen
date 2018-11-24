import collections

from .puzzle_type import PuzzleType


__all__ = [
    'config_get',
    'config_set',
    'config_getkey',
    'config_setkey',
]


CONFIG = collections.OrderedDict([
    ('puzzle_type', PuzzleType.BR),
    ('dpdb.dbdir', None),
    ('dpdb.memdb', None),
    ('dpdb.reflection', True),
])



def config_get():
    return CONFIG.copy()


def config_set(**kwargs):
    for key, value in kwargs.items():
        config_setkey(key, value)


def config_getkey(key):
    return CONFIG[key]


def config_setkey(key, value):
    if key in CONFIG:
        CONFIG[key] = value
    else:
        raise ValueError("unknown config key {}".format(key))
