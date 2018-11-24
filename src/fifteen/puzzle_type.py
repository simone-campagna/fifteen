import enum

__all__ = [
    'PuzzleType',
    'rot_puzzle_type',
]


class PuzzleType(enum.IntEnum):
    BR = 0
    UL = 1


ROT_PUZZLE_TYPE = {
    PuzzleType.BR: PuzzleType.UL,
    PuzzleType.UL: PuzzleType.BR,
}


def rot_puzzle_type(puzzle_type):
    return ROT_PUZZLE_TYPE[puzzle_type]
