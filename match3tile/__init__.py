from dataclasses import dataclass
from math import ceil, log2


def decode(action, board_width):
    a = (2 * board_width - 1)
    b = board_width - 1
    if action - a * int(action / a) >= b:
        column1 = action % a - b
        row1 = int((action - 3 - column1) / a)
        column2 = column1
        row2 = row1 + 1
    else:
        column1 = action % a
        row1 = int((action - column1) / a)
        column2 = column1 + 1
        row2 = row1

    return (row1, column1), (row2, column2)


def encode(tile1: tuple[int, int], tile2: tuple[int, int], board_width):
    source_row, source_column = tile1
    target_row, target_column = tile2
    assert (source_column == target_column and abs(source_row - target_row) == 1 or
            source_row == target_row and abs(source_column - target_column) == 1), \
        'source and target must be adjacent'
    a = 2 * board_width - 1
    b = board_width - 1 if source_column == target_column else 0
    return min(source_row, target_row) * a + b + min(source_column, target_column)


@dataclass
class Metadata:
    types = 6
    rows = 9
    columns = 9
    action_space = 144
    actions = {a: decode(a, 9) for a in range(144)}
    type_mask = 7           # 0 0 0 1 1 1
    special_type_mask = 24  # 0 1 1 0 0 0
    mega_token = 32         # 1 0 0 0 0 0

    def set_types(self, types):
        self.types = types
        bits = int(ceil(log2(self.types+1)))
        self.type_mask = 2 ** bits - 1
        self.special_type_mask = 2 ** (bits + 1) + 1 + self.type_mask
        self.mega_token = self.type_mask + self.special_type_mask + 1

    def set_shape(self, rows, columns):
        self.rows, self.columns = rows, columns
        self.action_space = self.rows * (self.columns - 1) * 2
        self.actions = {a: decode(a, self.columns) for a in range(self.action_space)}


metadata = Metadata()


__all__ = [
    'metadata',
    'decode',
    'encode'
]
