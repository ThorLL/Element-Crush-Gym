import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoardConfig:
    seed: int = None

    rows: int = 9
    columns: int = 9
    types: int = 6

    shape: tuple[int, int] = field(init=False)

    action_space: int = field(init=False)
    actions: dict[int, tuple[tuple[int, int], tuple[int, int]]] = field(init=False)

    type_mask: int = field(init=False)
    special_type_mask: int = field(init=False)

    h_line: int = field(init=False)
    v_line: int = field(init=False)
    bomb: int = field(init=False)
    mega_token: int = field(init=False)

    def __post_init__(self):
        action_space = self.rows * (self.columns - 1) * 2

        bits = int(np.ceil(np.log2(self.types + 1)))
        type_mask = 2 ** bits - 1
        special_type_mask = 2 ** (bits + 1) + 1 + type_mask
        h_line = type_mask + 1

        object.__setattr__(self, 'seed', self.seed or np.random.randint(0, 2 ** 31 - 1))
        object.__setattr__(self, 'shape', (self.rows, self.columns))
        object.__setattr__(self, 'action_space', action_space)
        object.__setattr__(self, 'actions', {a: self.decode(a) for a in range(action_space)})
        object.__setattr__(self, 'type_mask', type_mask)
        object.__setattr__(self, 'special_type_mask', special_type_mask)
        object.__setattr__(self, 'h_line', h_line)
        object.__setattr__(self, 'v_line', 2 * h_line)
        object.__setattr__(self, 'bomb', special_type_mask)
        object.__setattr__(self, 'mega_token', type_mask + special_type_mask + 1)

    def decode(self, action):
        a = (2 * self.columns - 1)
        b = self.columns - 1
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

    def encode(self, tile1: tuple[int, int], tile2: tuple[int, int]):
        source_row, source_column = tile1
        target_row, target_column = tile2
        assert (source_column == target_column and abs(source_row - target_row) == 1 or
                source_row == target_row and abs(source_column - target_column) == 1), \
            'source and target must be adjacent'
        a = 2 * self.columns - 1
        b = self.columns - 1 if source_column == target_column else 0
        return min(source_row, target_row) * a + b + min(source_column, target_column)
