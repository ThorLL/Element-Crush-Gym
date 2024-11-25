import math
import random
from collections import namedtuple

import numpy as np

Swap_Event = namedtuple('Swap_Event', ['action', 'init_board', 'event_boards'])

NONE_TOKEN = -2147483648


# match shapes
V_LINE = 0
H_LINE = 1
T_L_SHAPE = 2


def get_match_shape(match):
    h_line = all(match[0][0] == token[0] for token in match)
    v_line = all(match[0][1] == token[1] for token in match)
    return T_L_SHAPE if not h_line and not v_line else H_LINE if h_line else V_LINE


class Board:
    def __init__(self, shape, seed=0):
        self.seed = seed
        self.height, self.width, self.types = shape
        assert self.types > 1
        self.array = np.full((self.height, self.width), NONE_TOKEN, dtype=np.int32)
        self.fill_board()

        """
        example of 6 tokens:
        x    x x   x x x x
        
        0    x x   0 0 0 = 0 = wind
        0    x x   0 0 1 = 1 = dark
        0    x x   0 1 0 = 2 = earth
        0    x x   0 1 1 = 3 = fire
        0    x x   1 0 0 = 4 = sun
        0    x x   1 0 1 = 5 = water
        
        element mask:
        0    0 0   1 1 1 = 7
        
        types:
        x    0 1   x x x = +8   v_line
        x    1 0   x x x = +16  h_line
        x    1 1   x x x = +24  bomb
        
        type mask:
        
        x    1 1   0 0 0 = 24
        
        
        big bad = 
        
        1   x x    x x x = 32
        """

        required_bits = int(math.ceil(math.log2(self.types)))
        self.element_mask = (2 ** required_bits) - 1
        self.big_bad = 2 ** (required_bits + 2)

        self.v_line = 2 ** required_bits
        self.h_line = 2 ** (required_bits + 1)
        self.bomb = self.v_line + self.h_line

        self.token_type_mask = self.bomb

        np.random.seed(seed)
        matches = self.get_matches()
        while len(matches) > 0:
            for match in matches:
                for cell in match:
                    self.array[cell] = np.random.choice(range(self.types))
            matches = self.get_matches()

        self.actions = self.valid_actions()

    def get_token_element(self, token):
        return self.element_mask & token if not self.is_big_bad(token) else NONE_TOKEN

    def get_token_type(self, token):
        return self.token_type_mask & token

    def is_big_bad(self, token):
        return self.big_bad & token != 0

    def fill_board(self):
        self.seed += 1
        np.random.seed(self.seed)
        for row in range(self.height):
            for col in range(self.width):
                if self.array[row, col] != NONE_TOKEN:
                    continue
                self.array[row, col] = np.random.choice(range(self.types))

    def remove_matches(self, matches: list[list[tuple[int, int]]]) -> np.array:
        for match in matches:
            for row, col in match:
                self.array[row, col] = NONE_TOKEN

    def drop(self):
        def drop_column(c):
            col = self.array[:, c]

            tokens = col[col != NONE_TOKEN]
            non_tokens = np.full(self.height - tokens.size, NONE_TOKEN, dtype=np.int32)

            return np.concatenate((non_tokens, tokens))

        for column in range(self.width):
            self.array[:, column] = drop_column(column)

    def swap(self, action: int) -> tuple[int, Swap_Event]:
        self.seed += 1
        np.random.seed(self.seed)
        assert action in self.actions
        initial_obs = np.copy(self.array)
        self.swap_tokens(action)

        source, target = self.decode_action(action)

        matches = [match for match in [self.match_at(source), self.match_at(target)] if match]

        if self.is_big_bad(self.array[source]) or self.is_big_bad(self.array[target]):
            element = self.get_token_element(self.array[target if self.is_big_bad(self.array[source]) else source])
            matches.append([source if self.is_big_bad(self.array[source]) else target])
            for row in range(self.height):
                for col in range(self.width):
                    cell = (row, col)
                    if self.get_token_element(self.array[cell]) == element and not any([cell in match for match in matches]):
                        matches.append([cell])
        else:
            def select(start, end):
                start_row, start_col = start
                matches.append([start])
                while start != end:
                    if start[1] != end[1]:
                        start = (start_row, start[1] + 1)
                    else:
                        start = (start[0] + 1, start_col)
                    if 0 <= start[0] < self.height and 0 <= start[1] < self.width and not any([start in match for match in matches]):
                        matches.append([start])

            def handle_type(position):
                match self.get_token_type(self.array[position]):
                    case self.v_line:
                        select((0, position[1]), (self.height-1, position[1]))
                    case self.h_line:
                        select((position[0], 0), (position[0], self.width-1))
                    case self.bomb:
                        select((position[0]-1, position[1]-1), (position[0]+1, position[1]+1))

                match = self.match_at(position)
                if not match:
                    return

                if len(match) > 3:
                    shape = get_match_shape(match)
                    if shape == T_L_SHAPE:
                        self.array[position] = self.bomb + self.get_token_element(self.array[position])
                    elif len(match) >= 5:
                        self.array[position] = self.big_bad
                    elif shape == V_LINE:
                        self.array[position] = self.v_line + self.get_token_element(self.array[position])
                    else:
                        self.array[position] = self.h_line + self.get_token_element(self.array[position])
                    for match in matches:
                        if position in match:
                            match.remove(position)

            match (self.get_token_type(self.array[source]), self.get_token_type(self.array[target])):
                case (self.bomb, self.h_line):
                    select((source[0]-1, 0), (source[0]+1, self.width-1))
                case (self.bomb, self.v_line):
                    select((0, source[1]-1), (self.height-1, source[0]+1))
                case(self.v_line, self.bomb):
                    select((target[0]-1, 0), (target[0]+1, self.width-1))
                case(self.h_line, self.bomb):
                    select((0, target[1]-1), (self.height-1, target[0]+1))
                case _:
                    handle_type(source)
                    handle_type(target)

        events = []

        # while any matches exists remove them and update array
        while True:
            before_removing_match = np.copy(self.array)
            self.remove_matches(matches)
            after_removing_match = np.copy(self.array)

            self.drop()
            self.fill_board()
            events.append((before_removing_match, after_removing_match, np.copy(self.array)))

            matches = self.get_matches()
            if len(matches) == 0:
                break
        self.actions = self.valid_actions()
        return 0, Swap_Event(action, initial_obs, events)

    def swap_tokens(self, action: int) -> np.array:
        source, target = self.decode_action(action)
        source_value = self.array[source]
        self.array[source] = self.array[target]
        self.array[target] = source_value

    def get_matches(self) -> list[list[tuple[int, int]]]:
        matches = []
        for row in range(self.height):
            for col in range(self.width):
                if any([(row, col) in match for match in matches]):
                    continue
                match_at = self.match_at((row, col))
                if match_at:
                    matches.append(match_at)
        return matches

    def match_at(self, cell) -> None | list[tuple[int, int]]:
        row, col = cell
        token = self.get_token_element(self.array[cell])
        match = []

        def scan(row_offset, col_offset):
            next_row = row
            next_col = col

            scan_match = []

            while (0 <= next_row - row_offset and 0 <= next_col - col_offset and
                   self.array[next_row - row_offset, next_col - col_offset] != NONE_TOKEN and
                   self.get_token_element(self.array[next_row - row_offset, next_col - col_offset]) == token):
                next_row -= row_offset
                next_col -= col_offset
            scan_match.append((next_row, next_col))
            while (next_row + row_offset < self.height and next_col + col_offset < self.width and
                   self.array[next_row + row_offset, next_col + col_offset] != NONE_TOKEN and
                   self.get_token_element(self.array[next_row + row_offset, next_col + col_offset]) == token):
                next_row += row_offset
                next_col += col_offset
                scan_match.append((next_row, next_col))

            if len(scan_match) > 2:
                for scan_rc in scan_match:
                    if scan_rc not in match and scan_rc != cell:
                        match.append(scan_rc)

        scan(1, 0)
        scan(0, 1)

        if len(match) < 2:
            match = None
        else:
            for row, col in match:
                scan(1, 0)
                scan(0, 1)
            match = [cell] + match
        return match

    def is_valid_action(self, action: int) -> bool:
        cell1, cell2 = self.decode_action(action)
        if self.get_token_element(self.array[cell1]) == self.get_token_element(self.array[cell2]):
            return False
        board = np.copy(self.array)
        self.swap_tokens(action)
        is_valid = self.match_at(cell1) is not None or self.match_at(cell2) is not None
        self.array = board
        return is_valid

    def encode_action(self, tile1: tuple[int, int], tile2: tuple[int, int]) -> int:
        source_row, source_column = tile1
        target_row, target_column = tile2
        assert (source_column == target_column and abs(source_row - target_row) == 1 or
                source_row == target_row and abs(source_column - target_column) == 1), \
            'source and target must be adjacent'
        width = self.array.shape[1]
        a = 2 * width - 1
        b = width - 1 if source_column == target_column else 0
        return min(source_row, target_row) * a + b + min(source_column, target_column)

    def decode_action(self, action: int) -> tuple[tuple[int, int], tuple[int, int]]:
        width = self.array.shape[1]
        a = (2 * width - 1)
        b = width - 1
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

    def valid_actions(self) -> list[int]:
        actions = self.height * (self.width - 1) + self.width * (self.height - 1)
        return [i for i in range(actions) if self.is_valid_action(i)]

    def argmax(self, pred: np.array) -> int | list[int]:
        def argmax_batch(batch):
            batch = [v if self.is_valid_action(i) else -1 for i, v in enumerate(batch)]
            return np.argmax(np.array(batch)).item()
        if len(pred.shape) == 1:
            return argmax_batch(pred)
        if len(pred.shape) == 2:
            return [argmax_batch(p) for p in pred]

    def random_action(self) -> int:
        return random.choice(self.actions)
