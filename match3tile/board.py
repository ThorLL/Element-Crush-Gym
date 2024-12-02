import math
from collections import namedtuple

import numpy as np

Swap_Event = namedtuple('Swap_Event', ['action', 'init_board', 'event_boards', 'actions'])

NONE_TOKEN = -2147483648


# match shapes
V_LINE = 0
H_LINE = 1
T_L_SHAPE = 2


# points:
NORMAL_TOKEN = 2
LINE = 25
BOMB = 50
BIG_BAD = 250


def get_match_shape(match):
    h_line = all(match[0][0] == token[0] for token in match)
    v_line = all(match[0][1] == token[1] for token in match)
    return T_L_SHAPE if not h_line and not v_line else H_LINE if h_line else V_LINE


def get_center(match):
    shape = get_match_shape(match)
    if shape == V_LINE or shape == H_LINE:
        return match[int(len(match) / 2)]
    else:
        for a in match:
            v_line = [a]
            h_line = [a]
            a_row, a_col = a
            for b in match:
                b_row, b_col = b
                if a == b:
                    continue

                if a_row == b_row:
                    h_line.append(b)
                if a_col == b_col:
                    v_line.append(b)

            if len(v_line) > 2 and len(h_line) > 2:
                return a


class Board:
    def __init__(self, shape, seed=0):
        self.seed = seed
        self.height, self.width, self.types = shape
        assert self.types > 1
        self.array = np.full((self.height, self.width), NONE_TOKEN, dtype=np.int32)
        self.fill_board()

        """
        example of 6 tokens:
        x    x x   x x x
        
        0    x x   0 0 0 = 0 = wind
        0    x x   0 0 1 = 1 = dark
        0    x x   0 1 0 = 2 = earth
        0    x x   0 1 1 = 3 = fire
        0    x x   1 0 0 = 4 = sun
        0    x x   1 0 1 = 5 = water
        
        element mask:
        0    0 0   1 1 1 = 7
        
        types:
        x    0 0   x x x = +0   default
        x    0 1   x x x = +8   v_line
        x    1 0   x x x = +16  h_line
        x    1 1   x x x = +24  bomb
        
        type mask:
        
        x    1 1   0 0 0 = 24
        
        
        big bad = 
        
        1   0 0    0 0 0 = 32
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

    def swap(self, action: int, naive=False) -> tuple[int, Swap_Event]:
        self.seed += 1
        np.random.seed(self.seed)
        assert action in self.actions
        initial_obs = np.copy(self.array)
        self.swap_tokens(action)

        source, target = self.decode_action(action)

        matches = [match for match in [self.match_at(source), self.match_at(target)] if match]

        def select(start, end):
            start_row, start_col = start
            if 0 <= start[0] < self.height and 0 <= start[1] < self.width and not any([start in match for match in matches]):
                matches.append([start])
            while start != end:
                if start[1] != end[1]:
                    start = (start[0], start[1] + 1)
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

        if self.is_big_bad(self.array[source]) and self.is_big_bad(self.array[target]):
            matches = [[(row, col)] for row in range(self.height) for col in range(self.width)]
            self.array = np.full(self.array.shape, NONE_TOKEN, dtype=np.int32)
        elif self.is_big_bad(self.array[source]) or self.is_big_bad(self.array[target]):

            big_bad = target if self.is_big_bad(self.array[target]) else source
            other = source if self.is_big_bad(self.array[target]) else target
            element = self.get_token_element(self.array[other])
            matches.append([big_bad])
            if self.get_token_type(self.array[other]) != 0:  # line or bomb
                count = 0
                element_type = self.get_token_type(self.array[other])
                for row in range(self.height):
                    for col in range(self.width):
                        if self.get_token_element(self.array[row, col]) == element:
                            if element_type == self.bomb:
                                self.array[row, col] = self.array[other]
                            else:
                                if self.get_token_type(self.array[row, col]) != 0:
                                    self.array[row, col] = element + (self.v_line if count % 2 == 0 else self.h_line)
                                    count += 1
                            handle_type((row, col))
            else:  # normal token
                for row in range(self.height):
                    for col in range(self.width):
                        if self.get_token_element(self.array[row, col]) == element:
                            matches.append([(row, col)])

        else:
            match (self.get_token_type(self.array[source]), self.get_token_type(self.array[target])):
                case (self.h_line, self.v_line) | (self.v_line, self.h_line) | (self.v_line, self.v_line) | (self.h_line, self.h_line):
                    select((0, source[1]), (self.height - 1, source[1]))
                    select((source[0], 0), (source[0], self.width-1))
                case (self.h_line, self.bomb) | (self.v_line, self.bomb) | (self.bomb, self.v_line) | (self.bomb, self.h_line):
                    select((0, source[1]-1), (self.height - 1, source[1]+1))
                    select((source[0]-1, 0), (source[0]+1, self.width-1))
                case(self.bomb, self.bomb):
                    select((source[0]-2, source[1]-2), (source[0]+2, source[1]+2))
                case _:
                    if any([source in match for match in matches]):
                        handle_type(source)
                    if any([target in match for match in matches]):
                        handle_type(target)

        events = []

        # while any matches exists remove them and update array
        points = 0
        while True:
            for match in matches:
                for token in match:
                    if self.is_big_bad(self.array[token]):
                        points += BIG_BAD
                    else:
                        element_type = self.get_token_type(self.array[token])
                        match element_type:
                            case 0:
                                points += NORMAL_TOKEN
                            case self.h_line | self.v_line:
                                points += LINE
                            case self.bomb:
                                points += BOMB
                    handle_type(token)

                if len(match) > 3:
                    shape = get_match_shape(match)
                    element = self.get_token_element(self.array[match[0]])
                    center = get_center(match)
                    if shape == T_L_SHAPE:
                        self.array[center] = self.bomb + element
                    elif len(match) >= 5:
                        self.array[center] = self.big_bad
                    elif shape == V_LINE:
                        self.array[center] = self.v_line + element
                    else:
                        self.array[center] = self.h_line + element
                    for match in matches:
                        if center in match:
                            match.remove(center)
            if naive:
                break
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
        return points, Swap_Event(action, initial_obs, events, self.actions)

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
        if self.get_token_type(self.array[cell1]) != 0 and self.get_token_type(self.array[cell2]) != 0:
            return True
        if self.is_big_bad(self.array[cell1]) or self.is_big_bad(self.array[cell2]):
            return True
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
        return self.decode(action, self.array)

    @staticmethod
    def decode(action, board):
        width = board.shape[1]
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
        self.seed += 1
        np.random.seed(self.seed)
        return np.random.choice(self.actions)

    def fully_simulated_swap(self, action: int):
        board = np.copy(self.array)
        actions = self.actions
        reward, _ = self.swap(action, naive=True)
        self.array = board
        self.actions = actions
        return reward

    def best_action(self) -> int:
        best_action = None
        best_reward = -1
        for action in self.actions:
            reward = self.fully_simulated_swap(action)
            if reward > best_reward:
                best_reward = reward
                best_action = action

        return best_action

