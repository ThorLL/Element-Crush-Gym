import math
from collections import namedtuple, deque
from pickle import NONE
from random import choice

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
    def __init__(self, shape, seed=0, init_board=None):
        self.seed = seed
        self.height, self.width, self.types = shape
        assert self.types > 1

        required_bits = int(math.ceil(math.log2(self.types)))
        self.element_mask = (2 ** required_bits) - 1
        self.big_bad = 2 ** (required_bits + 2)

        self.v_line = 2 ** required_bits
        self.h_line = 2 ** (required_bits + 1)
        self.bomb = self.v_line + self.h_line

        self.token_type_mask = self.bomb

        if init_board is None:
            self.array = np.full((self.height, self.width), NONE_TOKEN, dtype=np.int32)
            self.fill_board()

            np.random.seed(seed)
            matches = self.get_matches()
            while len(matches) > 0:
                for match in matches:
                    for cell in match:
                        self.array[cell] = np.random.choice(range(self.types))
                matches = self.get_matches()
        else:
            self.array = init_board
        self.actions = self.get_valid_actions()

    def clone(self):
        return Board((self.height, self.width, self.types), self.seed, np.copy(self.array))

    def get_token_element(self, token):
        return self.element_mask & token if not self.is_big_bad(token) else NONE_TOKEN
    
    def quick_get_token_element(self, token):
        """ 
        Equivalent to get_token_element but does not check if the token is a big bad. 
        Ends up saving a lot of function calls and time 
        """
        return self.element_mask & token

    def get_token_type(self, token):
        return self.token_type_mask & token

    def is_big_bad(self, token):
        return self.big_bad & token != 0

    def is_special(self, token):
        """
        Returns True if the token is a special token (line or bomb)
        """
        return self.token_type_mask & token != 0

    def is_in_bounds(self, row, col):
        """
        Returns True if the cell is within the bounds of the board
        """
        return 0 <= row < self.height and 0 <= col < self.width

    def fill_board(self):
        self.seed = (1 + self.seed) % (2**32 - 1)
        np.random.seed(self.seed)
        changed = []
        for row in range(self.height):
            for col in range(self.width):
                if self.array[row, col] != NONE_TOKEN:
                    continue
                self.array[row, col] = np.random.choice(range(self.types))
                changed.append((row, col))

        if len(self.get_valid_actions()) == 0:
            for (row, col) in changed:
                self.array[row, col] = NONE_TOKEN
            self.fill_board()

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

    def select(self, start, end, matches):
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
        return matches

    def handle_type(self, position, matches):
        match self.get_token_type(self.array[position]):
            case self.v_line:
                self.select((0, position[1]), (self.height-1, position[1]), matches)
            case self.h_line:
                self.select((position[0], 0), (position[0], self.width-1), matches)
            case self.bomb:
                self.select((position[0]-1, position[1]-1), (position[0]+1, position[1]+1), matches)
        return matches

    def handle_swap_event(self, source, target, matches):
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
                                if self.get_token_type(self.array[row, col]) == 0:
                                    self.array[row, col] = element + (self.v_line if count % 2 == 0 else self.h_line)
                                    count += 1

                            matches = self.handle_type((row, col), matches)
            else:  # normal token
                for row in range(self.height):
                    for col in range(self.width):
                        if self.get_token_element(self.array[row, col]) == element:
                            matches.append([(row, col)])

        else:
            match (self.get_token_type(self.array[source]), self.get_token_type(self.array[target])):
                case (self.h_line, self.v_line) | (self.v_line, self.h_line) | (self.v_line, self.v_line) | (self.h_line, self.h_line):
                    matches = self.select((0, source[1]), (self.height - 1, source[1]), matches)
                    matches = self.select((source[0], 0), (source[0], self.width-1), matches)
                case (self.h_line, self.bomb) | (self.v_line, self.bomb) | (self.bomb, self.v_line) | (self.bomb, self.h_line):
                    matches = self.select((0, source[1]-1), (self.height - 1, source[1]+1), matches)
                    matches = self.select((source[0]-1, 0), (source[0]+1, self.width-1), matches)
                case(self.bomb, self.bomb):
                    matches = self.select((source[0]-2, source[1]-2), (source[0]+2, source[1]+2), matches)
                case _:
                    if any([source in match for match in matches]):
                        matches = self.handle_type(source, matches)
                    if any([target in match for match in matches]):
                        matches = self.handle_type(target, matches)
        return matches

    def swap(self, action: int, naive=False, recalculate_action=True) -> tuple[int, Swap_Event]:
        self.seed = (1 + self.seed) % (2**32 - 1)
        np.random.seed(self.seed)
        assert action in self.actions
        initial_obs = np.copy(self.array)
        self.swap_tokens(action)

        source, target = self.decode_action(action)

        # TODO: Here we should add logic that checks for big bads (and maybe specials?) such that we avoid calls to match_at
        # If one of the tokens is a big bad we can simply add the cells 1 above, below, left and right to the match
        # if self.is_big_bad(self.array[source]):
        #     for row, col in [(source[0] - 1, source[1]), (source[0] + 1, source[1]), (source[0], source[1] - 1), (source[0], source[1] + 1)]:
        #         if self.is_in_bounds(row, col):
        #             matches.append([(row, int(col))])
    
        # elif self.is_big_bad(self.array[target]):
        #     for row, col in [(target[0] - 1, target[1]), (target[0] + 1, target[1]), (target[0], target[1] - 1), (target[0], target[1] + 1)]:
        #         if self.is_in_bounds(row, col):
        #             matches.append([(row, int(col))])
        # else:
        matches = [match for match in [self.match_at(source), self.match_at(target)] if match]



        matches = self.handle_swap_event(source, target, matches)

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
                    matches = self.handle_type(token, matches)

                if len(match) > 3:
                    shape = get_match_shape(match)
                    element = self.quick_get_token_element(self.array[match[0]]) # A big bad is seemingly never here
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
        if recalculate_action:
            self.actions = self.get_valid_actions()
        return points, Swap_Event(action, initial_obs, events, self.actions)

    def swap_tokens(self, action: int) -> np.array:
        source, target = self.decode_action(action)
        source_value = self.array[source]
        self.array[source] = self.array[target]
        self.array[target] = source_value

    # Same as swap_tokens but with already decoded action
    def quick_swap_tokens(self, source: int, target: int) -> np.array:
        """
        Equivalent to swap_tokens but with already decoded action.
        """
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

    # Finds matches around a cell. This function is kinda bruh
    def match_at(self, cell) -> None | list[tuple[int, int]]:
        row, col = cell
        match = []

        # I don't think we should end up in here if the cell is a big bad - Fix the swap function such that this is the case
        token = self.get_token_element(self.array[cell])

        # Checks the cells in the vertical and horizontal directions
        def scan(row_offset, col_offset):
            next_row = row
            next_col = col
            scan_match = []

            # While for 
            # 1: We are in bounds
            # 2: The offset cell is not empty (NONE_TOKEN)
            # 3: The token has the same element as the cell we are scanning from
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
        token1, token2 = self.array[cell1], self.array[cell2]
        
        # Early return for quickly identifiable actions - sorted by likelihood to save on function calls
        # If the tokens are both special tokens we CAN swap them
        if self.is_special(token1) and self.is_special(token2):
            return True
        # If one of the tokens is a big bad we CAN swap them
        if self.is_big_bad(token1) or self.is_big_bad(token2):
            return True
        # If the tokens have the same elements we CANNOT swap them
        if self.quick_get_token_element(token1) == self.quick_get_token_element(token2):
            return False
        
        # Else we need to test if the swap will result in a match
        board = np.copy(self.array)
        self.quick_swap_tokens(cell1, cell2)
        is_valid = self.match_at(cell1) is not None or self.match_at(cell2) is not None
        self.array = board

        return is_valid

    def encode_action(self, tile1: tuple[int, int], tile2: tuple[int, int]) -> int:
        return self.encode(tile1, tile2, self.array)

    @staticmethod
    def encode(tile1: tuple[int, int], tile2: tuple[int, int], board):
        source_row, source_column = tile1
        target_row, target_column = tile2
        assert (source_column == target_column and abs(source_row - target_row) == 1 or
                source_row == target_row and abs(source_column - target_column) == 1), \
            'source and target must be adjacent'
        width = board.shape[1]
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

    def get_valid_actions(self) -> list[int]:
        all_actions = self.height * (self.width - 1) * 2
        valid_actions = [action for action in range(all_actions) if self.is_valid_action(action)]
        return valid_actions

    def argmax(self, pred: np.array) -> int | list[int]:
        def argmax_batch(batch):
            batch = [v if self.is_valid_action(i) else -1 for i, v in enumerate(batch)]
            return np.argmax(np.array(batch)).item()
        if len(pred.shape) == 1:
            return argmax_batch(pred)
        if len(pred.shape) == 2:
            return [argmax_batch(p) for p in pred]

    def random_action(self) -> int:
        return choice(self.actions)

    def simulate_swap(self, action: int, naive):
        board = np.copy(self.array)
        seed = self.seed
        reward, _ = self.swap(action, naive, False)
        self.array = board
        self.seed = seed
        return reward

    def naive_action(self):
        best_action = None
        best_reward = -1
        for action in self.actions:
            reward = self.simulate_swap(action, True)
            if reward > best_reward:
                best_reward = reward
                best_action = action
        return best_action

    def best_action(self) -> int:
        best_action = None
        best_reward = -1
        for action in self.actions:
            reward = self.simulate_swap(action, False)
            if reward > best_reward:
                best_reward = reward
                best_action = action
        return best_action
