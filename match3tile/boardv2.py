import random
from typing import List, Any

import numpy as np

from match3tile import metadata
from mctslib.abc.mcts import State
from util.quick_math import lower_clamp, upper_clamp


def get_match_shape(match):
    h_line = all(match[0][0] == token[0] for token in match)
    v_line = all(match[0][1] == token[1] for token in match)
    return metadata.special_type_mask if not h_line and not v_line else 2 * (metadata.type_mask + 1) if h_line else metadata.type_mask + 1


def get_center(match):
    shape = get_match_shape(match)
    if shape != metadata.special_type_mask:
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


class BoardV2(State):
    def __init__(self, n_actions, array=None, seed=None):
        self.seed = seed or random.randint(0, 2 ** 31 - 1)
        self.n_actions = n_actions

        self._reward = 0
        if array is not None:
            self.array = array
        else:
            np.random.seed(self.seed)
            self.array = np.random.randint(low=1, high=metadata.types + 1, size=(metadata.rows, metadata.columns))

            mask, matches = BoardV2.get_matches(self.array)
            while len(matches) > 0:
                new_rands = np.random.randint(low=1, high=metadata.types + 1, size=(metadata.rows, metadata.columns))
                self.array[mask] = new_rands[mask]
                mask, matches = BoardV2.get_matches(self.array)

        self._actions = []

    def clone(self):
        my_clone = BoardV2(self.n_actions, np.copy(self.array), self.seed)
        my_clone._reward = self._reward
        my_clone._actions = self._actions
        return my_clone

    @property
    def action_space(self):
        return metadata.action_space

    @property
    def legal_actions(self) -> List[Any]:
        if len(self._actions) > 0:
            return self._actions
        height, width = metadata.rows, metadata.columns

        def horizontal_check(left_token, right_token, left, right, arr):
            """
            Action is to swap X and Y, only have to cells x and y
            ::
                0 0 0 0 0 0  ->  0 0 0 0 0 0
                0 0 0 0 0 0  ->  0 0 y x 0 0
                0 0 0 0 0 0  ->  0 0 y x 0 0
                0 0 X-Y 0 0  ->  y y Y-X x x
                0 0 0 0 0 0  ->  0 0 y x 0 0
                0 0 0 0 0 0  ->  0 0 y x 0 0
            """
            (l_r, l_c), (r_r, r_c) = left, right
            if l_c - 2 >= 0 and arr[l_r, l_c - 2] == arr[l_r, l_c - 1] == left_token:
                return True

            if r_c + 2 < width and arr[r_r, r_c + 1] == arr[r_r, r_c + 2] == right_token:
                return True

            def check_above_and_below(r, c, token):
                above_is_same = r - 1 >= 0 and arr[r - 1, c] == token
                below_is_same = r + 1 < metadata.rows and arr[r + 1, c] == token

                if not (above_is_same or below_is_same):
                    return False
                if above_is_same and below_is_same:
                    return True
                if above_is_same and not below_is_same:
                    return r - 2 >= 0 and arr[r - 2, c] == token
                if not above_is_same and below_is_same:
                    return r + 2 < metadata.rows and arr[r + 2, c] == token

            return check_above_and_below(l_r, l_c, left_token) or check_above_and_below(r_r, r_c, right_token)

        def vertical_check(above_token, below_token, above, below, arr):
            """
            Same idea as for horizontal_check()
            ::
                0 0  0  0 0 0  ->  0 0  x  0 0 0
                0 0  0  0 0 0  ->  0 0  x  0 0 0
                0 0 ┌Y┐ 0 0 0  ->  x x ┌X┐ x x 0
                0 0 └X┘ 0 0 0  ->  y y └Y┘ y y 0
                0 0  0  0 0 0  ->  0 0  y  0 0 0
                0 0  0  0 0 0  ->  0 0  y  0 0 0
            """
            (b_r, b_c), (a_r, a_c) = below, above
            if b_r + 2 < height and arr[b_r + 1, b_c] == arr[b_r + 2, b_c] == below_token:
                return True

            if a_r - 2 >= 0 and arr[a_r - 2, a_c] == arr[a_r -1, a_c] == above_token:
                return True

            def check_left_and_right(r, c, token):
                left_is_same = c - 1 >= 0 and arr[r, c - 1] == token
                right_is_same = c + 1 < width and arr[r, c + 1] == token

                if not (left_is_same or right_is_same):
                    return False
                if left_is_same and right_is_same:
                    return True
                if left_is_same and not right_is_same:
                    return c - 2 >= 0 and arr[r, c - 2] == token
                if not left_is_same and right_is_same:
                    return c + 2 < width and arr[r, c + 2] == token

            return check_left_and_right(b_r, b_c, below_token) or check_left_and_right(a_r, a_c, above_token)

        token_board = self.array & metadata.type_mask
        for action, (cell1, cell2) in metadata.actions.items():
            token1, token2 = token_board[cell1], token_board[cell2]
            if token1 == 0 or token2 == 0 or (self.array[cell1] > metadata.type_mask and self.array[cell2] > metadata.type_mask):  # check special tokens
                self._actions.append(action)
                continue
            if token1 == token2:  # ignore same typed tokens
                continue
            is_vertical = cell1[1] == cell2[1]  # if columns are equal it is a vertical action
            if is_vertical:
                if vertical_check(token2, token1, cell1, cell2, token_board):
                    self._actions.append(action)
            else:
                if horizontal_check(token2, token1, cell1, cell2, token_board):
                    self._actions.append(action)
        if len(self._actions) > 0:
            return self._actions
        np.random.seed(self.seed)
        special_mask = (self.array > metadata.type_mask)
        special_tokens = np.zeros(self.array.shape)
        special_tokens[special_mask] = self.array[special_mask]

        np.random.shuffle(self.array)
        self.array[special_mask] = special_tokens[special_mask]
        return self.legal_actions

    @staticmethod
    def swap(array, source, target) -> np.ndarray:
        source_value = array[source]
        array[source] = array[target]
        array[target] = source_value
        return array

    @staticmethod
    def get_matches(array):
        rows, cols = array.shape
        mask = np.zeros_like(array, dtype=bool)
        matches = []

        def add_to_matches(match):
            for idx in range(len(matches)):
                if any([item in matches[idx] for item in match]):
                    matches[idx].extend([item for item in match if item not in matches])
                    return
            matches.append(match)

        for row in range(rows):
            for col in range(cols):
                value = array[row, col]
                if value == 0 or any([(row, col) in match for match in matches]):
                    continue
                match_indices = []
                # Check horizontal match
                if col <= cols - 3 and array[row, col] == array[row, col + 1] == array[row, col + 2]:
                    k = col
                    while k < cols and array[row, k] == value:
                        match_indices.append((row, k))
                        mask[row, k] = True
                        k += 1

                # Check vertical match
                if row <= rows - 3 and array[row, col] == array[row + 1, col] == array[row + 2, col]:
                    k = row
                    while k < rows and array[k, col] == value:
                        match_indices.append((k, col))
                        mask[k, col] = True
                        k += 1
                if len(match_indices) > 2:
                    add_to_matches(match_indices)
        return mask, matches

    @staticmethod
    def get_match_spawn_mask(matches, array):
        mask = np.zeros((metadata.rows, metadata.columns))
        for match in [match for match in matches if len(match) > 3]:
            shape = get_match_shape(match)
            if shape == metadata.special_type_mask:
                mask[get_center(match)] = array[match[0]] + shape
            elif len(match) >= 5:
                mask[get_center(match)] = metadata.mega_token
            else:
                mask[get_center(match)] = array[match[0]] + shape
        return mask

    def apply_action(self, action) -> 'BoardV2':
        np.random.seed(self.seed)
        reward = 0
        source, target = metadata.actions[action]

        board = np.copy(self.array)
        board = BoardV2.swap(board, source, target)

        token1, token2 = board[source], board[target]
        type_mask, special_type_mask, mega_token = metadata.type_mask, metadata.special_type_mask, metadata.mega_token
        height, width, types = metadata.rows, metadata.columns, metadata.types

        def point_board_vec(x):
            if x <= type_mask:         # normal token
                return 2
            if x == mega_token:        # mega token
                return 250
            if x < special_type_mask:  # line token
                return 25
            return 50                   # has to be bomb

        points_board = np.vectorize(point_board_vec)(board)
        special_tokens = np.where(board > type_mask, board, 0)
        token_board = board & type_mask
        token_spawn = np.zeros(board.shape)

        if token2 > token1:  # make sure larger token is first
            token1, token2 = token2, token1
        token_type = token2 & type_mask
        match (token1, token2):
            case (t1, t2) if t1 <= type_mask and t2 <= type_mask:                # normal, normal
                zeros_mask, matches = BoardV2.get_matches(token_board)
                token_board[zeros_mask] = 0
                token_spawn = BoardV2.get_match_spawn_mask(matches, board)
            case (t1, t2) if t1 == mega_token and t2 == mega_token:              # mega, mega
                token_board.fill(0)
            case (t1, t2) if t1 == mega_token and t2 > special_type_mask:        # mega, bomb
                mask = (token_board == token_type)
                token_board[mask] = 0
                special_tokens[mask] = token_type + special_type_mask
            case (t1, t2) if t1 == mega_token and t2 > type_mask:                # mega, line
                v_line, h_line = token_type + 2 * (type_mask+1), token_type + type_mask + 1
                mask = (token_board == token_type)
                token_board[mask] = 0
                for n, (i, j)  in enumerate(np.argwhere(mask)):
                    if special_tokens[i, j] == 0:
                        special_tokens[i, j] = v_line if n % 2 == 0 else h_line
            case (t1, t2) if t1 == mega_token and t2 <= type_mask:               # mega, normal
                token_board[(token_board == token_type)] = 0
            case (t1, t2) if t1 > special_type_mask and t2 > special_type_mask:  # bomb, bomb
                token_board[
                    lower_clamp(target[0]-2): upper_clamp(target[0]+2, height),
                    lower_clamp(target[1]-2): upper_clamp(target[1]+2, width)
                ] = 0
            case (t1, t2) if t1 > special_type_mask and t2 > type_mask:          # bomb, line
                token_board[0: height, lower_clamp(target[1]-2): upper_clamp(target[1]+2, width)] = 0
                token_board[lower_clamp(target[0]-2): upper_clamp(target[0]+2, height), 0: width] = 0
            case (t1, t2) if t1 > type_mask and t2 > type_mask:                  # line, line
                token_board[:target[1]] = 0
                token_board[target[0]:] = 0
        while True:
            special_tokens = np.where((token_board == 0), special_tokens, 0)
            for i, j in np.argwhere(special_tokens != 0):
                special_type = special_tokens[i, j] & special_type_mask
                match special_type:
                    case t if t == type_mask + 1:
                        token_board[i, :] = 0
                    case t if t == 2 * (type_mask + 1):
                        token_board[:, j] = 0
                    case t if t == special_type_mask:
                        start_row, end_row = i - 1, i + 1
                        start_col, end_col = j - 1, j + 1
                        token_board[start_col:end_col, start_row:end_row] = 0
            points_board = points_board[(token_board == 0)]
            reward += np.sum(points_board)

            board[(token_board == 0)] = 0
            board[(token_spawn != 0)] = token_spawn[(token_spawn != 0)]

            for col in range(width):
                column = board[:, col]
                tokens = column[column > 0]
                if tokens.size == height:
                    continue
                new_tokens = np.random.randint(1, types + 1, size=height - tokens.size)
                board[:, col] = np.concatenate((new_tokens, tokens))

            points_board = np.vectorize(point_board_vec)(board)
            special_tokens = np.where(board > type_mask, board, 0)
            token_board = board & type_mask

            zeros_mask, matches = BoardV2.get_matches(token_board)
            if len(matches) == 0:
                break
            token_board[zeros_mask] = 0
            token_spawn = BoardV2.get_match_spawn_mask(matches, board)

        board = BoardV2(self.n_actions-1, board, seed=self.seed)
        board._reward = self._reward + reward
        return board

    @property
    def naive_action(self):
        best_action = None
        best_reward = -1
        for action in self.legal_actions:
            next_state = self.apply_action(action)
            if next_state.reward > best_reward:
                best_reward = next_state.reward
                best_action = action
        return best_action

    @property
    def best_action(self) -> int:
        best_action = None
        best_reward = -1
        for action in self.legal_actions:
            next_state = self.apply_action(action)
            if next_state.reward > best_reward:
                best_reward = next_state.reward
                best_action = action
        return best_action

    @property
    def is_terminal(self) -> bool:
        return self.n_actions == 0

    @property
    def reward(self) -> float:
        return self._reward
