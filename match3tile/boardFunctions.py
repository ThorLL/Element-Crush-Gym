from typing import List

import numpy as np

from match3tile.boardConfig import BoardConfig


def get_center(match: list[tuple[int, int]]) -> tuple[int, int]:
    # Sort the match by row first, then by column
    match.sort(key=lambda x: (x[0], x[1]))

    # Return the middle item in the sorted list
    return match[len(match) // 2]


def shuffle(cfg: BoardConfig, array: np.ndarray):
    np.random.seed(cfg.seed)
    special_mask = (array > cfg.type_mask)
    special_tokens = np.zeros(array.shape)
    special_tokens[special_mask] = array[special_mask]

    np.random.shuffle(array)
    array[special_mask] = special_tokens[special_mask]


def legal_actions(cfg: BoardConfig, array: np.ndarray) -> List[int]:
    height, width = cfg.shape
    actions = []

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
            below_is_same = r + 1 < height and arr[r + 1, c] == token

            if not (above_is_same or below_is_same):
                return False
            if above_is_same and below_is_same:
                return True
            if above_is_same and not below_is_same:
                return r - 2 >= 0 and arr[r - 2, c] == token
            if not above_is_same and below_is_same:
                return r + 2 < height and arr[r + 2, c] == token

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

        if a_r - 2 >= 0 and arr[a_r - 2, a_c] == arr[a_r - 1, a_c] == above_token:
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

    token_board = array & cfg.type_mask
    for action, (cell1, cell2) in cfg.actions.items():
        token1, token2 = token_board[cell1], token_board[cell2]
        # check special tokens
        if token1 == 0 or token2 == 0 or (array[cell1] > cfg.type_mask and array[cell2] > cfg.type_mask):
            actions.append(action)
            continue
        if token1 == token2:  # ignore same typed tokens
            continue
        is_vertical = cell1[1] == cell2[1]  # if columns are equal it is a vertical action
        if is_vertical:
            if vertical_check(token2, token1, cell1, cell2, token_board):
                actions.append(action)
        else:
            if horizontal_check(token2, token1, cell1, cell2, token_board):
                actions.append(action)
    return actions


def swap(array: np.ndarray, source: tuple[int, int], target: tuple[int, int]) -> np.ndarray:
    new_arr = np.copy(array)
    new_arr[source], new_arr[target] = 0, 0
    return new_arr


def get_matches(array: np.ndarray) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]]]:
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


def get_match_spawn_mask(cfg: BoardConfig, matches):
    mask = np.zeros(cfg.shape)
    for match in [match for match in matches if len(match) > 3]:
        center = get_center(match)
        if all(match[0][0] == token[0] for token in match):                  # all rows are equal
            mask[center] = cfg.mega_token if len(match) > 4 else cfg.v_line  # line matches > 5 become mega tokens
        elif all(match[0][1] == token[1] for token in match):                # all columns are equal
            mask[center] = cfg.mega_token if len(match) > 4 else cfg.h_line  # line matches > 5 become mega tokens
        else:
            mask[center] = cfg.bomb
    return mask
