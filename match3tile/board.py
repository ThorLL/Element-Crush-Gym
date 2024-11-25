from collections import namedtuple

import numpy as np

Swap_Event = namedtuple('Swap_Event', ['action', 'init_board', 'event_boards'])


class Board:
    def __init__(self, shape, seed=0):
        self.seed = seed
        self.array, self.actions = self.generate_board(shape, seed)
        assert len(self.actions) > 0
        self.types = shape[-1]

    @staticmethod
    def generate_board(shape, seed):
        np.random.seed(seed)
        height, width, types = shape
        array = np.zeros((height, width), dtype=np.int32)

        cell_value_dict = {}

        column = 0
        row = 0
        reversing = False

        def get_possible_values(r, c, arr, min_val, max_val):
            values = set(range(min_val, max_val))

            if c >= 2 and arr[r, c - 1] == arr[r, c - 2]:
                values.discard(arr[r, c - 1].item())

            if r >= 2 and arr[r - 1, c] == arr[r - 2, c]:
                values.discard(arr[r - 1, c].item())
            return values

        def prev_cell(r, c):
            if (r, c) in cell_value_dict:
                del cell_value_dict[(row, column)]
            return (r-1, width-1) if c == 0 else (r, c - 1)

        def next_cell(r, c):
            return (r+1, 0) if c == width - 1 else (r, c+1)

        def aux(values, r, c):
            if len(values) == 0:
                return 0, prev_cell(r, c), True
            else:
                return np.random.choice(np.array(list(values))), next_cell(r, c), False

        cnt = 0
        while row < height and column < width:
            cnt += 1
            assert height * width * 1000 > cnt
            possible_values = get_possible_values(row, column, array, 1, types+1)

            if reversing:
                if len(possible_values) == 1:
                    array[row, column] = 0
                    row, column = prev_cell(row, column)
                    continue

                cell_value = array[row, column].item()
                if (row, column) in cell_value_dict:
                    cell_value_dict[(row, column)].append(cell_value)
                else:
                    cell_value_dict[(row, column)] = [cell_value]

                for value in cell_value_dict[(row, column)]:
                    possible_values.discard(value)

            array[row, column], (row, column), reversing = aux(possible_values, row, column)
        actions = Board.valid_actions(array)

        return array, actions

    @staticmethod
    def remove_matches(matches: list[list[tuple[int, int]]], board: np.array) -> np.array:
        board = np.copy(board)
        for match in matches:
            for row, col in match:
                board[row, col] = 0
        return board

    @staticmethod
    def drop(board: np.array):
        height, width = board.shape
        board = np.copy(board)

        def drop_column(c):
            col = board[:, c]

            # Move non-zero elements to the bottom and fill the top with zeros
            non_zeros = col[col != 0]
            zeros = np.zeros(height - non_zeros.size, dtype=np.int32)

            return np.concatenate((zeros, non_zeros))

        for column in range(width):
            board[:, column] = drop_column(column)
        return board

    def simulate_swap(self, action: int) -> int:
        board = np.copy(self.array)
        board = Board.swap_tokens(action, board)
        tokens_matched = 0
        chain_matches = 0
        matches = Board.get_matches(board)
        while len(matches) > 0:
            chain_matches += len(matches)
            tokens_matched += sum([len(match) for match in matches])
            board = Board.remove_matches(matches, board)
            board = Board.drop(board)
            matches = Board.get_matches(board)

        return tokens_matched + tokens_matched * (chain_matches - 1)

    def fully_simulated_swap(self, action: int):
        board = np.copy(self.array)
        actions = self.actions
        reward, _, _ = self.swap(action)
        self.array = board
        self.actions = actions
        return reward

    def swap(self, action: int) -> tuple[int, np.array, Swap_Event]:
        np.random.seed(self.seed)
        assert action in self.actions
        initial_obs = self.array
        array = Board.swap_tokens(action, self.array)
        # while any matches exists remove them and update array
        tokens_matched = 0
        chain_matches = 0
        events = []

        while True:
            matches = Board.get_matches(array)
            if len(matches) == 0:
                break
            before_removing_match = array
            chain_matches += len(matches)
            tokens_matched += sum([len(match) for match in matches])
            array = Board.remove_matches(matches, array)
            after_removing_match = array
            array = Board.drop(array)

            while True:
                array_cpy = np.copy(array)
                zero_mask = (array == 0)

                random_values = np.random.randint(1, self.types + 1, size=array.shape, dtype=np.int32)

                falls = np.zeros(array.shape, dtype=np.int32)
                falls[zero_mask] = random_values[zero_mask]
                array[zero_mask] = random_values[zero_mask]

                events.append((before_removing_match, after_removing_match, falls))
                actions = Board.valid_actions(array)
                if len(actions) > 0:
                    self.actions = actions
                    break
                else:
                    array = array_cpy
                    del events[-1]
        self.array = array
        return tokens_matched + tokens_matched * (chain_matches - 1), array, Swap_Event(action, initial_obs, events)

    def random_action(self) -> int:
        np.random.seed(self.seed)
        return np.random.choice(self.actions)

    def naive_actions(self):
        actions = []
        highest_reward = 0
        for action in self.actions:
            reward = self.simulate_swap(action)
            if reward > highest_reward:
                actions = [action]
                highest_reward = reward
            elif reward == highest_reward:
                actions.append(action)
        return actions

    def naive_action(self):
        return self.naive_actions()[0]

    def best_action(self) -> int:
        best_actions = self.naive_actions()
        if len(best_actions) > 1:
            best_actions = [max(best_actions, key=lambda a: self.fully_simulated_swap(a))]
        return best_actions[0]

    @staticmethod
    def swap_tokens(action: int, board: np.array) -> np.array:
        board = np.copy(board)
        source_row, source_column, target_row, target_column = Board.decode_action(action, board)
        source_value = board[source_row, source_column]
        board[source_row, source_column] = board[target_row, target_column]
        board[target_row, target_column] = source_value
        return board

    @staticmethod
    def scan_for_matches(arr):
        sequences = []
        start = 0

        while start < len(arr) - 1:
            end = start
            while end + 1 < len(arr) and arr[end] == arr[end + 1]:
                end += 1
            if end - start + 1 > 2:
                sequences.append(list(range(start, end + 1)))
            start = end + 1

        return sequences

    @staticmethod
    def get_matches(board: np.array) -> list[list[tuple[int, int]]]:
        height, width = board.shape
        matches = []
        for row in range(height):
            for i, columns in enumerate(Board.scan_for_matches(board[row])):
                matches.append([(row, column) for column in columns])

        for column in range(width):
            for i, rows in enumerate(Board.scan_for_matches(board[:, column])):
                matches.append([(row, column) for row in rows])

        matches = [match for match in matches if board[match[0]] != 0]

        if len(matches) == 0:
            return []

        merged = True
        while merged:
            merged = False
            new_matches = []
            skip_indices = set()  # Track which matches have already been merged

            for i in range(len(matches)):
                if i in skip_indices:
                    continue

                current_match = set(matches[i])  # Use a set to eliminate duplicates

                for j in range(i + 1, len(matches)):
                    if j in skip_indices:
                        continue

                    next_match = set(matches[j])

                    # Check for overlap between current match and next match
                    if current_match & next_match:  # If there's an intersection
                        current_match |= next_match  # Union the two sets
                        skip_indices.add(j)  # Mark next match as merged
                        merged = True

                # Add the merged match (or unmerged if no overlap)
                new_matches.append(list(current_match))

            matches = new_matches  # Update the matches list
        return matches

    @staticmethod
    def is_valid_action(action: int, board: np.array) -> bool:
        r1, c1, r2, c2 = Board.decode_action(action, board)
        board = np.copy(board)
        board = Board.swap_tokens(action, board)
        return (len(Board.scan_for_matches(board[r1])) > 0 or
                len(Board.scan_for_matches(board[:, c1])) > 0 or
                len(Board.scan_for_matches(board[r2])) > 0 or
                len(Board.scan_for_matches(board[:, c2])) > 0
                )

    @staticmethod
    def encode_action(tile1: tuple[int, int], tile2: tuple[int, int], board: np.array) -> int:
        source_row, source_column = tile1
        target_row, target_column = tile2
        assert (source_column == target_column and abs(source_row - target_row) == 1 or
                source_row == target_row and abs(source_column - target_column) == 1), \
            'source and target must be adjacent'
        width = board.shape[1]
        a = 2 * width - 1
        b = width - 1 if source_column == target_column else 0
        return min(source_row, target_row) * a + b + min(source_column, target_column)

    @staticmethod
    def decode_action(action: int, board: np.array) -> tuple[int, int, int, int]:
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

        return row1, column1, row2, column2

    @staticmethod
    def valid_actions(board: np.array) -> list[int]:
        height, width = board.shape
        actions = height * (width - 1) + width * (height - 1)
        return [i for i in range(actions) if Board.is_valid_action(i, board)]

    @staticmethod
    def argmax(pred: np.array, board: np.array) -> int | list[int]:
        def argmax_batch(batch):
            batch = [v if Board.is_valid_action(i, board) else -1 for i, v in enumerate(batch)]
            return np.argmax(np.array(batch)).item()
        if len(pred.shape) == 1:
            return argmax_batch(pred)
        if len(pred.shape) == 2:
            return [argmax_batch(p) for p in pred]
