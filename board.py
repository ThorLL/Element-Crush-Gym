import numpy as np


class Board:
    def __init__(self, width: int, height: int, n_types: int):
        self.height = height
        self.width = width
        self.n_types = n_types
        self._board = np.zeros((self.height, self.width), dtype=np.int32)
        for row in range(self.height):
            for column in range(self.width):
                possible_values = set(range(1, self.n_types + 1))

                # Remove values that would cause a horizontal match
                if column >= 2 and self._board[row, column - 1] == self._board[row, column - 2]:
                    possible_values.discard(self._board[row, column - 1])

                # Remove values that would cause a vertical match
                if row >= 2 and self._board[row - 1, column] == self._board[row - 2, column]:
                    possible_values.discard(self._board[row - 1, column])

                # Randomly pick a value from the remaining possible values
                self._board[row, column] = np.random.choice(list(possible_values))
        self.swap_events = {'valid': False}

    def swap(self, source_row: int, source_column: int, target_row: int, target_column: int) -> int:
        self.swap_events = {
            'action': (source_row, source_column, target_row, target_column),
            'init': self.observation,
            'events': [],  # state, state after removing matches, falls
        }
        self._swap(source_row, source_column, target_row, target_column)
        valid = (len(self.matches_on_row(target_row)) > 0 or
                 len(self.matches_in_column(target_column)) > 0 or
                 len(self.matches_on_row(source_row)) > 0 or
                 len(self.matches_in_column(source_column)) > 0
                 )
        self.swap_events['valid'] = valid
        if not valid:  # action was invalid: undo action
            self._swap(target_row, target_column, source_row, source_column)
            return 0

        # while any matches exists remove them and update board
        score = 0
        chain_matches = 0
        while True:
            matches = self.get_matches()
            if len(matches) == 0:
                return score + 5 * (chain_matches - 1)
            before_removing_match = self.observation
            chain_matches += len(matches)
            for match in matches:
                x = len(match)
                score += 0.5 * x ** 2 + 1.5 * x + 3
                for row, column in match:
                    self._board[row, column] = 0
            after_removing_match = self.observation
            for column in range(self.width):
                col = self._board[:, column]
                self._board[:, column] = np.concatenate((np.zeros(np.sum(col == 0), dtype=np.int32), col[col != 0]))

            zero_mask = (self._board == 0)
            random_values = np.random.randint(1, self.n_types + 1, size=self._board.shape)
            falls = np.zeros(self._board.shape, dtype=np.int32)
            falls[zero_mask] = random_values[zero_mask]
            self.swap_events['events'].append((before_removing_match, after_removing_match, falls))
            self._board[zero_mask] = random_values[zero_mask]

    def _swap(self, source_row: int, source_column: int, target_row: int, target_column: int):
        temp = self._board[target_row][target_column]
        self._board[target_row][target_column] = self._board[source_row][source_column]
        self._board[source_row][source_column] = temp

    @property
    def observation(self):
        return np.copy(self._board)

    @staticmethod
    def search_sequence_for_matches(lst: list[int]) -> list[list[int]]:
        sequences = []
        start = 0

        while start < len(lst) - 1:
            end = start
            while end + 1 < len(lst) and lst[end] == lst[end + 1]:
                end += 1
            if end - start + 1 > 2:
                sequences.append(list(range(start, end + 1)))
            start = end + 1

        return sequences

    def matches_in_column(self, column: int) -> list[list[int]]:
        return self.search_sequence_for_matches(self._board[:, column])

    def matches_on_row(self, row: int) -> list[list[int]]:
        return self.search_sequence_for_matches(self._board[row])

    def get_matches(self) -> list[list[tuple[int, int]]]:
        matches = []
        for row in range(self.height):
            for i, columns in enumerate(self.matches_on_row(row)):
                matches.append([(row, column) for column in columns])

        for column in range(self.width):
            for i, rows in enumerate(self.matches_in_column(column)):
                matches.append([(row, column) for row in rows])

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

    def __getitem__(self, key):
        return self._board[key]

    def __setitem__(self, key, value):
        self._board[key] = value

    def __str__(self):
        return str(self._board)

    def __del__(self):
        del self._board
