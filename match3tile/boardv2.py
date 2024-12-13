from typing import List

import numpy as np
from fontTools.misc.cython import returns

from match3tile.boardConfig import BoardConfig
from match3tile.boardFunctions import legal_actions, swap, get_matches, get_match_spawn_mask, shuffle
from mctslib.abc.mcts import State
from util.quickMath import lower_clamp, upper_clamp


class BoardV2(State):
    def __init__(self, n_actions: int, cfg=BoardConfig(), array: np.array = None):
        self.cfg = cfg
        self.n_actions = n_actions

        self._reward = 0
        if array is not None:
            self.array = array
        else:
            np.random.seed(cfg.seed)
            self.array = np.random.randint(low=1, high=cfg.types + 1, size=(cfg.rows, cfg.columns))

            mask, matches = get_matches(self.array)
            while len(matches) > 0:
                new_rands = np.random.randint(low=1, high=cfg.types + 1, size=(cfg.rows, cfg.columns))
                self.array[mask] = new_rands[mask]
                mask, matches = get_matches(self.array)

        self._actions = []

    @property
    def legal_actions(self) -> List[int]:
        if len(self._actions) == 0:
            self._actions = legal_actions(self.cfg, self.array)
        return self._actions

    def clone(self) -> 'BoardV2':
        my_clone = BoardV2(self.n_actions, self.cfg, np.copy(self.array))
        my_clone._reward = self._reward
        my_clone._actions = self._actions
        return my_clone

    def apply_action(self, action) -> 'BoardV2':
        if self.is_terminal:
            return self
        np.random.seed(self.cfg.seed)
        reward = 0
        source, target = self.cfg.actions[action]

        # create next board by swapping the tokens
        next_state = swap(self.array, source, target)

        # extract config variables
        type_mask, special_type_mask = self.cfg.type_mask, self.cfg.special_type_mask
        h_line, v_line, bomb, mega_token = self.cfg.h_line, self.cfg.v_line, self.cfg.bomb, self.cfg.mega_token
        height, width, types = self.cfg.rows, self.cfg.columns, self.cfg.types

        def point_board_vec(x):
            if x <= type_mask:         # normal token
                return 2
            if x >= mega_token:        # mega token
                return 250
            if x < special_type_mask:  # line token
                return 25
            return 50                   # has to be bomb

        # create sub boards
        points_board = np.vectorize(point_board_vec)(next_state)
        special_tokens = np.where(next_state > type_mask, next_state, 0)
        token_board = next_state & type_mask
        token_spawn = np.zeros(next_state.shape, dtype=np.int32)

        token1, token2 = self.array[source], self.array[target]
        token1_type, token2_type = special_tokens[source], special_tokens[target]

        token1_type = mega_token if token1 > mega_token else token1_type
        token2_type = mega_token if token2 > mega_token else token2_type


        def are(type1, type2):
            return (token1_type == type1 and token2_type == type2) or (token2_type == type1 and token1_type == type2)

        # removes all tokens
        if are(mega_token, mega_token):
            token_board[...] = 0
        # mega token + bomb (type: t) converts all non-special tokens of type t to bombs
        elif are(mega_token, bomb):
            token = max(token1, token2)  # one is 0 (mega token) other is the matched token
            token_mask = (token_board == token)
            special_mask = (special_tokens == 0)
            mask = token_mask & special_mask
            special_tokens[mask] = token + bomb
        # mega token + (v/h)_line (type: t) converts all non-special tokens of type t to alternating v and h lines
        elif are(mega_token, h_line) or are(mega_token, v_line):
            token = max(token1, token2)  # one is 0 (mega token) other is the matched token
            token_mask = (token_board == token)
            special_mask = (special_tokens == 0)
            mask = token_mask & special_mask  # mask of tokens with same type but not special
            token_board[mask] = 0
            for n, (i, j) in enumerate(np.argwhere(mask)):
                if special_tokens[i, j] == 0:
                    special_tokens[i, j] = v_line if n % 2 == 0 else h_line
        # mega token + normal token (type: t) removes all tokens of type t
        elif are(mega_token, 0):
            token = max(token1, token2)  # one is 0 (mega token) other is the matched token
            token_board[(token_board == token)] = 0
        # bomb + bomb
        # . . . . . . .
        # . ┌-------┐ .
        # . |       | .
        # . |  5x5  | .
        # . |       | .
        # . └-------┘ .
        # . . . . . . .
        elif are(bomb, bomb):
            token_board[
                lower_clamp(target[0]-2): upper_clamp(target[0]+2, height),
                lower_clamp(target[1]-2): upper_clamp(target[1]+2, width)
            ] = 0
        # bomb + (h_line or v_line):
        # . ┌---┐ .    plus shape clear with 3 x width and 3 x height
        # ┌-┘   └-┐
        # |       |
        # └-┐   ┌-┘
        # . └---┘ .
        elif are(bomb, h_line) or are(bomb, v_line):
            token_board[0: height, lower_clamp(target[1]-2): upper_clamp(target[1]+2, width)] = 0
            token_board[lower_clamp(target[0]-2): upper_clamp(target[0]+2, height), 0: width] = 0
        # two lines h or v
        # . | .    plus shape clear with 1 x width and 1 x height
        # --┼--
        # . | .
        elif are(h_line, v_line) or are(v_line, h_line):
            token_board[:target[1]] = 0
            token_board[target[0]:] = 0
        else:
            zeros_mask, matches = get_matches(token_board)
            token_board[zeros_mask] = 0
            token_spawn = get_match_spawn_mask(self.cfg, matches)

        while True:
            # trigger the effect of each special token that has been removed
            # a token is counted as removed if the token_board sub boards corresponding value is 0
            special_tokens = np.where((token_board == 0), special_tokens, 0)
            for i, j in np.argwhere(special_tokens != 0):
                # extract special token
                special_type = special_tokens[i, j] & special_type_mask
                # trigger effect
                match special_type:
                    case t if t == h_line:
                        token_board[i, :] = 0
                    case t if t == v_line:
                        token_board[:, j] = 0
                    case t if t == bomb:
                        start_row, end_row = i - 1, i + 2
                        start_col, end_col = j - 1, j + 2
                        token_board[start_row:end_row, start_col:end_col] = 0

            # extract points
            points_board = points_board[(token_board == 0)]
            reward += np.sum(points_board)

            # merge merge sub boards
            next_state[(token_board == 0)] = 0
            next_state[(token_spawn != 0)] += token_spawn[(token_spawn != 0)]

            # simulate gravity
            for col in range(width):
                column = next_state[:, col]
                tokens = column[column > 0]
                if tokens.size == height:
                    continue
                # fill with new random values
                new_tokens = np.random.randint(1, types + 1, size=height - tokens.size)
                next_state[:, col] = np.concatenate((new_tokens, tokens))

            # create new sub boards
            points_board = np.vectorize(point_board_vec)(next_state)
            special_tokens = np.where(next_state > type_mask, next_state, 0)
            token_board = next_state & type_mask

            # check for new matches
            zeros_mask, matches = get_matches(token_board)

            # moves | matches | should shuffle
            #  = 0  |   = 0   |     yes
            #  > 0  |   = 0   |      no
            #  > 0  |   > 0   |      no
            #  = 0  |   > 0   |      no
            while len(matches) == 0 and len(legal_actions(self.cfg, next_state)) == 0:
                shuffle(self.cfg, next_state)
                # create sub boards
                points_board = np.vectorize(point_board_vec)(next_state)
                special_tokens = np.where(next_state > type_mask, next_state, 0)
                token_board = next_state & type_mask
                zeros_mask, matches = get_matches(token_board)
            if len(matches) == 0:
                break

            # remove matches
            token_board[zeros_mask] = 0

            # get newly spawned token
            token_spawn = get_match_spawn_mask(self.cfg, matches)

        # creates the next board state object from the
        board = BoardV2(self.n_actions - 1, self.cfg, np.copy(next_state))
        board._reward = self._reward + reward  # add reward
        return board

    @property
    def greedy_action(self) -> int:
        best_action = None
        hightest_reward = -1
        state = BoardV2(self.n_actions, self.cfg, self.array)
        for action in self.legal_actions:
            next_state = state.apply_action(action)
            if next_state.reward > hightest_reward:
                hightest_reward = next_state.reward
                best_action = action
        return best_action

    @property
    def is_terminal(self) -> bool:
        return self.n_actions < 1

    @property
    def reward(self) -> float:
        return self._reward
