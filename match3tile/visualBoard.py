import os
import time
from typing import Callable

import numpy as np
import pygame
from pygame import Surface

from match3tile.boardConfig import BoardConfig
from match3tile.boardFunctions import swap, get_matches, get_match_spawn_mask, legal_actions, shuffle
from match3tile.boardv2 import BoardV2
from util.quickMath import lower_clamp, upper_clamp

BLOCK_SIZE = 70
PADDING = 3
IMAGE_RADIUS = 32


class VisualBoard(BoardV2):
    def __init__(self, n_actions: int, animation_speed, fps, cfg=BoardConfig(), array: np.array = None):
        assert cfg.types <= 6, "can't render more than 6 types"
        super().__init__(n_actions, cfg, array)
        self.animation_speed, self.fps = animation_speed, fps

        height, width = cfg.rows, cfg.columns
        self.window_size = (BLOCK_SIZE * width + 2, BLOCK_SIZE * height)

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()

        abs_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(abs_path)
        image_names = [image for image in os.listdir(f'{dir_path}/images/default')]

        self.images = {
            0: [pygame.image.load(f"{dir_path}/images/default/{image}") for image in image_names],
            cfg.v_line: [pygame.image.load(f"{dir_path}/images/Vline/{image}") for image in image_names],
            cfg.h_line: [pygame.image.load(f"{dir_path}/images/Hline/{image}") for image in image_names],
            cfg.bomb: [pygame.image.load(f"{dir_path}/images/bomb/{image}") for image in image_names],
        }
        self.avatar = pygame.image.load(f"{dir_path}/images/bigBad.png")

    def get_token_image(self, token):
        if token == self.cfg.mega_token or 0:
            return self.avatar
        special_type = token & self.cfg.special_type_mask
        token_type = token & self.cfg.type_mask
        return self.images[special_type][token_type - 1]

    def draw_token(self, canvas: Surface, token: int, row: int, col: int, size=1):
        canvas.blit(pygame.transform.scale(self.get_token_image(token), (IMAGE_RADIUS * size, IMAGE_RADIUS * size)),
                    (col * BLOCK_SIZE + PADDING, row * BLOCK_SIZE + PADDING))

    def draw(self,
             board: np.array,
             token_draw: Callable[[Surface, int, int], None] = None,
             early_draw: Callable[[Surface], None] = None,
             late_draw: Callable[[Surface], None] = None
             ) -> float:

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        if early_draw:
            early_draw(canvas)

        if not token_draw:
            def token_draw(can, r, c):
                self.draw_token(can, board[r, c], r, c)

        for row in range(self.cfg.rows):
            for col in range(self.cfg.columns):
                token_draw(canvas, row, col)

        if late_draw:
            late_draw(canvas)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        return self.clock.tick(self.fps)

    def draw_swap(self, action: tuple[tuple[int, int], tuple[int, int]]):
        self.clock = pygame.time.Clock()
        (target_row, target_column), (source_row, source_column) = action
        swap_time = 200 / self.animation_speed
        steps = swap_time / ((1 / max(1, self.fps)) * 1000)

        source_step = ((target_row - source_row) / steps, (target_column - source_column) / steps)
        target_step = ((source_row - target_row) / steps, (source_column - target_column) / steps)

        time = 0
        draw_counter = 0

        def draw(canvas, row, col):
            if row == source_row and col == source_column:
                self.draw_token(canvas, self.array[row, col], source_row + draw_counter * source_step[0],
                                source_column + draw_counter * source_step[1])
            elif row == target_row and col == target_column:
                self.draw_token(canvas, self.array[row, col], target_row + draw_counter * target_step[0],
                                target_column + draw_counter * target_step[1])
            else:
                self.draw_token(canvas, self.array[row, col], row, col)

        while swap_time > time:
            time += self.draw(self.array, draw)
            draw_counter += 1

    def draw_actions(self, board: np.array, actions: list[int], sleep_time=2):
        def draw_action(canvas):
            for action in actions:
                source, target = self.cfg.decode(action)
                pygame.draw.line(
                    canvas,
                    (0, 255, 0),
                    ((source[1] + 0.5) * BLOCK_SIZE, (source[0] + 0.5) * BLOCK_SIZE),
                    ((target[1] + 0.5) * BLOCK_SIZE, (target[0] + 0.5) * BLOCK_SIZE),
                    5
                )

        self.draw(board, early_draw=draw_action)
        time.sleep(sleep_time)

    def draw_matches(self, board: np.array, matches: np.ndarray):
        self.clock = pygame.time.Clock()
        highlight_time = 1000 / self.animation_speed
        blinking_speed = 300 / self.animation_speed
        blink = False
        blinker = 0
        time = 0

        def blinking_token_draw(canvas, row, col):
            if matches[row, col] and blink:
                pygame.draw.circle(
                    canvas,
                    (255, 255, 255),
                    (col * BLOCK_SIZE + PADDING + IMAGE_RADIUS, row * BLOCK_SIZE + PADDING + IMAGE_RADIUS),
                    IMAGE_RADIUS
                )
            else:
                self.draw_token(canvas, board[row, col], row, col)

        def token_draw(canvas, row, col):
            if not matches[row, col]:
                self.draw_token(canvas, board[row, col], row, col)

        while highlight_time > time:
            if blinker > blinking_speed:
                blink = not blink
                blinker = 0
            delta = self.draw(board, token_draw=blinking_token_draw)
            time += delta
            blinker += delta

        show_empty = 200 / self.animation_speed
        time = 0
        while show_empty > time:
            time += self.draw(board, token_draw=token_draw)

    def draw_change(self, before, after):
        self.clock = pygame.time.Clock()
        spawn_time = 500 / self.animation_speed
        current_time = 0

        def draw_fn(canvas, r, c):
            if before[r, c] != after[r, c]:
                self.draw_token(canvas, after[r, c], r, c, current_time / spawn_time)
            else:
                self.draw_token(canvas, after[r, c], r, c)

        while current_time < spawn_time:
            current_time += self.draw(after, draw_fn)

    def apply_action(self, action) -> 'VisualBoard':
        if self.is_terminal:
            return self

        # draw current state
        self.draw(self.array)

        np.random.seed(self.cfg.seed)
        reward = 0
        source, target = self.cfg.actions[action]

        # create next board by swapping the tokens
        next_state = swap(self.array, source, target)
        self.draw_swap((source, target))

        # extract config variables
        type_mask, special_type_mask = self.cfg.type_mask, self.cfg.special_type_mask
        h_line, v_line, bomb, mega_token = self.cfg.h_line, self.cfg.v_line, self.cfg.bomb, self.cfg.mega_token
        height, width, types = self.cfg.rows, self.cfg.columns, self.cfg.types

        def point_board_vec(x):
            if x <= type_mask:  # normal token
                return 2
            if x == mega_token:  # mega token
                return 250
            if x < special_type_mask:  # line token
                return 25
            return 50  # has to be bomb

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
                lower_clamp(target[0] - 2): upper_clamp(target[0] + 2, height),
                lower_clamp(target[1] - 2): upper_clamp(target[1] + 2, width)
            ] = 0
        # bomb + (h_line or v_line):
        # . ┌---┐ .    plus shape clear with 3 x width and 3 x height
        # ┌-┘   └-┐
        # |       |
        # └-┐   ┌-┘
        # . └---┘ .
        elif are(bomb, h_line) or are(bomb, v_line):
            token_board[0: height, lower_clamp(target[1] - 2): upper_clamp(target[1] + 2, width)] = 0
            token_board[lower_clamp(target[0] - 2): upper_clamp(target[0] + 2, height), 0: width] = 0
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
            self.draw_matches(next_state, (token_board == 0))
            # trigger the effect of each special token that has been removed
            # a token is counted as removed if the token_board sub boards corresponding value is 0
            special_tokens = np.where((token_board == 0), special_tokens, 0)

            changed = False
            for i, j in np.argwhere(special_tokens != 0):
                # extract special token
                special_type = special_tokens[i, j] & special_type_mask
                # trigger effect
                match special_type:
                    case t if t == h_line:
                        token_board[i, :] = 0
                        changed = True
                    case t if t == v_line:
                        token_board[:, j] = 0
                        changed = True
                    case t if t == bomb:
                        changed = True
                        start_row, end_row = i - 1, i + 2
                        start_col, end_col = j - 1, j + 2
                        token_board[start_row:end_row, start_col:end_col] = 0
            if changed:
                self.draw_matches(next_state, (token_board == 0))
                pass
            # extract points
            points_board = points_board[(token_board == 0)]
            reward += np.sum(points_board)

            next_state_before = np.copy(next_state)
            # merge merge sub boards
            next_state[(token_board == 0)] = 0
            next_state[(token_spawn != 0)] += token_spawn[(token_spawn != 0)] + next_state_before[(token_spawn != 0)]
            next_state = np.clip(next_state, 0, 32)

            before_drops = np.copy(next_state)

            # simulate gravity
            for col in range(width):
                column = next_state[:, col]
                tokens = column[column > 0]
                if tokens.size == height:
                    continue
                # fill with new random values
                new_tokens = np.random.randint(1, types + 1, size=height - tokens.size)
                next_state[:, col] = np.concatenate((new_tokens, tokens))

            self.draw_change(before_drops, next_state)

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
                before = np.copy(next_state)
                shuffle(self.cfg, next_state)
                self.draw_change(before, next_state)
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
        board = VisualBoard(self.n_actions - 1, self.animation_speed, self.fps, self.cfg, np.copy(next_state))
        board._reward = self._reward + reward  # add reward
        return board
