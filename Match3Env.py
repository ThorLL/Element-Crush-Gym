from collections.abc import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame import Surface

from board import Board


BLOCK_SIZE = 70
PADDING = 3
IMAGE_RADIUS = 32


def draw_token(canvas, icon, row, col):
    canvas.blit(icon, (col*BLOCK_SIZE+PADDING, row*BLOCK_SIZE+PADDING))


class Match3Env(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 60, 'animation_speed': 1}

    def __init__(
            self,
            width: int = 7,
            height: int = 9,
            num_types: int = 6,
            num_moves: int = 20,
            env_goal: int = 300,
            seed: int = None,
            render_mode=None,
    ):
        assert width >= 3 and height >= 3, f"Board size too small: min size: 3x3"

        self.width = width
        self.height = height
        self.num_types = num_types  # Number of different tile types
        self.env_goal = env_goal
        self.num_moves = num_moves
        self.score, self.moves_taken = 0, 0
        self.window_size = (BLOCK_SIZE*self.width+2, BLOCK_SIZE*self.height)

        np.random.seed(seed)
        self.board = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.images = None

    def init(self) -> np.ndarray[int, np.dtype[np.int32]]:
        self.board = Board(self.width, self.height, self.num_types)
        self.action_space = spaces.Discrete(self.board.n_actions)
        self.observation_space = spaces.Box(low=1, high=self.num_types + 1, shape=(self.height, self.width), dtype=np.int32)

        return self._get_obs()

    @property
    def n_actions(self):
        return self.board.n_actions

    def step(self, action):
        source_row, source_column, target_row, target_column = self.board.decode_action(action)
        move_score = self.board.swap(source_row, source_column, target_row, target_column)
        self.score += move_score
        self.moves_taken += 1

        truncated = self.score >= self.env_goal
        done = truncated or self.num_moves == self.moves_taken

        return self._get_obs(), move_score, done, truncated, {}

    def _get_obs(self):
        return self.board.observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        del self.board
        self.board = Board(self.width, self.height, self.num_types)
        self.score, self.moves_taken = 0, 0

        return self._get_obs(), {}

    def render(self):
        if self.render_mode is None:
            return
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.images is None:
            self.images = [pygame.image.load(f"images/{i+1}.png") for i in range(self.num_types)]

        def render_animations(_):
            if self.render_mode == 'human':
                events = self.board.swap_events
                if events['valid']:
                    self.show_swap(events['action'], events['init'])
                    for state, next_state, falls in events['events']:
                        self.show_matches(state, next_state)
                        self.show_falls(falls, next_state)

        def draw(row, col, canvas):
            draw_token(canvas, self.images[self.board[row, col]-1], row, col)

        self.draw_by_function(draw, early_draw=render_animations)



    def show_swap(self, action, board):
        source_row, source_column, target_row, target_column = action
        swap_time = 200 / self.metadata['animation_speed']
        steps = swap_time / ((1 / max(1, self.metadata['render_fps'])) * 1000)

        source_step = ((target_row - source_row) / steps, (target_column - source_column) / steps)
        target_step = ((source_row - target_row) / steps, (source_column - target_column) / steps)

        time = 0
        draw_counter = 0

        def draw(row, col, canvas):
            if row == source_row and col == source_column:
                draw_token(canvas, self.images[board[row, col]-1], source_row + draw_counter * source_step[0], source_column + draw_counter * source_step[1])
            elif row == target_row and col == target_column:
                draw_token(canvas, self.images[board[row, col]-1], target_row + draw_counter * target_step[0], target_column + draw_counter * target_step[1])
            else:
                draw_token(canvas, self.images[board[row, col]-1], row, col)

        while swap_time > time:
            time += self.draw_by_function(draw)
            draw_counter += 1

    def show_matches(self, board, next_board):
        highlight_time = 1000 / self.metadata['animation_speed']
        blinking_speed = 300 / self.metadata['animation_speed']
        blink = False
        blinker = 0
        time = 0

        def draw_blinking(row, col, canvas):
            draw_token(canvas, self.images[board[row, col]-1], row, col)
            if next_board[row, col] == 0 and blink:
                pygame.draw.circle(canvas, (255, 255, 255), (col*BLOCK_SIZE+PADDING+IMAGE_RADIUS, row*BLOCK_SIZE+PADDING+IMAGE_RADIUS), IMAGE_RADIUS)

        def draw_empty(row, col, canvas):
            if next_board[row, col] != 0:
                draw_token(canvas, self.images[next_board[row, col]-1], row, col)

        while highlight_time > time:
            if blinker > blinking_speed:
                blink = not blink
                blinker = 0
            delta = self.draw_by_function(draw_blinking)
            time += delta
            blinker += delta

        show_empty = 200 / self.metadata['animation_speed']
        time = 0
        while show_empty > time:
            time += self.draw_by_function(draw_empty)

    def show_falls(self, falls, board):
        fall_time = 100 / self.metadata['animation_speed']
        falling_tokens = {}

        # TODO merge logic for tokens falling on the grid and new tokens spawning in
        for column in range(self.width):
            for row in range(self.height-1, 0, -1):
                if board[row, column] != 0:
                    continue
                n_falls = 0
                for above_row in range(row, -1, -1):
                    if board[above_row, column] != 0:
                        break
                    n_falls += 1

                for above_row in range(row, -1, -1):
                    falling_tokens[above_row, column] = (n_falls, board[above_row, column])
                break
        falling_tokens = {key: value for key, value in falling_tokens.items() if board[key] != 0}

        for column in range(self.width):
            tiles_to_fall = sum([1 if value else 0 for value in (falls[:, column] != 0)])
            if tiles_to_fall == 0:
                continue
            out_of_screen_row = 0
            for row in range(self.height - 1, -1, -1):
                if falls[row, column] == 0:
                    continue
                out_of_screen_row -= 1
                falling_tokens[(out_of_screen_row, column)] = (tiles_to_fall, falls[row, column])

        fall_counter = 0
        settled_tokens = {}

        def draw_func(row, col, canvas):
            if board[row, col] == 0 or (row, col) in falling_tokens or (row, col) in settled_tokens:
                return
            if (row, col) in falling_tokens:
                draw_token(canvas, self.images[falling_tokens[(row, col)][1]-1], fall_counter + row + time/fall_time, col)
            elif (row, col) in settled_tokens:
                draw_token(canvas, self.images[settled_tokens[(row, col)][1]-1], settled_tokens[(row, col)][0] + row, col)
            else:
                draw_token(canvas, self.images[board[row, col]-1], row, col)

        def late_draw(canvas):
            for (row, col), (f, i) in falling_tokens.items():
                draw_token(canvas, self.images[i-1], fall_counter + row + time/fall_time, col)
            for (row, col), (f, i) in settled_tokens.items():
                draw_token(canvas, self.images[i-1], f + row, col)

        while len(falling_tokens) > 0:
            time = 0
            while fall_time > time:
                time += self.draw_by_function(draw_func, late_draw=late_draw)

            fall_counter += 1
            for key, (fall_steps, token_id) in list(falling_tokens.items()):
                if fall_steps == fall_counter:
                    settled_tokens[key] = (fall_counter, token_id)
                    del falling_tokens[key]

    def draw_by_function(
            self,
            token_draw: Callable[[int, int, Surface], None],
            early_draw: Callable[[Surface], None] = None,
            late_draw: Callable[[Surface], None] = None
    ) -> float:
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        if early_draw:
            early_draw(canvas)

        for row in range(self.height):
            for col in range(self.width):
                token_draw(row, col, canvas)

        if late_draw:
            late_draw(canvas)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        return self.clock.tick(self.metadata['render_fps'])