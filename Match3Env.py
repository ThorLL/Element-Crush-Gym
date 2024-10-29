import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from board import Board


def draw_token(canvas, icon, row, col):
    canvas.blit(icon, (col*70+3, row*70+3))


class Match3Env(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 60, 'animation_speed': 1}

    def __init__(
            self,
            width: int = 7,
            height: int = 9,
            num_types: int = 6,
            num_moves: int = 20,
            env_goal: int = 300,
            seed: int = 0,
            render_mode=None,
    ):
        assert width * height >= 6, (f"Board size must be larger or equal to 6, width({width}) * height({height}) "
                                     f"results in a board size of {width * height}")

        self.width = width
        self.height = height
        self.num_types = num_types  # Number of different tile types
        self.env_goal = env_goal
        self.num_moves = num_moves
        self.score = 0
        self.moves_taken = 0
        self.window_size = (70*self.width+2, 70*self.height)

        np.random.seed(seed)

        self.board = None

        self.n_actions = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.images = None

    def init(self):
        self.board = Board(self.width, self.height, self.num_types)

        self.n_actions = self.height * (self.width - 1) + self.width * (self.height - 1)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=1, high=self.num_types + 1, shape=(self.height, self.width), dtype=np.int32)

        return self._get_obs()

    def encode_action(self, source, target):
        source_row, source_column = source
        target_row, target_column = target
        assert (source_column == target_column and abs(source_row - target_row) == 1 or
                source_row == target_row and abs(source_column - target_column) == 1), \
            'source and target must be adjacent'

        a = (2 * self.width - 1)
        b = self.width - 1 if source_column == target_column else 0
        return min(source_row, target_row) * a + b + min(source_column, target_column)

    def decode_action(self, action):
        a = (2 * self.width - 1)
        b = self.width - 1
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

    def step(self, action):
        source_row, source_column, target_row, target_column = self.decode_action(action)
        self.score += self.board.swap(source_row, source_column, target_row, target_column)
        observation = self.board.observation
        reward = self.score
        self.moves_taken += 1
        truncated = self.score >= self.env_goal
        done = truncated or self.num_moves == self.moves_taken

        return observation, reward, done, truncated, {}

    def _get_obs(self):
        return self.board

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        del self.board
        self.board = Board(self.width, self.height, self.num_types)
        self.score = 0
        self.moves_taken = 0

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

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        if self.render_mode == 'human':
            events = self.board.swap_events
            if events['valid']:
                self.show_swap(events['action'], events['init'])
                for state, next_state, falls in events['events']:
                    self.show_matches(state, next_state)
                    self.show_falls(falls, next_state)

        for row in range(self.height):
            for col in range(self.width):
                draw_token(canvas, self.images[self.board[row, col]-1], row, col)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        frame_rate = self.metadata['render_fps'] if self.render_mode == 'human' else 0
        self.clock.tick(frame_rate)

    def show_swap(self, action, board):
        source_row, source_column, target_row, target_column = action
        swap_time = 500 / self.metadata['animation_speed']
        steps = swap_time / ((1 / 60) * 1000)

        source_pos = (source_row, source_column)
        target_pos = (target_row, target_column)
        source_step = ((target_row - source_row) / steps, (target_column - source_column) / steps)
        target_step = ((source_row - target_row) / steps, (source_column - target_column) / steps)

        time = 0

        while swap_time > time:
            canvas = pygame.Surface(self.window_size)
            canvas.fill((255, 255, 255))
            for row in range(self.height):
                for col in range(self.width):
                    if row == source_row and col == source_column:
                        source_pos = (source_pos[0] + source_step[0], source_pos[1] + source_step[1])
                        draw_token(canvas, self.images[board[row, col]-1], source_pos[0], source_pos[1])
                    elif row == target_row and col == target_column:
                        target_pos = (target_pos[0] + target_step[0], target_pos[1] + target_step[1])
                        draw_token(canvas, self.images[board[row, col]-1], target_pos[0], target_pos[1])
                    else:
                        draw_token(canvas, self.images[board[row, col]-1], row, col)

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            time += self.clock.tick(self.metadata['render_fps'])

    def show_matches(self, board, next_board):
        highlight_time = 1000 / self.metadata['animation_speed']
        blinking_speed = 100
        blink = False
        blinker = 0
        time = 0
        while highlight_time > time:
            if blinker > blinking_speed:
                blink = not blink
                blinker = 0
            canvas = pygame.Surface(self.window_size)
            canvas.fill((255, 255, 255))
            for row in range(self.height):
                for col in range(self.width):
                    draw_token(canvas, self.images[board[row, col]-1], row, col)
                    if next_board[row, col] == 0 and blink:
                        pygame.draw.circle(canvas, (255, 255, 255), (col*70+3+32, row*70+3+32), 32)

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            delta = self.clock.tick(self.metadata['render_fps'])
            time += delta
            blinker += delta

        show_empty = 500 / self.metadata['animation_speed']
        time = 0
        while show_empty > time:
            canvas = pygame.Surface(self.window_size)
            canvas.fill((255, 255, 255))
            for row in range(self.height):
                for col in range(self.width):
                    if next_board[row, col] != 0:
                        draw_token(canvas, self.images[next_board[row, col]-1], row, col)

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            time += self.clock.tick(self.metadata['render_fps'])

    def get_falling(self, board):
        in_fall = []
        for column in range(self.width):
            for row in range(self.height-1, 0, -1):
                if board[row, column] == 0:
                    n_falls = 0
                    for above_row in range(row, -1, -1):
                        if board[above_row, column] != 0:
                            break
                        n_falls += 1

                    for above_row in range(row, -1, -1):
                        in_fall.append((n_falls, board[above_row, column], (above_row, column)))
                    break
        return [(f, image, (row, column)) for f, image, (row, column) in in_fall if board[row, column] != 0]

    def show_falls(self, falls, board):
        fall_time = 500 / self.metadata['animation_speed']
        in_fall = self.get_falling(board)
        n_falls = 0
        has_fallen = []
        newsss = []
        for col in range(self.width):
            fall_ins = 0
            fall_amount = sum([1 if value else 0 for value in (falls[:, col] != 0)])
            if fall_amount == 0:
                continue
            for row in range(self.height - 1, -1, -1):
                if falls[row, col] == 0:
                    continue
                fall_ins += 1
                in_fall.append((fall_amount, falls[row, col], (0 - fall_ins, col)))
                newsss.append((fall_amount, falls[row, col], (0 - fall_ins, col)))

        while len(in_fall) > 0:
            time = 0
            while fall_time > time:
                canvas = pygame.Surface(self.window_size)
                canvas.fill((255, 255, 255))
                for row in range(self.height):
                    for col in range(self.width):
                        if board[row, col] == 0 or (row, col) in [(r, c) for _, _, (r, c) in in_fall]:
                            continue
                        draw_token(canvas, self.images[board[row, col]-1], row, col)
                for f, i, (row, col) in in_fall:
                    draw_token(canvas, self.images[i-1], n_falls + row + time/fall_time, col)
                for f, i, (row, col) in has_fallen:
                    draw_token(canvas, self.images[i-1], f + row, col)

                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()
                time += self.clock.tick(self.metadata['render_fps'])
            n_falls += 1
            has_fallen += [(n_falls, image, (row, col)) for f, image, (row, col) in in_fall if f == n_falls]
            in_fall = [(f, image, (row, col)) for f, image, (row, col) in in_fall if f != n_falls]
