import os
from typing import Callable
import numpy as np
import pygame
from pygame import Surface
from os import listdir

from match3tile.board import Board

BLOCK_SIZE = 70
PADDING = 3
IMAGE_RADIUS = 32


def draw_token(canvas: Surface, icon: Surface, row, col):
    canvas.blit(icon, (col * BLOCK_SIZE + PADDING, row * BLOCK_SIZE + PADDING))


class BoardAnimator:
    def __init__(self, animation_speed, shape, fps):
        self.animation_speed = animation_speed
        height, width = shape
        self.window_size = (BLOCK_SIZE*width+2, BLOCK_SIZE*height)
        self.fps = fps
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        abs_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(abs_path)
        self.images = {i+1: pygame.image.load(f"{dir_path}/images/{image}") for i, image in enumerate(listdir(f'{dir_path}/images'))}

    def show_swap(self, action, board):
        source_row, source_column, target_row, target_column = Board.decode_action(action, board)
        swap_time = 200 / self.animation_speed
        steps = swap_time / ((1 / max(1, self.fps)) * 1000)

        source_step = ((target_row - source_row) / steps, (target_column - source_column) / steps)
        target_step = ((source_row - target_row) / steps, (source_column - target_column) / steps)

        time = 0
        draw_counter = 0

        def draw(canvas, row, col):
            if row == source_row and col == source_column:
                draw_token(canvas, self.images[board[row, col].item()], source_row + draw_counter * source_step[0],
                           source_column + draw_counter * source_step[1])
            elif row == target_row and col == target_column:
                draw_token(canvas, self.images[board[row, col].item()], target_row + draw_counter * target_step[0],
                           target_column + draw_counter * target_step[1])
            else:
                draw_token(canvas, self.images[board[row, col].item()], row, col)

        while swap_time > time:
            time += self.draw(board, draw)
            draw_counter += 1

    def show_matches(self, board: np.array, next_board: np.array):
        highlight_time = 1000 / self.animation_speed
        blinking_speed = 300 / self.animation_speed
        blink = False
        blinker = 0
        time = 0

        def blinking_token_draw(canvas, row, col):
            draw_token(canvas, self.images[board[row, col].item()], row, col)
            if next_board[row, col] == 0 and blink:
                pygame.draw.circle(
                    canvas,
                    (255, 255, 255),
                    (col * BLOCK_SIZE + PADDING + IMAGE_RADIUS, row * BLOCK_SIZE + PADDING + IMAGE_RADIUS),
                    IMAGE_RADIUS
                )

        def token_draw(canvas, row, col):
            if next_board[row, col] != 0:
                draw_token(canvas, self.images[next_board[row, col].item()], row, col)

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

    def show_falls(self, falls, board: np.array):
        fall_time = 100 / self.animation_speed
        falling_tokens = {}

        height, width = board.shape

        # TODO merge logic for tokens falling on the grid and new tokens spawning in
        for column in range(width):
            for row in range(height - 1, 0, -1):
                if board[row, column] != 0:
                    continue
                n_falls = 0
                for above_row in range(row, -1, -1):
                    if board[above_row, column] != 0:
                        break
                    n_falls += 1

                for above_row in range(row, -1, -1):
                    falling_tokens[above_row, column] = (n_falls, board[above_row, column].item())
                break
        falling_tokens = {key: value for key, value in falling_tokens.items() if board[key] != 0}

        for column in range(width):
            tiles_to_fall = sum([1 if value else 0 for value in (falls[:, column] != 0)])
            if tiles_to_fall == 0:
                continue
            out_of_screen_row = 0
            for row in range(height - 1, -1, -1):
                if falls[row, column] == 0:
                    continue
                out_of_screen_row -= 1
                falling_tokens[(out_of_screen_row, column)] = (tiles_to_fall, falls[row, column].item())

        fall_counter = 0
        settled_tokens = {}

        def token_draw(canvas, r, c):
            if board[r, c] == 0 or (r, c) in falling_tokens or (r, c) in settled_tokens:
                return
            if (r, c) in falling_tokens:
                draw_token(canvas, self.images[falling_tokens[(r, c)][1]],
                           fall_counter + r + time / fall_time, c)
            elif (r, c) in settled_tokens:
                draw_token(canvas, self.images[settled_tokens[(r, c)][1]], settled_tokens[(r, c)][0] + r,
                           c)
            else:
                draw_token(canvas, self.images[board[r, c].item()], r, c)

        def late_draw(canvas):
            for (r, c), (f, i) in falling_tokens.items():
                draw_token(canvas, self.images[i], fall_counter + r + time / fall_time, c)
            for (r, c), (f, i) in settled_tokens.items():
                draw_token(canvas, self.images[i], f + r, c)

        while len(falling_tokens) > 0:
            time = 0
            while fall_time > time:
                time += self.draw(board, token_draw=token_draw, late_draw=late_draw)

            fall_counter += 1
            for key, (fall_steps, token_id) in list(falling_tokens.items()):
                if fall_steps == fall_counter:
                    settled_tokens[key] = (fall_counter, token_id)
                    del falling_tokens[key]

    def draw(
            self,
            board: np.array,
            token_draw: Callable[[Surface, int, int], None] = None,
            early_draw: Callable[[Surface], None] = None,
            late_draw: Callable[[Surface], None] = None
    ) -> float:
        height, width = board.shape

        canvas = pygame.Surface((BLOCK_SIZE * width + 2, BLOCK_SIZE * height))
        canvas.fill((255, 255, 255))
        if early_draw:
            early_draw(canvas)

        if not token_draw:
            def token_draw(can, r, c):
                draw_token(can, self.images[board[r, c].item()], r, c)

        for row in range(height):
            for col in range(width):
                token_draw(canvas, row, col)

        if late_draw:
            late_draw(canvas)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        return self.clock.tick(self.fps)
