# import os
# from typing import Callable
# import numpy as np
# import pygame
# from pygame import Surface
# from os import listdir
#
# from match3tile.boardConfig import BoardConfig
#
# BLOCK_SIZE = 70
# PADDING = 3
# IMAGE_RADIUS = 32
#
#
# class BoardAnimator:
#     def __init__(self, cfg: BoardConfig, animation_speed, fps):
#         self.animation_speed = animation_speed
#         height, width = cfg.rows, cfg.columns
#         self.window_size = (BLOCK_SIZE*width+2, BLOCK_SIZE*height)
#         self.fps = fps
#         pygame.init()
#         pygame.display.init()
#         self.window = pygame.display.set_mode(self.window_size)
#         self.clock = pygame.time.Clock()
#
#         self.cfg = cfg
#
#         abs_path = os.path.abspath(__file__)
#         dir_path = os.path.dirname(abs_path)
#         image_names = [image for image in listdir(f'{dir_path}/images/default')]
#
#         images = {
#             0: [pygame.image.load(f"{dir_path}/images/default/{image}") for image in image_names],
#             cfg.v_line: [pygame.image.load(f"{dir_path}/images/Vline/{image}") for image in image_names],
#             cfg.h_line: [pygame.image.load(f"{dir_path}/images/Hline/{image}") for image in image_names],
#             cfg.bomb: [pygame.image.load(f"{dir_path}/images/bomb/{image}") for image in image_names],
#         }
#         big_bad = pygame.image.load(f"{dir_path}/images/bigBad.png")
#
#         self.get_token_image = (
#             lambda token:
#             big_bad if token == cfg.mega_token or token == 0
#             else images[(token & cfg.special_type_mask)][(token & cfg.type_mask) - 1]
#         )
#
#     def draw_token(self, canvas: Surface, token, row, col):
#         canvas.blit(self.get_token_image(token), (col * BLOCK_SIZE + PADDING, row * BLOCK_SIZE + PADDING))
#
#     def show_swap(self, action, board):
#         (source_row, source_column), (target_row, target_column) = action
#         swap_time = 200 / self.animation_speed
#         steps = swap_time / ((1 / max(1, self.fps)) * 1000)
#
#         source_step = ((target_row - source_row) / steps, (target_column - source_column) / steps)
#         target_step = ((source_row - target_row) / steps, (source_column - target_column) / steps)
#
#         time = 0
#         draw_counter = 0
#
#         def draw(canvas, row, col):
#             if row == source_row and col == source_column:
#                 self.draw_token(canvas, board[row, col], source_row + draw_counter * source_step[0], source_column + draw_counter * source_step[1])
#             elif row == target_row and col == target_column:
#                 self.draw_token(canvas, board[row, col], target_row + draw_counter * target_step[0], target_column + draw_counter * target_step[1])
#             else:
#                 self.draw_token(canvas, board[row, col], row, col)
#
#         while swap_time > time:
#             time += self.draw(board, draw)
#             draw_counter += 1
#
#     def draw_actions(self, board: np.array, actions: list[int]):
#         def draw_action(canvas):
#             for action in actions:
#                 source, target = self.cfg.decode(action)
#                 pygame.draw.line(
#                     canvas,
#                     (0, 255, 0),
#                     ((source[1] + 0.5) * BLOCK_SIZE, (source[0] + 0.5) * BLOCK_SIZE),
#                     ((target[1] + 0.5) * BLOCK_SIZE, (target[0] + 0.5) * BLOCK_SIZE),
#                     5
#                 )
#         self.draw(board, early_draw=draw_action)
#
#     def show_matches(self, board: np.array, next_board: np.array):
#         highlight_time = 1000 / self.animation_speed
#         blinking_speed = 300 / self.animation_speed
#         blink = False
#         blinker = 0
#         time = 0
#
#         def blinking_token_draw(canvas, row, col):
#             self.draw_token(canvas, board[row, col], row, col)
#             if next_board[row, col] == 0 and blink:
#                 pygame.draw.circle(
#                     canvas,
#                     (255, 255, 255),
#                     (col * BLOCK_SIZE + PADDING + IMAGE_RADIUS, row * BLOCK_SIZE + PADDING + IMAGE_RADIUS),
#                     IMAGE_RADIUS
#                 )
#
#         def token_draw(canvas, row, col):
#             if next_board[row, col] != 0:
#                 self.draw_token(canvas, board[row, col], row, col)
#
#         while highlight_time > time:
#             if blinker > blinking_speed:
#                 blink = not blink
#                 blinker = 0
#             delta = self.draw(board, token_draw=blinking_token_draw)
#             time += delta
#             blinker += delta
#
#         show_empty = 200 / self.animation_speed
#         time = 0
#         while show_empty > time:
#             time += self.draw(board, token_draw=token_draw)
#
#     def show_falls(self, falls, board: np.array):
#         fall_time = 100 / self.animation_speed
#         falling_tokens = {}
#
#         height, width = board.shape
#
#         for col in range(width):
#             col_cnt = 0
#             for row in range(height):
#                 if board[row, col] == 0:
#                     col_cnt += 1
#
#             for row in range(height):
#                 if row >= col_cnt:
#                     falls[row, col] = 0
#
#         # TODO merge logic for tokens falling on the grid and new tokens spawning in
#         for column in range(width):
#             for row in range(height - 1, 0, -1):
#                 if board[row, column] != 0:
#                     continue
#                 n_falls = 0
#                 for above_row in range(row, -1, -1):
#                     if board[above_row, column] != 0:
#                         break
#                     n_falls += 1
#
#                 for above_row in range(row, -1, -1):
#                     falling_tokens[above_row, column] = (n_falls, board[above_row, column].item())
#                 break
#         falling_tokens = {key: value for key, value in falling_tokens.items() if board[key] != 0}
#
#         for column in range(width):
#             tiles_to_fall = sum([1 if value else 0 for value in (falls[:, column] != 0)])
#             if tiles_to_fall == 0:
#                 continue
#             out_of_screen_row = 0
#             for row in range(height - 1, -1, -1):
#                 if falls[row, column] == 0:
#                     continue
#                 out_of_screen_row -= 1
#                 falling_tokens[(out_of_screen_row, column)] = (tiles_to_fall, falls[row, column].item())
#
#         fall_counter = 0
#         settled_tokens = {}
#
#         def token_draw(canvas, r, c):
#             if board[r, c] == 0 or (r, c) in falling_tokens or (r, c) in settled_tokens:
#                 return
#             if (r, c) in falling_tokens:
#                 self.draw_token(canvas, falling_tokens[(r, c)][1], fall_counter + r + time / fall_time, c)
#             elif (r, c) in settled_tokens:
#                 self.draw_token(canvas, settled_tokens[(r, c)][1], settled_tokens[(r, c)][0] + r, c)
#             else:
#                 self.draw_token(canvas, board[r, c], r, c)
#
#         def late_draw(canvas):
#             for (r, c), (f, i) in falling_tokens.items():
#                 self.draw_token(canvas, i, fall_counter + r + time / fall_time, c)
#             for (r, c), (f, i) in settled_tokens.items():
#                 self.draw_token(canvas, i, f + r, c)
#
#         while len(falling_tokens) > 0:
#             time = 0
#             while fall_time > time:
#                 time += self.draw(board, token_draw=token_draw, late_draw=late_draw)
#
#             fall_counter += 1
#             for key, (fall_steps, token_id) in list(falling_tokens.items()):
#                 if fall_steps == fall_counter:
#                     settled_tokens[key] = (fall_counter, token_id)
#                     del falling_tokens[key]
#
#     def draw(
#             self,
#             board: np.array,
#             token_draw: Callable[[Surface, int, int], None] = None,
#             early_draw: Callable[[Surface], None] = None,
#             late_draw: Callable[[Surface], None] = None
#     ) -> float:
#         height, width = board.shape
#
#         canvas = pygame.Surface((BLOCK_SIZE * width + 2, BLOCK_SIZE * height))
#         canvas.fill((255, 255, 255))
#         if early_draw:
#             early_draw(canvas)
#
#         if not token_draw:
#             def token_draw(can, r, c):
#                 self.draw_token(can, board[r, c], r, c)
#
#         for row in range(height):
#             for col in range(width):
#                 token_draw(canvas, row, col)
#
#         if late_draw:
#             late_draw(canvas)
#
#         self.window.blit(canvas, canvas.get_rect())
#         pygame.event.pump()
#         pygame.display.update()
#         return self.clock.tick(self.fps)
#
#     def __del__(self):
#         pygame.quit()
#