from random import randint

import numpy as np

from match3tile.boardv2 import BoardV2


class Match3Env:
    metadata = {"render_modes": ["human"], 'render_fps': 60, 'animation_speed': 1}

    def __init__(
            self,
            width: int = 9,
            height: int = 9,
            num_types: int = 6,
            num_moves: int = 20,
            env_goal: int = 500,
            seed: int = None,
            render_mode: str = None
    ):

        if seed is not None:
            self.seed = seed
        else:
            self.seed = randint(0, 2**32 - 1)

        assert width >= 3 and height >= 3, f"Board size too small: min size: 3x3"

        self.width = width
        self.height = height
        self.num_types = num_types
        self.env_goal = env_goal
        self.num_moves = num_moves
        self.score, self.moves_taken = 0, 0

        self.action_space = height * (width - 1) + width * (height - 1)

        self.board = BoardV2(self.num_moves, seed=self.seed)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.board_animator = None
        self.event = None

    def init(self):
        return self.board.array

    def step(self, action: int) -> tuple[np.array, int, bool, bool, dict[str, any]]:
        self.actions = self.board.legal_actions
        move_score, self.event = self.board.apply_action(action)
        self.score += move_score
        self.moves_taken += 1

        truncated = self.score >= self.env_goal
        done = truncated or self.num_moves == self.moves_taken
        return self.board.array, move_score, done, truncated, {}

    def reset(self, seed=None) -> tuple[np.array, dict[str, any]]:
        if seed is not None:
            self.seed = seed
        else:
            self.seed = (1 + self.seed) % 2**32 - 1
        self.score, self.moves_taken = 0, 0
        self.board = BoardV2(self.num_moves, seed=self.seed)
        return self.board.array, {}

    def render(self):
        if self.render_mode is None:
            return
        if self.board_animator is None:
            from match3tile.draw_board import BoardAnimator
            self.board_animator = BoardAnimator(self.metadata['animation_speed'], self.metadata['render_fps'], self.board)

        if self.render_mode == 'human':
            self.board_animator.draw(self.event.init_board)
            # self.board_animator.draw_actions(self.event.init_board, self.actions)
            # # time.sleep(1)
            # self.board_animator.show_swap(self.board.decode_action(self.event.action), self.event.init_board)
            # for state, next_state, falls in self.event.event_boards:
            #     self.board_animator.show_matches(state, next_state)
            #     self.board_animator.show_falls(falls, next_state)
            # self.board_animator.draw(self.board.array)
