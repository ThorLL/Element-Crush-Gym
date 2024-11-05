import numpy as np

from match3tile.board import Board


class Match3Env:
    metadata = {"render_modes": ["human"], 'render_fps': 60, 'animation_speed': 1}

    def __init__(
            self,
            width: int = 7,
            height: int = 9,
            num_types: int = 6,
            num_moves: int = 20,
            env_goal: int = 300,
            seed: int = 0,
            render_mode: str = None
    ):

        np.random.seed(seed)
        assert width >= 3 and height >= 3, f"Board size too small: min size: 3x3"

        self.width = width
        self.height = height
        self.num_types = num_types
        self.env_goal = env_goal
        self.num_moves = num_moves
        self.score, self.moves_taken = 0, 0
        self.seed = seed

        self.observation_space = (height, width, num_types)
        self.action_space = height * (width - 1) + width * (height - 1)

        self.board = Board(self.observation_space, seed)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.board_animator = None
        self.event = None

    def init(self):
        return self.board.array

    def step(self, action: int) -> tuple[np.array, int, bool, bool, dict[str, any]]:
        move_score, board, self.event = self.board.swap(action)
        self.score += move_score
        self.moves_taken += 1

        truncated = self.score >= self.env_goal
        done = truncated or self.num_moves == self.moves_taken
        return board, move_score, done, truncated, {}

    def reset(self, seed=None) -> tuple[np.array, dict[str, any]]:
        if seed is not None:
            self.seed = seed
        else:
            self.seed += 1
        self.score, self.moves_taken = 0, 0
        return self.board.array, {}

    def render(self):
        if self.render_mode is None:
            return
        if self.board_animator is None:
            from match3tile.draw_board import BoardAnimator
            self.board_animator = BoardAnimator(self.metadata['animation_speed'], self.event.init_board.shape, self.metadata['render_fps'])

        if self.render_mode == 'human':
            self.board_animator.draw(self.event.init_board)
            self.board_animator.show_swap(self.event.action, self.event.init_board)
            for state, next_state, falls in self.event.event_boards:
                self.board_animator.show_matches(state, next_state)
                self.board_animator.show_falls(falls, next_state)
