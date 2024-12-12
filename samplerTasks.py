import numpy as np

from match3tile.boardConfig import BoardConfig
from match3tile.boardv2 import BoardV2
from mctslib.nn.mcts import NeuralNetworkMCTS
from mctslib.standard.mcts import MCTS


def random_task():
    state = BoardV2(20, BoardConfig())
    np.random.seed(state.cfg.seed)
    while not state.is_terminal:
        state = state.apply_action(np.random.choice(state.legal_actions))
    return state.reward


def best_task():
    state = BoardV2(20, BoardConfig())
    np.random.seed(state.cfg.seed)
    while not state.is_terminal:
        state = state.apply_action(state.best_action)
    return state.reward


def mcts_task():
    state = BoardV2(20, BoardConfig())
    np.random.seed(state.cfg.seed)
    mcts = MCTS(state, 2, 100, False, deterministic=False)
    while not state.is_terminal:
        action, _, _, = mcts()
        state = state.apply_action(action)
    return state.reward


def nn_mcts_task(model):
    state = BoardV2(20, BoardConfig())
    np.random.seed(state.cfg.seed)
    algo = NeuralNetworkMCTS(model, state, 3, 100, verbose=False)
    while not state.is_terminal:
        action, _, _ = algo()
        state = state.apply_action(action)
    return state.reward
