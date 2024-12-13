from random import choice

from match3tile.boardConfig import BoardConfig
from match3tile.boardv2 import BoardV2
from mctslib.nn.mcts import NeuralNetworkMCTS
from mctslib.standard.mcts import MCTS


def random_task(moves, cfg: BoardConfig):
    cfg = BoardConfig(rows=cfg.rows, columns=cfg.columns, types=cfg.types)
    state = BoardV2(moves, cfg)
    while not state.is_terminal:
        state = state.apply_action(choice(state.legal_actions))
    return state.reward


def greedy_test(moves, cfg: BoardConfig):
    cfg = BoardConfig(rows=cfg.rows, columns=cfg.columns, types=cfg.types)
    state = BoardV2(moves, cfg)
    while not state.is_terminal:
        state = state.apply_action(state.greedy_action)
    return state.reward


def mcts_task(moves, ew, sims, cfg: BoardConfig):
    cfg = BoardConfig(rows=cfg.rows, columns=cfg.columns, types=cfg.types)
    state = BoardV2(moves, cfg)
    mcts = MCTS(state, ew, sims, False, deterministic=False)
    while not state.is_terminal:
        action, _, _, = mcts()
        state = state.apply_action(action)
    return state.reward


def nn_mcts_task(moves, ew, sims, model, cfg):
    cfg = BoardConfig(rows=cfg.rows, columns=cfg.columns, types=cfg.types)
    state = BoardV2(moves, cfg)
    algo = NeuralNetworkMCTS(model, state, ew, sims, verbose=False)
    while not state.is_terminal:
        action, _, _ = algo()
        state = state.apply_action(action)
    return state.reward
