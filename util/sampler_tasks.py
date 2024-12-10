import numpy as np

from match3tile.boardv2 import BoardV2
from mctslib.nn.mcts import NeuralNetworkMCTS
from mctslib.standard.mcts import MCTS
from util import checkpointing
from util.mp import async_pbar_auto_batcher
from util.plotter import plot_distribution


def random_task():
    state = BoardV2(20, seed=100)
    np.random.seed(state.seed)
    while not state.is_terminal:
        state = state.apply_action(np.random.choice(state.legal_actions))
    return state.reward


def best_task():
    state = BoardV2(20, seed=100)
    while not state.is_terminal:
        state = state.apply_action(state.best_action)
    return state.reward


def mcts_task():
    state = BoardV2(20)
    mcts = MCTS(state, 2, 100, False, deterministic=False)
    while not state.is_terminal:
        action, _, _, = mcts()
        state = state.apply_action(action)
    return state.reward


def nn_mcts_task():
    state = BoardV2(20)
    model = checkpointing.load('model')
    algo = NeuralNetworkMCTS(model, state, 3, 100, verbose=True)
    while not state.is_terminal:
        action, _, _ = algo()
        state = state.apply_action(action)
    return state.reward


def sample(sample_size=100):
    random_action_scores = async_pbar_auto_batcher(random_task, sample_size)
    best_score = async_pbar_auto_batcher(best_task, sample_size)
    mcts_score = async_pbar_auto_batcher(mcts_task, sample_size)
    # nn_mcts_score = async_pbar_auto_batcher(nn_mcts_task, sample_size)
    nn_mcts_score = [nn_mcts_task() for _ in range(sample_size)]

    plot_distribution({
        'Random actions': random_action_scores,
        'Best actions': best_score,
        'MCTS actions': mcts_score,
        'NN MCTS actions': nn_mcts_score,
    })


def main():
    sample(10)

if __name__ == '__main__':
    nn_mcts_task()
