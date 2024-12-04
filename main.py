from MCTS import MCTS
from match3tile.env import Match3Env
from model import Model
from util.dataset import get_train_and_test_data
from util.mp import async_pbar_auto_batcher


def random_task():
    env = Match3Env()
    score = 0
    while env.moves_taken != env.num_moves:
        _, reward, _, _, _ = env.step(env.board.random_action())
        score += reward
    return score


def naive_task():
    env = Match3Env()
    score = 0
    while env.moves_taken != env.num_moves:
        _, reward, _, _, _ = env.step(env.board.naive_action())
        score += reward
    return score


def best_task():
    env = Match3Env()
    score = 0
    while env.moves_taken != env.num_moves:
        _, reward, _, _, _ = env.step(env.board.best_action())
        score += reward
    return score


def mcts_task():
    env = Match3Env()
    algo = MCTS(env, verbal=False)
    score = 0
    while env.moves_taken != env.num_moves:
        _, reward, _, _, _ = env.step(algo())
        score += reward
    return score


def nn_mcts_task():
    env = Match3Env()
    algo = MCTS(env, verbal=False)
    score = 0
    while env.moves_taken != env.num_moves:
        _, reward, _, _, _ = env.step(algo())
        score += reward
    return score


def train_model():
    env = Match3Env()

    height, width, channels = env.observation_space
    model = Model(
        height=height,
        width=width,
        channels=channels,
        action_space=env.action_space,
        learning_rate=0.005,
        momentum=0.9,
    )

    train_ds, test_ds = get_train_and_test_data()
    model.fit(train_ds, test_ds)


if __name__ == '__main__':
    import os
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

    samples_size = 1000
    train_model()

    random_action_scores = async_pbar_auto_batcher(random_task, samples_size)
    naive_score = async_pbar_auto_batcher(naive_task, samples_size)
    best_score = async_pbar_auto_batcher(best_task, samples_size)
    mcts_score = async_pbar_auto_batcher(mcts_task, samples_size)
