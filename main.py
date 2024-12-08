from fileinput import filename
from MCTS import MCTS
from elementGO.MCTSModel import Model
from match3tile.env import Match3Env
from util.dataset import get_train_and_test_data
from util.mp import async_pbar_auto_batcher
from util.plotter import plot_distribution
from pickle import load, dump

import cProfile
import pstats
from pstats import SortKey


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
        action_space=env.action_space,
        channels=channels,
        features=256,
        learning_rate=0.005,
        momentum=0.9,
    )

    train_ds, test_ds = get_train_and_test_data()
    model.train(train_ds, test_ds)

# Using seed 100
# Time before refactoring: 20.025s

if __name__ == "__main__":
    import os
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
    RUN_PROFILER = True

    env = Match3Env(seed=100)
    mcts = MCTS(env, 100, True)

    if not os.path.exists("mcts.prof") or RUN_PROFILER:
        cProfile.run("mcts.__call__()", "mcts.prof")
    
    p = pstats.Stats("mcts.prof")  # Load the profile stats


    def override_pstats_prints():
        """
        By default, pstats only prints 3 decimal places. This function overrides the print functions to print 6 decimal places and slightly adjust the title position.
        Should probably be upgraded to use Tabulate or something similar.
        """
        def f8_alt(x):
            return "%9.6f" % x

        def print_title(self):
            print('   ncalls   tottime   percall   cumtime   percall', end=' ', file=self.stream)
            print('filename:lineno(function)', file=self.stream)
        pstats.f8 = f8_alt
        pstats.Stats.print_title = print_title
    override_pstats_prints()

    # Specify the files which we are interested in (Otherwise a lot of built-in stuff)
    included_files = ["board.py", "MCTS.py", "profiler.py"]

    p.stats = {
        key: value
        for key, value in p.stats.items()
        if any(file in key[0] for file in included_files)
    }
    
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats()
