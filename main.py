
import time
from MCTS import MCTS
from elementGO.MCTSModel import Model
from match3tile.env import Match3Env
from util.dataset import get_train_and_test_data

import cProfile
import pstats
from pstats import SortKey

import os
import argparse

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

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

# Using seed 100
# Time before refactoring: 20.025s
def perform_profiling():
    SEED = 100
    RUN_PROFILER = True

    env = Match3Env(seed=SEED)
    mcts = MCTS(env, 100, True)
    if not os.path.exists("mcts.prof") or RUN_PROFILER:
        profiler = cProfile.Profile()
        profiler.runctx("mcts.__call__()", globals(), locals())
        profiler.dump_stats("mcts.prof")
    p = pstats.Stats("mcts.prof")

    override_pstats_prints()

    # Specify the files which we are interested in (Otherwise a lot of built-in stuff)
    included_files = ["board.py", "MCTS.py", "profiler.py"]

    p.stats = {
        key: value
        for key, value in p.stats.items()
        if any(file in key[0] for file in included_files)
    }
    
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats()



def mcts_samples():
    seeds = [100, 150, 200, 250, 300]
    rewards = []

    move_count = 20 # 20 default
    goal = 500      # 500 default

    start_time = time.time()

    for seed in seeds:
        env = Match3Env(seed=seed, num_moves=move_count, env_goal=goal)
        mcts = MCTS(env, 100, True)
        mcts_moves = []
        total_reward = 0

        while env.moves_taken != env.num_moves:
            action = mcts()
            _, reward, done, won, _ = env.step(action)
            total_reward += reward
            mcts_moves.append(action)
        rewards.append(total_reward)
        print(f" - Seed: {seed}")
        print(f" - Total reward: {total_reward}")
        print(f" - MCTS moves: {mcts_moves}")
        print()
    
    print("-" * 50)
    print(f"Time taken: {time.time() - start_time}")
    print(f"Rewards: {rewards}")
    print(f"Average reward: {sum(rewards) / len(rewards)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", default=False, help="Run the profiler")
    args = parser.parse_args()

    if args.profile:
        perform_profiling()
        exit()

    seed = 100
    env = Match3Env(seed=seed, num_moves=20, env_goal=500)
    mcts = MCTS(env, 100, False)

    mcts_moves = []
    total_reward = 0
    
    print("Performing \"optimized\" MCTS")
    print("-" * 50)
    start_time = time.time()
    while env.moves_taken != env.num_moves:
        action = mcts()
        _, reward, done, won, _ = env.step(action)
        total_reward += reward
        mcts_moves.append(action)

    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(f"Total reward: {total_reward}")
    print(f"MCTS moves: {mcts_moves}")


    # Show the MCTS moves
    env_copy = Match3Env(seed=seed, num_moves=20, env_goal=500, render_mode="human")
    for move in mcts_moves:
        env_copy.step(move)
        env_copy.render()