import time

from numpy import full
from MCTS import MCTS
from elementGO.MCTSModel import Model
from match3tile.env import Match3Env

import cProfile
import pstats
from pstats import SortKey

import os
import argparse

from util.dataset import Dataset

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

    train_ds, test_ds = Dataset(50000, type_switching=True, types=channels, type_switch_limit=24).with_batching(128).get_split(0.2)
    model.train(train_ds, test_ds, 2, 50)


def override_pstats_prints():
    """
    By default, pstats only prints 3 decimal places. This function overrides the print functions to print 6 decimal places and slightly adjust the title position.
    Should probably be upgraded to use Tabulate or something similar.
    """

    def f8_alt(x):
        return "%9.6f" % x

    def print_title(self):
        print(
            "   ncalls   tottime   percall   cumtime   percall",
            end=" ",
            file=self.stream,
        )
        print("filename:lineno(function)", file=self.stream)

    pstats.f8 = f8_alt
    pstats.Stats.print_title = print_title


# Using seed 100
# Time before refactoring: 20.025s
def perform_profiling(mode="full", sort_key="time", seed=100):
    """
    Runs the profiler on the MCTS algorithm and prints the results.
    mode: "full" or "quick".
            "quick" will only look at the MCTS.prof file (provided it exists)
            "full" will run the profiler and save the results to MCTS.prof
    sort_key: "calls", "cumtime", "time". Sorts the profiler output by the specified key.
    """

    print(f"Running profiler: \n - Mode: {mode} \n - Sort key: {sort_key} \n - Seed: {seed}")
    print("-" * 50)

    env = Match3Env(seed=seed)
    mcts = MCTS(env, 100, True)

    if mode == "full" or not os.path.exists("mcts.prof"):
        profiler = cProfile.Profile()
        profiler.runctx("mcts.__call__()", globals(), locals())
        profiler.dump_stats("mcts.prof")
    p = pstats.Stats("mcts.prof")

    override_pstats_prints()

    # Specify the files which we are interested in (Otherwise a lot of built-in stuff)
    included_files = ["board.py", "MCTS.py", "quick_math.py"]

    p.stats = {
        key: value
        for key, value in p.stats.items()
        if any(file in key[0] for file in included_files)
    }

    print()
    p.strip_dirs().sort_stats(sort_key).print_stats()


def mcts_samples():
    verbose = False
    seeds = list(range(50, 1001, 50))
    rewards = []

    move_count = 20  # 20 default
    goal = 500  # 500 default

    start_time = time.time()

    for seed in seeds:
        print(f"Running MCTS with seed {seed}")
        mcts_start_time = time.time()
        env = Match3Env(seed=seed, num_moves=move_count, env_goal=goal)
        mcts = MCTS(env, 100, verbose)
        mcts_moves = []
        total_reward = 0

        while env.moves_taken != env.num_moves:
            action = mcts()
            _, reward, done, won, _ = env.step(action)
            total_reward += reward
            mcts_moves.append(action)
        rewards.append(total_reward)
        print(f" - Time taken: {time.time() - mcts_start_time:.2f} seconds")
        print(f" - Total reward: {total_reward}")
        print(f" - MCTS moves: {mcts_moves}")
        print()

    print("-" * 50)
    print("Results from running MCTS sample (optimized branch)")
    print(f" - Sample count: {len(seeds)}")
    print(f" - Seeds: {seeds}")
    print(f" - Time taken: {time.time() - start_time:.2f} seconds")
    print(f" - Rewards: {rewards}")
    print(f" - Average reward: {sum(rewards) / len(rewards)}")


def mcts_single(seed=100, move_count=20, goal=500, render=False):
    env = Match3Env(seed=seed, num_moves=move_count, env_goal=goal)
    mcts = MCTS(env, 10, False, True)

    mcts_moves = []
    total_reward = 0

    print(
        f'Performing "optimized" MCTS (seed: {seed}, moves: {move_count}, goal: {goal})'
    )
    print("-" * 50)
    start_time = time.time()
    while env.moves_taken != env.num_moves:
        action = mcts()
        print(env.board.actions)
        _, reward, done, won, _ = env.step(action)
        total_reward += reward
        mcts_moves.append(action)

    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(f"Total reward: {total_reward}")
    print(f"MCTS moves: {mcts_moves}")

    if render:
        env_copy = Match3Env(
            seed=seed, num_moves=move_count, env_goal=goal, render_mode="human"
        )
        for move in mcts_moves:
            env_copy.step(move)
            env_copy.render()


if __name__ == "__main__":
    # mcts_single(move_count=10, render=True)
    # mcts_single(move_count=10, render=False)
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        nargs="?",
        action="store",
        const="full",
        type=str,
        help="Run the profiler",
        choices=["quick", "full"],
    )
    parser.add_argument(
        "--sort",
        action="store",
        default="time",
        type=str,
        help="Sort the profiler output",
        choices=["calls", "cumtime", "time"],
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        default=100,
        help="Seed for the environment",
    )
    args = parser.parse_args()

    if args.profile:
        perform_profiling(args.profile, args.sort, args.seed)
        exit()

