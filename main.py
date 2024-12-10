import argparse
import cProfile
import os
import pstats
import time

import numpy as np

import match3tile
from elementGO.MCTSModel import Model
from match3tile.boardv2 import BoardV2
from match3tile.draw_board import BoardAnimator
from mctslib.standard.mcts import MCTS
from util import checkpointing
from util.dataset import Dataset
from util.mp import async_pbar_auto_batcher
from util.plotter import plot_distribution
from util.pstate_override import override_pstats_prints

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


def random_task():
    state = BoardV2(20)
    np.random.seed(state.seed)
    while not state.is_terminal:
        state = state.apply_action(np.random.choice(state.legal_actions))
    return state.reward


def naive_task():
    state = BoardV2(20)
    while not state.is_terminal:
        state = state.apply_action(state.naive_action)
    return state.reward


def best_task():
    state = BoardV2(20)
    while not state.is_terminal:
        state = state.apply_action(state.best_action)
    return state.reward


def mcts_task():
    state = BoardV2(20)
    mcts = MCTS(state, 3, 666, False, deterministic=False)
    while not state.is_terminal:
        action, _, _, = mcts()
        state = state.apply_action(action)
    return state.reward


def deterministic_mcts_task():
    state = BoardV2(20)
    mcts = MCTS(state, 3, 666, False, deterministic=True)
    while not state.is_terminal:
        action, _, _, = mcts()
        state = state.apply_action(action)
    return state.reward


def nn_mcts_task():
    state = BoardV2(20)
    # algo = NeuralNetworkMCTS(state.board, 3, 100, verbose=False)
    while not state.is_terminal:
        state = state.apply_action(0)
    return state.reward


def train_model():
    model = Model(
        match3tile.metadata.rows, match3tile.metadata.columns, match3tile.metadata.action_space,
        channels=match3tile.metadata.types,
        features=64,
        learning_rate=0.005,
        momentum=0.9,
    )

    train_ds, test_ds = Dataset(10, fat_cache=True, mirroring=True, type_switching=True, types=match3tile.metadata.types, type_switch_limit=256).with_batching(128).get_split(0.1)
    model.train(train_ds, test_ds, 3, len(test_ds))
    return model


def create_Board(seed=100, move_count=20, goal=500):
    return BoardV2(n_actions=move_count, seed=seed)


def create_mcts(board, exploration_weight=3, simulations=100, verbose=False):
    return MCTS(board, exploration_weight, simulations, verbose)


def perform_profiling(mode="full", sort_key="time", mcts: MCTS = None, file="mcts_new.prof"):
    """
    Runs the profiler on the MCTS algorithm and prints the results.
    mode: "full" or "quick".
            "quick" will only look at the MCTS.prof file (provided it exists)
            "full" will run the profiler and save the results to MCTS.prof
    sort_key: "calls", "cumtime", "time". Sorts the profiler output by the specified key.
    """

    assert mcts is not None, "MCTS object must be provided for profiling"

    if mode == "full" or not os.path.exists(file):
        profiler = cProfile.Profile()
        profiler.runctx("mcts.__call__()", globals(), locals())
        profiler.dump_stats(file)
    p = pstats.Stats(file)

    override_pstats_prints()

    # Specify the files which we are interested in (Otherwise a lot of built-in stuff)
    included_files = ["boardv2.py", "MCTS.py", "quick_math.py"]

    p.stats = {
        key: value
        for key, value in p.stats.items()
        if any(file in key[0] for file in included_files)
    }

    print()
    p.strip_dirs()
    p.sort_stats(sort_key).print_stats()


def mcts_samples():
    verbose = False
    seeds = list(range(50, 1001, 50))
    rewards = []

    move_count = 20  # 20 default
    simulations = 100  # 100 default
    goal = 500  # 500 default

    start_time = time.time()

    for seed in seeds:
        print(f"Running MCTS with seed {seed}")
        mcts_start_time = time.time()
        state = BoardV2(move_count, seed=seed)
        mcts = MCTS(state, 3, simulations, verbose)
        mcts_moves = []
        total_reward = 0

        while not state.is_terminal:
            action, _, _ = mcts()
            state = state.apply_action(action)
            total_reward += state.reward
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


def mcts_single(seed=100, move_count=20, goal=500, simulations=100, render=False, verbose=False, deterministic=False):
    print(f"Performing MCTS (seed: {seed}, moves: {move_count}, goal: {goal})")
    print("-" * 50)

    state = BoardV2(move_count, seed=seed)
    mcts = MCTS(state, 3, simulations, verbose)
    mcts_moves = []

    start_time = time.time()
    while not state.is_terminal:
        action, _, _ = mcts()
        state = state.apply_action(action)
        mcts_moves.append(action)

    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(f"Total reward: {state.reward}")
    print(f"MCTS moves: {mcts_moves}")

    if render:
        board = BoardV2(move_count, seed=seed)
        renderer = BoardAnimator(1, 60, board.array)
        for move in mcts_moves:
            time.sleep(2)
            board.apply_action(move)
            renderer.draw(board.array)


def sample(sample_size=100):
    random_action_scores = async_pbar_auto_batcher(random_task, sample_size)
    naive_score = async_pbar_auto_batcher(naive_task, sample_size)
    best_score = async_pbar_auto_batcher(best_task, sample_size)
    mcts_score = async_pbar_auto_batcher(mcts_task, sample_size)
    dmt_mcts_score = async_pbar_auto_batcher(deterministic_mcts_task, sample_size)

    plot_distribution({
        'Random actions': random_action_scores,
        'Naive actions': naive_score,
        'Best actions': best_score,
        'MCTS actions': mcts_score,
        'DMT MCTS actions': dmt_mcts_score,
    })


def test_save_load():
    model1 = Model(
        match3tile.metadata.rows, match3tile.metadata.columns, match3tile.metadata.action_space,
        channels=match3tile.metadata.types,
        features=64,
        learning_rate=0.005,
        momentum=0.9,
    )

    train_ds_1, test_ds_2 = Dataset(10, fat_cache=True, mirroring=True, type_switching=True, types=match3tile.metadata.types, type_switch_limit=256).with_batching(128).get_split(0.1)
    model1.train(train_ds_1, test_ds_2, 3, len(test_ds_2), plot=False)

    model2 = Model(
        match3tile.metadata.rows, match3tile.metadata.columns, match3tile.metadata.action_space,
        channels=match3tile.metadata.types,
        features=32,
        learning_rate=0.01,
        momentum=0.8,
    )
    train_ds_2, test_ds_2 = Dataset(10, fat_cache=True, mirroring=True, type_switching=True, types=match3tile.metadata.types, type_switch_limit=256).with_batching(128).get_split(0.1)
    model2.train(train_ds_2, test_ds_2, 3, len(test_ds_2), plot=False)

    checkpointing.save(model1, "model1", False)
    model1_loaded = checkpointing.load("model1", False)

    assert checkpointing.compare_models(model1, model1_loaded), "Model1 and Model1_loaded are not the same"
    assert not checkpointing.compare_models(model1, model2), "Model1 and Model2 are the same"
    print("All tests passed")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", nargs="?", action="store", const="full", choices=["quick", "full"])
    parser.add_argument("--sort", action="store", default="time", choices=["calls", "cumtime", "time"])

    parser.add_argument("--seed", action="store", type=int, default=100)
    parser.add_argument("--moves", action="store", type=int, default=20)
    parser.add_argument("--goal", action="store", type=int, default=500)
    parser.add_argument("--sims", action="store", type=int, default=1000)
    parser.add_argument("--exploration_weight", action="store", type=float, default=3.0)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)

    parser.add_argument("--mcts_samples", default=False, action="store_true")
    parser.add_argument("--mcts_single", default=False, action="store_true")
    args = parser.parse_args()

    if args.profile:
        board = create_Board()
        mcts = create_mcts(board, args.exploration_weight, args.sims, args.verbose)

        print("-" * 50)
        print(f" Performing {args.profile} MCTS Profiling")
        print(f" - Seed: {args.seed}")
        print(f" - Moves: {args.moves}")
        print(f" - Goal: {args.goal}")
        print(f" - Simulations: {args.sims}")
        print(f" - Verbose: {args.verbose}")
        print(f" - Deterministic: {args.deterministic}")
        print("-" * 50)

        perform_profiling(args.profile, args.sort, mcts)
        exit()

    if args.mcts_single:
        mcts_single(args.seed, args.moves, args.goal, args.sims, args.render, args.verbose)
        exit()

    if args.mcts_samples:
        mcts_samples()
        exit()

