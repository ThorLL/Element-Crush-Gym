import argparse
import os
import sys
import time
from contextlib import redirect_stdout

from optax import sgd
from tqdm import tqdm

import gui
from elementCrush import ElementCrush
from match3tile.boardConfig import BoardConfig
from match3tile.boardv2 import BoardV2
from match3tile.draw_board import BoardAnimator
from mctslib.standard.mcts import MCTS
from dataset import Dataset
from samplerTasks import random_task, best_task, mcts_task, nn_mcts_task
from util.multiprocessingAutoBatcher import async_pbar_auto_batcher
from util.profiler import perform_profiling
from visualisers.plotter import plot_distribution


def mcts_samples():
    verbose = False
    seeds = list(range(50, 1001, 50))
    rewards = []

    move_count = 20  # 20 default
    simulations = 100  # 100 default

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


def mcts_single(seed, move_count, simulations, render=False, verbose=False, deterministic=False):
    print(f"Performing MCTS (seed: {seed}, moves: {move_count})")
    print("-" * 50)

    state = BoardV2(move_count, seed=seed)
    mcts = MCTS(state, 3, simulations, verbose, deterministic)
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


def parse_args():
    parser = argparse.ArgumentParser()

    # profiler args
    parser.add_argument("--profile", nargs="?", action="store", const="full", choices=["quick", "full"])
    parser.add_argument("--sort", action="store", default="time", choices=["calls", "cumtime", "time"])

    # general args
    parser.add_argument("--train_em_all", action="store_true", default=False)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--sample_size", action="store", type=int, default=100)
    # board args
    parser.add_argument("--height", action="store", type=int, default=9)
    parser.add_argument("--width", action="store", type=int, default=9)
    parser.add_argument("--types", action="store", type=int, default=6)

    # model args
    parser.add_argument("--residual_layer", action="store", type=int, default=10)
    parser.add_argument("--features", action="store", type=int, default=128)

    parser.add_argument("--learning_rate", action="store", type=float, default=1e-3)
    parser.add_argument("--momentum", action="store", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true", default=False)

    parser.add_argument("--training_plot", action="store_true", default=False)
    parser.add_argument("--epochs", action="store", type=int, default=1)
    parser.add_argument("--eval_every", action="store", type=int, default=100)
    # dataset args
    parser.add_argument("--dataset_size", action="store", type=int, default=1000)
    parser.add_argument("--dataset_caching", action="store_true", default=True)
    parser.add_argument("--data_split", action="store", type=float, default=0.8)
    parser.add_argument("--batch_size", action="store", type=int, default=64)

    parser.add_argument("--mirroring", action="store_true", default=True)
    parser.add_argument("--type_switching", action="store_true", default=True)
    parser.add_argument("--type_switch_limit", type=int, default=128)

    parser.add_argument("--seed", action="store", type=int, default=100)
    parser.add_argument("--moves", action="store", type=int, default=20)
    parser.add_argument("--sims", action="store", type=int, default=1000)
    parser.add_argument("--exploration_weight", action="store", type=float, default=3.0)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    # mcts args

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random_scores = []
    best_score = []
    mcts_score = []
    nn_mcts_score = []

    profiling = gui.Variable('Profiling', args.profile)
    sample_size = gui.Variable('Sample Size', args.sample_size, 'size')

    seed = gui.Variable('Seed', args.seed)

    exploration_weight = gui.Variable('Exploration Weight', args.exploration_weight, 'ew')
    simulations = gui.Variable('Simulations', args.sims, 'sims')
    deterministic = gui.Variable('Deterministic', args.deterministic)
    render = gui.Variable('Render', args.render)
    verbose = gui.Variable('Verbose', args.verbose)
    residual_layer = gui.Variable('Residual Layers', args.residual_layer, 'rl')
    features = gui.Variable('Features', args.features)

    board = gui.Variable('Board', None)

    def get_config():
        return BoardConfig(seed.val, *shape.val)


    def build_board():
        cfg = get_config()
        board.set(BoardV2(moves.val, cfg), force=True)
        dataset.set(Dataset(cfg, moves.val), force=True)


    moves = gui.Variable('Moves', args.moves, callback=build_board)
    shape = gui.Variable('Shape', (args.height, args.height, args.types), callback=build_board)

    model = gui.Variable('Current model', None)

    epochs = gui.Variable('Epochs', args.epochs)
    eval_every = gui.Variable('Eval every', args.eval_every, 'eval')
    show_training_plot = gui.Variable('Show training plot', args.training_plot, 'plot')

    learning_rate = gui.Variable('Learning Rate', args.learning_rate, 'lr')
    momentum = gui.Variable('Momentum', args.momentum)
    nesterov = gui.Variable('nesterov', args.nesterov)

    dataset = gui.Variable('Dataset', Dataset(get_config(), moves.val))
    dataset_caching = gui.Variable('Caching', args.dataset_size,)
    dataset_size = gui.Variable(
        'Size',
        args.dataset_size,
        callback=lambda: dataset.sample(dataset_size.val, dataset_caching.val)
    )
    data_split = gui.Variable('Data_split', args.data_split, 'split')
    batch_size = gui.Variable(
        'Batch Size',
        args.batch_size,
        'batch',
        callback=lambda: dataset.with_batching(batch_size.val)
    )
    mirrored = gui.Variable(
        'Is mirrored',
        args.mirroring,
        'mirror',
        callback=lambda: dataset.with_mirroring(mirrored.val)
    )
    type_switched = gui.Variable(
        'Is type switched',
        args.type_switching,
        'switch',
        callback=lambda: dataset.with_type_switching(type_switched.val, type_switch_limit.val)
    )
    type_switch_limit = gui.Variable(
        'Type switch limit',
        args.type_switch_limit,
        'limit',
        callback=lambda: dataset.with_type_switching(type_switched.val, type_switch_limit.val)
    )

    dataset.with_batching(batch_size.val).with_mirroring(mirrored.val).with_type_switching(type_switched.val, type_switch_limit.val)

    def show_board():
        with redirect_stdout(sys.__stdout__):
            anim = BoardAnimator(1, 1)
            anim.draw(board.array)
            input('Press enter to continue')
            del anim

    def train_model():
        if model.val is None:
            print('No model selected, create or load one')
            return
        try:
            train, test = dataset.sample(dataset_size.val, dataset_caching.val).get_split(data_split.val)
            model.train(train, test, epochs.val, eval_every.val)
            nn_mcts_score.clear()
        except Exception as e:
            print(e)

    def load_model():
        with redirect_stdout(sys.__stdout__):
            model.set(ElementCrush.load(), force=True)
        nn_mcts_score.clear()

    def sample():
        size = sample_size.val
        if len(random_scores) < size:
            missing = size - len(random_scores)
            random_scores.extend(async_pbar_auto_batcher(random_task, missing))

        if len(best_score) < size:
            missing = size - len(best_score)
            best_score.extend(async_pbar_auto_batcher(best_task, missing))

        if len(mcts_score) < size:
            missing = size - len(mcts_score)
            mcts_score.extend(async_pbar_auto_batcher(mcts_task, missing))

        if len(nn_mcts_score) < size and model != None:
            missing = size - len(nn_mcts_score)
            nn_mcts_score.extend([nn_mcts_task(model.val) for _ in tqdm(range(missing))])

        data = {
            'Random actions': random_scores[:size],
            'Best actions': best_score[:size],
            'MCTS actions': mcts_score[:size],
        }

        if any(nn_mcts_score):
            data['NN MCTS actions'] = nn_mcts_score[:size]

        plot_distribution(data)

    def save_model():
        if model.val is None:
            print('No model selected, create or load one')
            return
        model.val.save()

    if args.gui:
        main_gui = gui.Menu(
            'Welcome to the element crush AIGS exam project',
            info=gui.Info(model),
            variables=gui.Variables(
                # profiling, would be cool to be able to enable the profiler at any time
                sample_size
            ),
            actions=gui.Actions(gui.Action('sample', sample)),
            submenus=[
                gui.Menu(
                    'Element Crush AI',
                    short_hand='ai',
                    info=gui.Info('The model uses the sgd optimizer', model),
                    variables=gui.Variables(
                        features,
                        residual_layer,
                        epochs,
                        eval_every,
                        show_training_plot,
                        learning_rate,
                        momentum,
                        nesterov
                    ),
                    actions=gui.Actions(
                        gui.Action('Create model', lambda: model.set(ElementCrush(
                            get_config(),
                            residual_layer.val,
                            features.val,
                            sgd(learning_rate.val, momentum.val, nesterov=nesterov.val)
                        ), force=True), 'new'),
                        gui.Action('Save', save_model),
                        gui.Action('Load', load_model),
                        gui.Action('Train', train_model),
                    )
                ),
                gui.Menu(
                    'Board',
                    short_hand='b',
                    variables=gui.Variables(seed, moves, shape),
                    actions=gui.Actions(gui.Action('Show', show_board))
                ),
                gui.Menu(
                    'Dataset',
                    short_hand='ds',
                    info=gui.Info('The Dataset auto-refreshes when variables are changed'),
                    variables=gui.Variables(
                        dataset_size,
                        data_split,
                        batch_size,
                        mirrored,
                        type_switched,
                        type_switch_limit,
                    ),
                ),
            ]
        )

        main_gui.start()
    model.val = ElementCrush.load()
    sample()

    if args.train_em_all:
        train, test = dataset.sample(dataset_size.val, dataset_caching.val).get_split(data_split.val)
        model = ElementCrush(get_config(), 5, 64)
        model.train(train, test, epochs.val, eval_every.val)
        model.save()
        model = ElementCrush(get_config(), 5, 128)
        model.train(train, test, epochs.val, eval_every.val)
        model.save()
        model = ElementCrush(get_config(), 10, 64)
        model.train(train, test, epochs.val, eval_every.val)
        model.save()

    if args.profile:
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
