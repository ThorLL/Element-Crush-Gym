from pickle import dump, load

import numpy as np

from MCTS import MCTS
from match3tile.board import Board
from match3tile.env import Match3Env
from util.mp import batched_async_pbar


def task(data):
    callback, batch_size = data
    env = Match3Env()
    obs = env.init()
    data = {
        'observations': [],
        'actions': []
    }
    for _ in range(batch_size):
        data['observations'].append(obs)
        data['actions'].append(MCTS(env, verbal=False)())
        obs, reward, done, won, _ = env.step(env.board.random_action())
        callback()
        if done:
            obs, _ = env.reset()
    return [data]


def get_train_and_test_data(
        batches: int = 1000,
        batch_size: int = 128,
        split: float = 0.2
) -> tuple[list[dict[str, list]], list[dict[str, list[int]]]]:
    pickle_path = f'{batches}x{batch_size}'
    train_ds_path = f'{pickle_path}_training_data.ds'
    test_ds_path = f'{pickle_path}_test_data.ds'
    try:
        with open(train_ds_path, 'rb') as train, open(test_ds_path, 'rb') as test:
            training_data = load(train)
            test_data = load(test)
    except:
        data = batched_async_pbar(task, int(batches/2) * batch_size)
        training_data = []
        test_data = []

        # mirror actions and observations:
        def create_mirror(a, o):
            (r1, c1), (r2, c2) = Board.decode(a, o)
            width = obs.shape[1] - 1
            c1 = width - c1
            c2 = width - c2
            return Board.encode((r1, c1), (r2, c2), o), np.fliplr(obs)

        mirrored_data = []
        for data_samp in data:
            for action, obs in zip(data_samp['actions'], data_samp['observations']):
                mirrored_data.append((action, obs))
                mirrored_data.append(create_mirror(action, obs))

        split = int(batches * batch_size * split)
        for i, (action, obs) in enumerate(mirrored_data):
            if i % batch_size == 0:

                (training_data if i >= split else test_data).append(
                    {
                        'actions': [],
                        'observations': []
                    }
                )
            (training_data if i >= split else test_data)[-1]['actions'].append(action)
            (training_data if i >= split else test_data)[-1]['observations'].append(obs)
            i += 1

        with open(train_ds_path, 'wb') as handle:
            dump(training_data, handle)
        with open(test_ds_path, 'wb') as handle:
            dump(test_data, handle)

    return training_data, test_data
