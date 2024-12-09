import math
import jax.numpy as jnp
from itertools import permutations
from pickle import dump, load

import numpy as np
from tqdm import tqdm

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
        'policies': []
    }
    mcts = MCTS(env, verbal=False)
    for _ in range(batch_size):
        action, value, policy = mcts.analyse()
        policies = np.zeros((env.action_space,))
        for a, p in zip(env.board.actions, policy):
            policies[a] = p

        data['observations'].append(obs)
        data['policies'].append(policies)

        obs, reward, done, won, _ = env.step(action)
        callback()
        if done:
            obs, _ = env.reset()
            mcts = MCTS(env, verbal=False)

    return [data]


class Dataset:
    def __init__(self, size, fat_cache=False, mirroring=True, type_switching=False, types=None, type_switch_limit=-1):
        assert not type_switching or type_switching and types is not None

        dataset_path = f'{size}.ds'
        self.dataset = {
            'observations': [],
            'policies': []
        }

        self.size = size

        fat_cache_file = 'fat_'
        if mirroring:
            fat_cache_file += 'm_'
        if type_switching:
            fat_cache_file += f't-s({types}'
            if type_switch_limit == -1:
                type_switch_limit = math.perm(types+1)
            fat_cache_file += f'-{type_switch_limit})_'
        fat_cache_file += dataset_path
        fat_cache_failed = True
        if fat_cache and (mirroring or type_switching):
            try:
                with open(fat_cache_file, 'rb') as read_handle:
                    self.dataset = load(read_handle)
                fat_cache_failed = False
                if mirroring:
                    self.size *= 2 * type_switch_limit
            except:
                print('No fat cache found')

        if fat_cache_failed:
            self._load_or_create(dataset_path)

            if type_switching:
                self._type_switch(types, type_switch_limit)

            if mirroring:
                self.mirror = {}
                self._mirror()

            for data_key in self.dataset:
                np.random.seed(0)
                np.random.shuffle(self.dataset[data_key])

            if fat_cache:
                if len(self.dataset['observations']) > 1_718_000:
                    print('Warning dataset has over 1_718_000 entries or (2GB), pickler might crash')
                with open(fat_cache_file, 'wb') as write_handle:
                    dump(self.dataset, write_handle)

        self.dataset['values'] = [np.max(policies) for policies in self.dataset['policies']]
        self._batching = None

    def _load_or_create(self, dataset_path):
        try:
            with open(dataset_path, 'rb') as read_handle:
                self.dataset = load(read_handle)

        except:
            print('No cache found, creating dataset')
            mcts_data = batched_async_pbar(task, self.size)
            for d in mcts_data:
                for data_key, data_value in d.items():
                    self.dataset[data_key].extend(data_value)

            with open(dataset_path, 'wb') as write_handle:
                dump(self.dataset, write_handle)

    def _mirror(self):
        print(f'Mirroring enabled - doubling dataset size from {self.size} to {self.size * 2}')
        self.size *= 2

        def mirror_action(_action, _board):
            if _action in self.mirror:
                return self.mirror[_action]
            (r1, c1), (r2, c2) = Board.decode(_action, _board)
            width = _board.shape[1] - 1
            c1 = width - c1
            c2 = width - c2
            mirrored_action = Board.encode((r1, c1), (r2, c2), _board)
            self.mirror[_action] = mirrored_action
            self.mirror[mirrored_action] = _action
            return mirrored_action

        with tqdm(total=self.size) as pbar:
            for observation, policies in list(zip(*self.dataset.values())):
                self.dataset['observations'].append(np.fliplr(observation))

                mirrored_policy = np.zeros((len(policies),))

                for idx, policy in enumerate(policies):
                    mirrored_policy[mirror_action(idx, observation)] = policy

                self.dataset['policies'].append(mirrored_policy)
                pbar.n += 2
                pbar.refresh()

    def _type_switch(self, types, limit):
        bits = int(math.ceil(math.log2(types)))
        mask = 7 * (2 ** bits)
        big_bad = 2 ** (bits + 2)

        value_to_letter = {val: chr(97 + val) for val in range(types+1)}
        print(f'Type switching enabled - increasing dataset size {limit} times, from {self.size} to {self.size * limit}')
        self.size *= limit
        new_observations = []
        new_policies = []

        with tqdm(total=self.size) as pbar:
            for observation, policies in list(zip(*self.dataset.values())):
                special_tokes = observation & mask
                observation -= special_tokes
                pattern = np.vectorize(value_to_letter.get)(observation)
                all_permutations = permutations(range(types + 1))
                new_boards = []
                for i, perm in enumerate(all_permutations):
                    if i == limit:
                        break
                    letter_to_value = {letter: value for letter, value in zip(value_to_letter.values(), perm)}
                    new_board = np.vectorize(letter_to_value.get)(pattern)
                    new_board += special_tokes
                    new_board = np.clip(new_board, 0, big_bad)
                    new_boards.append(new_board)
                    pbar.n += 1
                    pbar.refresh()

                new_observations.extend(new_boards)
                new_policies.extend([policies] * len(new_boards))
        self.dataset['observations'] = new_observations
        self.dataset['policies'] = new_policies

    def with_batching(self, batch_size):
        self._batching = batch_size
        return self

    def get_split(self, test_data_split=0.2):
        test_data_split = int(test_data_split * self.size)
        test_data = {k: jnp.array(v[:test_data_split]) for k, v in self.dataset.items()}
        training_data = {k: v[test_data_split:] for k, v in self.dataset.items()}

        if self._batching is not None:
            train = [
                {
                    key: jnp.array(data[lower: min(self.size - test_data_split, lower + self._batching)])
                    for key, data in training_data.items()
                }
                for lower in range(0, self.size - test_data_split, self._batching)
            ]
            training_data = train
        else:
            training_data = {k: jnp.array(v) for k, v in training_data.items()}
        return training_data, test_data
