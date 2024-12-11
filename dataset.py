import math
import os

from itertools import permutations
from pickle import dump, load

import numpy as np
from tqdm import tqdm

from match3tile.boardConfig import BoardConfig
from match3tile.boardv2 import BoardV2
from mctslib.standard.mcts import MCTS
from util.multiprocessingAutoBatcher import batched_async_pbar


def mcts_task(data):
    (callback, cfg, moves), batch_size = data
    data = {
        'observations': [],
        'policies': [],
        'values': []
    }
    count = 0

    while True:
        if count > batch_size:
            break
        state = BoardV2(moves, cfg)
        mcts = MCTS(state, 3, 256, False, False)
        while not state.is_terminal:
            action, _, policy_logits = mcts()
            policies = np.zeros_like(state.array)
            for a, p in zip(state.legal_actions, policy_logits):
                policies[a] = p

            data['observations'].append(state.array)
            data['policies'].append(policies)
            state = state.apply_action(action)
            count += 1
            callback()
        data['values'].extend([state.reward] * moves)

    return [data]


class Dataset:
    def __init__(self, cfg: BoardConfig, moves=20):
        self.cfg = cfg
        self.moves = moves

        self._size = 0

        self._mirroring = False
        self._batching = 1
        self._type_switching = False
        self._type_switching_limit = -1

        self.dataset = {
            'observations': [],
            'policies': [],
            'values': []
        }
        self._type_switched_dataset: list[dict[str, list]] = []

    def sample(self, size, caching=True):
        size = 20 * math.ceil(size / 20)
        file = str((*self.cfg.shape, self.cfg.types)) + '.ds'
        if caching and os.path.isfile(file) and len(self.dataset['values']) == 0:
            with open(file, 'rb') as read_handle:
                self.dataset = load(read_handle)

        missing_data = size - len(self.dataset['values'])
        if missing_data > 0:
            data = batched_async_pbar(mcts_task, missing_data, (self.cfg, self.moves))
            for data_batch in data:
                for data_key, data_value in data_batch.items():
                    self.dataset[data_key].extend(data_value)
            if caching:
                with open(file, 'wb') as write_handle:
                    # noinspection PyTypeChecker
                    dump(self.dataset, write_handle)

        self._size = size
        return self

    def mirror(self, data):
        if not self._mirroring:
            return data

        with tqdm(total=len(data['values']) * 2) as pbar:
            pbar.n = len(data['values'])
            pbar.refresh()
            for observation, policies, values in list(zip(*data.values())):
                data['observations'].append(np.fliplr(observation))

                mirrored_policy = np.zeros((len(policies),))

                for idx, policy in enumerate(policies):
                    if policy == 0:
                        continue
                    (r1, c1), (r2, c2) = self.cfg.decode(idx)
                    c1 = self.cfg.columns - 1 - c1
                    c2 = self.cfg.columns - 1 - c2

                    mirrored_action = self.cfg.encode((r1, c1), (r2, c2))
                    mirrored_policy[mirrored_action] = policy

                data['policies'].append(mirrored_policy)
                data['values'].append(values)
                pbar.n += 1
                pbar.refresh()
            return data

    def type_switch(self):
        if not self._type_switching:
            return
        limit = self._type_switching_limit
        if limit <= 0:
            limit = math.factorial(self.cfg.types)
        limit -= 1
        value_to_letter = {val: chr(97 + val) for val in range(1, self.cfg.types+2)}
        value_to_letter[0] = 'x'

        with tqdm(total=limit * self._size + self._size) as pbar:
            pbar.n = self._size

            def switch(obs: np.ndarray, lower) -> list[np.ndarray]:
                if lower >= limit:
                    return []
                switched_obs = []

                special_tokes = obs & self.cfg.special_type_mask
                tokens = obs & self.cfg.type_mask

                pattern = np.vectorize(value_to_letter.get)(tokens)

                all_permutations = permutations(range(1, self.cfg.types + 2))
                for i, perm in enumerate(all_permutations):
                    if i == 0:
                        continue
                    if i == limit+1:
                        break
                    pbar.n += 1
                    pbar.refresh()
                    if i < lower:
                        continue
                    del value_to_letter[0]
                    letter_to_value = {letter: value for letter, value in zip(value_to_letter.values(), perm)}
                    letter_to_value['x'] = self.cfg.mega_token
                    value_to_letter[0] = 'x'

                    new_board = np.vectorize(letter_to_value.get)(pattern)
                    new_board += special_tokes
                    switched_obs.append(new_board)
                return switched_obs

            for i, observation in enumerate(self.dataset['observations'][:self._size]):
                if i < len(self._type_switched_dataset):
                    current_switched = len(self._type_switched_dataset[i]['observations'])
                    if current_switched >= limit:  # 1 - 10 = 9
                        pbar.n += limit
                        pbar.refresh()
                        continue
                    self._type_switched_dataset[i]['observations'].extend(switch(observation, current_switched))
                    self._type_switched_dataset[i]['policies'].extend(self.dataset['policies'][i] * (limit - current_switched))
                    self._type_switched_dataset[i]['values'].extend(self.dataset['values'][i] * (limit - current_switched))
                else:
                    self._type_switched_dataset.append({
                        'observations': switch(observation, 1),
                        'policies': [self.dataset['policies'][i]] * limit,
                        'values': [self.dataset['values'][i]] * limit,
                    })

    def with_mirroring(self, should_mirror):
        self._mirroring = should_mirror
        return self

    def with_batching(self, batch_size):
        self._batching = batch_size
        return self

    def with_type_switching(self, should_switch, switch_limit):
        self._type_switching = should_switch
        self._type_switching_limit = min(switch_limit, math.factorial(self.cfg.types+1))
        return self

    def get_split(self, split=0.8):
        data = {key: list(values[: self._size]) for key, values in self.dataset.items()}
        self.type_switch()
        for switch in self._type_switched_dataset:
            for key, values in switch.items():
                data[key].extend(values[:self._size])
        data = self.mirror(data)

        if not (0 < split < 1):
            raise ValueError("Split value must be between 0 and 1.")

        # Ensure all data arrays are the same length
        obs = np.array(data['observations'])
        pol = np.array(data['policies'])
        val = np.array(data['values'])

        if not (len(obs) == len(pol) == len(val)):
            raise ValueError("All input data arrays must have the same length.")

        # Shuffle data
        indices = np.arange(len(obs))
        np.random.shuffle(indices)

        obs, pol, val = obs[indices], pol[indices], val[indices]

        # Split data
        split_idx = int(len(obs) * split)

        train_data = {
            'observations': obs[:split_idx],
            'policies': pol[:split_idx],
            'values': val[:split_idx]
        }
        test_data = {
            'observations': obs[split_idx:],
            'policies': pol[split_idx:],
            'values': val[split_idx:]
        }

        # Helper function to create batches
        def batchify(data):
            n_batches = math.ceil(len(data['observations']) / self._batching)
            batches = []
            for i in range(n_batches):
                start_idx = i * self._batching
                end_idx = start_idx + self._batching
                batch = {
                    'observations': data['observations'][start_idx:end_idx],
                    'policies': data['policies'][start_idx:end_idx],
                    'values': data['values'][start_idx:end_idx]
                }
                batches.append(batch)
            return batches

        return batchify(train_data), batchify(test_data)
