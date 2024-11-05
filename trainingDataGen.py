from pickle import dump, load
from multiprocessing import Pool, cpu_count, Value
from tqdm import tqdm
from numpy import zeros, int32, array
from match3tile.env import Match3Env


def create_batches(
        n_batches: int,
        batch_size: int,
        width: int,
        height: int,
        num_types: int,
        seed: int
) -> tuple[array, array]:
    global progress_counter
    env = Match3Env(width, height, num_types, seed=seed)
    obs = env.init()
    observations = zeros((n_batches, batch_size, height, width), dtype=int32)
    actions = zeros((n_batches, batch_size), dtype=int32)

    for i in range(n_batches):
        for j in range(batch_size):
            action = env.board.random_action()
            best_action = env.board.best_action()
            observations[i][j] = obs
            actions[i][j] = best_action
            obs, reward, done, won, _ = env.step(action)

            with progress_counter.get_lock():
                progress_counter.value += 1

            if done:
                obs, _ = env.reset()

    return observations, actions


def initializer(counter: Value):
    global progress_counter
    progress_counter = counter


def create_data(
        batches: int,
        batch_size: int,
        width: int,
        height: int,
        num_types: int,
        seed: int,
        split: float
) -> tuple[list[tuple[array, array]], list[tuple[array, array]]]:
    cpus = min(batches, cpu_count())
    batches_pr_cpu = int(batches / cpus)
    batches_pr_cpu = [
        batches_pr_cpu if i >= batches - batches_pr_cpu * cpus else batches_pr_cpu + 1
        for i in range(cpus)
    ]
    progress_counter = Value('i', 0)
    total = batches * batch_size
    print(f'Starting {cpus} processes to create the {total} observations')
    with tqdm(total=total) as pbar:
        with Pool(initializer=initializer, initargs=(progress_counter,)) as pool:
            batches = [
                pool.apply_async(
                    create_batches,
                    args=(batches_count, batch_size, width, height, num_types, seed + i,)
                )
                for i, batches_count in enumerate(batches_pr_cpu)
            ]
            while any(not batch.ready() for batch in batches):
                with progress_counter.get_lock():
                    pbar.n = progress_counter.value
                pbar.refresh()
            pbar.n = progress_counter.value
            pbar.refresh()
    batches = [batch.get() for batch in batches]
    batch_split = len(batches) - int(len(batches) * split)
    return batches[:batch_split], batches[batch_split::]


def get_train_and_test_data(
        batches: int = 1000,
        batch_size: int = 128,
        width: int = 7,
        height: int = 9,
        num_types: int = 6,
        seed: int = 0,
        split: float = 0.2
) -> tuple[list[dict[str, list]], list[dict[str, list[int]]]]:
    pickle_path = f'{seed}-({batches}x{batch_size})x{width}x{height}x{num_types}'
    train_ds_path = f'{pickle_path}_training_data.ds'
    test_ds_path = f'{pickle_path}_test_data.ds'
    try:
        with open(train_ds_path, 'rb') as train, open(test_ds_path, 'rb') as test:
            training_data = load(train)
            test_data = load(test)
    except:
        training_data, test_data = create_data(batches, batch_size, width, height, num_types, seed, split)
        with open(train_ds_path, 'wb') as handle:
            dump(training_data, handle)
        with open(test_ds_path, 'wb') as handle:
            dump(test_data, handle)

    return training_data, test_data
