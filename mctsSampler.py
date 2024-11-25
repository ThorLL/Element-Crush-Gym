from multiprocessing import Pool, cpu_count, Value
from time import sleep

from tqdm import tqdm

from MCTS import MCTS
from match3tile.env import Match3Env


def initializer(counter: Value):
    global progress_counter
    progress_counter = counter


def batched_mcts(batches):
    global progress_counter
    batch_scores = []
    for b in range(batches):
        mcts_env = Match3Env()
        mcts_algo = MCTS(mcts_env, simulations=50, verbal=False)
        score = 0

        while mcts_env.moves_taken != mcts_env.num_moves:
            _, reward, done, won, _ = mcts_env.step(mcts_algo())
            score += reward
            with progress_counter.get_lock():
                progress_counter.value += 1
        batch_scores.append(score)
    return batch_scores


def gen_samples(samples):
    cpus = min(samples, cpu_count())
    batches_pr_cpu = int(samples / cpus)
    batches_pr_cpu = [batches_pr_cpu] * cpus

    for i in range(samples - sum(batches_pr_cpu)):
        batches_pr_cpu[i] += 1

    progress_counter = Value('i', 0)
    with tqdm(total=samples) as pbar:
        with Pool(initializer=initializer, initargs=(progress_counter,)) as pool:
            batches = pool.map_async(batched_mcts, batches_pr_cpu)

            while not batches.ready():
                sleep(0.5)
                with progress_counter.get_lock():
                    pbar.n = progress_counter.value/20
                pbar.refresh()
            pbar.n = samples
            pbar.refresh()
    return [data for batch in batches.get() for data in batch]
