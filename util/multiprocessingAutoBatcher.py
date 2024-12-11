from multiprocessing import Value, Pool
from os import cpu_count
from time import sleep

from tqdm import tqdm


def initializer(counter: Value):
    global progress_counter
    progress_counter = counter


def update_pbar():
    global progress_counter
    with progress_counter.get_lock():
        progress_counter.value += 1


def async_pbar(func, data, bar_total):
    progress_counter = Value('i', 0)

    with tqdm(total=bar_total) as pbar:
        with Pool(initializer=initializer, initargs=(progress_counter,)) as pool:
            batches = pool.map_async(func, data)

            while not batches.ready():
                sleep(0.5)
                with progress_counter.get_lock():
                    pbar.n = progress_counter.value
                pbar.refresh()
            pbar.n = bar_total
            pbar.refresh()

    return [data for batch in batches.get() for data in batch]


def batch_data(data_size, insert=None):
    cpus = min(data_size, cpu_count())
    batches_pr_cpu = int(data_size / cpus)
    underflow = data_size - batches_pr_cpu * cpus
    if insert:
        return [(insert, batches_pr_cpu+1) if i < underflow else (insert, batches_pr_cpu) for i in range(cpus)]
    return [batches_pr_cpu+1 if i < underflow else batches_pr_cpu for i in range(cpus)]


def batcher(data):
    f, batched_data = data
    batched_results = []
    for _ in range(batched_data):
        batched_results.append(f())
        update_pbar()
    return batched_results


def async_pbar_auto_batcher(func, data_size):
    return async_pbar(batcher, batch_data(data_size, func), data_size)


def batched_async_pbar(func, data_size):
    return async_pbar(func, batch_data(data_size, update_pbar), data_size)
