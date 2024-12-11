import cProfile
import os

from mctslib.standard.mcts import MCTS


import pstats


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


def perform_profiling(mode="full", sort_key="time", mcts: MCTS = None, file="mcts_new.prof"):
    """
    Runs the profiler on the MCTS algorithm and prints the results.
    mode: "full" or "quick".
            "quick" will only look at the MCTS.prof file (provided it exists)
            "full" will run the profiler and save the results to MCTS.prof
    sort_key: "calls", "cumtime", "time". Sorts the profiler output by the specified key.
    """

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
