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