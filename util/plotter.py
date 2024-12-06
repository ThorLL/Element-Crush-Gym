import math
import random

import numpy as np
from matplotlib import pyplot as plt

from util.table import build_table


def plot_distribution(data):
    analyses = []

    bin_end = 0
    for name, values in data.items():
        values = sorted(values)
        n_values = len(values)
        mean = round(sum(values) / n_values, 2)
        variance = sum((x - mean) ** 2 for x in values) / n_values
        q0 = values[0]
        q5 = values[-1]
        analyses.append({
            'name': name,
            'mean': mean,
            'min': q0,
            'q1': values[int(0.25 * n_values)],
            'median': values[int(0.5 * n_values)],
            'q3': values[int(0.75 * n_values)],
            'max': q5,
            'standard_deviation': round(math.sqrt(variance), 2)
        })

        bin_end = max(bin_end, q5)

    # Plot each distribution

    print(build_table(
        ['Dataset', 'Mean', 'Min(Q0)', 'Q1', 'Median(Q2)', 'Q3', 'Max(Q5)', 'Standard Deviation'],
        [analysis.values() for analysis in analyses]
    ))

    plt.figure(figsize=(10, 6))
    for analysis in analyses:
        mean = analysis['mean']
        std_dev = analysis['standard_deviation']

        # Create a range of values for the normal distribution
        x = np.linspace(0, bin_end, 500)
        pdf = (1 / (std_dev * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        # Plot the Gaussian curve
        plt.plot(x, pdf, label=analysis['name'], linewidth=2)

    # Add legend and labels
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Normal Distributions of Data')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot_distribution({
        'Random actions': random.sample(range(1, 50), 7),
        'Naive actions': random.sample(range(1, 50), 7),
        'Best actions': random.sample(range(1, 50), 7),
        'MCTS actions': random.sample(range(1, 50), 7),
    })