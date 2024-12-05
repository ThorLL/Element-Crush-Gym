import math

import numpy as np
from matplotlib import pyplot as plt


def plot_distribution(data: dict[str, list[int]]):
    analyses = [
        {
            'name': 'Dataset',
            'mean': 'Mean',
            'min': 'Min(Q0)',
            'q1': 'Q1',
            'median': 'Median(Q2)',
            'q3': 'Q3',
            'max': 'Max(Q5)',
            'standard_deviation': 'Standard Deviation'
        }
    ]

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
    plt.figure(figsize=(10, 6))

    min_widths = {}

    for analysis in analyses:
        for column, value in analysis.items():
            min_widths[column] = max(min_widths[column], len(str(value))) if column in min_widths else max(len(str(value)), 8)
    min_widths = {key: val+2 if val % 2 == 0 else val+3 for key, val in min_widths.items()}
    row_seperator = '|'
    for length in min_widths.values():
        row_seperator += '-' * length + '|'

    print()
    print(row_seperator)

    for analysis in analyses:
        row = '|'
        for column, value in analysis.items():
            column_width = min_widths[column]
            missing_spaces = column_width - len(str(value))
            left = int(missing_spaces/2)
            right = missing_spaces - left
            row += ' ' * left + str(value) + ' ' * right + '|'
        print(row)
        print(row_seperator)

    for analysis in analyses[1:]:
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
