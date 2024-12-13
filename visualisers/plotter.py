import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from visualisers.table import build_table


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
            'standard_deviation': math.sqrt(variance) + 1e-10
        })

        bin_end = max(bin_end, q5)

    # Plot each distribution

    print(build_table(
        'Samples table',
        ['Dataset', 'Mean', 'Min(Q0)', 'Q1', 'Median(Q2)', 'Q3', 'Max(Q4)', 'Standard Deviation'],
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
    plt.xlabel('Game Score')
    plt.ylabel('Density')
    plt.title('Normal Distributions of Game Score')
    plt.grid()
    plt.show()


class SubPlot:
    def __init__(self, label, x_values, y_values, x_step=1):
        self.label = label
        self.x_values = x_values
        self.y_values = y_values
        self.x_step = x_step

        self.plot = None

    def add_data(self, x, y):
        self.x_values.append(x)
        self.y_values.append(y)

    def add_value(self, y):
        self.x_values.append(self.x_values[-1] + self.x_step if len(self.x_values) > 0 else self.x_step)
        self.y_values.append(y)

    def set_data(self, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values

    def set_y_data(self, y_values):
        self.set_data(
            list(range(self.x_step, self.x_step * (len(y_values) + 1), self.x_step)),
            y_values
        )

    def update(self):
        self.plot.set_data(self.x_values, self.y_values)


class Plot:
    def __init__(self, x_axis_label, y_axis_label, title=None):
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.title = title or f'{y_axis_label} over {x_axis_label}'

        self.plots: dict[str, SubPlot] = {}

    def add_plot(self, label, x_values=None, y_values=None, x_step=1):
        x_values = x_values or []
        y_values = y_values or []
        assert len(x_values) == len(y_values)
        self.plots[label] = SubPlot(label, x_values, y_values, x_step)
        return self.plots[label]

    def update(self):
        for plot in self.plots.values():
            plot.update()


class LivePlotter:
    def __init__(self):
        self.plots: dict[str, Plot] = {}
        self.fig: Figure | None = None
        self.axes = []

    def add_view(self, x_axis_label, y_axis_label, title=None):
        plot = Plot(x_axis_label, y_axis_label, title)
        self.plots[plot.title] = plot
        return plot

    def build(self):
        plt.ion()
        n_plots = len(self.plots)

        rows, cols = None, None
        for i in range(1, int(n_plots**0.5) + 1):
            if n_plots % i == 0:
                rows, cols = i, n_plots // i

        self.fig, self.axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))

        for i, (title, plot) in enumerate(self.plots.items()):
            r = i % rows
            c = int(i/rows)
            idx = (r, c) if r > 1 and c > 1 else i
            for label, sub_plot in plot.plots.items():
                sub_plot.plot = self.axes[r, c].plot(sub_plot.x_values, sub_plot.y_values, label=label, marker='o')[0]
            self.axes[r, c].set_title(plot.title)
            self.axes[r, c].set_xlabel(plot.x_axis_label)
            self.axes[r, c].set_ylabel(plot.y_axis_label)
            self.axes[r, c].legend()

    def add_value_for(self, label, value):
        for plot in self.plots.values():
            if label in plot.plots:
                plot.plots[label].add_value(value)

    def update(self):
        n_plots = len(self.plots)

        rows, cols = None, None
        for i in range(1, int(n_plots**0.5) + 1):
            if n_plots % i == 0:
                rows, cols = i, n_plots // i

        for plot in self.plots.values():
            plot.update()
        for ax in self.axes:
            if rows > 1 and cols > 1:
                for ay in ax:
                    ay.relim()
                    ay.autoscale_view()
            else:
                ax.relim()
                ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show(self):
        plt.ioff()
        plt.show()

    def save(self, name):
        self.fig.savefig(f'plots/{name}.png')
