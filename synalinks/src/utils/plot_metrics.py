# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import os

import matplotlib.pyplot as plt

from synalinks.src.api_export import synalinks_export
from synalinks.src.utils.plot_utils import generate_distinct_colors


@synalinks_export("synalinks.utils.plot_metrics")
def plot_metrics(
    metrics,
    to_file="evaluation_metrics.png",
    to_folder=None,
    xlabel="Metrics",
    ylabel="Scores",
    title="Evaluation metrics",
    grid=True,
    **kwargs,
):
    """Plots the evaluation metrics of a program and saves it to a file.

    Code Example:

    ```python
    program.compile(...)
    metrics = await program.evaluate(...)

    synalinks.utils.plot_metrics(metrics)
    ```

    Args:
        metrics (dict): The metrics from a program evaluation.
        to_file (str): The file path where the plot will be saved.
            Default to "evaluation_metrics.png".
        xlabel (str): Optional. The label for the x-axis. Default to "Metrics".
        ylabel (str): Optional. The label for the y-axis. Default to "Scores".
        title (str): Optional. The title of the plot. Default to "Evaluation metrics".
        grid (bool): Whether to display the grid on the plot. Default to True.
        **kwargs (keyword arguments): Addtional keyword arguments
            forwarded to `plt.bar()`

    Raises:
        ValueError: If there are unrecognized keyword arguments.

    Returns:
        (IPython.display.Image | marimo.Image | str):
            If running in a Jupyter notebook, returns an IPython Image object
            for inline display. If running in a Marimo notebook returns a marimo image.
            Otherwise returns the filepath where the image have been saved.
    """

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    colors = generate_distinct_colors(len(metric_names))

    plt.bar(metric_names, metric_values, color=colors, **kwargs)

    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(grid)
    if to_folder:
        to_file = os.path.join(to_folder, to_file)
    plt.savefig(to_file)
    plt.close()
    try:
        import marimo as mo

        if mo.running_in_notebook():
            return mo.image(src=to_file).center()
    except ImportError:
        pass
    try:
        from IPython import display

        return display.Image(filename=to_file)
    except ImportError:
        pass
    return to_file
