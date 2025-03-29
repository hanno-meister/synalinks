# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from synalinks.src.api_export import synalinks_export
from synalinks.src.utils.plot_utils import generate_distinct_colors


@synalinks_export("synalinks.utils.plot_history")
def plot_history(
    history,
    to_file="training_history.png",
    to_folder=None,
    xlabel="Epochs",
    ylabel="Scores",
    title="Training history",
    grid=True,
    **kwargs,
):
    """Plots the training history of a program and saves it to a file.

    Code Example:

    ```python
    program.compile(...)
    history = await program.fit(...)

    synalinks.utils.plot_history(history)
    ```

    Example:

    ![training_history.png](../../assets/training_history.png)

    Args:
        history (History): The training history.
        to_file (str): The file path where the plot will be saved.
            Default to "training_history.png".
        xlabel (str): Optional. The label for the x-axis. Default to "Epochs".
        ylabel (str): Optional. The label for the y-axis. Default to "Scores".
        title (str): Optional. The title of the plot. Default to "Training history".
        grid (bool): Whether to display the grid on the plot. Default to True.
        **kwargs (keyword arguments): Addtional keyword arguments
            forwarded to `plt.plot()`

    Raises:
        ValueError: If there are unrecognized keyword arguments.

    Returns:
        (IPython.display.Image | marimo.Image | str):
            If running in a Jupyter notebook, returns an IPython Image object
            for inline display. If running in a Marimo notebook returns a marimo image.
            Otherwise returns the filepath where the image have been saved.
    """

    colors = generate_distinct_colors(len(history.history))

    for (metric, value), color in zip(history.history.items(), colors):
        plt.plot(value, label=metric, color=color, **kwargs)

    if xlabel:
        plt.xlabel(xlabel)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

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
