from syn_grid.utils.paths_util import get_project_path
from syn_grid.plot.plot_helpers import (
    Color,
    _EVAL,
    _setup_plotting,
    _plot_series,
    _finalize_plot,
)

import pandas as pd
from pathlib import Path

# ================= #
#       Plots       #
# ================= #


def plot_reward(csv_dir: Path, plots_dir: Path) -> None:
    name = _setup_plotting("rewards")

    for file in csv_dir.glob("*.csv"):
        data = pd.read_csv(file)
        window = len(data) // 30

        _plot_series(data["reward"], Color.RED, window, name)

    _finalize_plot(name, plots_dir)


def plot_average_reward(csv_dir: Path, plots_dir: Path) -> None:
    name = _setup_plotting("average_rewards")

    for file in csv_dir.glob("*.csv"):
        data = pd.read_csv(file)
        window = len(data) // 30
        average_reward = data["reward"] / data["length"]

        _plot_series(average_reward, Color.RED, window, name)

    _finalize_plot(name, plots_dir)


def plot_episode_length(csv_dir: Path, plots_dir: Path) -> None:
    name = _setup_plotting("steps")

    for file in csv_dir.glob("*.csv"):
        data = pd.read_csv(file)
        window = len(data) // 30

        _plot_series(data["length"], Color.RED, window, name)

    _finalize_plot(name, plots_dir)


def plot_chain_outcomes(csv_dir: Path, plots_dir: Path) -> None:
    name = _setup_plotting("chain_outcomes")

    for file in csv_dir.glob("*.csv"):
        data = pd.read_csv(file)
        window = len(data) // 30
        chains_completed = data["chains_completed"]
        chains_broken = data["chains_broken"]

        _plot_series(chains_completed, Color.ORANGE, window, "Completed chains")
        _plot_series(chains_broken, Color.TEAL, window, "Broken chains")

    _finalize_plot(name, plots_dir)


def plot_completion_rate(csv_dir: Path, plots_dir: Path) -> None:
    name = _setup_plotting("chain_completion_rate")

    for file in csv_dir.glob("*.csv"):
        data = pd.read_csv(file)
        window = len(data) // 30
        completion_rate = data["chains_completed"] / (
            data["chains_completed"] + data["chains_broken"]
        )

        _plot_series(completion_rate, Color.PINK, window, name)

    _finalize_plot(name, plots_dir)


# ================= #
#       Main        #
# ================= #

if __name__ == "__main__":
    # The logger have a header with [episode,reward,length,chains_completed,chains_broken]
    csv_dir = get_project_path(
        "output", "results", "logs", "experiments", "just_checking", _EVAL
    )
    plots_dir = get_project_path("output", "results", "plots", _EVAL)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_reward(csv_dir, plots_dir)
    plot_average_reward(csv_dir, plots_dir)
    plot_episode_length(csv_dir, plots_dir)
    plot_chain_outcomes(csv_dir, plots_dir)
    plot_completion_rate(csv_dir, plots_dir)
