from syn_grid.utils.paths_util import get_project_path
from syn_grid.gymnasium.utils.episode_logging.log_keys import LogKey

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from enum import Enum
import re

# ================= #
#     Constants     #
# ================= #


_BASE_LOG_DIR = get_project_path(
    "output", "results", "logs", "thesis_real", "tier_5", "threshold"
)
_BASE_PLOT_DIR = get_project_path("output", "results", "plots", "tier_5", "threshold")


# ================= #
#      Colors       #
# ================= #


class Color(str, Enum):
    BLUE = "steelblue"
    RED = "crimson"
    GREEN = "limegreen"
    PURPLE = "rebeccapurple"
    ORANGE = "darkorange"
    TEAL = "teal"
    PINK = "hotpink"
    GREY = "slategrey"


# ====================================================== #
#                         Plots                          #
#                                                        #
#                 The logger's header:                   #
# [episode,reward,length,chains_completed,chains_broken] #
# ====================================================== #


def plot_reward(csv_dir: Path, plots_dir: Path) -> None:
    _figsize()
    for i, file in enumerate(csv_dir.glob(_get_files())):
        label, color = _get_label_and_color(file, i)
        data, window = _get_data_and_window(file)

        _plot_series(data[LogKey.REWARD], color, window, label)

    _finalize_plot("rewards", plots_dir)


def plot_episode_length(csv_dir: Path, plots_dir: Path) -> None:
    _figsize()
    for i, file in enumerate(csv_dir.glob(_get_files())):
        label, color = _get_label_and_color(file, i)
        data, window = _get_data_and_window(file)

        print(f"\n{file} color: {color}\n")

        _plot_series(data[LogKey.LENGTH], color, window, label)

    _finalize_plot("steps", plots_dir)


def plot_average_reward(csv_dir: Path, plots_dir: Path) -> None:
    _figsize()
    for i, file in enumerate(csv_dir.glob(_get_files())):
        label, color = _get_label_and_color(file, i)
        data, window = _get_data_and_window(file)
        average_reward = data[LogKey.REWARD] / data["length"]

        _plot_series(average_reward, color, window, label)

    _finalize_plot("average_rewards", plots_dir)


def plot_chain_progression_steps(csv_dir: Path, plots_dir: Path) -> None:
    _figsize()
    for i, file in enumerate(csv_dir.glob(_get_files())):
        label, color = _get_label_and_color(file, i)
        data, window = _get_data_and_window(file)

        _plot_series(data[LogKey.CHAIN_PROGRESSED], color, window, label)

    _finalize_plot("chain_progression_steps", plots_dir)


def plot_chain_outcomes(csv_dir: Path, plots_dir: Path) -> None:
    _figsize()
    for i, file in enumerate(csv_dir.glob(_get_files())):
        label, color = _get_label_and_color(file, i * 2)
        color2 = list(Color)[(i * 2 + 1) % len(Color)]
        data, window = _get_data_and_window(file)
        chains_completed = data[LogKey.CHAINS_COMPLETED]
        chains_broken = data[LogKey.CHAINS_BROKEN]

        _plot_series(chains_completed, color, window, f"{label} Completed chains")
        _plot_series(chains_broken, color2, window, f"{label} Broken chains")

    _finalize_plot("chain_outcomes", plots_dir)


def plot_completion_rate(csv_dir: Path, plots_dir: Path) -> None:
    _figsize()
    for i, file in enumerate(csv_dir.glob(_get_files())):
        label, color = _get_label_and_color(file, i)
        data, window = _get_data_and_window(file)
        completion_rate = data[LogKey.CHAINS_COMPLETED] / (
            data[LogKey.CHAINS_COMPLETED] + data[LogKey.CHAINS_BROKEN]
        )

        _plot_series(completion_rate, color, window, label)

    _finalize_plot("chain_completion_rate", plots_dir)


# === Single chain mode plots === #


def plot_success(csv_dir: Path, plots_dir: Path) -> None:
    _figsize()
    for i, file in enumerate(csv_dir.glob(_get_files())):
        label, color = _get_label_and_color(file, i)
        data, window = _get_data_and_window(file)

        _plot_series(data[LogKey.CHAINS_COMPLETED], color, 1, label)

    _finalize_plot("Reached max tier", plots_dir)


def plot_failure(csv_dir: Path, plots_dir: Path) -> None:
    _figsize()
    for i, file in enumerate(csv_dir.glob(_get_files())):
        label, color = _get_label_and_color(file, i)
        data, window = _get_data_and_window(file)

        _plot_series(data[LogKey.CHAINS_BROKEN], color, 1, label)

    _finalize_plot("Broke the chain", plots_dir)


# ================= #
#      Helpers      #
# ================= #


def _figsize() -> None:
    plt.figure(figsize=(15, 8))


def _get_files() -> str:
    return "*.csv"


def _get_label_and_color(file: Path, i: int) -> tuple[str, Color]:
    match = re.search(r"seed\d+_[^_]+", file.stem)
    label = match.group() if match else "unknown"
    color = list(Color)[i % len(Color)]
    return label, color


def _get_data_and_window(file: Path) -> tuple[DataFrame, int]:
    data = pd.read_csv(file)
    return data, len(data) // 10


def _plot_series(data, color, window: int, label: str) -> int:
    if window <= 1:
        smoothed = data.fillna(0)
    else:
        smoothed = data.fillna(0).rolling(window=window).mean()

    plt.plot(data, alpha=0.2, color=color)
    plt.plot(smoothed, label=label, color=color)
    return 1


def _finalize_plot(plot_id: str, plots_dir: Path):
    plt.xlabel("Episode")
    plt.ylabel(plot_id)
    plt.title(f"{plot_id} per episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"{plot_id}.png")
    plt.close()
