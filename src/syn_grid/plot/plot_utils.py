from syn_grid.utils.paths_util import get_project_path
from syn_grid.gymnasium.utils.episode_logging.log_keys import LogKey

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from enum import Enum
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================= #
#     Constants     #
# ================= #

_BASE_LOG_DIR = get_project_path(
    "output",
    "results",
    "logs",
    "thesis_new",
    "tier_3",
    "fully_pomdp",
    "single_chain",
    "max_tier",
    "tensorboard",
)

_BASE_PLOT_DIR = get_project_path("output", "results", "plots", "delay")


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

        _plot_series(data[LogKey.CHAINS_COMPLETED], color, window, label)

    _finalize_plot("Reached max tier", plots_dir)


def plot_failure(csv_dir: Path, plots_dir: Path) -> None:
    _figsize()
    for i, file in enumerate(csv_dir.glob(_get_files())):
        label, color = _get_label_and_color(file, i)
        data, window = _get_data_and_window(file)

        _plot_series(data[LogKey.CHAINS_BROKEN], color, window, label)

    _finalize_plot("Broke the chain", plots_dir)


# === TensorBoard plots === #

# One entry per architecture: label -> subfolder name under _BASE_LOG_DIR
# that contains that run's events.out.tfevents.* file.
# Adjust the subfolder names to match your actual layout.
_RUNS: dict[str, str] = {
    "RPPO_5x5": "RPPO_5x5",
    "RPPO_6x6": "RPPO_6x6",
    "RPPO_7x7": "RPPO_7x7",
    "FSPPO_5x5": "FSPPO_5x5",
    "FSPPO_6x6": "FSPPO_6x6",
    "FSPPO_7x7": "FSPPO_7x7",
    "PPO_5x5": "PPO_5x5",
    "PPO_6x6": "PPO_6x6",
    "PPO_7x7": "PPO_7x7",
}

_ARCH_COLOR = {
    "PPO_5x5": "#595959",
    "PPO tier 5": "#797979",
    "PPO_6x6": "#949494",
    "PPO tier 7": "#BBBBBB",
    "PPO_7x7": "#DADADA",
    "FSPPO_5x5": "#00ccff",
    "FSPPO_6x6": "#71e3ff",
    "FSPPO_7x7": "#bff2ff",
    "RPPO_5x5": "#e6b800",
    "RPPO_6x6": "#ffdf61",
    "RPPO_7x7": "#ffec9f",
}

# The scalar tag to plot. Run SummaryReader(...).scalars and print
# df.tag.unique() to find the exact name your LogKey logs under.
_REWARD_TAG = "rollout/ep_rew_mean"   # <-- replace with your actual tag
_LENGTH_TAG = "rollout/ep_len_mean"   # <-- replace with your actual tag
_EV_TAG = "train/explained_variance"
_ENTROPY_TAG = "train/entropy_loss"


def plot_ts_reward() -> None:
    _figsize()
    for label, subdir in _RUNS.items():
        steps, values = _get_scalar(_BASE_LOG_DIR / subdir, _REWARD_TAG)
        color = _ARCH_COLOR[label]
        window = max(len(values) // 50, 1)
        legend_label = "_nolegend_" if "ext" in subdir else label
        _plot_ts_series(steps, values, color, window, legend_label)
    _finalize_ts_plot(-1, "rew", "Mean episode reward", _BASE_PLOT_DIR)


def plot_ts_length() -> None:
    _figsize()
    for i, (label, subdir) in enumerate(_RUNS.items()):
        steps, values = _get_scalar(_BASE_LOG_DIR / subdir, _LENGTH_TAG)
        color = _ARCH_COLOR[label]
        window = max(len(values) // 50, 1)
        legend_label = "_nolegend_" if "ext" in subdir else label
        _plot_ts_series(steps, values, color, window, legend_label)
    _finalize_ts_plot(None, "len", "Mean episode length", _BASE_PLOT_DIR)

def plot_ts_explained_variance() -> None:
    _figsize()
    for i, (label, subdir) in enumerate(_RUNS.items()):
        steps, values = _get_scalar(_BASE_LOG_DIR / subdir, _EV_TAG)
        color = _ARCH_COLOR[label]
        window = max(len(values) // 50, 1)
        _plot_ts_series(steps, values, color, window, label)
    _finalize_ts_plot(0, "ev", "Explained variance", _BASE_PLOT_DIR)


def plot_ts_entropy_loss() -> None:
    _figsize()
    for i, (label, subdir) in enumerate(_RUNS.items()):
        steps, values = _get_scalar(_BASE_LOG_DIR / subdir, _ENTROPY_TAG)
        color = _ARCH_COLOR[label]
        window = max(len(values) // 50, 1)
        _plot_ts_series(steps, values, color, window, label)
    _finalize_ts_plot(-1.1, "ent", "Entropy loss", _BASE_PLOT_DIR)

def _get_scalar(run_dir: Path, tag: str):
    """Read one scalar tag from an event-file directory.
    Returns (steps, values) as pandas Series, sorted by step."""
    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    available = acc.Tags().get("scalars", [])
    if tag not in available:
        raise ValueError(
            f"Tag '{tag}' not found in {run_dir}.\n"
            f"Available tags: {', '.join(sorted(available))}"
        )
    events = acc.Scalars(tag)
    steps = pd.Series([e.step for e in events])
    values = pd.Series([e.value for e in events])
    order = steps.argsort()
    return steps.iloc[order].reset_index(drop=True), values.iloc[order].reset_index(drop=True)


def _plot_ts_series(steps, values, color, window: int, label: str) -> None:
    if window <= 1:
        smoothed = values.fillna(0)
    else:
        smoothed = values.fillna(0).rolling(window=window).mean()

    plt.plot(steps, values, alpha=0.2, color=color)
    plt.plot(steps, smoothed, label=label, color=color)


def _finalize_ts_plot(bottom_limit: float| None, plot_id: str, ylabel: str, plots_dir: Path) -> None:
    plt.ylim(bottom=bottom_limit)
    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} at sparse training")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"spatial_{plot_id}.pdf")  # PDF for the thesis
    plt.close()


# ================= #
#      Helpers      #
# ================= #


def _figsize() -> None:
    plt.figure(figsize=(6, 4))


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



if __name__ == "__main__":
    _BASE_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plot_ts_reward()
    plot_ts_length()
    # plot_ts_entropy_loss()
    # plot_ts_explained_variance()
