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
_REWARD_TAG = "rollout/ep_rew_mean"  # <-- replace with your actual tag
_LENGTH_TAG = "rollout/ep_len_mean"  # <-- replace with your actual tag
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
    return steps.iloc[order].reset_index(drop=True), values.iloc[order].reset_index(
        drop=True
    )


def _plot_ts_series(steps, values, color, window: int, label: str) -> None:
    if window <= 1:
        smoothed = values.fillna(0)
    else:
        smoothed = values.fillna(0).rolling(window=window).mean()

    plt.plot(steps, values, alpha=0.2, color=color)
    plt.plot(steps, smoothed, label=label, color=color)


def _finalize_ts_plot(
    bottom_limit: float | None, plot_id: str, ylabel: str, plots_dir: Path
) -> None:
    plt.ylim(bottom=bottom_limit)
    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} in the spatial scenario")
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


def time_line(plots_dir: Path):
    """Final roadmap Gantt chart for SYNGrid thesis — Appendix A companion figure.
    Data transcribed from GitHub Projects roadmap screenshot (verify dates against repo).
    Regenerate: python roadmap_gantt.py
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import date

    # (label, issue_no, start, end)
    items = [
        ("Write ex-job proposal", 56, date(2026, 3, 21), date(2026, 3, 27)),
        (
            "See if the TierBase resource can be discarded",
            48,
            date(2026, 3, 21),
            date(2026, 3, 31),
        ),
        (
            "Move _chained_tiers list into the agent for more unified handling",
            47,
            date(2026, 3, 21),
            date(2026, 3, 31),
        ),
        (
            "Test if learning stabilizes earlier with move penalty as a reward",
            46,
            date(2026, 3, 21),
            date(2026, 3, 31),
        ),
        (
            "Change how world boundary works in BaseResource",
            40,
            date(2026, 3, 21),
            date(2026, 3, 31),
        ),
        (
            "Move the human control loop into the GridWorld class",
            45,
            date(2026, 3, 21),
            date(2026, 3, 31),
        ),
        ("Fix the combo based reward signal", 42, date(2026, 3, 21), date(2026, 3, 31)),
        (
            "Introduction, background and related work",
            63,
            date(2026, 3, 31),
            date(2026, 4, 5),
        ),
        ("Implement configurable runs file", 23, date(2026, 3, 31), date(2026, 4, 10)),
        (
            "Abstract base class for observation spaces + concrete observations",
            32,
            date(2026, 4, 11),
            date(2026, 4, 20),
        ),
        (
            "Find a neat solution for plug and play for different agents",
            33,
            date(2026, 4, 21),
            date(2026, 4, 22),
        ),
        (
            "Implement \u201cplug and play\u201d ability",
            34,
            date(2026, 4, 23),
            date(2026, 5, 3),
        ),
        ("Method and preliminary results", 64, date(2026, 4, 6), date(2026, 5, 3)),
        (
            "Midterm check: abstract, intro, background, related work, method",
            65,
            date(2026, 5, 3),
            date(2026, 5, 4),
        ),
        (
            "Ablation studies for fully Markovian observation spaces",
            81,
            date(2026, 5, 3),
            date(2026, 5, 5),
        ),
        ("Max Tier Only reward variant", 88, date(2026, 5, 3), date(2026, 5, 5)),
        ("Add persistent result metrics", 39, date(2026, 5, 4), date(2026, 5, 10)),
        ("Add performance metrics logging", 57, date(2026, 5, 4), date(2026, 5, 10)),
        (
            "Converter and hazard orbs (if time allows)",
            49,
            date(2026, 5, 11),
            date(2026, 5, 21),
        ),
        ("Result, discussion", 66, date(2026, 5, 4), date(2026, 5, 24)),
        ("Completeness check", 67, date(2026, 5, 24), date(2026, 5, 25)),
        ("Video submission", 68, date(2026, 5, 31), date(2026, 6, 1)),
        ("Discussion day with examiners", 69, date(2026, 6, 2), date(2026, 6, 3)),
        ("Finalize the ex-job report", 54, date(2026, 5, 1), date(2026, 6, 10)),
        (
            "Set up CI workflow to run tests on push and pull requests",
            35,
            date(2026, 5, 4),
            date(2026, 6, 10),
        ),
        ("Add docstrings where needed", 37, date(2026, 6, 1), date(2026, 6, 10)),
        ("Write contributing file", 53, date(2026, 6, 1), date(2026, 6, 10)),
        ("Build upon the readme file", 55, date(2026, 6, 1), date(2026, 6, 10)),
        ("Final submission", 70, date(2026, 6, 11), date(2026, 6, 12)),
    ]

    DEADLINE = date(2026, 6, 12)

    BAR = "#ffdf61"
    EDGE = "#e6b800"
    TEXT = "#000000"
    NUMC = "#888888"
    GRID = "#d9d9d9"

    fig, ax = plt.subplots(figsize=(9.2, 6.4))

    ys = range(len(items))

    # ---- label + gray number as one annotation ----
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for y, (label, num, start, end) in zip(ys, items):
        ax.barh(
            y,
            (end - start).days,
            left=mdates.date2num(start),
            height=0.62,
            color=BAR,
            edgecolor=EDGE,
            linewidth=0.8,
            zorder=3,
        )
        right_side = end <= date(2026, 5, 20)
        if right_side:
            x, ha = mdates.date2num(end) + 1.0, "left"
        else:
            x, ha = mdates.date2num(start) - 1.0, "right"
        near_deadline = end >= date(2026, 6, 8)
        if near_deadline:
            ax.text(
                mdates.date2num(start) - 1.0,
                y,
                f"#{num}  {label}",
                va="center",
                ha="right",
                fontsize=7.2,
                color=TEXT,
                zorder=4,
            )
            continue
        ax.text(
            x, y, f"{label}", va="center", ha=ha, fontsize=7.2, color=TEXT, zorder=4
        )
        # number rides at the opposite end of the bar, inside-adjacent
        nx = mdates.date2num(start) - 1.0 if right_side else mdates.date2num(end) + 1.0
        nha = "right" if right_side else "left"
        ax.text(
            nx, y, f"#{num}", va="center", ha=nha, fontsize=6.6, color=NUMC, zorder=4
        )

    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_ylim(len(items) - 0.4, -0.6)

    # x axis: weekly minor grid, monthly major labels
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%B %Y"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.grid(which="major", axis="x", color="#595959", linewidth=0.9, zorder=0)
    ax.tick_params(axis="x", which="major", labelsize=8.5, length=0, pad=6)
    ax.tick_params(axis="x", which="minor", length=0)

    ax.set_xlim(mdates.date2num(date(2026, 3, 20)), mdates.date2num(date(2026, 6, 15)))

    # deadline marker
    ax.axvline(mdates.date2num(DEADLINE), color="#00ccff", linewidth=1.1, zorder=2)
    ax.text(
        mdates.date2num(DEADLINE) - 1.0,
        -0.55,
        "final deadline (June 12) ",
        fontsize=7.5,
        color="#595959",
        ha="right",
        va="bottom",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(plots_dir / "final_roadmap.pdf", bbox_inches="tight")
    fig.savefig(plots_dir / "final_roadmap_preview.png", dpi=150, bbox_inches="tight")
    print("done")


if __name__ == "__main__":
    _BASE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    # time_line(_BASE_PLOT_DIR)
    plot_ts_reward()
    plot_ts_length()
    # plot_ts_entropy_loss()
    # plot_ts_explained_variance()
