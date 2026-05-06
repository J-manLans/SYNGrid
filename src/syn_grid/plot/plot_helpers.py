import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum

# ================= #
#      Colors       #
# ================= #


class Color(str, Enum):
    BLUE = "steelblue"
    RED = "coral"
    GREEN = "mediumseagreen"
    PURPLE = "mediumpurple"
    ORANGE = "darkorange"
    TEAL = "teal"
    PINK = "hotpink"
    GREY = "slategrey"


# ================= #
#     Constants     #
# ================= #


_TRAIN = "train"
_EVAL = "eval"

# ================= #
#      Helpers      #
# ================= #


def _setup_plotting(name: str) -> str:
    plt.figure(figsize=(15, 8))
    return name


def _plot_series(data, color, window: int, name: str) -> int:
    plt.plot(data, alpha=0.2, color=color)
    plt.plot(data.fillna(0).rolling(window=window).mean(), label=name, color=color)
    return 1


def _finalize_plot(name: str, plots_dir: Path):
    plt.xlabel("Episode")
    plt.ylabel(name)
    plt.title(f"{name} per episode")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=999)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{name}.png")
    plt.close()
