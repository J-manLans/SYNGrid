from enum import Enum
from typing import Final

STATS_KEY: Final[str] = "episode"  # Mirrors RecordEpisodeStatistics variable
SYN_STATS_KEY: Final[str] = "syn_grid_episode"


class LogKey(str, Enum):
    """The episode statistics ``RecordEpisodeStatistics`` records. A scenario's own columns
    are the names of its metrics."""

    REWARD = "r"
    LENGTH = "l"
    TIME = "t"
