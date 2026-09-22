from enum import Enum
from typing import Final

STATS_KEY: Final[str] = "episode"  # Mirrors RecordEpisodeStatistics variable
SYN_STATS_KEY: Final[str] = "syn_grid_episode"


class LogKey(str, Enum):
    REWARD = "r"
    LENGTH = "l"
    TIME = "t"
    CHAINS_BROKEN = "chains_broken"
    CHAIN_PROGRESSED = "chain_progressed"
    CHAINS_COMPLETED = "chains_completed"
