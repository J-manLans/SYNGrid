from dataclasses import dataclass
from collections.abc import Callable
from syn_grid.core.grid_world import GridWorld


@dataclass(frozen=True)
class HudStat:
    label: str
    read: Callable[[GridWorld], float]

@dataclass(frozen=True)
class HudBar:
    label: str
    maximum: float
    read: Callable[[GridWorld], float]

HudElement = HudStat | HudBar                # a countdown joins when effects exist