from dataclasses import dataclass
from collections.abc import Callable
from syn_grid.core.grid_world import GridWorld


@dataclass(frozen=True)
class Metric:
    name: str                                # static: the logger's column
    read: Callable[[GridWorld], float]       # live: called at episode end