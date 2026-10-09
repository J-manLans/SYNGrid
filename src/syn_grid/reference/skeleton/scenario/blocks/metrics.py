"""
What a scenario reports at the end of an episode.

This is the scenario's side of the logger. A scenario lists its metrics; the
logger writes whatever names it is given.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from syn_grid.core.grid_world import GridWorld


@dataclass(frozen=True)
class Metric:
    """One named number a scenario reports.

    Attributes:
        name: the metric's name, known before the first episode; the logger's
            column.
        read: returns the value from the world it is handed, at episode end.
    """

    name: str
    read: Callable[[GridWorld], float]
