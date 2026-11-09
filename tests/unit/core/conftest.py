"""What the core tests share: letting steps of the world pass for a timer."""

from collections.abc import Callable

import pytest

from syn_grid.core.utils.timer import Timer

# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def tick() -> Callable[[Timer, int], Timer]:
    """`tick(timer, steps)` lets `steps` steps of the world pass for a timer, which the world
    ticks once a step, and returns the timer. An orb's timer is its `TIMER`."""

    def _tick(timer: Timer, steps: int) -> Timer:
        for _ in range(steps):
            timer.tick()

        return timer

    return _tick
