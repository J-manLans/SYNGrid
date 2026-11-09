"""How an orb moves on and off the field: its lifespan on it, its cool-down off it, and None for never."""

from collections.abc import Callable

import pytest

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.direct.fixed_reward_orb import FixedRewardOrb
from syn_grid.core.orbs.orb_meta import DirectKind
from syn_grid.core.utils.timer import Timer

# ================= #
#     Constants     #
# ================= #


COOL_DOWN = 2
LIFE_SPAN = 3
CELL = [1, 2]

type Tick = Callable[[Timer, int], Timer]


# ================= #
#      Helpers      #
# ================= #


def _orb(
    cool_down: int | None = COOL_DOWN, life_span: int | None = LIFE_SPAN
) -> BaseOrb:
    """An orb with this cool-down and lifespan. A fixed reward orb is the plainest one."""

    return FixedRewardOrb(DirectKind.NEGATIVE, -1.0, cool_down, life_span)


def _spawned(orb: BaseOrb) -> BaseOrb:
    orb.spawn(list(CELL))

    return orb


def _leave(orb: BaseOrb, how: str) -> None:
    """Take a spawned orb off the field, eaten by the droid or removed by the world."""

    if how == "eaten":
        assert orb.consume() is orb
    else:
        orb.de_spawn()


# ================= #
#       Tests       #
# ================= #


class TestNewOrb:
    """An orb before it first reaches the field."""

    def test_it_is_off_the_field_and_ready_to_spawn(self):
        orb = _orb()

        assert not orb.is_active
        assert orb.is_ready()
        assert not orb.has_expired()


class TestOnTheField:
    """A spawned orb, and its lifespan."""

    def test_it_sits_in_its_cell_and_is_not_ready(self):
        orb = _spawned(_orb())

        assert orb.is_active
        assert orb.position == CELL
        assert not orb.is_ready()

    def test_it_expires_on_the_last_step_of_its_lifespan_and_not_before(
        self, tick: Tick
    ):
        orb = _spawned(_orb())

        tick(orb.TIMER, LIFE_SPAN - 1)
        assert not orb.has_expired()
        tick(orb.TIMER, 1)
        assert orb.has_expired()
        # Expired is not ready: the orb is still on the field until the world removes it
        assert not orb.is_ready()

    def test_with_no_lifespan_it_never_expires(self, tick: Tick):
        orb = _spawned(_orb(life_span=None))

        tick(orb.TIMER, 100)

        assert not orb.has_expired()


class TestOffTheField:
    """An orb eaten by the droid or removed by the world, and its cool-down."""

    @pytest.mark.parametrize("how", ["eaten", "removed"])
    def test_it_is_ready_on_the_last_step_of_its_cool_down_and_not_before(
        self, how: str, tick: Tick
    ):
        orb = _spawned(_orb())

        _leave(orb, how)

        assert not orb.is_active
        tick(orb.TIMER, COOL_DOWN - 1)
        assert not orb.is_ready()
        tick(orb.TIMER, 1)
        assert orb.is_ready()
        assert not orb.has_expired()

    @pytest.mark.parametrize("how", ["eaten", "removed"])
    def test_with_no_cool_down_it_does_not_come_back(self, how: str, tick: Tick):
        orb = _spawned(_orb(cool_down=None))

        _leave(orb, how)

        tick(orb.TIMER, 100)

        assert not orb.is_ready()

    def test_leaving_partway_through_its_lifespan_starts_the_cool_down_in_full(
        self, tick: Tick
    ):
        orb = _spawned(_orb(cool_down=LIFE_SPAN, life_span=LIFE_SPAN))
        tick(orb.TIMER, 1)

        _leave(orb, "eaten")

        tick(orb.TIMER, LIFE_SPAN - 1)
        assert not orb.is_ready()
        tick(orb.TIMER, 1)
        assert orb.is_ready()


class TestReset:
    """A reset, at the start of an episode."""

    def test_it_takes_an_orb_off_the_field_and_makes_it_ready(self):
        on_the_field = _spawned(_orb())
        not_coming_back = _spawned(_orb(cool_down=None))
        _leave(not_coming_back, "eaten")

        on_the_field.reset()
        not_coming_back.reset()

        assert not on_the_field.is_active
        assert on_the_field.is_ready()
        assert not_coming_back.is_ready()
