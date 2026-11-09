"""What a tier orb is: its tier, what that tier is worth, and the timing it is given."""

from collections.abc import Callable

import pytest

from syn_grid.core.orbs.orb_meta import SynergyKind
from syn_grid.core.orbs.synergy.chain.tier_orb import TierOrb
from syn_grid.core.utils.timer import Timer

# ================= #
#     Constants     #
# ================= #


MAX_TIER = 4

type Tick = Callable[[Timer, int], Timer]


# ================= #
#      Helpers      #
# ================= #


def _tier_orb(
    tier: int,
    *,
    base_reward: float = 1.0,
    growth_factor: float = 1.0,
    cool_down: int | None = None,
    life_span: int | None = None,
) -> TierOrb:
    return TierOrb(tier, MAX_TIER, base_reward, growth_factor, cool_down, life_span)


# ================= #
#       Tests       #
# ================= #


class TestWhatItIs:
    """The kind and the tier it is made with, up to the max tier."""

    def test_it_is_a_tier_orb_of_its_tier(self):
        orb = _tier_orb(2)

        assert orb.META.KIND == SynergyKind.TIER
        assert orb.META.TIER == 2

    def test_the_max_tier_is_allowed_and_a_tier_above_it_is_refused(self):
        assert _tier_orb(MAX_TIER).META.TIER == MAX_TIER

        with pytest.raises(ValueError):
            _tier_orb(MAX_TIER + 1)


class TestReward:
    """A tier's worth: the base reward times the tier raised to the growth factor."""

    def test_it_grows_with_the_tier_by_the_growth_factor(self):
        # 10 * 3 ** 2, and 10 * 3 ** 1 when the growth is linear
        assert _tier_orb(3, base_reward=10.0, growth_factor=2.0).REWARD == 90.0
        assert _tier_orb(3, base_reward=10.0, growth_factor=1.0).REWARD == 30.0


class TestTiming:
    """The cool-down and lifespan it is made with, which its base counts."""

    def test_it_expires_after_its_lifespan_and_returns_after_its_cool_down(
        self, tick: Tick
    ):
        orb = _tier_orb(1, cool_down=2, life_span=1)

        orb.spawn([0, 0])
        tick(orb.TIMER, 1)

        assert orb.has_expired()

        orb.de_spawn()
        tick(orb.TIMER, 2)

        assert orb.is_ready()
