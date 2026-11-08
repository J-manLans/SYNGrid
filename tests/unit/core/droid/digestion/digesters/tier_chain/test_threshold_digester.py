"""What threshold scoring pays: a chain holds its worth while it grows, and pays it when it ends."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from syn_grid.core.droid.digestion.digesters.tier_chain.step_wise_digester import (
    StepWiseDigester,
)
from syn_grid.core.droid.digestion.digesters.tier_chain.threshold_digester import (
    ThresholdDigester,
)
from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.scenario.blocks.populations.tier_chain_population import (
    TierChainPopulation,
)

# ================= #
#     Constants     #
# ================= #


MAX_TIER = 3


# Tier orbs are worth 10, 20 and 30
BASE_REWARD = 10.0


GROWTH_FACTOR = 1.0


CHAIN_BREAK_PENALTY = -0.5


# ================= #
#      Helpers      #
# ================= #


def _eat(digester: TierDigester, orbs: list[BaseOrb], *tiers: int) -> list[float]:
    """Digest a tier orb of each tier, in the order given; returns what each paid."""

    return [
        digester.digest(next(o for o in orbs if o.META.TIER == tier)) for tier in tiers
    ]


def _tier_orbs() -> list[BaseOrb]:
    """A three-tier chain worth 10, 20 and 30."""

    return TierChainPopulation(
        max_tier=MAX_TIER,
        base_reward=BASE_REWARD,
        growth_factor=GROWTH_FACTOR,
        cool_down=None,
        life_span=None,
    ).create()


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def orbs() -> list[BaseOrb]:
    return _tier_orbs()


@pytest.fixture
def digester() -> ThresholdDigester:
    return ThresholdDigester(max_tier=MAX_TIER, chain_break_penalty=CHAIN_BREAK_PENALTY)


# ================= #
#       Tests       #
# ================= #


class TestProgress:
    """The next tier, not the last."""

    def test_an_orb_that_grows_the_chain_pays_nothing_and_is_held(
        self, digester: ThresholdDigester, orbs: list[BaseOrb]
    ):
        paid = _eat(digester, orbs, 1, 2)

        assert paid == [0.0, 0.0]
        assert digester.undigested() == 10.0 + 20.0


class TestCompletion:
    """The next tier, and the last."""

    def test_the_orb_that_completes_the_chain_pays_its_full_worth(
        self, digester: ThresholdDigester, orbs: list[BaseOrb]
    ):
        paid = _eat(digester, orbs, 1, 2, 3)

        assert paid[-1] == 10.0 + 20.0 + 30.0
        assert digester.undigested() == 0.0


class TestBreak:
    """A tier orb eaten out of order."""

    def test_a_tier_1_that_breaks_a_chain_pays_the_penalty_and_what_the_chain_held(
        self, digester: ThresholdDigester, orbs: list[BaseOrb]
    ):
        # The tier 1 that breaks is not held, so nothing is owed afterwards
        paid = _eat(digester, orbs, 1, 2, 1)

        assert paid[-1] == CHAIN_BREAK_PENALTY + 10.0 + 20.0
        assert digester.undigested() == 0.0


class TestSameTotalAsStepWise:
    """Threshold pays what step-wise pays, only later."""

    @given(tiers=st.lists(st.integers(min_value=1, max_value=MAX_TIER), max_size=12))
    def test_any_sequence_pays_the_same_total_once_what_is_held_is_paid(
        self, tiers: list[int]
    ):
        orbs = _tier_orbs()
        threshold = ThresholdDigester(MAX_TIER, CHAIN_BREAK_PENALTY)
        step_wise = StepWiseDigester(MAX_TIER, CHAIN_BREAK_PENALTY)

        threshold_total = sum(_eat(threshold, orbs, *tiers)) + threshold.undigested()
        step_wise_total = sum(_eat(step_wise, orbs, *tiers))

        assert threshold_total == pytest.approx(step_wise_total)
