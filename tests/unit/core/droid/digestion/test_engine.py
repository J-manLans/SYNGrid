"""How the digestion engine routes an eaten orb to the digester of its kind, resets and sums its
digesters, and finds one for a reader."""

import pytest

from syn_grid.core.droid.digestion.digesters.direct_digester import (
    DirectDigester,
    NegDigester,
    PosDigester,
)
from syn_grid.core.droid.digestion.digesters.tier_chain.threshold_digester import (
    ThresholdDigester,
)
from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.core.droid.digestion.engine import DigestionEngine
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.direct.fixed_reward_orb import FixedRewardOrb
from syn_grid.core.orbs.orb_meta import DirectKind
from syn_grid.scenario.blocks.populations.tier_chain_population import (
    TierChainPopulation,
)

# ================= #
#     Constants     #
# ================= #


MAX_TIER = 3


CHAIN_BREAK_PENALTY = -0.5


NEG_REWARD = -1.0


# Not 1.0, so a digester that paid 1 whatever the orb would be caught
POS_REWARD = 2.0


# What the stand-in holds unpaid
HELD = 5.0


# ================= #
#    Stand-ins      #
# ================= #


class _Holding:
    """A positive orb digester that holds a fixed amount it has not paid, so the engine's sum has
    two parts to add."""

    KIND = DirectKind.POSITIVE

    def reset(self) -> None:
        pass

    def digest(self, orb: BaseOrb) -> float:
        return 0.0

    def undigested(self) -> float:
        return HELD


# ================= #
#      Helpers      #
# ================= #


def _tier(orbs: list[BaseOrb], tier: int) -> BaseOrb:
    return next(o for o in orbs if o.META.TIER == tier)


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def tier_orbs() -> list[BaseOrb]:
    """A three-tier chain worth 10, 20 and 30."""

    return TierChainPopulation(
        max_tier=MAX_TIER,
        base_reward=10.0,
        growth_factor=1.0,
        cool_down=None,
        life_span=None,
    ).create()


@pytest.fixture
def neg_orb() -> BaseOrb:
    return FixedRewardOrb(DirectKind.NEGATIVE, NEG_REWARD, cool_down=0, life_span=1)


@pytest.fixture
def pos_orb() -> BaseOrb:
    return FixedRewardOrb(DirectKind.POSITIVE, POS_REWARD, cool_down=0, life_span=1)


@pytest.fixture
def threshold_digester() -> ThresholdDigester:
    return ThresholdDigester(max_tier=MAX_TIER, chain_break_penalty=CHAIN_BREAK_PENALTY)


@pytest.fixture
def neg_digester() -> NegDigester:
    return NegDigester()


@pytest.fixture
def pos_digester() -> PosDigester:
    return PosDigester()


@pytest.fixture
def engine(
    threshold_digester: ThresholdDigester,
    neg_digester: NegDigester,
    pos_digester: PosDigester,
) -> DigestionEngine:
    """An engine with a tier digester and both direct digesters."""

    return DigestionEngine([threshold_digester, neg_digester, pos_digester])


# ================= #
#       Tests       #
# ================= #


class TestBuilding:
    """The digesters an engine is made with."""

    def test_two_digesters_for_one_kind_are_rejected(self):
        with pytest.raises(ValueError, match="Two digesters"):
            DigestionEngine([NegDigester(), NegDigester()])


class TestDigest:
    """Where an eaten orb goes."""

    def test_an_orb_goes_to_the_digester_of_its_kind_and_no_other(
        self,
        engine: DigestionEngine,
        threshold_digester: ThresholdDigester,
        neg_digester: NegDigester,
        pos_digester: PosDigester,
        pos_orb: BaseOrb,
    ):
        paid = engine.digest(pos_orb)

        assert paid == POS_REWARD
        assert pos_digester.eaten == 1
        assert neg_digester.eaten == 0
        assert (
            threshold_digester.chain_progresses + threshold_digester.chains_broken == 0
        )

    def test_a_direct_orb_eaten_mid_chain_leaves_the_chain_as_it_was(
        self,
        engine: DigestionEngine,
        threshold_digester: ThresholdDigester,
        tier_orbs: list[BaseOrb],
        neg_orb: BaseOrb,
    ):
        engine.digest(_tier(tier_orbs, 1))
        engine.digest(_tier(tier_orbs, 2))
        engine.digest(neg_orb)

        engine.digest(_tier(tier_orbs, 3))

        assert threshold_digester.chains_completed == 1
        assert threshold_digester.chains_broken == 0

    def test_an_orb_whose_kind_has_no_digester_is_rejected(
        self, threshold_digester: ThresholdDigester, neg_orb: BaseOrb
    ):
        engine = DigestionEngine([threshold_digester])

        with pytest.raises(KeyError, match="NEGATIVE"):
            engine.digest(neg_orb)


class TestReset:
    """A new episode."""

    def test_reset_clears_every_digester(
        self,
        engine: DigestionEngine,
        threshold_digester: ThresholdDigester,
        neg_digester: NegDigester,
        pos_digester: PosDigester,
        tier_orbs: list[BaseOrb],
        neg_orb: BaseOrb,
        pos_orb: BaseOrb,
    ):
        for orb in (_tier(tier_orbs, 1), neg_orb, pos_orb):
            engine.digest(orb)

        engine.reset()

        assert threshold_digester.chain_tier == 0
        assert neg_digester.eaten == 0
        assert pos_digester.eaten == 0


class TestUndigested:
    """What the digesters are owed together when the episode ends."""

    def test_it_is_the_sum_of_what_each_digester_is_owed(
        self, threshold_digester: ThresholdDigester, tier_orbs: list[BaseOrb]
    ):
        engine = DigestionEngine([threshold_digester, _Holding()])
        engine.digest(_tier(tier_orbs, 1))
        engine.digest(_tier(tier_orbs, 2))

        assert engine.undigested() == 10.0 + 20.0 + HELD


class TestGet:
    """How a reader finds a digester."""

    def test_the_exact_class_finds_its_digester(
        self, engine: DigestionEngine, pos_digester: PosDigester
    ):
        assert engine.get(PosDigester) is pos_digester

    def test_a_base_class_finds_the_one_digester_of_it(
        self, engine: DigestionEngine, threshold_digester: ThresholdDigester
    ):
        # A reader need not know which scoring mode the scenario uses
        assert engine.get(TierDigester) is threshold_digester

    def test_a_class_with_no_digester_is_rejected(self, neg_digester: NegDigester):
        engine = DigestionEngine([neg_digester])

        with pytest.raises(KeyError, match="PosDigester"):
            engine.get(PosDigester)

    def test_a_class_that_more_than_one_digester_is_of_is_rejected(
        self, engine: DigestionEngine
    ):
        # Both direct digesters are direct digesters: the engine refuses to pick one
        with pytest.raises(LookupError, match="DirectDigester"):
            engine.get(DirectDigester)
