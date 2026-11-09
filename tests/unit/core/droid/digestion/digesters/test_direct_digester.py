"""What a direct digester does with an eaten orb: pays what it is worth, counts it, and owes nothing
after."""

import pytest

from syn_grid.core.droid.digestion.digesters.direct_digester import (
    NegDigester,
    PosDigester,
)
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.direct.fixed_reward_orb import FixedRewardOrb
from syn_grid.core.orbs.orb_meta import DirectKind

# ================= #
#     Constants     #
# ================= #


NEG_REWARD = -1.0


# Not 1.0, so a digester that paid 1 whatever the orb would be caught
POS_REWARD = 2.0


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def negative_orb() -> BaseOrb:
    return FixedRewardOrb(DirectKind.NEGATIVE, NEG_REWARD, cool_down=0, life_span=1)


@pytest.fixture
def positive_orb() -> BaseOrb:
    return FixedRewardOrb(DirectKind.POSITIVE, POS_REWARD, cool_down=0, life_span=1)


# ================= #
#       Tests       #
# ================= #


class TestDigest:
    """What eating an orb pays, and how it is counted."""

    def test_an_orb_pays_what_it_is_worth(self, positive_orb: BaseOrb):
        assert PosDigester().digest(positive_orb) == POS_REWARD

    def test_each_orb_eaten_counts_once(self, negative_orb: BaseOrb):
        digester = NegDigester()

        paid = [digester.digest(negative_orb), digester.digest(negative_orb)]

        assert paid == [NEG_REWARD, NEG_REWARD]
        assert digester.eaten == 2


class TestUndigested:
    """What the digester is owed when the episode ends."""

    def test_nothing_is_owed_even_after_eating(self, negative_orb: BaseOrb):
        digester = NegDigester()
        digester.digest(negative_orb)

        assert digester.undigested() == 0.0


class TestReset:
    """A new episode."""

    def test_reset_clears_the_count(self, negative_orb: BaseOrb):
        digester = NegDigester()
        digester.digest(negative_orb)

        digester.reset()

        assert digester.eaten == 0
