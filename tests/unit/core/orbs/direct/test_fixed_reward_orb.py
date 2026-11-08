"""What a fixed reward orb is: a direct orb of the kind it is given, paying the reward it is given."""

from syn_grid.core.orbs.direct.fixed_reward_orb import FixedRewardOrb
from syn_grid.core.orbs.orb_meta import DirectKind

# ================= #
#       Tests       #
# ================= #


class TestWhatItIs:
    """The kind and the reward it is made with."""

    def test_it_carries_the_kind_and_the_reward_it_is_given_and_no_tier(self):
        orb = FixedRewardOrb(DirectKind.POSITIVE, 2.0, cool_down=0, life_span=1)

        assert orb.META.KIND == DirectKind.POSITIVE
        assert orb.REWARD == 2.0
        assert orb.META.TIER is None


class TestIdentity:
    """What the agent tells orbs apart by."""

    def test_a_negative_and_a_positive_orb_have_different_identities(self):
        negative = FixedRewardOrb(DirectKind.NEGATIVE, -1.0, cool_down=0, life_span=1)
        positive = FixedRewardOrb(DirectKind.POSITIVE, 1.0, cool_down=0, life_span=1)

        assert negative.META.IDENTITY != positive.META.IDENTITY
