"""What a tier digester recipe makes: a new digester of its scoring mode, with its max tier and penalty."""

from syn_grid.core.droid.digestion.digesters.tier_chain.step_wise_digester import (
    StepWiseDigester,
)
from syn_grid.core.orbs.synergy.chain.tier_orb import TierOrb
from syn_grid.scenario.blocks.recipes.tier_digester_recipe import TierDigesterRecipe

# ================= #
#     Constants     #
# ================= #


MAX_TIER = 2
CHAIN_BREAK_PENALTY = -3.0


# ================= #
#      Helpers      #
# ================= #


def _recipe() -> TierDigesterRecipe:
    return TierDigesterRecipe(
        scoring_mode=StepWiseDigester,
        max_tier=MAX_TIER,
        chain_break_penalty=CHAIN_BREAK_PENALTY,
    )


def _tier_orb(tier: int) -> TierOrb:
    return TierOrb(tier, MAX_TIER, 1.0, 1.0, cool_down=None, life_span=None)


# ================= #
#       Tests       #
# ================= #


class TestCall:
    """The digester each call makes."""

    def test_it_is_the_scoring_mode_completing_at_the_max_tier_and_paying_the_penalty(
        self,
    ):
        recipe = _recipe()
        completed, broken = recipe(), recipe()

        completed.digest(_tier_orb(1))
        completed.digest(_tier_orb(2))
        # A tier 2 eaten first breaks the chain, which step-wise pays as the penalty alone
        paid = broken.digest(_tier_orb(2))

        assert isinstance(completed, StepWiseDigester)
        assert completed.chains_completed == 1
        assert paid == CHAIN_BREAK_PENALTY

    def test_each_call_makes_a_new_digester_that_shares_no_chain(self):
        recipe = _recipe()
        first, second = recipe(), recipe()

        first.digest(_tier_orb(1))

        assert first is not second
        assert second.chain_tier == 0
