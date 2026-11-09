"""What the orb config blocks accept."""

import pytest
from pydantic import ValidationError

from syn_grid.config.models.scenario.orb_models import NegOrbConf, TierOrbConf

# ================= #
#      Helpers      #
# ================= #


def _negative(*, reward: float = -1.0, cool_down: int = 5) -> NegOrbConf:
    return NegOrbConf(enabled=True, reward=reward, cool_down=cool_down)


def _tiers(
    *,
    max_tier: int = 3,
    growth_factor: float = 1.0,
    chain_break_penalty: float = -0.1,
) -> TierOrbConf:
    return TierOrbConf(
        max_tier=max_tier,
        base_reward=1.0,
        growth_factor=growth_factor,
        chain_break_penalty=chain_break_penalty,
    )


# ================= #
#       Tests       #
# ================= #


class TestNegOrbConf:
    """The negative orb's block."""

    def test_it_costs_or_is_free_and_never_pays(self):
        _negative(reward=0.0)

        with pytest.raises(ValidationError, match="reward"):
            _negative(reward=0.01)

    def test_it_can_come_back_at_once_and_never_waits_less(self):
        _negative(cool_down=0)

        with pytest.raises(ValidationError, match="cool_down"):
            _negative(cool_down=-1)


class TestTierOrbConf:
    """The tier orb's block."""

    def test_a_chain_has_at_least_two_tiers(self):
        # One tier would be an orb that just pays, with no order to keep
        _tiers(max_tier=2)

        with pytest.raises(ValidationError, match="max_tier"):
            _tiers(max_tier=1)

    def test_the_worth_grows_with_the_tier(self):
        # At 0 every tier would be worth the same, and below it a higher tier would be worth less
        _tiers(growth_factor=0.01)

        with pytest.raises(ValidationError, match="growth_factor"):
            _tiers(growth_factor=0.0)

    def test_breaking_the_chain_never_pays(self):
        _tiers(chain_break_penalty=0.0)

        with pytest.raises(ValidationError, match="chain_break_penalty"):
            _tiers(chain_break_penalty=0.01)
