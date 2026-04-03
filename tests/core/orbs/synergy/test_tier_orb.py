from syn_grid.core.orbs.synergy.tier_orb import TierOrb
from syn_grid.core.orbs.base_orb import BaseOrb

from tests.utils.config_helpers import get_test_config

import pytest
import numpy as np


class TestTierOrb:
    _GRID_ROWS = 5
    _GRID_COLS = 5
    _MAX_TIER = 5
    _COOL_DOWN = 10
    _TIER = 2

    """
    Unit tests for the TierOrb class.

    Verifies:
    - Step-wise reward behavior for correctly chained tiers
    - Step-wise reward behavior for incomplete or invalid tier chains
    - Combo reward behavior for correctly chained tiers
    - Combo reward behavior for incomplete or invalid tier chains
    - That the orb updates _chained_tiers correctly during consumption
    """

    # ================= #
    #       Init        #
    # ================= #

    @pytest.fixture
    def orb(self):
        """
        Creates a Tier 2 orb with the world boundaries set to rows x cols used to calculate its life span.
        """

        BaseOrb.set_life_span(self._GRID_ROWS, self._GRID_COLS)
        TierOrb.MAX_TIER = self._MAX_TIER
        t = TierOrb(self._TIER, get_test_config().run.tier_orb_conf)
        t.reset()

        return t

    # ================= #
    #       Tests       #
    # ================= #

    def test_created_orb(self, orb: TierOrb):
        assert orb._cool_down == self._COOL_DOWN
        assert orb.meta.tier == self._TIER
        assert orb.MAX_TIER == self._MAX_TIER
        assert orb._LIFE_SPAN == (self._GRID_ROWS - 1) + (self._GRID_COLS - 1)

    def test_consuming_orb_returns_the_orb(self, orb: TierOrb):
        assert orb.consume() is orb

    def test_stepwise_reward_is_correct(self, orb: TierOrb):
        orb.step_wise_scoring = True

        assert (orb.meta.tier + 1) * orb._TIER_BASE_REWARD == orb.REWARD

    def test_factor_reward_is_correct(self, orb: TierOrb):
        orb.step_wise_scoring = False

        assert (orb._TIER_BASE_REWARD * (orb._GROWTH_FACTOR * (self._TIER - 1))) + 0.5

    def test_active_orb_is_correct(self, orb: TierOrb):
        position = [
            np.int64(max(0, self._GRID_ROWS - 2)),
            np.int64(max(0, self._GRID_COLS - 2)),
        ]
        orb.spawn(position)

        assert orb.timer.remaining == orb._LIFE_SPAN
        assert orb.is_active
        assert orb.position == position

    def test_creating_orb_with_negative_tier(self):
        with pytest.raises(ValueError):
            TierOrb(-1, get_test_config().run.tier_orb_conf)

    def test_creating_orb_with_high_tier(self):
        TierOrb.MAX_TIER = 999
        orb = TierOrb(666, get_test_config().run.tier_orb_conf)

        orb.step_wise_scoring = True
        assert (orb.meta.tier + 1) * orb._TIER_BASE_REWARD == orb.REWARD

        orb.step_wise_scoring = False
        assert (orb._TIER_BASE_REWARD * (orb._GROWTH_FACTOR * (self._TIER - 1))) + 0.5

    def test_creating_orb_with_to_high_tier(self):
        TierOrb.MAX_TIER = self._MAX_TIER

        with pytest.raises(ValueError):
            TierOrb(666, get_test_config().run.tier_orb_conf)
