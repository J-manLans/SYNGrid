from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.synergy.tier_orb import TierOrb
from syn_grid.core.droid.digestion_engine import DigestionEngine

from tests.utils.config_helpers import get_test_config

import pytest


class TestDigestionEngine:
    _MAX_TIER = 10
    TierOrb.max_tier = _MAX_TIER

    # ================= #
    #      Helpers      #
    # ================= #

    @staticmethod
    def _tier_params(max_tier=_MAX_TIER) -> list[TierOrb]:
        tierOrbs = [
            (TierOrb(t, get_test_config().world.tier_orb_conf))
            for t in range(1, max_tier + 1)
        ]

        for t in tierOrbs:
            t.reset()

        return tierOrbs

    # ================= #
    #      Fixtures     #
    # ================= #

    @pytest.fixture
    def digestion_engine(self) -> DigestionEngine:
        BaseOrb.set_life_span(5, 5)
        d = DigestionEngine(-1.0, 1.0)
        d.reset()
        return d

    @pytest.fixture
    def parameterize_reset(self):
        # adjust max tier so we don't tap out
        TierOrb.max_tier = self._MAX_TIER + 1

    # ================= #
    #       Tests       #
    # ================= #

    # === Step wise scoring === #

    @pytest.mark.parametrize("orb", _tier_params())
    def test_in_order_consumption_gives_reward_and_builds_chain(
        self,
        parameterize_reset,
        digestion_engine: DigestionEngine,
        orb: TierOrb,
    ):
        # prep the "chain" by giving it a tier value 1 lower than current orb
        digestion_engine.chained_tiers = orb.META.TIER - 1

        assert digestion_engine.digest(orb) == orb.REWARD
        assert digestion_engine.chained_tiers == orb.META.TIER

    def test_max_tier_consumption_rewards_and_resets_chain(
        self, digestion_engine: DigestionEngine
    ):
        max_orb = TierOrb(self._MAX_TIER, get_test_config().world.tier_orb_conf)
        max_orb.max_tier = self._MAX_TIER

        # prep the "chain" by giving it a tier value 1 lower than max_orb
        digestion_engine.chained_tiers = max_orb.META.TIER - 1

        assert digestion_engine.digest(max_orb) == max_orb.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_out_of_order_consumption_returns_zero_and_resets_chain(
        self, digestion_engine: DigestionEngine
    ):
        orb = TierOrb(self._MAX_TIER - 2, get_test_config().world.tier_orb_conf)

        # force out-of-order consumption for orb
        digestion_engine.chained_tiers = self._MAX_TIER - 1

        assert digestion_engine.digest(orb) == -1.0
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_base_tier_consumption_rewards_and_starts_chain(
        self, digestion_engine: DigestionEngine
    ):
        base_orb = TierOrb(1, get_test_config().world.tier_orb_conf)

        # force out-of-order consumption for base tier
        digestion_engine.chained_tiers = self._MAX_TIER

        assert digestion_engine.digest(base_orb) == base_orb.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._BASE_TIER

    # === Delayed scoring === #

    @pytest.mark.parametrize("orb", _tier_params())
    def test_in_order_consumption_return_zero_and_builds_chain(
        self,
        parameterize_reset,
        digestion_engine: DigestionEngine,
        orb: TierOrb,
    ):
        # set correct scoring type
        orb.step_wise_scoring = False
        orb.threshold_scoring = True

        # prep the "chain" by giving it a tier value 1 lower than current orb
        digestion_engine.chained_tiers = orb.META.TIER - 1

        assert digestion_engine.digest(orb) == 0
        assert digestion_engine.chained_tiers == orb.META.TIER

    def test_threshold_scoring_max_tier_consumption_rewards_and_resets_chain(
        self, digestion_engine: DigestionEngine
    ):
        max_orb = TierOrb(self._MAX_TIER, get_test_config().world.tier_orb_conf)
        # set correct scoring type and max tier
        max_orb.step_wise_scoring = False
        max_orb.threshold_scoring = True
        max_orb.max_tier = self._MAX_TIER

        # prep the "chain" by giving it a tier value 1 lower than max_orb
        digestion_engine.chained_tiers = max_orb.META.TIER - 1

        assert digestion_engine.digest(max_orb) == max_orb.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_out_of_order_consumption_rewards_and_resets_chain(
        self, digestion_engine: DigestionEngine
    ):
        out_of_order_orb = TierOrb(
            self._MAX_TIER - 3, get_test_config().world.tier_orb_conf
        )
        in_order_orb = TierOrb(
            self._MAX_TIER - 2, get_test_config().world.tier_orb_conf
        )
        # set correct scoring type
        out_of_order_orb.step_wise_scoring = False
        out_of_order_orb.threshold_scoring = True
        in_order_orb.step_wise_scoring = False
        in_order_orb.threshold_scoring = True

        # force out-of-order consumption for out_of_order_orb and prep the reward
        digestion_engine.chained_tiers = in_order_orb.META.TIER
        digestion_engine._pending_reward = in_order_orb.REWARD

        assert digestion_engine.digest(out_of_order_orb) == in_order_orb.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_base_tier_consumption_returns_pending_reward_and_starts_chain(
        self, digestion_engine: DigestionEngine
    ):
        base_orb = TierOrb(1, get_test_config().world.tier_orb_conf)
        in_order_orb = TierOrb(self._MAX_TIER, get_test_config().world.tier_orb_conf)
        # set correct scoring type
        base_orb.step_wise_scoring = False
        base_orb.threshold_scoring = True
        in_order_orb.step_wise_scoring = False
        in_order_orb.threshold_scoring = True

        # force out-of-order consumption for base_orb and prep the reward
        digestion_engine.chained_tiers = in_order_orb.META.TIER
        digestion_engine._pending_reward = in_order_orb.REWARD

        assert digestion_engine.digest(base_orb) == in_order_orb.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._BASE_TIER
