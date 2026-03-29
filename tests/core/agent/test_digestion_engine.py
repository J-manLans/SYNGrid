import pytest

from syn_grid.core.resources.base_resource import BaseResource
from syn_grid.core.resources.synergy.tier_resource import TierResource
from syn_grid.core.agent.digestion_engine import DigestionEngine


class TestDigestionEngine:
    _MAX_TIER = 10
    TierResource.MAX_TIER = _MAX_TIER

    # ================= #
    #      Helpers      #
    # ================= #

    @staticmethod
    def _tier_params(max_tier=_MAX_TIER) -> list[TierResource]:
        tierResources = [(TierResource(t)) for t in range(0, max_tier + 1)]

        for t in tierResources:
            t.reset()

        return tierResources

    # ================= #
    #      Fixtures     #
    # ================= #

    @pytest.fixture
    def digestion_engine(self) -> DigestionEngine:
        BaseResource.set_life_span(5, 5)
        d = DigestionEngine()
        d.reset()
        return d

    @pytest.fixture
    def reset_resource(self):
        # restore state
        TierResource.MAX_TIER = self._MAX_TIER
        TierResource.step_wise_scoring_type = True
        TierResource._linear_reward_growth = True

    @pytest.fixture
    def parameterize_reset(self):
        # adjust max tier so we don't tap out
        TierResource.MAX_TIER = self._MAX_TIER + 1
        TierResource.step_wise_scoring_type = True
        TierResource._linear_reward_growth = True

    # ================= #
    #       Tests       #
    # ================= #

    # === Step wise scoring === #

    @pytest.mark.parametrize("resource", _tier_params())
    def test_in_order_consumption_gives_reward_and_builds_chain(
        self,
        parameterize_reset,
        digestion_engine: DigestionEngine,
        resource: TierResource,
    ):
        # prep the "chain" by giving it a tier value 1 lower than current resource
        digestion_engine.chained_tiers = resource.meta.tier - 1

        assert digestion_engine.digest(resource) == resource.REWARD
        assert digestion_engine.chained_tiers == resource.meta.tier

    def test_max_tier_consumption_rewards_and_resets_chain(
        self, reset_resource, digestion_engine: DigestionEngine
    ):
        max_resource = TierResource(self._MAX_TIER)

        # prep the "chain" by giving it a tier value 1 lower than max_resource
        digestion_engine.chained_tiers = max_resource.meta.tier - 1

        assert digestion_engine.digest(max_resource) == max_resource.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_out_of_order_consumption_returns_zero_and_resets_chain(
        self, reset_resource, digestion_engine: DigestionEngine
    ):
        resource = TierResource(self._MAX_TIER - 2)

        # force out-of-order consumption for resource
        digestion_engine.chained_tiers = self._MAX_TIER - 1

        assert digestion_engine.digest(resource) == 0
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_base_tier_consumption_rewards_and_starts_chain(
        self, reset_resource, digestion_engine: DigestionEngine
    ):
        base_resource = TierResource(0)

        # force out-of-order consumption for base tier
        digestion_engine.chained_tiers = self._MAX_TIER - 1

        assert digestion_engine.digest(base_resource) == base_resource.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._BASE_TIER

    # === Delayed scoring === #

    @pytest.mark.parametrize("resource", _tier_params())
    def test_in_order_consumption_return_zero_and_builds_chain(
        self,
        parameterize_reset,
        digestion_engine: DigestionEngine,
        resource: TierResource,
    ):
        # set correct scoring type
        resource.step_wise_scoring_type = False

        # prep the "chain" by giving it a tier value 1 lower than current resource
        digestion_engine.chained_tiers = resource.meta.tier - 1

        assert digestion_engine.digest(resource) == 0
        assert digestion_engine.chained_tiers == resource.meta.tier

    def test_delayed_scoring_max_tier_consumption_rewards_and_resets_chain(
        self, reset_resource, digestion_engine: DigestionEngine
    ):
        # set correct scoring type
        TierResource.step_wise_scoring_type = False

        max_resource = TierResource(self._MAX_TIER)

        # prep the "chain" by giving it a tier value 1 lower than max_resource
        digestion_engine.chained_tiers = max_resource.meta.tier - 1

        assert digestion_engine.digest(max_resource) == max_resource.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_out_of_order_consumption_rewards_and_resets_chain(
        self, reset_resource, digestion_engine: DigestionEngine
    ):
        # set correct scoring type
        TierResource.step_wise_scoring_type = False

        out_of_order_resource = TierResource(self._MAX_TIER - 3)
        in_order_resource = TierResource(self._MAX_TIER - 2)

        # force out-of-order consumption for out_of_order_resource and prep the reward
        digestion_engine.chained_tiers = in_order_resource.meta.tier
        digestion_engine._pending_reward = in_order_resource.REWARD

        assert (
            digestion_engine.digest(out_of_order_resource) == in_order_resource.REWARD
        )
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_base_tier_consumption_returns_pending_reward_and_starts_chain(
        self, reset_resource, digestion_engine: DigestionEngine
    ):
        # set correct scoring type
        TierResource.step_wise_scoring_type = False

        base_resource = TierResource(0)
        in_order_resource = TierResource(self._MAX_TIER - 1)

        # force out-of-order consumption for base_resource and prep the reward
        digestion_engine.chained_tiers = in_order_resource.meta.tier
        digestion_engine._pending_reward = in_order_resource.REWARD

        assert digestion_engine.digest(base_resource) == in_order_resource.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._BASE_TIER

