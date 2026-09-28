import pytest

from syn_grid.config.models import ScoringMode
from syn_grid.core.droid.digestion_engine import DigestionEngine
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.synergy.tier_orb import TierOrb
from tests.utils.config_helpers import get_test_config


class TestDigestionEngine:
    _MAX_TIER = 10

    # ================= #
    #      Helpers      #
    # ================= #

    @staticmethod
    def _tier_params(max_tier=_MAX_TIER) -> list[TierOrb]:
        """One orb per tier, all sitting below the world's ceiling.

        The ceiling is deliberately one above the top orb. The engine treats a
        consumed orb whose tier equals the ceiling as the end of a chain, so
        including the ceiling tier here would test completion on every case
        rather than progression. This used to be arranged by bumping a class
        attribute after the orbs were built; passing the ceiling in is the same
        arrangement without the shared state.
        """

        conf = get_test_config().world.tier_orb_conf
        tierOrbs = [TierOrb(t, conf, max_tier + 1) for t in range(1, max_tier + 1)]

        for t in tierOrbs:
            t.reset()

        return tierOrbs

    @staticmethod
    def _tier_orb(tier: int, max_tier: int, scoring: ScoringMode) -> TierOrb:
        return TierOrb(tier, get_test_config().world.tier_orb_conf, max_tier)

    @staticmethod
    def _as(orb: TierOrb, scoring: ScoringMode) -> TierOrb:
        """Re-score an orb.

        Scoring used to be three booleans an orb carried and a test overwrote one
        at a time. It is one value now, so this says which one it means.
        """

        orb.scoring = scoring
        return orb

    # ================= #
    #      Fixtures     #
    # ================= #

    @pytest.fixture
    def digestion_engine(self) -> DigestionEngine:
        BaseOrb.set_life_span(5, 5)
        d = DigestionEngine(-1.0, 1.0, -0.1)
        d.reset()
        return d

    # ================= #
    #       Tests       #
    # ================= #

    # === Step wise scoring === #

    @pytest.mark.parametrize("orb", _tier_params())
    def test_in_order_consumption_gives_reward_and_builds_chain(
        self,
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
        max_orb = TierOrb(
            self._MAX_TIER, get_test_config().world.tier_orb_conf, self._MAX_TIER
        )

        # prep the "chain" by giving it a tier value 1 lower than max_orb
        digestion_engine.chained_tiers = max_orb.META.TIER - 1

        assert digestion_engine.digest(max_orb) == max_orb.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_out_of_order_consumption_returns_zero_and_resets_chain(
        self, digestion_engine: DigestionEngine
    ):
        orb = TierOrb(
            self._MAX_TIER - 2, get_test_config().world.tier_orb_conf, self._MAX_TIER
        )

        # force out-of-order consumption for orb
        digestion_engine.chained_tiers = self._MAX_TIER - 1

        assert digestion_engine.digest(orb) == -1.0
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_base_tier_consumption_rewards_and_starts_chain(
        self, digestion_engine: DigestionEngine
    ):
        base_orb = TierOrb(1, get_test_config().world.tier_orb_conf, self._MAX_TIER)

        # force out-of-order consumption for base tier
        digestion_engine.chained_tiers = self._MAX_TIER

        assert digestion_engine.digest(base_orb) == base_orb.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._BASE_TIER

    # === Delayed scoring === #

    @pytest.mark.parametrize("orb", _tier_params())
    def test_in_order_consumption_return_zero_and_builds_chain(
        self,
        digestion_engine: DigestionEngine,
        orb: TierOrb,
    ):
        orb = self._as(orb, ScoringMode.THRESHOLD)

        # prep the "chain" by giving it a tier value 1 lower than current orb
        digestion_engine.chained_tiers = orb.META.TIER - 1

        assert digestion_engine.digest(orb) == 0
        assert digestion_engine.chained_tiers == orb.META.TIER

    def test_threshold_scoring_max_tier_consumption_rewards_and_resets_chain(
        self, digestion_engine: DigestionEngine
    ):
        max_orb = TierOrb(
            self._MAX_TIER, get_test_config().world.tier_orb_conf, self._MAX_TIER
        )
        max_orb.scoring = ScoringMode.THRESHOLD

        # prep the "chain" by giving it a tier value 1 lower than max_orb
        digestion_engine.chained_tiers = max_orb.META.TIER - 1

        assert digestion_engine.digest(max_orb) == max_orb.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_out_of_order_consumption_rewards_and_resets_chain(
        self, digestion_engine: DigestionEngine
    ):
        out_of_order_orb = TierOrb(
            self._MAX_TIER - 3, get_test_config().world.tier_orb_conf, self._MAX_TIER
        )
        in_order_orb = TierOrb(
            self._MAX_TIER - 2, get_test_config().world.tier_orb_conf, self._MAX_TIER
        )
        out_of_order_orb.scoring = ScoringMode.THRESHOLD
        in_order_orb.scoring = ScoringMode.THRESHOLD

        # force out-of-order consumption for out_of_order_orb and prep the reward
        digestion_engine.chained_tiers = in_order_orb.META.TIER
        digestion_engine._pending_reward = in_order_orb.REWARD

        assert digestion_engine.digest(out_of_order_orb) == in_order_orb.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_base_tier_consumption_returns_pending_reward_and_starts_chain(
        self, digestion_engine: DigestionEngine
    ):
        base_orb = TierOrb(1, get_test_config().world.tier_orb_conf, self._MAX_TIER)
        in_order_orb = TierOrb(
            self._MAX_TIER, get_test_config().world.tier_orb_conf, self._MAX_TIER
        )
        base_orb.scoring = ScoringMode.THRESHOLD
        in_order_orb.scoring = ScoringMode.THRESHOLD

        # force out-of-order consumption for base_orb and prep the reward
        digestion_engine.chained_tiers = in_order_orb.META.TIER
        digestion_engine._pending_reward = in_order_orb.REWARD

        assert digestion_engine.digest(base_orb) == in_order_orb.REWARD
        assert digestion_engine.chained_tiers == digestion_engine._BASE_TIER

    # === Max tier scoring === #

    def test_max_tier_scoring_pays_nothing_until_the_chain_completes(
        self, digestion_engine: DigestionEngine
    ):
        """The property the spatial and sparse-scaling scenarios are built on: an
        incomplete chain is worth exactly zero, so the only reward in the world
        is the full chain."""

        for tier in range(1, self._MAX_TIER):
            orb = TierOrb(tier, get_test_config().world.tier_orb_conf, self._MAX_TIER)
            orb.scoring = ScoringMode.MAX_TIER
            digestion_engine.chained_tiers = tier - 1

            assert digestion_engine.digest(orb) == 0.0

    def test_max_tier_scoring_pays_the_full_reward_on_completion(
        self, digestion_engine: DigestionEngine
    ):
        orb = TierOrb(
            self._MAX_TIER, get_test_config().world.tier_orb_conf, self._MAX_TIER
        )
        orb.scoring = ScoringMode.MAX_TIER
        digestion_engine.chained_tiers = self._MAX_TIER - 1

        assert digestion_engine.digest(orb) == orb.REWARD

    # === Counters and chain state === #

    def test_a_broken_chain_pays_the_chain_break_penalty(
        self, digestion_engine: DigestionEngine
    ):
        orb = TierOrb(
            self._MAX_TIER - 2, get_test_config().world.tier_orb_conf, self._MAX_TIER
        )
        orb.scoring = ScoringMode.MAX_TIER
        digestion_engine.chained_tiers = self._MAX_TIER - 1

        digestion_engine.digest(orb)

        assert digestion_engine.chains_broken == 1
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_a_negative_orb_breaks_a_running_chain(
        self, digestion_engine: DigestionEngine
    ):
        from syn_grid.config.models import NegativeConf
        from syn_grid.core.orbs.direct.negative_orb import NegativeOrb

        orb = NegativeOrb(NegativeConf(reward=-3.0, cool_down=5))
        digestion_engine.chained_tiers = 2

        reward = digestion_engine.digest(orb)

        assert reward == -3.0
        assert digestion_engine.chains_broken == 1
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_a_negative_orb_with_no_chain_running_is_not_a_break(
        self, digestion_engine: DigestionEngine
    ):
        from syn_grid.config.models import NegativeConf
        from syn_grid.core.orbs.direct.negative_orb import NegativeOrb

        orb = NegativeOrb(NegativeConf(reward=-3.0, cool_down=5))

        assert digestion_engine.digest(orb) == -3.0
        assert digestion_engine.chains_broken == 0

    def test_reset_clears_the_counters_because_they_cover_one_episode(
        self, digestion_engine: DigestionEngine
    ):
        """The counters are per-episode, so reset() zeroes them along with the
        chain. The docstring on _reset_chain says the opposite -- that they
        "cover the whole episode and stay" -- which is only true within an
        episode, and reads as though they survive reset()."""

        digestion_engine._chains_broken = 3
        digestion_engine._chains_completed = 2
        digestion_engine._chains_progressed = 7

        digestion_engine.reset()

        assert digestion_engine.chains_broken == 0
        assert digestion_engine.chains_completed == 0
        assert digestion_engine.chains_progressed == 0
        assert digestion_engine.chained_tiers == digestion_engine._NO_CHAIN

    def test_reset_clears_a_pending_reward(self, digestion_engine: DigestionEngine):
        digestion_engine._pending_reward = 4.2

        digestion_engine.reset()

        assert digestion_engine.pending_reward == 0.0

    def test_stats_reports_the_counters(self, digestion_engine: DigestionEngine):
        assert digestion_engine.stats == {
            "chains_progressed": 0,
            "chains_broken": 0,
            "chains_completed": 0,
        }
