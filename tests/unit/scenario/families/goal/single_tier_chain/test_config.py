"""What the single tier chain configs accept, and the steps delay's config gives an episode."""

import copy
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.scenario.families.goal.single_tier_chain.config import (
    TierDelayScenarioConf,
)
from syn_grid.scenario.registry import SCENARIOS

# ================= #
#     Constants     #
# ================= #


DELAY = ScenarioName.GOAL_SINGLE_TIER_CHAIN_DELAY


# ================= #
#      Helpers      #
# ================= #


def _with(
    base: dict[str, Any], *, delay: int, rows: int = 5, cols: int = 5, max_tier: int = 3
) -> dict[str, Any]:
    raw = copy.deepcopy(base)
    raw["delay_conf"]["steps"] = delay
    raw["world_conf"]["grid_conf"] = {"grid_rows": rows, "grid_cols": cols}
    raw["world_conf"]["tier_orb_conf"]["max_tier"] = max_tier

    return raw


# === Delay tunables === #


def _run(**extra: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "world_conf": {"neg_orb_conf": {"enabled": False}},
        "delay_conf": {"steps": 390},
        "obs_conf": {"observation_handler_conf": {"perception": "vector_markovian"}},
    }
    for key, value in extra.items():
        raw[key] = {**raw.get(key, {}), **value}

    return raw


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def delay_base() -> dict[str, Any]:
    lock = SCENARIOS[DELAY].lock
    assert lock is not None

    with lock.base_config_path.open("r") as f:
        return yaml.safe_load(f)


# ================= #
#       Tests       #
# ================= #


class TestDelayMaxSteps:
    """The steps delay's config model gives an episode: the silences, plus the spare steps."""

    def test_spare_steps_follow_rows_and_cols(self, delay_base: dict[str, Any]):
        conf = TierDelayScenarioConf(**_with(delay_base, delay=100, rows=10, cols=10))

        assert conf.world_conf.max_steps == 2 * 100 + 100

    def test_one_silence_per_tier_orb_but_the_last(self, delay_base: dict[str, Any]):
        conf = TierDelayScenarioConf(**_with(delay_base, delay=100, max_tier=4))

        assert conf.world_conf.max_steps == 3 * 100 + 50

    def test_a_stated_max_steps_is_rejected(self, delay_base: dict[str, Any]):
        raw = _with(delay_base, delay=100)
        raw["world_conf"]["max_steps"] = 250

        with pytest.raises(ValidationError, match="max_steps"):
            TierDelayScenarioConf(**raw)

    def test_a_config_missing_a_block_is_told_which(self, delay_base: dict[str, Any]):
        raw = _with(delay_base, delay=100)
        del raw["delay_conf"]

        with pytest.raises(ValidationError, match="delay_conf"):
            TierDelayScenarioConf(**raw)


class TestChainFitsGrid:
    """A chain needs a cell for each tier orb, the negative orb and the droid."""

    def test_a_chain_that_leaves_the_negative_orb_and_the_droid_a_cell_is_accepted(
        self, delay_base: dict[str, Any]
    ):
        TierDelayScenarioConf(
            **_with(delay_base, delay=100, rows=2, cols=2, max_tier=2)
        )

    def test_the_negative_orbs_cell_counts_while_it_is_switched_off(
        self, delay_base: dict[str, Any]
    ):
        raw = _with(delay_base, delay=100, rows=2, cols=2, max_tier=3)
        assert not raw["world_conf"]["neg_orb_conf"]["enabled"]

        with pytest.raises(ValidationError, match="max_tier 3 does not fit .* 4 cells"):
            TierDelayScenarioConf(**raw)


class TestDelayTunables:
    """What a delay run may set: the delay, the negative orb's switch and the perception."""

    def test_a_run_sets_the_delay_and_gets_50_spare_steps_on_5x5(self):
        conf = SCENARIOS[DELAY].full_conf(_run())

        assert conf.world_conf.max_steps == 2 * 390 + 50

    def test_a_run_turns_the_negative_orb_on(self):
        conf = SCENARIOS[DELAY].full_conf(
            _run(world_conf={"neg_orb_conf": {"enabled": True}})
        )

        assert conf.world_conf.neg_orb_conf.enabled

    def test_a_fixed_value_is_rejected(self):
        with pytest.raises(ValidationError):
            SCENARIOS[DELAY].full_conf(
                _run(world_conf={"droid_conf": {"step_penalty": -0.001}})
            )
