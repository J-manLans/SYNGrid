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
#     Fixtures      #
# ================= #

DELAY = ScenarioName.GOAL_SINGLE_TIER_CHAIN_DELAY


@pytest.fixture
def delay_base() -> dict[str, Any]:
    lock = SCENARIOS[DELAY].lock
    assert lock is not None

    with lock.base_config_path.open("r") as f:
        return yaml.safe_load(f)


def _with(
    base: dict[str, Any], *, delay: int, rows: int = 5, cols: int = 5, max_tier: int = 3
) -> dict[str, Any]:
    raw = copy.deepcopy(base)
    raw["delay_conf"]["steps"] = delay
    raw["world_conf"]["grid_conf"] = {"grid_rows": rows, "grid_cols": cols}
    raw["world_conf"]["tier_orb_conf"]["max_tier"] = max_tier

    return raw


# ================= #
#      Tests        #
# ================= #


class TestDelayMaxSteps:
    """The steps delay's config model gives an episode: the silences, plus the spare steps."""

    @pytest.mark.parametrize("delay", [30, 70, 130, 260, 390])
    def test_spare_steps_on_5x5_are_50_at_any_delay(
        self, delay_base: dict[str, Any], delay: int
    ):
        conf = TierDelayScenarioConf(**_with(delay_base, delay=delay))

        assert conf.world_conf.max_steps - 2 * delay == 50

    @pytest.mark.parametrize(
        ("rows", "cols", "spare"), [(4, 4, 40), (4, 6, 50), (7, 7, 70), (10, 10, 100)]
    )
    def test_spare_steps_follow_the_grid(
        self, delay_base: dict[str, Any], rows: int, cols: int, spare: int
    ):
        conf = TierDelayScenarioConf(
            **_with(delay_base, delay=100, rows=rows, cols=cols)
        )

        assert conf.world_conf.max_steps == 2 * 100 + spare

    def test_one_silence_per_tier_orb_but_the_last(self, delay_base: dict[str, Any]):
        conf = TierDelayScenarioConf(**_with(delay_base, delay=100, max_tier=4))

        assert conf.world_conf.max_steps == 3 * 100 + 50

    def test_a_stated_max_steps_is_rejected(self, delay_base: dict[str, Any]):
        raw = _with(delay_base, delay=100)
        raw["world_conf"]["max_steps"] = 250

        with pytest.raises(ValidationError, match="max_steps"):
            TierDelayScenarioConf(**raw)


def _run(**extra: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "world_conf": {"neg_orb_conf": {"enabled": False}},
        "delay_conf": {"steps": 390},
        "obs_conf": {"observation_handler_conf": {"perception": "vector_markovian"}},
    }
    for key, value in extra.items():
        raw[key] = {**raw.get(key, {}), **value}

    return raw


class TestDelayTunables:
    """What a delay run may set: the delay, the negative orb's switch and the perception."""

    def test_a_run_sets_the_delay_and_gets_its_steps(self):
        conf = SCENARIOS[DELAY].full_conf(_run())

        assert conf.delay_conf.steps == 390
        assert conf.world_conf.max_steps == 830

    def test_a_run_turns_the_negative_orb_on(self):
        conf = SCENARIOS[DELAY].full_conf(
            _run(world_conf={"neg_orb_conf": {"enabled": True}})
        )

        assert conf.world_conf.neg_orb_conf.enabled

    @pytest.mark.parametrize(
        "extra",
        [
            {"world_conf": {"max_steps": 850}},
            {"world_conf": {"grid_conf": {"grid_rows": 6, "grid_cols": 6}}},
            {"world_conf": {"droid_conf": {"step_penalty": -0.001}}},
        ],
    )
    def test_anything_else_is_rejected(self, extra: dict[str, Any]):
        with pytest.raises(ValidationError):
            SCENARIOS[DELAY].full_conf(_run(**extra))
