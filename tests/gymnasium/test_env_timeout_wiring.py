"""Wiring test: does `DroidConf.timeout_penalty` actually reach the terminal reward?

The unit tests in `tests/gymnasium/utils/test_episode_termination.py` call
`check_episode_end` directly, so they cannot see whether the environment passes
the configured value or the wrong one. They all pass even if `environment.py`
hands the function something else entirely.

That gap is exactly where the original defect lived: the parameter was named
`timeout_penalty` but the caller passed `chain_break_penalty`, and nothing failed
for a long time. These tests drive the real environment to the step limit so the
whole path -- YAML -> FullConf -> SYNGridEnv -> check_episode_end -- is covered.

Note on isolation: constructing the env sets orb class attributes (see AGENTS.md,
"Global mutable state"). The config is therefore held constant across the
parametrised cases and only `timeout_penalty` varies, so the orb state observed
here is the same in every case.
"""

from pathlib import Path

import pytest
import yaml

from syn_grid.legacy.config_legacy.models import FullConf
from syn_grid.legacy.gymnasium_legacy.utils.env_factory import make, register_env

CONFIG = Path("src/syn_grid/config/configs.yaml")


def _conf(timeout_penalty: float) -> FullConf:
    raw = yaml.safe_load(CONFIG.read_text())
    droid = raw["world"]["droid_conf"]
    droid["timeout_penalty"] = timeout_penalty
    # Isolate the terminal reward from per-step costs so the assertion is on the
    # timeout penalty alone, and keep the droid alive so the episode cannot end
    # early on score depletion.
    droid["step_penalty"] = 0.0
    droid["boundary_penalty"] = 0.0
    droid["starting_score"] = 1e9
    return FullConf(**raw)


@pytest.fixture(scope="module", autouse=True)
def _register() -> None:
    register_env()


def _terminal_reward(conf: FullConf) -> float:
    env = make(None, conf.world, conf.obs)
    try:
        env.reset(seed=7)
        horizon = conf.obs.observation_handler.max_steps
        reward = 0.0
        for _ in range(horizon + 5):
            _, reward, terminated, truncated, _ = env.step(1)
            if terminated or truncated:
                return float(reward)
        raise AssertionError(f"episode did not end within {horizon} steps")
    finally:
        env.close()


@pytest.mark.parametrize("timeout_penalty", [-0.1, -0.25, -1.0, -5.0])
def test_configured_timeout_penalty_reaches_the_terminal_reward(
    timeout_penalty: float,
):
    reward = _terminal_reward(_conf(timeout_penalty))

    assert reward == pytest.approx(timeout_penalty)


def test_terminal_reward_tracks_config_rather_than_a_constant():
    """Two configs, two different terminal rewards. If the env were still
    hardcoding -1 or passing the chain-break penalty, these would be equal."""

    cheap = _terminal_reward(_conf(-0.25))
    harsh = _terminal_reward(_conf(-5.0))

    assert cheap != harsh
    assert cheap == pytest.approx(-0.25)
    assert harsh == pytest.approx(-5.0)


def test_env_exposes_both_penalties_separately():
    """The env must carry the timeout penalty as its own attribute, not reuse
    the chain-break value under a confusing name."""

    conf = _conf(-2.5)
    assert conf.world.droid_conf.chain_break_penalty == -0.01
    env = make(None, conf.world, conf.obs)
    try:
        assert env.unwrapped.timeout_penalty == -2.5
        assert not hasattr(env.unwrapped, "chain_break_penalty")
    finally:
        env.close()
