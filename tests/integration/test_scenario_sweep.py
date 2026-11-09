"""Every registered scenario, built from its base config and made the way a run makes its env, keeps
Gymnasium's contract and plays a whole episode."""

import warnings

import pytest

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.gymnasium.utils.env_factory import check_my_env, make, register_env
from syn_grid.scenario.registry import SCENARIOS, build_scenario

# ================= #
#       Tests       #
# ================= #


class TestEveryScenario:
    """One case per registered scenario, so a new scenario is swept with no change here."""

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_it_keeps_gymnasiums_contract_and_plays_an_episode_to_its_end(
        self, name: ScenarioName
    ):
        conf = SCENARIOS[name].base_conf()
        register_env()
        env = make(build_scenario(name, conf), render_mode=None)

        try:
            with warnings.catch_warnings():
                # check_env warns about what it cannot verify, such as rendering with no render mode
                warnings.simplefilter("ignore")
                check_my_env(env)

            obs, _ = env.reset(seed=0)
            env.action_space.seed(0)
            assert env.observation_space.contains(obs)

            # Random play usually breaks the chain early; the clock ends it by max_steps at the latest.
            # The clock itself is tested in the family's termination tests
            for _ in range(conf.world_conf.max_steps):
                obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
                assert env.observation_space.contains(obs)
                if terminated or truncated:
                    break
            else:
                pytest.fail(f"{name.value} played max_steps without ending")
        finally:
            env.close()
