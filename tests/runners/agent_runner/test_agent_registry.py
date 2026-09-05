import pytest

from syn_grid.runners.agent_runners.agent_registry import ALGORITHMS, build_runner
from syn_grid.runners.agent_runners.sb3.stateless_ppo import StatelessPPO
from tests.utils.config_helpers import get_test_config, update_conf


class TestBuildRunner:
    # ================= #
    #       Tests       #
    # ================= #

    def test_builds_registered_algorithm(self):
        full_conf = get_test_config()
        full_conf = update_conf(
            full_conf, {"agent": {"global_agent_conf": {"alg": "PPO"}}}
        )

        runner = build_runner(full_conf.world, full_conf.obs, full_conf.agent)

        assert isinstance(runner, StatelessPPO)

    @pytest.mark.parametrize("alg", list(ALGORITHMS.keys()))
    def test_builds_every_registered_algorithm(self, alg: str):
        full_conf = get_test_config()
        full_conf = update_conf(
            full_conf, {"agent": {"global_agent_conf": {"alg": alg}}}
        )

        runner = build_runner(full_conf.world, full_conf.obs, full_conf.agent)

        assert isinstance(runner, ALGORITHMS[alg])

    def test_raises_key_error_for_unregistered_algorithm(self):
        full_conf = get_test_config()
        full_conf = update_conf(
            full_conf, {"agent": {"global_agent_conf": {"alg": "not_a_real_algo"}}}
        )

        with pytest.raises(KeyError, match="not_a_real_algo"):
            build_runner(full_conf.world, full_conf.obs, full_conf.agent)
