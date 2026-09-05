import pytest

from syn_grid.runners.agent_runners.agent_bundle import AgentBundle
from syn_grid.runners.agent_runners.agent_registry import ALGORITHMS, build_runner
from syn_grid.runners.agent_runners.sb3.stateless_ppo import StatelessPPO
from tests.utils.config_helpers import get_test_config, update_conf


class TestBuildRunner:
    # ================= #
    #     Fixtures      #
    # ================= #

    @pytest.fixture
    def make_agent_bundle(self):
        def _make(alg: str) -> AgentBundle:
            full_conf = get_test_config()
            full_conf = update_conf(
                full_conf, {"agent": {"global_agent_conf": {"alg": alg}}}
            )

            return AgentBundle(
                world_conf=full_conf.world,
                obs_conf=full_conf.obs,
                agent_conf=full_conf.agent,
            )

        return _make

    # ================= #
    #       Tests       #
    # ================= #

    def test_builds_registered_algorithm(self, make_agent_bundle):
        runner = build_runner(make_agent_bundle("PPO"))

        assert isinstance(runner, StatelessPPO)

    @pytest.mark.parametrize("alg", list(ALGORITHMS.keys()))
    def test_builds_every_registered_algorithm(self, make_agent_bundle, alg: str):
        runner = build_runner(make_agent_bundle(alg))

        assert isinstance(runner, ALGORITHMS[alg])

    def test_raises_key_error_for_unregistered_algorithm(self, make_agent_bundle):
        with pytest.raises(KeyError, match="not_a_real_algo"):
            build_runner(make_agent_bundle("not_a_real_algo"))
