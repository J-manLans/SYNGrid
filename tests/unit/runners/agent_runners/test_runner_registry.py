"""Which runner each algorithm in the runner config gets, and what each SB3 algorithm is run with."""

from pathlib import Path

import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.runner_models import Algorithm, RunnerConf
from syn_grid.runners.agent_runners.runner_registry import RUNNER, build_runner
from syn_grid.runners.agent_runners.sb3.sb3_runner import SB3Runner, SB3Spec
from syn_grid.scenario.registry import SCENARIOS

# ================= #
#     Constants     #
# ================= #


SPATIAL = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL


# ================= #
#      Helpers      #
# ================= #


def _runner_conf(alg: Algorithm) -> RunnerConf:
    base = ConfigManager().load_config("runner_config.yaml", RunnerConf)
    return base.model_copy(
        update={"common_conf": base.common_conf.model_copy(update={"alg": alg})}
    )


# ================= #
#       Tests       #
# ================= #


class TestRunner:
    """What each algorithm is run with."""

    def test_every_algorithm_in_the_config_has_a_runner(self):
        assert set(RUNNER) == set(Algorithm)

    def test_ppo_runs_on_the_cpu_with_no_memory(self):
        assert RUNNER[Algorithm.PPO] == SB3Spec(PPO, device="cpu")

    def test_fsppo_is_ppo_with_four_stacked_observations(self):
        assert RUNNER[Algorithm.FSPPO] == SB3Spec(PPO, device="cpu", frame_stack=4)

    def test_rppo_is_recurrent_ppo_on_the_gpu_with_one_256_unit_lstm_per_network(self):
        assert RUNNER[Algorithm.RPPO] == SB3Spec(
            RecurrentPPO,
            device="cuda",
            hyper_parameters={
                "policy_kwargs": {
                    "lstm_hidden_size": 256,
                    "n_lstm_layers": 1,
                    "shared_lstm": False,
                }
            },
        )


class TestBuildRunner:
    """Making the configured algorithm's runner."""

    def test_it_makes_the_configured_algorithms_runner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        scenario = SCENARIOS[SPATIAL].build(
            SPATIAL.value, SCENARIOS[SPATIAL].base_conf()
        )

        runner = build_runner(scenario, _runner_conf(Algorithm.FSPPO))

        # The run id names the algorithm, and a PPO run's would not say FSPPO
        assert isinstance(runner, SB3Runner)
        assert Algorithm.FSPPO.value in runner.get_unique_model_id()

    def test_an_algorithm_with_no_runner_is_refused_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delitem(RUNNER, Algorithm.RPPO)

        with pytest.raises(KeyError, match="RPPO"):
            build_runner(object(), _runner_conf(Algorithm.RPPO))  # type: ignore[arg-type]
