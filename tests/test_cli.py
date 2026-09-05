from unittest.mock import MagicMock

import pytest

from syn_grid.app import dispatch, load_experiment_configs
from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models import (
    AgentConfig,
    ExperimentConfig,
    ObsConfig,
    WorldConfig,
)
from syn_grid.runners.agent_runners.agent_bundle import AgentBundle

# ================= #
#  Global Fixtures  #
# ================= #


@pytest.fixture
def config_manager() -> ConfigManager:
    return ConfigManager("test_configs.yaml")


class TestLoadExperimentConfig:
    # ================= #
    #       Tests       #
    # ================= #

    def test_loads_full_conf_and_experiments_conf(self, config_manager):
        agent_bundle, experiments_conf = load_experiment_configs(config_manager)

        assert isinstance(agent_bundle.world_conf, WorldConfig)
        assert isinstance(agent_bundle.obs_conf, ObsConfig)
        assert isinstance(agent_bundle.agent_conf, AgentConfig)
        assert isinstance(experiments_conf, ExperimentConfig)


class TestDispatch:
    # ================= #
    #     Fixtures      #
    # ================= #

    @pytest.fixture
    def bundle(
        self, config_manager: ConfigManager
    ) -> tuple[AgentBundle, ExperimentConfig]:
        agent_bundle, experiments_conf = load_experiment_configs(config_manager)
        return agent_bundle, experiments_conf

    @pytest.fixture
    def runner(self) -> MagicMock:
        runner = MagicMock()
        runner.get_unique_model_id.return_value = "fake_model_id"
        return runner

    # ================= #
    #       Tests       #
    # ================= #

    def test_snapshot_enabled_saves_snapshot_and_returns(
        self,
        runner: MagicMock,
        config_manager: ConfigManager,
        bundle: tuple[AgentBundle, ExperimentConfig],
        capsys,
    ):
        agent_bundle, experiment_conf = bundle

        experiment_conf = experiment_conf.model_copy(
            update={
                "snapshot": experiment_conf.snapshot.model_copy(
                    update={"enabled": True}
                )
            }
        )

        save_snapshot = MagicMock()
        config_manager.save_snapshot = save_snapshot

        dispatch(
            runner,
            config_manager,
            agent_bundle,
            experiment_conf,
        )

        save_snapshot.assert_called_once_with("fake_model_id")
        runner.train.assert_not_called()
        runner.eval.assert_not_called()
        assert "Config snapshot saved" in capsys.readouterr().out

    def test_training_true_calls_train(
        self,
        runner: MagicMock,
        config_manager: ConfigManager,
        bundle: tuple[AgentBundle, ExperimentConfig],
    ):
        agent_bundle, experiment_conf = bundle
        agent_bundle.agent_conf.global_agent_conf.training = True

        dispatch(runner, config_manager, agent_bundle, experiment_conf)

        runner.train.assert_called_once()
        runner.eval.assert_not_called()

    def test_training_false_calls_eval(
        self,
        runner: MagicMock,
        config_manager: ConfigManager,
        bundle: tuple[AgentBundle, ExperimentConfig],
    ):
        agent_bundle, experiment_conf = bundle
        agent_bundle.agent_conf.global_agent_conf.training = False

        dispatch(runner, config_manager, agent_bundle, experiment_conf)

        runner.eval.assert_called_once()
        runner.train.assert_not_called()
