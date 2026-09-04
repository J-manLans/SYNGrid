from syn_grid.cli import load_experiment_config, dispatch, ExperimentBundle
from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models import ExperimentConfig, FullConf

from unittest.mock import MagicMock
import pytest


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
        bundle = load_experiment_config(config_manager)

        assert isinstance(bundle.full_conf, FullConf)
        assert isinstance(bundle.experiments_conf, ExperimentConfig)

    def test_no_overrides_leaves_config_unchanged(self, config_manager):
        bundle = load_experiment_config(config_manager)
        baseline = config_manager.load_config(FullConf)

        assert bundle.full_conf.agent == baseline.agent

    def test_applies_overrides_onto_loaded_config(self, config_manager):
        bundle = load_experiment_config(
            config_manager, overrides={"human_control": True}
        )

        assert bundle.full_conf.agent.global_agent_conf.human_control is True

    def test_empty_overrides_dict_leaves_config_unchanged(self, config_manager):
        bundle = load_experiment_config(config_manager, overrides={})
        baseline = config_manager.load_config(FullConf)

        assert bundle.full_conf.agent == baseline.agent


class TestDispatch:
    # ================= #
    #     Fixtures      #
    # ================= #

    @pytest.fixture
    def bundle(self, config_manager) -> ExperimentBundle:
        return load_experiment_config(config_manager)

    @pytest.fixture
    def runner(self) -> MagicMock:
        runner = MagicMock()
        runner._get_model_id.return_value = "fake_model_id"
        return runner

    # ================= #
    #       Tests       #
    # ================= #

    def test_snapshot_enabled_saves_snapshot_and_exits(
        self, runner, config_manager, bundle
    ):
        bundle.experiments_conf = bundle.experiments_conf.model_copy(
            update={"snapshot": bundle.experiments_conf.snapshot.model_copy(
                update={"enabled": True}
            )}
        )
        save_snapshot = MagicMock()
        config_manager.save_snapshot = save_snapshot

        with pytest.raises(SystemExit):
            dispatch(runner, config_manager, bundle)

        save_snapshot.assert_called_once_with("fake_model_id")
        runner.train.assert_not_called()
        runner.eval.assert_not_called()

    def test_training_true_calls_train(self, runner, config_manager, bundle):
        bundle.full_conf.agent.global_agent_conf.training = True

        dispatch(runner, config_manager, bundle)

        runner.train.assert_called_once()
        runner.eval.assert_not_called()

    def test_training_false_calls_eval(self, runner, config_manager, bundle):
        bundle.full_conf.agent.global_agent_conf.training = False

        dispatch(runner, config_manager, bundle)

        runner.eval.assert_called_once()
        runner.train.assert_not_called()
