from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import GlobalConf
from syn_grid.config.models.runner_models import RunnerConf
from syn_grid.config.models.scenario.common_models import ScenarioConf
from syn_grid.gymnasium.utils.env_factory import register_env
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.runners.agent_runners.runner_bundle import RunnerBundle
from syn_grid.runners.agent_runners.runner_registry import build_runner
from syn_grid.runners.human_runner.human_runner import HumanRunner
from syn_grid.scenario.registry import SCENARIOS, build_scenario

# ================= #
#        APP        #
# ================= #


def main() -> None:
    register_env()
    config_manager = ConfigManager()
    global_conf, scenario_conf, runner_conf = load_experiment_configs(config_manager)

    scenario = build_scenario(global_conf.scenario, scenario_conf)

    # Human control overrides everything else: no snapshot, training or eval.
    if global_conf.human_control:
        HumanRunner(scenario).play()
        return

    runner_bundle = RunnerBundle(scenario=scenario, runner_conf=runner_conf)
    runner = build_runner(runner_bundle)

    dispatch(runner, config_manager, runner_bundle, global_conf)


# ================= #
#      Helpers      #
# ================= #


def load_experiment_configs(
    config_manager: ConfigManager,
) -> tuple[GlobalConf, ScenarioConf, RunnerConf]:
    """
    Load the global, scenario, and runner configurations.

    Args:
        config_manager: Manager used to load the YAML configuration files.
    Returns:
        A tuple containing the global, scenario and runner configuration.
    """

    global_conf = config_manager.load_config("global_config.yaml", GlobalConf)
    scenario_conf = config_manager.load_config(
        config_manager.scenario_config_path(global_conf.scenario),
        SCENARIOS[global_conf.scenario].conf_class,
    )
    runner_conf = config_manager.load_config("runner_config.yaml", RunnerConf)

    return (global_conf, scenario_conf, runner_conf)


def dispatch(
    runner: BaseAgentRunner,
    config_manager: ConfigManager,
    runner_bundle: RunnerBundle,
    global_conf: GlobalConf,
) -> None:
    """
    Run an agent runner according to the loaded experiment configuration.

    Handles the snapshot, training, and evaluation modes.

    Args:
        runner: The agent runner to dispatch to (train or eval).
        config_manager: Used to save a config snapshot if enabled.
        runner_bundle: The loaded experiment configuration.
        global_conf: The global configuration, read for the snapshot
            setting and the scenario's name.
    """

    if global_conf.snapshot.enabled:
        config_manager.save_snapshot(
            config_manager.scenario_config_path(global_conf.scenario),
            runner.get_unique_model_id(),
        )
        print("Config snapshot saved. Exiting.")
        return

    if runner_bundle.runner_conf.common_conf.training:
        runner.train()
    else:
        runner.eval()
