from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import GlobalConf
from syn_grid.config.models.runner_models import RunnerConf
from syn_grid.config.models.scenario.scenario_common import ScenarioConf
from syn_grid.config.models.scenario.scenario_models import SCENARIO_MODELS
from syn_grid.gymnasium.utils.env_factory import register_env
from syn_grid.runners.agent_runners.agent_registry import build_runner
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.runners.agent_runners.runner_bundle import RunnerBundle
from syn_grid.scenario.registry import build_scenario

# ================= #
#        APP        #
# ================= #


def main() -> None:
    register_env()
    config_manager = ConfigManager()
    global_conf, scenario_conf, runner_conf = load_experiment_configs(config_manager)
    scenario = build_scenario(global_conf.scenario, scenario_conf)
    runner_bundle = RunnerBundle(scenario, runner_conf)
    runner = build_runner(global_conf.human_control, runner_bundle)

    dispatch(runner, config_manager, (runner_bundle), global_conf)


# ================= #
#      Helpers      #
# ================= #


def load_experiment_configs(config_manager: ConfigManager) -> tuple[
    GlobalConf,
    ScenarioConf,
    RunnerConf
]:
    """
    Load the global, scenario, and runner configurations.

    Args:
        config_manager: Manager used to load the YAML configuration files.

    Returns:
        A tuple containing the global, scenario and runner configuration.
    """

    global_conf = config_manager.load_config("global_config.yaml", GlobalConf)
    scenario_conf = config_manager.load_config(
        f"{global_conf.scenario.value}.yaml",
        SCENARIO_MODELS[global_conf.scenario]
    )
    runner_conf = config_manager.load_config("runner_config.yaml", RunnerConf)

    return (global_conf, scenario_conf, runner_conf)


def dispatch(
    runner: BaseAgentRunner,
    config_manager: ConfigManager,
    agent_bundle: RunnerBundle,
    global_conf: GlobalConf,
) -> None:
    """
    Run an agent runner according to the loaded experiment configuration.

    Handles the snapshot, training, and evaluation modes. Not used for
    HumanRunner, which is driven directly via `human_player_loop()`
    since it doesn't participate in snapshot/train/eval dispatch.

    Args:
        runner: The agent runner to dispatch to (train or eval).
        config_manager: Used to save a config snapshot if enabled.
        bundle: The loaded experiment configuration.
    """

    if global_conf.snapshot.enabled:
        config_manager.save_snapshot(
            f"{global_conf.scenario.value}.yaml",
            runner.get_unique_model_id()
        )
        print("Config snapshot saved. Exiting.")
        return

    if agent_bundle.runner_conf.common_conf.training:
        runner.train()
    else:
        runner.eval()
