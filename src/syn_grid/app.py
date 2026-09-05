from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models import (
    ExperimentConfig,
    FullConf,
)
from syn_grid.gymnasium.utils.env_factory import register_env
from syn_grid.runners.agent_runners.agent_bundle import AgentBundle
from syn_grid.runners.agent_runners.agent_registry import build_runner
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.runners.human_runner.human_runner import HumanRunner

# ================= #
#        APP        #
# ================= #


def main() -> None:
    register_env()

    config_manager = ConfigManager("configs.yaml")

    agent_bundle, experiment_conf = load_experiment_configs(config_manager)

    if agent_bundle.agent_conf.global_agent_conf.human_control:
        runner = HumanRunner(
            agent_bundle.world_conf,
            agent_bundle.obs_conf.observation_handler.max_steps,
        )
        runner.human_player_loop()
        return

    runner = build_runner(agent_bundle)
    dispatch(runner, config_manager, agent_bundle, experiment_conf)


# ================= #
#      Helpers      #
# ================= #


def load_experiment_configs(
    config_manager: ConfigManager,
) -> tuple[AgentBundle, ExperimentConfig]:
    """
    Load the full experiment configuration.

    Args:
        config_manager: Manager pointed at the YAML config file to load.

    Returns:
        An ExperimentBundle with the world, obs, agent config and experiment settings.
    """

    full_conf = config_manager.load_config(FullConf)

    return (
        AgentBundle(
            world_conf=full_conf.world,
            obs_conf=full_conf.obs,
            agent_conf=full_conf.agent,
        ),
        config_manager.load_config(ExperimentConfig),
    )


def dispatch(
    runner: BaseAgentRunner,
    config_manager: ConfigManager,
    agent_bundle: AgentBundle,
    experiment_conf: ExperimentConfig,
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

    if experiment_conf.snapshot.enabled:
        config_manager.save_snapshot(runner.get_unique_model_id())
        print("Config snapshot saved. Exiting.")
        return

    if agent_bundle.agent_conf.global_agent_conf.training:
        runner.train()
    else:
        runner.eval()
