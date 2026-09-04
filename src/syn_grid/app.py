from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models import (
    ExperimentConfig,
    FullConf,
    WorldConfig,
    ObsConfig,
    AgentConfig,
)
from syn_grid.runners.human_runner.human_runner import HumanRunner
from syn_grid.runners.agent_runners.agent_registry import build_runner
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.gymnasium.utils.env_factory import register_env

from dataclasses import dataclass

# ================= #
#        APP        #
# ================= #


def main() -> None:
    register_env()

    config_manager = ConfigManager("configs.yaml")

    bundle = load_experiment_config(config_manager)

    if bundle.agent.global_agent_conf.human_control:
        runner = HumanRunner(
            bundle.world,
            bundle.obs.observation_handler.max_steps,
        )
        runner.human_player_loop()
        return

    runner = build_runner(bundle.world, bundle.obs, bundle.agent)
    dispatch(runner, config_manager, bundle)


# ================= #
#      Helpers      #
# ================= #


@dataclass
class ExperimentBundle:
    """Bundled configuration needed to build and run an experiment."""

    world: WorldConfig
    obs: ObsConfig
    agent: AgentConfig
    experiments_conf: ExperimentConfig


def load_experiment_config(config_manager: ConfigManager) -> ExperimentBundle:
    """
    Load the full experiment configuration.

    Args:
        config_manager: Manager pointed at the YAML config file to load.

    Returns:
        An ExperimentBundle with the world, obs, agent config and experiment settings.
    """

    full_conf = config_manager.load_config(FullConf)
    experiments_conf = config_manager.load_config(ExperimentConfig)

    return ExperimentBundle(
        world=full_conf.world,
        obs=full_conf.obs,
        agent=full_conf.agent,
        experiments_conf=experiments_conf,
    )


def dispatch(
    runner: BaseAgentRunner, config_manager: ConfigManager, bundle: ExperimentBundle
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

    if bundle.experiments_conf.snapshot.enabled:
        config_manager.save_snapshot(runner._get_model_id())
        print("Config snapshot saved. Exiting.")
        return

    if bundle.agent.global_agent_conf.training:
        runner.train()
    else:
        runner.eval()
