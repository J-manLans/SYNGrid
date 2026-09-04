from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models import ExperimentConfig, FullConf
from syn_grid.utils.args_utils import parse_args, args_to_overrides
from syn_grid.config.overrides import apply_overrides
from syn_grid.runners.human_runner.human_runner import HumanRunner
from syn_grid.runners.agent_runners.agent_registry import build_runner
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.gymnasium.utils.env_factory import register_env

from dataclasses import dataclass
import sys


@dataclass
class ExperimentBundle:
    """Bundled configuration needed to build and run an experiment."""

    full_conf: FullConf
    experiments_conf: ExperimentConfig


# ================= #
#    Config setup   #
# ================= #


def load_experiment_config(
    config_manager: ConfigManager, overrides: dict | None = None
) -> ExperimentBundle:
    """
    Load the full experiment configuration and apply any overrides.

    Args:
        config_manager: Manager pointed at the YAML config file to load.
        overrides: Optional dict of agent-config overrides, e.g. from CLI
            args (via `args_to_overrides`) or a future GUI. Keys must
            match fields on GlobalAgentConf, TrainAgentConf, or
            EvalAgentConf.

    Returns:
        An ExperimentBundle with the full config and experiment settings.
    """

    full_conf = config_manager.load_config(FullConf)
    experiments_conf = config_manager.load_config(ExperimentConfig)

    if overrides:
        apply_overrides(full_conf.agent, overrides)

    return ExperimentBundle(full_conf=full_conf, experiments_conf=experiments_conf)


# ================= #
#      Dispatch     #
# ================= #


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
        sys.exit("Config snapshot saved. Exiting.")

    if bundle.full_conf.agent.global_agent_conf.training:
        runner.train()
    else:
        runner.eval()


# ================= #
#        CLI        #
# ================= #


def main() -> None:
    register_env()

    config_manager = ConfigManager("configs.yaml")

    overrides = {}
    if len(sys.argv) > 1:
        overrides = args_to_overrides(parse_args())

    bundle = load_experiment_config(config_manager, overrides)
    agent_conf = bundle.full_conf.agent

    if agent_conf.global_agent_conf.human_control:
        runner = HumanRunner(
            bundle.full_conf.world,
            bundle.full_conf.obs.observation_handler.max_steps,
        )
        runner.human_player_loop()
        return

    runner = build_runner(agent_conf, bundle.full_conf.obs, bundle.full_conf.world)
    dispatch(runner, config_manager, bundle)
