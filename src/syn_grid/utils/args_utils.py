from syn_grid.config.models import AgentConfig
from syn_grid.runners.agent_runners.agent_runner import AgentRunner

import argparse
from argparse import Namespace


def parse_args() -> Namespace:
    """
    Parse command-line arguments for running the agent.

    All arguments use `None` as a sentinel default, allowing the merge
    with the configuration file to detect which values were explicitly
    set by the user and should override the defaults.

    Run `python -m experiments -h` for detailed usage information.
    """

    parser = argparse.ArgumentParser(description="Run agent experiments.")

    # === Global values === #

    parser.add_argument(
        "--alg-index",
        type=int,
        default=None,
        choices=range(len(AgentRunner.algorithms)),
        help="Algorithm index",
    )

    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        metavar=": str",
        help="Number of steps of the chosen agent",
    )

    parser.add_argument(
        "--id",
        type=str,
        default=None,
        metavar=": str",
        help="Identifier to use for the saved model",
    )

    parser.add_argument(
        "--human_controls",
        dest="human_control",
        action="store_true",
        default=None,
        help="Manually control of the game if set",
    )

    parser.add_argument(
        "--train", action="store_true", default=None, help="Enable training if set"
    )

    # === Train values === #

    parser.add_argument(
        "--cont",
        action="store_true",
        default=None,
        help="Continue training from a saved model if set",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        metavar=": int",
        help="Number of timesteps per iteration (a checkpoint is saved after this many steps)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        metavar=": int",
        help="Number of training iterations",
    )

    # === Eval values === #

    parser.add_argument(
        "--trained-model",
        action="store_true",
        default=None,
        help="Use trained model for eval instead of random sampling if set",
    )

    return parser.parse_args()


def update_agent_conf_from_args(args: Namespace, agent_conf: AgentConfig) -> None:
    """
    Update an AgentConfig with values from parsed CLI arguments.

    Only arguments that are not None will overwrite the corresponding
    fields in the agent's global, training, or evaluation configuration.
    """

    for key, val in vars(args).items():
        if val is not None:
            if hasattr(agent_conf.global_agent_conf, key):
                setattr(agent_conf.global_agent_conf, key, val)
            elif hasattr(agent_conf.train_agent_conf, key):
                setattr(agent_conf.train_agent_conf, key, val)
            elif hasattr(agent_conf.eval_agent_conf, key):
                setattr(agent_conf.eval_agent_conf, key, val)
