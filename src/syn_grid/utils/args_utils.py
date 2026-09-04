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

    # Imported lazily: ALGORITHMS pulls in the full SB3/gymnasium chain,
    # and it's only needed here for --alg-index's choices range. Keeping
    # it out of the module-level imports lets other functions in this
    # module (and anything importing them) stay lightweight.
    from syn_grid.runners.agent_runners.agent_registry import ALGORITHMS

    parser = argparse.ArgumentParser(description="Run agent experiments.")

    # === Global values === #

    parser.add_argument(
        "--alg-index",
        type=int,
        default=None,
        choices=range(len(ALGORITHMS)),
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


def args_to_overrides(args: Namespace) -> dict:
    """
    Convert parsed CLI arguments into a plain overrides dict.

    Filters out arguments left at their `None` sentinel default, so only
    values explicitly set by the user are included. The resulting dict
    has the same shape `apply_overrides` (see `syn_grid.config.overrides`)
    expects regardless of where it came from, which lets other override
    sources (e.g. a future GUI) reuse the same downstream code path
    without touching argparse.

    Args:
        args: Parsed CLI arguments from `parse_args()`.

    Returns:
        A dict of {field_name: value} for every explicitly set argument.
    """

    return {key: val for key, val in vars(args).items() if val is not None}
