from synergygrid.config.configs import algorithms
import argparse
from argparse import Namespace
import re

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Run agent experiments.")

    parser.add_argument(
        "--alg",
        type=int,
        default=0,
        choices=range(len(algorithms)),
        help="Algorithm index",
    )

    parser.add_argument(
        "--no-agent",
        action="store_false",
        dest="agent",
        help="Use random sampling of action (default: use agent)",
    )
    parser.set_defaults(agent=True)

    parser.add_argument(
        "--run",
        action="store_false",
        dest="train",
        help="Run a trained agent (default: train it instead)",
    )
    parser.set_defaults(train=True)

    parser.add_argument(
        "--cont", action="store_true", help="Continue training from a saved model"
    )

    parser.add_argument(
        "--steps",
        type=_positive_int_str,
        default="0",
        help="Steps of the chosen agent, whole numbers 0 > infinity (--cont must be chosen)",
    )

    parser.add_argument(
        "--timesteps", type=int, default=10000, help="Number of timesteps"
    )

    parser.add_argument(
        "--iterations", type=int, default=10000, help="Number of iterations"
    )

    parser.add_argument(
        "--human_controls",
        action="store_true",
        help="If set, you will be in control of the game",
    )

    args = parser.parse_args()

    if args.train and args.steps != "0" and not args.cont:
        parser.error("--steps can only be used if --cont is set")

    return args


# ================= #
#      Helpers      #
# ================= #


def _positive_int_str(value: str):
    if not re.fullmatch(r"\d+", value):
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer string")
    return value
