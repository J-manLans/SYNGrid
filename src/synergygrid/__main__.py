import argparse
import sys
import re
from synergygrid import *


def __parse_args():
    def positive_int_str(value: str):
        if not re.fullmatch(r"\d+", value):
            raise argparse.ArgumentTypeError(
                f"{value} is not a positive integer string"
            )
        return value

    parser = argparse.ArgumentParser(description="Run agent experiments.")

    parser.add_argument(
        "--alg",
        type=str,
        default="PPO",
        choices=algorithms,
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
        type=positive_int_str,
        default="0",
        help="Steps of the chosen agent, whole numbers 0 > infinity (--cont must be chosen)",
    )

    parser.add_argument(
        "--timesteps", type=int, default=10000, help="Number of timesteps"
    )

    parser.add_argument(
        "--iterations", type=int, default=10000, help="Number of iterations"
    )

    args = parser.parse_args()

    if args.train and args.steps != "0" and not args.cont:
        parser.error("--steps can only be used if --cont is set")

    return args


def main():
    register_env()
    env = list(environment.keys())[0]
    alg = list(algorithms.keys())

    if len(sys.argv) == 1:
        # Pick algorithm to train or evaluate
        algorithm = alg[0]
        # Choose to use an agent or just random sampling (for debugging the environment)
        agent = True
        # If we want to test the game our selves
        # Choose to train or run the agent
        training = False if agent == False else False
        # Continue training from a saved model
        continue_training = False
        # Model that we shall continue to train
        agent_steps = "716800"
        # Num of timesteps for training or model selection when running
        timesteps = 50000
        # Number of training iterations
        iterations = 15
    else:
        args = __parse_args()  # python -m experiments -h for info
        algorithm = args.alg
        agent = args.agent
        training = args.train
        continue_training = args.cont
        agent_steps = args.steps
        timesteps = args.timesteps
        iterations = args.iterations

    # For human controls
    control = False

    runner = AgentRunner(environment=env, algorithm=algorithm)

    if control:
        SynergyGridEnv(render_mode="human", control=control)
        return

    if training:
        # Train agent
        train_agent(
            runner,
            continue_training=continue_training,
            agent_steps=agent_steps,
            timesteps=timesteps,
            iterations=iterations,
        )
    else:

        # Run environment with agent
        evaluate_agent(runner, agent_steps, agent)


if __name__ == "__main__":
    main()
