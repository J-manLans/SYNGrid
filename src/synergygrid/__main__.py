import sys
from synergygrid.agentrunner.base import AgentRunner
from synergygrid.agentrunner.train import train_agent
from synergygrid.agentrunner.eval import evaluate_agent
from synergygrid.utils.parse_args import parse_args
from synergygrid.gymnasium.environment import SYNGridEnv


def main():
    if len(sys.argv) == 1:
        # Pick algorithm to train or evaluate
        algorithm = 0
        # Choose to use an agent or just random sampling (for debugging the environment)
        agent = True
        # If we want to test the game our selves
        # Choose to train or run the agent
        training = True
        # Human control to test hte game
        human_control = False
        # Continue training from a saved model
        continue_training = False
        # Model that we shall continue to train
        agent_steps = "12288"
        # Num of timesteps for training or model selection when running
        timesteps = 5000
        # Number of training iterations
        iterations = 2
    else:
        args = parse_args()  # python -m experiments -h for info
        algorithm = args.alg
        agent = args.agent
        training = args.train
        human_control = args.human_controls
        continue_training = args.cont
        agent_steps = args.steps
        timesteps = args.timesteps
        iterations = args.iterations

    runner = AgentRunner(algorithm, "test")

    if human_control:
        SYNGridEnv(render_mode="human", human_control=human_control)
    elif training:
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
