import sys
from synergygrid import (
    algorithms,
    environment,
    register_env,
    AgentRunner,
    train_agent,
    evaluate_agent,
    SYNGridEnv,
    parse_args
)

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
        training = True
        # Continue training from a saved model
        continue_training = False
        # Model that we shall continue to train
        agent_steps = "1505280"
        # Num of timesteps for training or model selection when running
        timesteps = 100000
        # Number of training iterations
        iterations = 15
        human_control = False
    else:
        args = parse_args()  # python -m experiments -h for info
        algorithm = args.alg
        agent = args.agent
        training = args.train
        continue_training = args.cont
        agent_steps = args.steps
        timesteps = args.timesteps
        iterations = args.iterations
        human_control = args.human_controls

    runner = AgentRunner(environment=env, algorithm=algorithm)

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
