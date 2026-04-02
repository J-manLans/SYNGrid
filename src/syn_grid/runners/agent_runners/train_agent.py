from syn_grid.config.models import TrainAgentConf
from syn_grid.runners.agent_runners.agent_runner import AgentRunner
from syn_grid.gymnasium.env_factory import make
from syn_grid.utils.paths_util import get_project_path

import datetime
from pathlib import Path
from stable_baselines3.common.monitor import Monitor


# TODO: this only accompanies the stable baselines3 models as of now. We need to crete a more
# modular agent base class that can accompany many different models
def train_agent(runner: AgentRunner, conf: TrainAgentConf) -> None:
    """
    Train an agent, either from scratch or by continuing from a saved checkpoint.

    :param continue_training: If True, the training continues from an existing model checkpoint.
    :type continue_training: bool
    :param agent_steps: The specific checkpoint steps of the model to continue training from.
    :type agent_steps: str
    :param timesteps: Number of steps to train before saving the agent.
    :type timesteps: int
    :param iterations: Number of training loops, each consisting of `timesteps` steps.
    :type iterations: int
    """

    # Get current date and time and create a identifier for unique file naming
    date = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    # Create directories for saving models and logs
    model_dir = Path(get_project_path("output", "models"))
    log_dir = Path(get_project_path("output", "results", "logs"))

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create and wrap the training environment
    env = make(None, runner.run_conf, runner.obs_conf)
    # Wrap the environment with a Monitor for logging.
    # The created csv is needed for plotting our own graphs with matplotlib later.
    monitor_file = Path(log_dir) / f"{runner.identifier}_{runner.algorithm}_{date}.csv"
    env = Monitor(env, filename=str(monitor_file))

    model = None
    if conf.continue_training:
        print("Loading existing training data")
        # Get the model with the desired steps to continue its training
        model = runner.get_model(env)
    else:
        # Initialize a fresh model
        model = runner.AlgorithmClass(
            env=env,
            verbose=1,
            tensorboard_log=str(log_dir),
            **runner.agent_hyper_parameters,
        )

    try:
        # This loop will keep training based on timesteps and iterations.
        # After the timesteps are completed, the model is saved and training
        # continues for the next iteration. When training is done, start another
        # cmd prompt and launch Tensorboard:
        # tensorboard --logdir results/logs/<env_name>
        # Once Tensorboard is loaded, it will print a URL. Follow the URL to see
        # the status of the training.
        for i in range(1, conf.iterations + 1):
            print(f"\nTraining starting for iteration: {i}\n")

            # Train the model
            model.learn(
                total_timesteps=conf.timesteps,
                tb_log_name=f"{runner.identifier}_{runner.algorithm}_{date}",
                reset_num_timesteps=False,
            )

            # Save the model
            save_path = (
                Path(model_dir)
                / f"{runner.identifier}_{runner.algorithm}_{model.num_timesteps}_{date}"
            )
            model.save(save_path)
            print(f"\nModel saved with {model.num_timesteps} time steps")
    finally:
        env.close()
