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
        policy_kwargs = {
            "features_extractor_class": TinyGridCNN,
            "features_extractor_kwargs": {"features_dim": 64},
            "normalize_images": False
        }

        model = runner.AlgorithmClass(
            env=env,
            verbose=1,
            tensorboard_log=str(log_dir),
            policy_kwargs = policy_kwargs,
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

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TinyGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[-1]  # channels last: (H, W, C)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape after conv layers
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            # SB3 expects channels last, but Conv2d expects channels first
            sample = sample.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Convert from channels-last (H, W, C) to channels-first (C, H, W)
        x = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(x))