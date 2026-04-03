from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models import ExperimentConfig, FullConf
from syn_grid.runners.agent_runners.agent_runner import AgentRunner
from syn_grid.runners.agent_runners.train_agent import train_agent
from syn_grid.runners.agent_runners.evaluate_agent import evaluate_agent
from syn_grid.utils.args_utils import parse_args, update_agent_conf_from_args
from syn_grid.runners.human_runner.human_runner import HumanRunner

import sys


def main():
    # Load full experiment configuration and snapshot settings
    config_manager = ConfigManager("configs.yaml")
    full_conf = config_manager.load_config(FullConf)
    experiments_conf = config_manager.load_config(ExperimentConfig)

    # Save a snapshot if snapshot is enabled
    if experiments_conf.snapshot.enabled:
        config_manager.save_snapshot(full_conf, experiments_conf.snapshot.id)
        print(f"Config snapshot saved. Exiting.")
        return

    # Extract individual configs for use
    run_conf = full_conf.run
    obs_conf = full_conf.obs
    agent_conf = full_conf.agent

    # Parse command-line arguments and update agent config for any overrides
    if len(sys.argv) > 1:
        args = parse_args()
        update_agent_conf_from_args(args, agent_conf)

    # Initialize the appropriate runner:
    # - HumanRunner if manual control is enabled
    # - AgentRunner otherwise
    if agent_conf.global_agent_conf.human_control:
        runner = HumanRunner(run_conf, obs_conf.observation_handler.max_steps)
        runner.human_player_loop()
    else:
        runner = AgentRunner(agent_conf.global_agent_conf, run_conf, obs_conf)

        # Decide between training or evaluating the agent based on config
        if agent_conf.global_agent_conf.training:
            train_agent(runner, agent_conf.train_agent_conf)
        else:
            evaluate_agent(runner, agent_conf.eval_agent_conf)


if __name__ == "__main__":
    main()
