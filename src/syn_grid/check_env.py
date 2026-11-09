"""
Manual environment sanity check.

Run this after changing anything in the environment implementation
(observation space, action space, reset/step logic) to verify it still
satisfies the Gymnasium API contract. Not part of the training/eval
flow — this is a standalone dev tool, run explicitly when needed.

Usage:
    python -m syn_grid.check_env
"""

from syn_grid.app import load_experiment_configs
from syn_grid.config.config_manager import ConfigManager
from syn_grid.gymnasium.utils.env_factory import check_my_env, make, register_env
from syn_grid.scenario.registry import build_scenario


def main() -> None:
    register_env()

    global_conf, scenario_conf, _ = load_experiment_configs(ConfigManager())
    scenario = build_scenario(global_conf.scenario, scenario_conf)

    env = make(scenario, render_mode=None)
    try:
        check_my_env(env)
        print("Environment is fine.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
