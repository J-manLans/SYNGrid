"""
Manual environment sanity check.

Run this after changing anything in the environment implementation
(observation space, action space, reset/step logic) to verify it still
satisfies the Gymnasium API contract. Not part of the training/eval
flow — this is a standalone dev tool, run explicitly when needed.

Usage:
    python -m syn_grid.check_env
"""

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models import FullConf
from syn_grid.gymnasium.utils.env_factory import register_env, make, check_my_env


def main() -> None:
    register_env()

    config_manager = ConfigManager("configs.yaml")
    full_conf = config_manager.load_config(FullConf)

    env = make(None, full_conf.world, full_conf.obs)
    try:
        check_my_env(env)
        print("Environment is fine.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
