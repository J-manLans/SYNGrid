from syn_grid.plot.plot_utils import (
    _BASE_LOG_DIR,
    _BASE_PLOT_DIR,
    plot_chain_progression_steps,
    plot_completion_rate,
    plot_episode_length,
    plot_failure,
    plot_reward,
    plot_success,
)

_EVAL = "eval"

# ================= #
#       Main        #
# ================= #

if __name__ == "__main__":
    csv_dir = _BASE_LOG_DIR / _EVAL
    plots_dir = _BASE_PLOT_DIR / _EVAL
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_reward(csv_dir, plots_dir)
    plot_episode_length(csv_dir, plots_dir)
    # plot_average_reward(csv_dir, plots_dir)
    plot_chain_progression_steps(csv_dir, plots_dir)
    # plot_chain_outcomes(csv_dir, plots_dir)
    plot_completion_rate(csv_dir, plots_dir)
    plot_success(csv_dir, plots_dir)
    plot_failure(csv_dir, plots_dir)
