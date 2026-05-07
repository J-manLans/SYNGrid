from syn_grid.plot.plot_utils import (
    _BASE_LOG_DIR,
    _BASE_PLOT_DIR,
    plot_reward,
    plot_average_reward,
    plot_episode_length,
    plot_chain_progression_steps,
    plot_chain_outcomes,
    plot_completion_rate,
)


_TRAIN = "train"

# ================= #
#       Main        #
# ================= #

if __name__ == "__main__":
    csv_dir = _BASE_LOG_DIR / _TRAIN
    plots_dir = _BASE_PLOT_DIR / _TRAIN
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_reward(csv_dir, plots_dir)
    plot_average_reward(csv_dir, plots_dir)
    plot_episode_length(csv_dir, plots_dir)
    plot_chain_progression_steps(csv_dir, plots_dir)
    plot_chain_outcomes(csv_dir, plots_dir)
    plot_completion_rate(csv_dir, plots_dir)
