# Reproducing the Thesis Experiments
Setup and installation: see [README](../README.md). This file covers reproducing the experiments and verifying the results reported in the thesis.

## Reproducing a run
1. Copy the contents of the relevant scenario YAML file from `reproduction_package/scenario_x/version` into [configs.yaml](/src/syn_grid/config/configs.yaml)
2. Set the seed you want to run in `global_agent_conf` in the Agent Configuration section of the `configs.yaml` file (thesis runs used 3, 7, and 42, and 3 was used for the thesis Figures)
3. Run:

```bash
python3 -m syn_grid
```

Each scenario config is self-contained: agent type, observation, penalties, reward structure, training length, and output folder structure are all defined in the file. No code changes are required to reproduce any scenario.

The seed field in the config seeds both the environment and the SB3 agent.

Training used 16 parallel environments. This is set in the configs — note that it affects both throughput and learning dynamics (rollout batch diversity), so results are not comparable across different environment counts.

## Continuation order (spatial scenario)
The spatial scenario is staged. Grid sizes must be trained in order, as each stage continues from the previous checkpoint. The continuation loading resolves checkpoints by path, so run all three stages with the attached configs as-is — renaming output folders mid-sequence will break the next stage's checkpoint loading. Rename to grid-size-specific names afterwards if desired:

1. `spatial_5x5` — trained from scratch
2. `spatial_6x6` — continues from the 5×5 checkpoint
3. `spatial_7x7` — continues from the 6×6 checkpoint

All other scenarios train from scratch and can be run independently.

## About the YAML config files in the reproduction package
When you enter a scenario (the spatial 5x5 scenario, for example) you will see three YAML files:

- `fsppo.yaml`
- `ppo.yaml`
- `rppo.yaml`

This means the scenario was run with all three models. To retrain one, follow the description above.

Some scenarios contain fewer files — sparse tier scaling at tier 5, for example, has only `ppo.yaml` and `rppo.yaml`. This means only those models were run at that tier, following the probing protocol described in the [thesis](Evaluating%20Syngrid%3A%20A%20Configurable%20Benchmark%20Environment%20for%20Temporal%20Credit%20Assignment%20in%20Reinforcement%20Learning.pdf). The same training procedure as above applies for them.

## Evaluation runs
### Your own re-trained model
To watch them in action, navigate to the `global_agent_conf` in the Agent Configuration section and change `training` to `false` for the model you retrained, the path is auto resolved from the config file itself.

### From the reproduction package
Copy the desired scenario config into `configs.yaml` as described above, and set `training` to `false`. Then copy the corresponding run files from the reproduction package (matching scenario and seed) into the locations the framework loads from:

Example:

```yaml
save_folder: 'reproduction_package/tier_scaling_sparse/ppo'
```

The files for this scenario goes into:
- logs: `output/results/logs/reproduction_package/tier_scaling_sparse/ppo`
- vecnorm stats: `output/models_vec_norms/reproduction_package/tier_scaling_sparse/ppo`
- model checkpoints: `output/models/reproduction_package/tier_scaling_sparse/ppo`

**To reproduce the RQ2 path traces, the config seed must be set to 3**.

## Viewing of training curves
Example:

```bash
tensorboard --logdir reproduction_package/spatial_scenario/
```

This will bring up all runs and seeds for the spatial scenario from 5x5 -> 7x7

### Verifying a thesis figure (new values are stated once, then repeated below with '---')
| Thesis figure/table | Path | Tensorboard regex | Notes |
|---|---|---|---|
| Fig. 5 (spatial reward) | reproduction_package/spatial_scenario | seed3 | |
| Fig. 6 (spatial episode length) | --- | --- | |
| Fig. 7 (tier scaling reward, sparse PPO) | reproduction_package/tier_scaling_scenario | sparse.\*seed3.\*\_ppo\_ |  |
| Fig. 10 (tier scaling length, sparse PPO) | --- | --- | |
| Fig. 8 (tier scaling reward, sparse FSPPO) | --- | sparse.\*seed3.\*\_fsppo\_ |  |
| Fig. 11 (tier scaling length, sparse FSPPO) | --- | --- | |
| Fig. 9 (tier scaling reward, sparse RPPO) | --- | sparse.\*seed3.\*\_rppo\_ | |
| Fig. 12 (tier scaling length, sparse RPPO) | --- | --- | |
| Figs. 13 (tier scaling reward, dense PPO) | --- | dense.\*seed3.\*\_ppo\_ | |
| Figs. 16 (tier scaling length, dense PPO) | --- | --- | |
| Figs. 14 (tier scaling reward, dense FSPPO) |--- | dense.\*seed3.\*\_fsppo\_ | |
| Figs. 17 (tier scaling length, dense FSPPO) | --- | --- | |
| Figs. 15 (tier scaling reward, dense RPPO) | --- | dense.\*seed3.\*\_rppo\_ | |
| Figs. 18 (tier scaling length, dense RPPO) | --- | --- | |
| Figs. 19 (tier scaling reward combined, dense) | --- | dense.*seed3 | |
| Figs. 20 (tier scaling length combined, dense) | --- | --- | |
| Figs. 21 (delay reward, PPO) | reproduction_package/delay_scenario | seed3\*\_ppo\_ | Delays 30, 70, 130, 260, 390 |
| Figs. 22 (delay length, PPO) | --- | --- | --- |
| Figs. 23 (delay reward, PPO and FSPPO comparison) | --- | delay\_390\_seed3.\*(\_ppo\_\|fsppo) | Delay 390 |
| Figs. 24 (delay reward, PPO, FSPPO and RPPO) | --- | delay\_30\_seed3.\*(\_ppo\_\|fsppo\|rppo) | Delay 30 |
| Figs. 25 (delay length, PPO, FSPPO and RPPO) | --- | --- | --- |
| Figs. 26 (delay explained variance, PPO, FSPPO and RPPO) | --- | --- | --- |
| Figs. 27 (delay entropy loss, PPO, FSPPO and RPPO) | --- | --- | --- |

## Notes on reproducibility
- **Reruns within this codebase are exact, with a caveat**. Same configuration and seed produce identical results. Compared to the thesis-era logs, trajectories diverge a little after the initial episodes due to minor post-thesis code revisions without deserting the overall trends. The exceptions are described in the point below.
- **The RPPO result in the spatial scenario could not be re-established** for any of the seeds after post-submission code revisions, the original logs and checkpoints from seed 3 are preserved and remain the authoritative record for the thesis figures. See the [thesis](Evaluating%20Syngrid%3A%20A%20Configurable%20Benchmark%20Environment%20for%20Temporal%20Credit%20Assignment%20in%20Reinforcement%20Learning.pdf) Limitations section for details. The other two agents and most other scenarios reproduce comparable results. However, to what extent wasn't made clear before the submission of the thesis. A rerun of all seeds and tiers, except where it flatlined, were conduced for PPO in the sparse tier scaling scenario to get a notion of the damage. These are the results:

    - tier 4: all seeds converged
    - tier 5: seed 7 and 47 converged
    - tier 6: seed 7 converged

- **Some runs were cut at compute budget rather than at convergence**: This is specifically true to the spatial and delay scenario and for RPPO in general. The corresponding curves should not be read as final performance ceilings and more like trends that show significant distinction between architectures.
- **Hardware**: all experiments ran on a consumer-grade laptop. Stateless PPO and FSPPO trained on CPU; RPPO on GPU. Wall-clock expectations should be set accordingly (the RPPO delay-30 run took ~10 hours).