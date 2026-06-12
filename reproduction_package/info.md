# Reproducing the Thesis Experiments

Setup and installation: see [README](../README.md). This file covers reproducing the experiments and verifying the results reported in the thesis.

## Reproducing a run

1. Copy the contents of the relevant scenario file from `reproduction_package/scenario_x/` into `configs.yaml`
2. Set the seed in the Agent Configuration section (thesis runs used 3, 7, and 42, and 3 was used for the thesis Figures)
3. Run:

```bash
python3 -m syn_grid
```

Each scenario config is self-contained: agent type, observation, penalties, reward structure, training length, and output folder structure are all defined in the file. No code changes are required to reproduce any scenario.

The seed field in the config seeds both the environment and the SB3 agent. <!-- VERIFY: confirm this covers all RNG consumers (env, SB3, torch/numpy) -->

Training used 16 parallel environments. This is set in the configs — note that it affects both throughput and learning dynamics (rollout batch diversity), so results are not comparable across different environment counts.

## Continuation order (spatial scenario)

The spatial scenario is staged. Grid sizes must be trained in order, as each stage continues from the previous checkpoint:

1. `spatial_5x5` — trained from scratch
2. `spatial_6x6` — continues from the 5×5 checkpoint
3. `spatial_7x7` — continues from the 6×6 checkpoint

All other scenarios train from scratch and can be run independently.

## Verifying a thesis figure

| Thesis figure/table | Config | Notes |
|---|---|---|
| Fig. X (spatial reward) | `spatial_5x5_<agent>.yaml` → `6x6` → `7x7` | all three agents |
| Fig. X (spatial episode length) | same as above | |
| Fig. X (tier scaling, sparse) | `tier_scaling_sparse_t<N>_<agent>.yaml` | PPO probed t4–t7; FSPPO t6–t7; RPPO t5–t6 |
| Fig. X (tier scaling, dense) | `tier_scaling_dense_t<N>_<agent>.yaml` | at each agent's failing tier |
| Table X (human vs. agent) | — | human baseline numbers in `<path>` |
| Fig. X (delay probing) | `delay_<D>_ppo.yaml` | D ∈ {30, 70, 130, 260, 390} |
| Fig. X (delay 390, PPO vs FSPPO) | `delay_390_{ppo,fsppo}.yaml` | |
| Fig. X (delay 30, all agents) | `delay_30_<agent>.yaml` | |
| Figs. X (RQ2 path traces) | evaluation runs from trained checkpoints, see below | |

Each config defines its own output folder. Training results (CSV logs, TensorBoard events, checkpoints, VecNormalize statistics) for all three seeds are included per run.

### Evaluation runs

The behavioral observations in RQ2 (path traces, looping counts) come from fixed-seed evaluation runs of trained checkpoints:

```bash
<evaluation command here>
```

## Trained models

Model checkpoints and VecNormalize statistics are too large for the repository and are attached to [Release vX.X](<release-link>).

## Notes on reproducibility

- **Exact trajectories diverge after roughly 10 episodes** even under identical seeds, due to RNG/library nondeterminism across environments. The reward model has been verified against the original logs to floating-point precision episode-by-episode up to the divergence point. Aggregate learning curves are the meaningful comparison standard.
- **The RPPO result in the spatial scenario could not be re-established** after post-submission code revisions; the original logs and checkpoints are preserved and remain the authoritative record for the thesis figures. See the thesis Limitations section for details. The other two agents and all other scenarios reproduce comparable results.
- **Some runs were cut at compute budget rather than at convergence**: delay 130 and above (RPPO), and the extended stateless PPO probe in the spatial scenario. The corresponding curves should not be read as final performance ceilings.
- **Hardware**: all experiments ran on a consumer-grade laptop. Stateless PPO and FSPPO trained on CPU; RPPO on GPU. Wall-clock expectations should be set accordingly (the RPPO delay-30 run took ~10 hours).