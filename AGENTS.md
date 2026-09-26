# AGENTS.md

Guidance for AI agents working in this repository. Verified against the repo at
`cleanup-after-versioning`; trust executable config over prose.

## Setup and commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # required; see the editable-install trap below
```

```bash
pytest                                          # full suite
pytest tests/core/orbs -v                       # subset
pytest tests/gymnasium/test_environment.py::TestX::test_y   # single test
ruff check src/syn_grid tests                   # lint  (CI scope — not the whole repo)
ruff format --check src/syn_grid tests          # format
./scripts/tidy.sh                               # check --fix + format; needs venv on PATH
python3 -m syn_grid                             # run (there is no [project.scripts] entry point)
python3 -m syn_grid.check_env                   # gymnasium contract self-check
```

There is no Makefile, no pre-commit config, and no single test entry point.

## The editable-install trap (check this first)

`import syn_grid` resolves through the editable install, which can point at a
**different checkout**. A sibling worktree at `../syngrid-known-good` has been
the target in the past, which makes local runs silently test the wrong code —
or fail collection entirely with 12 `ModuleNotFoundError`s, because the two
checkouts have different module layouts.

Always confirm before trusting any test or run:

```bash
python -c "import syn_grid; print(syn_grid.__file__)"
```

To probe a revision without touching the venv, `PYTHONPATH=src` takes precedence
over the `.pth` file. `PYTHONPATH=../syngrid-known-good/src` does the same for
that worktree. Library versions are shared, so this isolates code only.

## Config system traps

Config is **inside the package**: `ConfigManager("configs.yaml")` resolves via
`get_syn_grid_path`, so the default config is `src/syn_grid/config/configs.yaml`.
Editing it requires an editable install, and there is no `--config` flag yet.

- **No `ConfigDict` anywhere in `src/`** → pydantic defaults to `extra='ignore'`.
  Unknown keys are silently dropped. `check_env:` appears in every shipped YAML
  and is read by nothing. A typo in a config key trains on a different
  experiment than the file describes, with no warning.
- **YAML anchors couple unrelated values.** `step_penalty: &step_penalty` is
  reused as `tier_consumption_penalty: *step_penalty`. Changing the step penalty
  silently rescales the tier-consumption penalty (it moved 1000× in one historical
  commit). Break the alias before tuning penalties.
- **Fields are duplicated across blocks and must agree**, but only grid dimensions
  are cross-validated. `grid_rows`/`grid_cols` appear in 4 blocks, `max_tier` in 3,
  `max_steps` in 2, plus `single_chain_mode`, `max_active_orbs`, `curriculum_training`.
  Update every copy or the run does not mean what the file says.
- **Scoring modes are mutually exclusive** (`step_wise_scoring`,
  `threshold_scoring`, `max_tier_scoring`), validated on `TierConf`. Changing
  `max_tier_scoring` in `grid_world_conf` does **not** change
  `tier_orb_conf.max_tier_scoring` unless the anchor propagates — the engine
  reads the `tier_orb_conf` copy.
- `human_control: true` routes `app.main()` to `HumanRunner` and skips training
  entirely. `snapshot.enabled` short-circuits: it saves a config copy and exits
  without training or evaluating.

## Global mutable state

`OrbFactory.__init__` sets **class attributes** on every construction:

```python
BaseOrb.set_life_span(grid_rows, grid_cols)   # sets cls._life_span
TierOrb.max_tier = ...                        # sets a class attribute
```

Training uses `n_envs: 16` in a single process, so all envs share this. Two
`GridWorld`s with different geometry will silently rewrite each other's lifespan,
and `base_perception.py` snapshots the value at construction, so a declared
`Box` high can stop matching the timers it emits. Several tests depend on
alphabetical collection order because of this; don't run them in isolation or
under `pytest-xdist` without checking.

## Observations alias one buffer

Every perception returns `self._obs_data` — the same object each step, and
`environment.py` hands that straight back. So `obs_t is obs_{t+1}`. SB3 copies
into rollout buffers so training is unaffected, but anything retaining two
consecutive observations (video recording, logging, `np.array_equal` on history)
gets silently corrupted data. `get_orb_positions()` and friends likewise return
internal lists by reference.

## Test suite

Helpers live in `tests/utils/config_helpers.py`: `get_test_config()` loads
`test_configs.yaml`, `update_conf(conf, {...})` returns a modified copy
(pydantic models are frozen but `model_copy` works).

Two deliberate non-green states — do **not** "fix" these silently:

- `tests/gymnasium/utils/test_episode_termination.py` fails. The test asserts
  `reward == -10.0`; the code hardcodes `reward = -1` at timeout. This is an open
  research decision (which value produced the published figures), tracked in
  `docs/dev/todo.md`.
- 4 `xfail(strict=True)` cases in `tests/core/orbs/test_orb_factory.py` document
  the `OrbFactory._normalize_counts` weight inversion. They will turn XPASS and
  demand attention when the bug is fixed.

## Architecture

`core/` is the simulation and deliberately has **no gymnasium import** — it is
steppable without SB3 installed. `gymnasium/` wraps it and owns spaces, obs
encoding and episode termination. `runners/` owns training/eval, artifact IO and
naming. Config is pydantic v2 models in `config/models.py` loaded from YAML.

`orb_meta.py` is a value object, **not** a metaclass — there is no orb registry.
Adding an orb type means editing `orb_meta.py`, `orb_factory.py`, both
`TypesConf`/`EnabledOrbsConf`, `composite_grid_markovian.py`'s channel map, and
`pygame_renderer.py`. Adding a *perception* needs `observation_handler.PERCEPTIONS`
plus a duplicate string list in `models.py`.

`core/orbs/effects/` contains only a `dummy_file` and is not a package.

`plot/` is a thesis scratch module (hardcoded CWD paths evaluated at import,
`"<-- replace with your actual tag"`), and needs `matplotlib`/`pandas` from the
optional `extra` group that CI never installs. It is being rewritten.

## Repo layout

`reproduction_package/` is **committed and large** — 246 model / vec-norm /
tfevents binaries; the repo packs to ~207 MiB, dominated by it. It is slated for
removal before publication, so do not add to it.

`output/` and `.venv/` are gitignored. Note that `.gitignore` lists `.vscode/`
then `!.vscode/launch.json`; the negation looks like a no-op but `launch.json` is
in fact tracked and `settings.json` correctly ignored.

`pyproject.toml` pins `requires-python = ">=3.10,<3.11"`. Nothing in the code
needs the ceiling, but don't assume a newer interpreter works.

## Replay harness

`scripts/replay_probe.py` replays a fixed seeded action sequence through any
revision and dumps the `(obs, reward, terminated, truncated)` stream;
`replay_sweep.sh` sweeps history via sparse worktrees and `replay_diff.py`
reports the first differing step. Use it to compare revisions without training.

Gotchas learned the hard way, encoded in the script:
- Source the config from `--config-src`, **not** the target revision's own
  `test_configs.yaml` — historical revisions ship YAML with undefined anchors,
  and older schemas require fields the current one dropped.
- Pin `chain_break_penalty` in **both** config blocks; hold it constant when
  comparing revisions, or you will "confirm" nothing.
- Pin `starting_score` high, or the droid's score drains on wall contact and
  every episode ends early — silently skipping the timeout branch entirely.

## Pygame

Constructing a `render_mode="human"` env needs a video device; CI sets
`SDL_VIDEODRIVER=dummy`. `pygame.mixer` is initialised in a `finally` block in
`base_sb3_runner.py` and will raise on machines with no audio device.

## In-flight work

`docs/dev/todo.md` is the current v1.0.0 list, ordered by dependency.
`docs/dev/rppo-regression.md` is an in-flight bug hunt (RPPO stopped reproducing
the May spatial result) with the evidence, ruled-out causes and open suspects —
read it before touching the reward path, and do not resolve the timeout-penalty
question without checking what it records.
`docs/code_style.md` documents conventions, but its §3 mandates Black — the repo
actually uses ruff, and it names a path (`src/synergygrid`) that does not exist.
Trust `scripts/tidy.sh` and CI. `docs/workflow.md` describes a two-person
branch/PR flow.
