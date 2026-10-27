# AGENTS.md

How an agent works with the owner and in this repo: the working rules, the doc map, where each kind of fact goes,
commands and traps. No state, progress, decisions or history: state is in the roadmap's table, decisions are in
`docs/dev/design-decisions.md`, and history is in `git log`. Trust the code over any doc, and say so when they
disagree.

## The project

SYNGrid is a grid, a droid and orbs, and a scenario is a choice of orbs, observation and goal. Read
`docs/vision.md` first for where it is heading, then the roadmap for what is next.

The scenario layer is being rebuilt on `rebuilding-scenario`, file by file: the owner and the agent design each piece
together, the agent writes it, and the owner reviews it.

| Branch | Role |
|---|---|
| `rebuilding-scenario` | the rebuild |
| `cleanup-after-versioning` | last stable state before the rebuild |
| `refactor-of-scenario-refactor` | the reference: read it, do not build on it |
| `main` | older (v0.4.0) |

The pushed branch can be behind the owner's local tree. Read local files.

## The docs

| File | What it is for |
|---|---|
| `docs/vision.md` | Where the project is heading, and the questions to ask of a new idea |
| `docs/dev/scenario-rebuild-roadmap.md` | What is left, in order, and what is stub, legacy or not yet run |
| `docs/dev/design-decisions.md` | What is settled, why, and what was turned down |
| `docs/dev/open-questions.md` | What is not decided yet, by the phase that needs it |
| `docs/dev/codebase-ideas.md` | Ideas worth returning to that are not problems now |
| `docs/dev/glossary.md` | What the words mean. Use its terms |
| `docs/dev/landscape-changes.md` | Changes that could move what the agent sees or is paid, to check against legacy |
| `docs/dev/archive/` | Finished work kept on purpose |
| `docs/code_style.md` | How code is written |
| `CONTRIBUTING.md`, `README.md` | For people outside: how to contribute, what SYNGrid is |

Read `design-decisions.md` and `open-questions.md` before giving a design opinion, and do not treat an open question
as decided.

## Where things go

| When | Update |
|---|---|
| A decision is made | `design-decisions.md`, in the present tense |
| A question is answered | remove it from `open-questions.md`; the answer goes to `design-decisions.md` |
| A question comes up | `open-questions.md`, under the phase that needs it |
| A roadmap step is done | remove it from the roadmap, and update the has-it-run cell in its table |
| Code is first run, or first learns | the has-it-run cell in the roadmap's table |
| A change touches reward, observation, layout or how often something appears | `landscape-changes.md` |
| A word is coined | `glossary.md` |
| An idea that is not a problem now | `codebase-ideas.md` |
| Finished work worth keeping | `docs/dev/archive/`, on purpose |
| What was done, and what was run | the commit message |

Each file in `docs/dev/` opens with what it is for and what stays out of it. Read that before adding to it.

## Commits

A commit message says what changed, why, and what was run to check it, or that nothing was run. The dated progress
notes the docs used to carry belong here, in the commit that made them true. The owner leaves commit messages mostly
to the agent, so they carry the record.

**Never commit code before the owner says so.** This is a hard rule: the owner reviews changes before they are
committed. The exception is documentation, which the agent may commit when it is done.

## How to work with the owner

- **Design together, the agent writes, the owner reviews.** A design is talked through first and the owner decides.
  The agent then writes the code and the docs that go with it. Do not offer "which should I implement" choices.
- **When the direction is clear enough, go ahead and say what was done.** Ask "shall I?" in one line only when the
  choice is genuinely open. Clear enough means the owner has stated a preference, even loosely, not that one can be
  guessed.
- **Spitball before verdicts.** In a design talk, open with what the code does and one question, and reason step by
  step together. Do not come back with a numbered list of calls. Give an opinion when asked and label it a lean. A
  restated option is not a decision: something is decided when the owner says so.
- **Confirm before refining.** When the owner's summary is right, say yes plainly first. Add a detail only when it
  changes the picture.
- **Flag workarounds, argue for hunches.** Say when a proposal patches a limitation, and name the option of removing
  the limitation. When the owner says something feels off, argue for their hunch before defending the proposal. Do
  not present fluent reasoning as experience to defer to.
- **Audit and discuss before writing.** Say what exists, what is missing and what you would create, then wait.
- **Pull first.** Before the first edit of a session, say once "pull first, then we start", and wait for the owner
  to confirm.
- **Agree on where a new file or folder goes before creating it.** Show the tree it gives and say what sort of thing
  lives at that level. The owner reads the structure to picture the design. This covers renames and moves.
- **Make the work reviewable.** After writing, list each file touched and what changed in it, say what was run and
  what was only read, and name anything written against code that does not exist yet.
- **Plain language.** Report what you verified and what you did not.
- **Mention anything surprising, but do not fix it unless asked.**
- **Docstrings say what a thing is and does.** No `STUB:`, "not written yet", "until phase N", "for now", "used to
  be" or "here and not there". A docstring must still read correctly when the rebuild is done.
- **Work still to do goes in a `# TODO` or `# NOTE` comment,** which the owner can search for. Put a tag in brackets
  when several belong together, as in `# TODO(delay):`.
- **Do not add, migrate or rewrite tests unless asked.** The owner curates the suite. A test that is asked for must
  fail when its behaviour is broken: reintroduce the defect and confirm it goes red.

## The two strains

The working strain is `src/syn_grid/`. A frozen copy of the stable code is in `src/syn_grid/legacy/`, so it can be
trained as it was and its curves compared with the rebuild. Each legacy package and loose file has a `_legacy` suffix
and imports only other legacy copies. `USE_LEGACY` in `src/syn_grid/__main__.py` picks the strain. Nothing under
`legacy/` is edited, and it is deleted when the rebuild is done.

Legacy is a reference and not a target. Several decisions change rewards on purpose, so the rebuild is held against
legacy's learning curves and not its rewards step for step.

- Only one strain is imported per process. Both register the Gymnasium id `syn_grid-v0`.
- The legacy config names its images as `assets/sprites/...` while the folder is `assets_legacy/`, so legacy
  rendering stops with a `FileNotFoundError`.
- The legacy run id has its own scheme. Give legacy runs their own tag or output folder.
- The config schemas differ, so a comparison means something only if both configs describe the same world. Compare
  several seeds per side.

## Reading the reference

Read a file from `refactor-of-scenario-refactor` with `git show refactor-of-scenario-refactor:src/syn_grid/<path>`.
Its traps:

- `delay_on_consume` is `int | None`: `None` means off and `False` turns it on.
- A `negative` block in a tier-chain config registers a digester but creates no orbs.
- Scoring mode is passed by each builder, not set in config.
- Threshold scoring and `NegativeDigester` were never executed there.
- `TierChainTermination` is cut to what spatial reaches. The removed branches are in
  `scenario/rules/termination.py` at `c105f7a`.
- Five perceptions beyond the `Perception` enum read a config that is gone, and `WeightedPopulation` names a class
  that is gone.
- Its `docs/dev/scenario-refactor.md` is stale.

`docs/dev/rppo-regression.md` is on `cleanup-after-versioning` only. It investigates why spatial learned worse after
the first refactor; read it before touching the reward path.

## Running things

`import syn_grid` resolves through the editable install, which can point at another checkout. Check first:

```bash
.venv/bin/python -c "import syn_grid; print(syn_grid.__file__)"
```

`PYTHONPATH=src` overrides it for one command.

```bash
pip install -e ".[dev]"                    # required
python3 -m syn_grid                        # run; USE_LEGACY in __main__.py picks the strain
python3 -m syn_grid.check_env              # Gymnasium contract self-check
ruff check src/syn_grid tests              # lint, CI scope
./scripts/tidy.sh                          # check --fix + format
```

`python` is not on PATH outside the venv. `requires-python` is `>=3.10,<3.11`. Rendering needs
`SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy` on a machine without devices.

## Verify before asserting

Each of these cost a wrong conclusion in this repo.

- **A config value's effective setting is in the code that consumes it,** not where it appears in YAML.
- **When comparing, confirm the variable was free to vary.** Pinning it makes "no change" vacuous.
- **A passing diff means the code agreed, not that the question was asked.** Check the resolved config.
- **No visible prompt does not mean no permission decision.** Prompts are client-side.
- **Written is not run.** Check the has-it-run column in the roadmap before saying something works.
- For where a behaviour came from or when it changed, prefer `git log -S`, a replay probe or loading the artifact
  over inference from a name or a comment.
- The replay harness runs the legacy strain. Pin `starting_score` high, or episodes end early and skip the timeout
  branch. Comparisons are exact, so read the size of a difference before calling it behavioural.

## Traps

- **`GridWorld.reset` needs an rng.** Pass `np.random.default_rng(seed)`, or let the environment pass its seeded
  `np_random`.
- **Every perception returns the same buffer each step,** so `obs_t is obs_{t+1}`. Anything that keeps two
  observations must copy.
- **Tests and `scripts/replay_probe.py` import the legacy strain.**
- **`runners/agent_runners/utils/extractors.py` is imported by nothing** and reads observation keys from before the
  rebuild.
- **`reproduction_package/` is committed and large.** Do not add to it. It records what the thesis runs did.
- **`plot/` is a thesis scratch module** with hardcoded paths evaluated at import, and needs the `extra`
  dependencies.
- **A pygame window and a model load crash together on the owner's machine.** Opening a window loads Mesa's LLVM on
  the AMD iGPU, and Triton's own LLVM then segfaults when SB3 loads a model. It hits eval with a renderer and training
  with one. Start through `scripts/run_app.sh` or the `launch.json` entry, which put the window on the NVIDIA card, or
  set `SDL_FRAMEBUFFER_ACCELERATION=0`. See `docs/dev/archive/pygame-triton-segfault.md`.
