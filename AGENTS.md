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
| `docs/dev/design-decisions.md` | What is settled, why, and what was turned down: the principles, and an index of every decision by title. The decisions themselves are in `docs/dev/decisions/`, one file per area |
| `docs/dev/open-questions.md` | What is not decided yet, by the phase that needs it |
| `docs/dev/codebase-ideas.md` | Ideas worth returning to that are not problems now |
| `docs/dev/glossary.md` | What the words mean. Use its terms |
| `docs/dev/code-map.md` | Where the rebuilt code lives, and the path a run and a step take through it. Read it before reading code |
| `docs/dev/landscape-changes.md` | Changes that could move what the agent sees or is paid, to check against legacy |
| `docs/dev/archive/` | Finished work kept on purpose |
| `docs/code_style.md` | How code is written |
| `CONTRIBUTING.md`, `README.md` | For people outside: how to contribute, what SYNGrid is |

Read the `design-decisions.md` index, the decision files for the area in full, and `open-questions.md` before giving
a design opinion, and do not treat an open question as decided. Read `docs/code_style.md` before writing code.

## Where things go

| When | Update |
|---|---|
| A decision is made | its area's file in `decisions/`, in the present tense, and its title in the `design-decisions.md` index |
| A question is answered | remove it from `open-questions.md`; the answer goes to its area's file in `decisions/` |
| A question comes up | `open-questions.md`, under the phase that needs it |
| A roadmap step is done | remove it from the roadmap, and update the has-it-run cell in its table |
| Code is first run, or first learns | the has-it-run cell in the roadmap's table |
| A change touches reward, observation, layout or how often something appears | `landscape-changes.md` |
| A word is coined | `glossary.md` |
| A file or folder is added, moved or renamed, or a run's or a step's path changes | `code-map.md`, in the same change |
| An idea that is not a problem now | `codebase-ideas.md` |
| Finished work worth keeping | `docs/dev/archive/`, on purpose |
| What was done, and what was run | the commit message |

Each file in `docs/dev/` opens with what it is for and what stays out of it. Read that before adding to it.

A fact is written in the file that owns it, by the table above, and another file that needs it points there instead
of repeating it. The traps below are the exception: each stays a short warning here, and the owning file holds the
detail.

## The docs sweep

A pass over the docs that ties them back together at the end of a session. When the owner signals that the session is
ending, say once, "want a docs sweep before we stop?", and drop it if they decline. The owner can also ask for one at
any time.

A sweep is four checks:

1. The roadmap and `open-questions.md` against what was done: remove finished steps and answered questions, and update
   the has-it-run cells.
2. Decisions from the session that are missing from `decisions/`, or from the `design-decisions.md` index.
3. New terms that belong in `glossary.md`.
4. Anything that landed in the wrong file, by the table above.

Say what changed, and flag borderline calls instead of guessing. Whether something is a decision or still open, and
whether it is an idea or a problem, is the owner's to say.

If the owner says to go ahead and leaves, do only the mechanical parts: removing finished items, moving things to the
right file, adding glossary terms. Leave a borderline call as it is and list it in the commit message, so it waits for
the owner. Then commit the docs.

A sweep has no subject of its own: it finishes the session's work. So it is almost always amended into the session's
last commit, under the rule in "Commits", with its borderline calls added to that commit's message. It is a commit
of its own only when that commit is pushed or is the owner's.

## Commits

A commit message says what changed, why, and what was run to check it, or that nothing was run. The dated progress
notes the docs used to carry belong here, in the commit that made them true. The owner leaves commit messages mostly
to the agent, so they carry the record.

**Never commit code before the owner says so.** This is a hard rule: the owner reviews changes before they are
committed. The exception is documentation, which the agent may commit when it is done. Comments and docstrings
the owner asks for count as documentation, even inside a code file. Code is anything that does something, in a `.py`
file, YAML or any other file, and it waits for review. When a coding job also updates the docs, stage and commit the
doc files on their own and leave every code file uncommitted for review. A file that mixes the two waits whole.

Amend the agent's own last commit when the change finishes or corrects it and it has not been pushed; otherwise
make a new commit. When in doubt, ask the owner in one line whether to amend. A docs-only session is one commit,
amended as the session goes, unless the owner says otherwise.

## How to work with the owner

- **Design together, the agent writes, the owner reviews.** A design is talked through first and the owner decides.
  The agent then writes the code and the docs that go with it. Do not offer "which should I implement" choices.
- **When the direction is clear enough, go ahead and say what was done.** Ask "shall I?" in one line only when the
  choice is genuinely open. Clear enough means the owner has stated a preference, even loosely, not that one can be
  guessed.
- **Spitball before verdicts.** In a design talk, open with what the code does and one question, and reason step by
  step together. Do not come back with a numbered list of calls, unless the owner asks for a recommendation. When a
  choice is open, give a lean with it, labelled as a lean and with its reason, even unasked: the owner wants to hear
  it. A restated option is not a decision: something is decided when the owner
  says so.
- **Confirm before refining.** When the owner's summary is right, say yes plainly first. Add a detail only when it
  changes the picture.
- **Flag workarounds, argue for hunches.** Say when a proposal patches a limitation, and name the option of removing
  the limitation. When the owner says something feels off, argue for their hunch before defending the proposal. Do
  not present fluent reasoning as experience to defer to.
- **Audit and discuss before writing.** Say what exists, what is missing and what you would create, then wait.
- **Pull first.** Before the first edit of a session on the owner's local machine, say once "pull first, then we
  start", and wait for the owner to confirm. An agent in a cloud or web session starts from a fresh clone and does
  not ask.
- **`scripts/` is the agent's.** The probes, the sweep and the helper scripts are the agent's to design, change,
  move and tidy without asking. The owner learns how they work before a 1.0 release. A change to one is still listed
  in the report and the commit message.
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
- **Every code change ships with its tests,** reviewed together with the code. Do not migrate or rewrite old tests
  unless asked: the suite is rebuilt from scratch, see the roadmap. Where tests go and how they are grouped is in
  `docs/code_style.md`. How they are written is in "Writing tests" below.

## Writing tests

New rules get their tests before their code, and every change ends with a sweep of the files it touched. Why:
"Tests come first, from the owner's rules" in `docs/dev/design-decisions.md`.

### New code: tests first

1. The owner states the rules as behaviour: "when X happens, Y must happen", or "Y must never happen". Add the edge
   cases and rules the owner missed, and ask where a rule is unclear.
2. Close in on the rules together before writing anything, also when they come as a finished prompt. Review them
   against the code and give an opinion: what is missing, what is unclear, and how each rule would be tested. Write
   the tests once the owner and the agent agree.
3. Write the tests, then stop. The owner reviews them before any code is written. They should read as the owner's
   rules.
4. Write the code with the tests as the guide, and see each test go red before it goes green.

Tests go through the public entry points the rest of the code uses, and describe behaviour, not structure. Code whose
design is still moving, such as a mechanic under ablation, is tried first and gets its tests once its rule is settled
in `design-decisions.md`.

### The sweep

Every change ends with it, over the source files it touched, and tightening an existing test file is the same sweep.
coverage and mutmut are a reading aid, not targets: a file is done when every survivor and every uncovered line is
either tested or listed with its reason.

1. **Run** pytest, the regression test included, so no locked scenario drifted. Then coverage of each touched file
   (`coverage run -m pytest`, then `coverage report -m`), and mutmut on it. mutmut runs only `tests/unit` and
   `tests/integration`, so a survivor means no targeted test guards that code.
2. **Sort** each survivor and each uncovered line by one question: would a caller, through the public surface,
   notice if this broke?
   - Yes: a real gap, which gets a test.
   - No behaviour can change: an equivalent mutant, listed and not tested.
   - Nobody decided what should happen: a question for the owner, not a test. A test would decide it.
   - Nothing depends on the code, such as a defensive branch or output text: listed, not propped up with a test.
   - The observation and render packages: skipped, since they are rebuilt.
3. **Test the behaviour, not the mutant.** A test that reads a private attribute only to kill a mutant pins how the
   code works, not what it does.
4. **Confirm it goes red** with its mutant (`mutmut run <name>`), or, where mutmut does not reach, with the code
   broken by hand and restored.
5. **Look for redundant tests,** only when a file's tests are tightened, not on every change.
   `coverage html --show-contexts` lists on each line the tests that ran it. A test
   whose lines all list others too, and which checks the same outcome, is a suspect: comment it out and run mutmut
   on the file again. If nothing new survives, propose removing it.
6. **Report** per file: coverage and mutmut before and after, the tests added or removed, and what is left with its
   reason, briefly.

Traps:

- mutmut does not mutate a decorated method: one marked `@override`, which most strategy methods are, or a pydantic
  validator. A clean mutmut run says nothing about them. Break each one in a touched file by hand, run the tests,
  restore it, and say in the report which were checked that way.
- After a change to `[tool.mutmut]`, `rm -rf mutants` before the next run, or mutmut keeps the old copy.
- A test that builds a runner works in `tmp_path` with `monkeypatch.chdir`, so its `output/` lands there.
- The run controls read pygame's queue in one batch and drop what follows Space in it. An event meant for the pause
  screen is posted with `pygame.time.set_timer`.

## The two strains

The working strain is `src/syn_grid/`. A frozen copy of the stable code is in `src/syn_grid/legacy/`, so it can be
trained as it was and its curves compared with the rebuild. Each legacy package and loose file has a `_legacy` suffix
and imports only other legacy copies. `USE_LEGACY` in `src/syn_grid/__main__.py` picks the strain. Nothing under
`legacy/` is edited, and it is deleted when phase 7b's perceptions are done.

Legacy is a reference and not a target. Several decisions change rewards on purpose, so the rebuild is held against
legacy's learning curves and not its rewards step for step.

- Only one strain is imported per process. Both register the Gymnasium id `syn_grid-v0`.
- The legacy config names its images as `assets/sprites/...` while the folder is `assets_legacy/`, so legacy
  rendering stops with a `FileNotFoundError`.
- The legacy run id has its own scheme. Give legacy runs their own tag or output folder.
- The config schemas differ, so a comparison means something only if both configs describe the same world.

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
coverage run -m pytest && coverage report -m   # coverage, with branches and the probe's subprocess
HYPOTHESIS_PROFILE=dev pytest              # the invariants searched far; plain pytest runs CI's small profile
./scripts/tidy.sh                          # check --fix + format
```

`python` is not on PATH outside the venv. `requires-python` is `>=3.12,<3.13`. Rendering needs
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
- The replay probe (`scripts/replay/probe.py`) records a locked scenario from its base config, so a run's tunables do
  not move it; `--run-yaml` records it as a run sees it. The regression test checks each locked scenario against its
  baseline in `tests/regression/baselines/`. Comparisons are exact, so read the size of a difference before calling
  it behavioural.
- **A change meant to leave behaviour alone is probed before and after,** for each scenario whose code it touches,
  and compared with `scripts/replay/diff.py`. Say whether the traces matched and what the probe does not reach:
  rendering, logging, the runners, and any branch the tape's random walk does not hit.
- **A change to the runners is fingerprinted before and after** with `scripts/runner_probe.py`, which trains each
  algorithm briefly with a seed, saves and loads it through `train()` and `eval()`, and digests the weights, the eval
  actions and the CSV rows. The replay probe does not reach the runners, and neither probe reaches the run controls
  or video recording.

## Traps

- **`GridWorld.reset` needs an rng.** Pass `np.random.default_rng(seed)`, or let the environment pass its seeded
  `np_random`.
- **`runners/agent_runners/utils/extractors.py` is imported by nothing** and reads observation keys from before the
  rebuild. It stays on purpose: phase 7b works from it, not from scratch.
- **`reproduction_package/` is committed and large.** Do not add to it. It records what the thesis runs did, and it
  stays until the owner decides how it connects to the thesis.
