# AGENTS.md

How an agent works with the owner and in this repo: the working rules, the doc map, where each kind of fact goes,
commands and traps. No state, progress, decisions or history: state is in the roadmap's table, decisions are in
`docs/dev/design-decisions.md`, and history is in `git log`.

## Hard rules

Breaking one of these costs the owner something. Each is detailed in the section named.

- Never commit code before the owner says commit or push. Docs about the code wait staged with it; other docs are
  committed. (Commits)
- Never edit anything under `src/syn_grid/legacy/`. (The two strains)
- Trust the code over any doc, and say so when they disagree.
- On the owner's machine, say "pull first, then we start" and wait before the first edit. (How to work with the owner)
- Agree on where a new file or folder goes before creating it, renames and moves included. (How to work with the
  owner)

## The project

SYNGrid is a grid, a droid and orbs, and a scenario is a choice of orbs, observation and goal.

The scenario layer is being rebuilt, file by file: the owner and the agent design each piece together, the agent
writes it, and the owner reviews it. `main` holds the rebuild so far, and each phase is worked on a small branch of its
own off `main`, merged back through a pull request with "Rebase and merge", so `main` stays a straight line.

The pushed branch can be behind the owner's local tree. Read local files.

## The docs

| File | What it is for |
|---|---|
| `docs/vision.md` | Where the project is heading, and the questions to ask of a new idea |
| `docs/dev/scenario-rebuild-roadmap/` | What is left, in order, and what is stub, legacy or not yet run. `README.md` holds **Next**, the table and the phases; each phase is a file |
| `docs/dev/design-decisions.md` | What is settled, why, and what was turned down: the principles, and an index of every decision by title. The decisions themselves are in `docs/dev/decisions/`, one file per area |
| `docs/dev/open-questions.md` | What is not decided yet, by the phase that needs it |
| `docs/dev/codebase-ideas/` | Ideas worth returning to that are not problems now, one file per idea. Its `README.md` is the index and the only part read outside a docs sweep |
| `docs/dev/glossary/` | What the words mean, one file per area. Its `README.md` indexes every term. Use its terms |
| `docs/dev/code-map.md` | Where the rebuilt code lives, and the path a run and a step take through it. Read it before reading code |
| `docs/dev/landscape-changes.md` | Changes that could move what the agent sees or is paid, to check against legacy |
| `docs/dev/procedures/` | How to do a job only some sessions need: writing tests, the test sweep, checking behaviour, the docs sweep, running legacy. Read the one the job needs, before starting it |
| `docs/dev/archive/` | Closed investigations and dropped ideas worth keeping, with an index in its `README.md` |
| `docs/code_style.md` | How code is written |
| `CONTRIBUTING.md`, `README.md` | For people outside: how to contribute, what SYNGrid is |

## Starting a session

Read in this order, and do not ask the owner what these already answer:

1. `docs/vision.md`, for where the project is heading.
2. The roadmap's `README.md`, for what is next and what has run, and the file of the phase its **Next** line names.
3. Before a design opinion: the `design-decisions.md` index, the decision files for the area in full,
   `open-questions.md`, and the index in `docs/dev/archive/README.md`, so a dropped idea is not offered again. Do
   not treat an open question as decided.
4. Before reading code, `docs/dev/code-map.md`. Before writing code, `docs/code_style.md` and
   `docs/dev/procedures/writing-tests.md`.

## Where things go

| When | Update |
|---|---|
| A decision is made | its area's file in `decisions/`, in the present tense, and its title in the `design-decisions.md` index |
| A question is answered | remove it from `open-questions.md`; the answer goes to its area's file in `decisions/` |
| A question comes up | `open-questions.md`, under the phase that needs it |
| A roadmap step is done | remove it from its phase's file, and update the has-it-run cell in the README's table; when a phase is done, its file and its line go |
| Code is first run, or first learns | the has-it-run cell in the roadmap's table |
| A change touches reward, observation, layout or how often something appears | `landscape-changes.md` |
| A word is coined | its area's file in `glossary/`, and its name in the index |
| A file or folder is added, moved or renamed, or a run's or a step's path changes | `code-map.md`, in the same change |
| An idea that is not a problem now | a new file in `codebase-ideas/`, and its line in the index |
| How a job only some sessions need is done | its file in `docs/dev/procedures/` |
| An investigation closes, or an idea is dropped, and both would cost to redo | `docs/dev/archive/`, by its `README.md`; otherwise delete it |
| What was done, and what was run | the commit message |

Each file in `docs/dev/` opens with what it is for and what stays out of it. Read that before adding to it.

A fact is written in the file that owns it, by the table above, and another file that needs it points there instead
of repeating it. The traps below are the exception: each stays a short warning here, and the owning file holds the
detail. When two docs disagree, the file that owns the fact wins and the other is fixed, `README.md` included.

## The docs sweep

A pass over the docs that ties them back together at the end of a session. When the owner signals that the session is
ending, say once, "want a docs sweep before we stop?", and drop it if they decline. The owner can also ask for one at
any time. How a sweep is run is in `docs/dev/procedures/docs-sweep.md`.

## Commits

A commit message says what changed, why, and what was run to check it, or that nothing was run. The dated progress
notes the docs used to carry belong here, in the commit that made them true. The owner leaves commit messages mostly
to the agent, so they carry the record.

**Docs are the agent's to commit; code waits for the owner.** The owner reviews code in the list of unstaged
changes, so the agent keeps docs out of it.

- **Docs not about the code under review,** asked for or not, are committed when done, naming the files
  (`git commit -- <files>`), so nothing staged goes with them. Several in a row are amended into the agent's last
  docs commit while it is unpushed.
- **Docs about the code under review** are staged and wait. When the owner says commit, they go in one commit with
  the code. A file with changes of both kinds waits with the code.
- **Code is never committed before the owner says commit or push.** This is a hard rule. A go to write it is not a
  go to commit it. Code is anything that does something, in a `.py` file, YAML, `pyproject.toml` or any other file.
  A file that mixes code and docs counts as code. Comments and docstrings the owner asks for count as docs, even
  inside a code file.

A new commit is made only when the last one is pushed or is the owner's; otherwise the change is amended into it,
whatever its subject. Before an amend, `git status -sb` shows whether the branch is ahead. When in doubt, ask the
owner in one line whether to amend.

**A commit on an unmerged branch is cited by its title, not its hash.** "Rebase and merge" gives every commit of the
branch a new hash on `main`, and the old ones vanish with the branch. A hash is cited only for a commit already on
`main`. After a merge, switch to `main` and pull before anything else, since the local branch still holds the old
hashes.

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
- **Every code change ships with its tests,** reviewed together with the code. A new rule gets its tests first, and
  the owner reviews them before any code is written. Where tests go and how they are grouped is in
  `docs/code_style.md`; how they are written is in `docs/dev/procedures/writing-tests.md`.

## The two strains

The working strain is `src/syn_grid/`. A frozen copy of the stable code is in `src/syn_grid/legacy/`, so it can be
trained as it was and its curves compared with the rebuild. Each legacy package and loose file has a `_legacy` suffix
and imports only other legacy copies. `USE_LEGACY` in `src/syn_grid/__main__.py` picks the strain. Nothing under
`legacy/` is edited, and it is deleted when phase 7b's perceptions are done.

Before running legacy or comparing it with the rebuild, read `docs/dev/procedures/legacy.md`.

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
pyright src/syn_grid tests                 # types, CI scope; needs the venv's python first on PATH
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
- **Before changing anything under `src/` without meaning to change behaviour,** a refactor, a rename, a move or a
  tidy, **and before any change under `src/syn_grid/runners/`,** read `docs/dev/procedures/checking-behaviour.md`. It says which
  probe checks the change against the stable version, and what the probes do not reach.

## Traps

- **`GridWorld.reset` needs an rng.** Pass `np.random.default_rng(seed)`, or let the environment pass its seeded
  `np_random`.
- **`runners/agent_runners/utils/extractors.py` is imported by nothing** and reads observation keys from before the
  rebuild. It stays on purpose: phase 7b works from it, not from scratch.
- **`reproduction_package/` is committed and large.** Do not add to it. It records the thesis runs, which live at the
  `v0.4.0` tag, and it leaves `main` with legacy at the end of phase 7b.
