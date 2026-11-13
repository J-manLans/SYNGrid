# Decisions: Testing, CI

What CI runs and when: the tests and the Hypothesis invariants, the types, the coverage floor and tidy. How the suite
is built and measured is in `suite.md`. The two files are one area, testing, and are read together before a design
opinion on it.

Written in the present tense, in the format of the template in `../../design-decisions.md`. A decision added here
also gets its title in that index.

---

## Hypothesis checks the world's invariants, and CI runs the tests and ruff (2026-11-07)

**Decision:** Hypothesis is in the `dev` extras, at 6.170.15, for property-based tests of the rules that hold for any
seed and any sequence of actions: the observation stays in its space, the droid on the grid, the same seed and actions
give the same trace, no orb is lost, no reward exceeds what the chain pays, and an episode ends by `max_steps`. These
are a handful of tests at the world and environment level, and only these; everything else is an example test, which
stays the form for one specific behaviour. CI runs them in `pytest` on a minimal, derandomized profile, so they are
quick and cannot fail by chance. Locally they run on a profile as extensive as wanted. The profiles are registered in
`tests/conftest.py`: `ci`, 25 examples, is the default, and `dev`, 1000, is chosen with `HYPOTHESIS_PROFILE=dev`. The
world's invariants are in the world's test file and the episode's in the environment's step test, each class named
`TestInvariants`; the observation's waits for the rewritten perceptions. An episode's actions mix a repeated pattern
with steps towards the next tier orb, since a random walk almost never completes a chain and the reward bound would
then check nothing. mutmut is local only, run while a file is worked on, and never in CI. coverage runs locally the
same way, and in CI only for the floor; see "CI fails below a coverage floor of 85%". CI installs the CPU build of torch
from PyTorch's own index before the project: the runner has no GPU, no test needs one, and the default Linux build
brings several GB of CUDA libraries. Its torch version can drift from a local install; it is pinned in that step if a
test ever fails in CI alone because of it.

**Why:** Example tests check an invariant only where someone thought to look; a property checks it over many generated
inputs and shrinks a failure to its smallest case. The invariants are what catch the subtlest breakage, so they run on
every push. mutmut is not pass or fail: its survivors are read, and some are always equivalent. A coverage floor
would only fix today's 70% in place while the suite is rebuilt.

**Rejected:**

- Hypothesis kept out of CI: a change that breaks an invariant would pass CI.
- mutmut in CI: slow, and there is no sensible rule for when it fails.
- A coverage floor in CI during phase 6b. Taken up when 6b closed: see "CI fails below a coverage floor of 85%".

**Revisit if:** the suite is rebuilt at the end of phase 6b, for coverage in CI that fails only below a threshold:
`fail_under` in `[tool.coverage.report]` for the total, or `diff-cover` for the lines a change touches, which matters
once others send pull requests. Or the Hypothesis profile makes CI slow.

## CI checks types with Pyright (2026-11-13)

**Decision:** Pyright is in the `dev` extras, as `pyright[nodejs]`, which brings Node through pip, and CI runs
`pyright src/syn_grid tests` on every push. Its settings are in `pyproject.toml`'s `[tool.pyright]`, which Pylance reads
too, so the editor and CI agree. `runners/agent_runners/utils/extractors.py` is excluded: it is imported by nothing
and kept as it is for phase 7b. pylint, which brings pyreverse for the class diagrams, is in an extra of its own,
`diagrams`, which CI does not install.

**Why:** Ruff lints and formats but checks no types. Pyright ran only inside VS Code, through Pylance, and only on the
files open there, so a missing `@override`, or a class that did not match the protocol it is handed to, could reach
`main` unseen. Protocols are matched, not inherited (see `docs/code_style.md`), so Pyright is what checks them.

**Rejected:**

- Pylance alone: it checks only open files, and only on a machine with VS Code.
- The PyPI `pyright` without `[nodejs]`: it needs Node installed apart from the project.
- pylint in `dev`: CI would install it on every run and never use it.

**Revisit if:** Pyright's checks slow CI noticeably, or `extractors.py` is reworked in phase 7b, when its exclude
goes.

## CI fails below a coverage floor of 85% (2026-11-10)

**Decision:** `fail_under = 85` in `[tool.coverage.report]`, so `coverage report` fails when the total is below 85%.
The total, as configured, was 88% when phase 6b closed. CI checks it where branches meet `main`: on a pull request
into `main` and on a push to it, CI runs the suite as `coverage run -m pytest`, which runs every test as `pytest` does,
and then `coverage report`. Every other push runs plain `pytest`. Work goes in small branches merged often, so a drop
shows within a branch or two of the change that made it.

**Why:** A floor is a backstop: at 85 it fails on about 55 untested statements, a dropped test file or a mid-sized
module with none. At 78 it would let about 180 through, more than a whole untested module. It sits three points
under the total because phases 7 and 8 rebuild the observation and render code, part of what is untested today, and
the total moves while they do. It does not check that new code is tested; the rule that every change ships with its
tests, and the sweep, do that.

**Rejected:**

- 78: so low it fails only when a test file is deleted outright.
- 88, the total at the time: harmless moves during the rebuild would fail it.
- The floor on every push: a few seconds a run, for a check that a merge into `main`, made often, already gives.
- `diff-cover` on the lines a change touches: what matters once others send pull requests; until then the sweep
  covers it.

**Revisit if:** phases 7 and 8 are done, to raise the floor toward the total then. Or others send pull requests, for
`diff-cover`.

## CI checks tidy where branches meet main (2026-11-13)

**Decision:** CI runs `./scripts/tidy.sh --check` on a pull request into `main` and on a push to it, the moments the
coverage floor is checked, and not on other pushes. `--check` changes nothing: it runs ruff's lint and its format
check without fixing, and lists the markdown lines past 120 characters, and fails on any of them. The plain
`./scripts/tidy.sh` fixes with ruff and lists the long lines without failing. Tests and Pyright still run on every
push.

**Why:** Lint and formatting are run in the session, with `tidy.sh`; what must not slip is what reaches `main`. A
check at the merge is a reminder that tidy was not run, and a branch's pushes are never blocked by a lint or a docs
nit. The width of the docs had no check at all, since ruff formats Python only, and one command now covers both. A
long line is rewrapped by hand: a markdown formatter refills whole paragraphs, so a one-word edit would show as a
paragraph rewritten in the diff.

**Rejected:**

- The ruff checks on every push, and only the doc width at `main`: two rules where one does.
- The doc width only listed by `tidy.sh`, never in CI: easy to forget.
- The doc width checked on every push: a branch blocked over a docs nit.
- A script of its own for the width check: `tidy.sh` already runs where it is needed.
- `mdformat` with a wrap of 120: it refills whole paragraphs, turns `---` into a row of underscores, and escapes
  characters such as the underscore in `Pop!_OS`.

**Revisit if:** others send pull requests and want lint feedback before the merge, or the width check flags lines it
should skip.
