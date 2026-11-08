# Decisions: Testing

How the suite is built and measured: the regression test, coverage, mutmut, Hypothesis and tests first.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## The regression test runs the replay probe on each locked scenario's base config (2026-11-07)

**Decision:** `tests/regression/` runs the replay probe, `scripts/replay/probe.py`, as a subprocess for every locked
scenario, with `PYTHONPATH=src`, and checks its `stream_digest` against the scenario's baseline. The probe builds a
locked scenario from its base config, not from its run YAML. The baselines are committed in
`tests/regression/baselines/`, a `.json` and an `.npz` per scenario, named after the scenario; the `.json` records
the commit it was recorded at. A locked scenario with no baseline fails the test, so locking a scenario means
recording its baseline. The probe needs no model: its tape is a seeded random walk.

**Why:** The baselines are recorded by the probe, so the test checks them with the same code. `sweep.sh` runs today's
probe against older commits, which works only while the probe carries its own loop. The base config changes only
when the scenario does, so an experiment's tunables do not move the test. `PYTHONPATH=src` keeps the subprocess on
this tree, whatever the editable install points at. Committed baselines reach CI, and the `.npz` lets `diff.py`
show where a failing scenario moved.

**Rejected:**

- The tape loop in the package, imported by the test: the probe would keep a copy for `sweep.sh`, and two loops
  would have to give the same digests.
- Recording from the run YAML: changing a tunable for an experiment would fail the test with no code changed.
- The digests as strings in the test: a second record of what the baseline files hold.
- Baselines in `output/`: it is gitignored, so CI would have none, and a file was a baseline only by its name.

**Revisit if:** the probe stops being the tool that records the baselines.

## The suite is measured by coverage.py alone, and mutmut (2026-11-07)

**Decision:** Coverage is coverage.py on its own, run as `coverage run -m pytest` and read with `coverage report -m`.
Its settings are in `[tool.coverage.run]` in `pyproject.toml`: branch coverage, `syn_grid` as the source, `legacy/`
left out, and `patch = ["subprocess"]`, so the replay probe the regression test starts as a subprocess is measured
too, and `dynamic_context = "test_function"`, so `coverage html --show-contexts` shows which tests ran each line.
Mutation testing is mutmut. Both are in the `dev` extras, at coverage 7.17.2 and mutmut 3.8.1. mutmut tests the
mutated copy and not the editable install, and runs only `tests/unit` and `tests/integration`, set by
`pytest_add_cli_args_test_selection` in `[tool.mutmut]`.

**Why:** `pytest-cov` only starts coverage from pytest's command line; the settings live in coverage's own config
either way, and since its version 7 it leaves subprocesses to coverage's `patch` setting too. One dependency fewer.
Without `patch`, the regression test alone reaches 25% of the code; with it, 54%. The regression test starts the
replay probe as a subprocess, which imports `syn_grid` from `src/` and never sees a mutant, so under mutmut it adds
time and kills nothing, and what only it checks shows as survived. Without it, a survivor says what mutmut is for:
that no targeted test guards the code. The regression test still runs in every `pytest` and in CI.

**Rejected:**

- `pytest-cov`: a second way to start the same tool.
- The probe importing from mutmut's copy under mutmut: the replay tape would count as guarding a rule, though it only
  says that something changed, not which rule broke.

## Hypothesis checks the world's invariants, and CI runs only pytest and ruff (2026-11-07)

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
then check nothing. coverage and mutmut are local
only, run while a file is worked on, and never in CI.

**Why:** Example tests check an invariant only where someone thought to look; a property checks it over many generated
inputs and shrinks a failure to its smallest case. The invariants are what catch the subtlest breakage, so they run on
every push. mutmut is not pass or fail: its survivors are read, and some are always equivalent. A coverage floor
would only fix today's 70% in place while the suite is rebuilt.

**Rejected:**

- Hypothesis kept out of CI: a change that breaks an invariant would pass CI.
- mutmut in CI: slow, and there is no sensible rule for when it fails.
- A coverage floor in CI during phase 6b.

**Revisit if:** the suite is rebuilt at the end of phase 6b, for coverage in CI that fails only below a threshold:
`fail_under` in `[tool.coverage.report]` for the total, or `diff-cover` for the lines a change touches, which matters
once others send pull requests. Or the Hypothesis profile makes CI slow.

## Tests come first, from the owner's rules (2026-11-07)

**Decision:** When new code is added and its rules are decided, the owner states the rules as behaviour, "when X
happens, Y must happen" or "Y must never happen", and the agent adds the edge cases and asks where a rule is unclear.
The agent writes the tests first, and the owner reviews them before any code is written. The code is then written
with the tests as the guide, each test seen red before green. Tests go through the public entry points and describe
behaviour, not structure. Code whose design is still moving, such as a mechanic under ablation, is tried first and
gets its tests once its rule is settled. A "must never" rule outside the world's handful of invariants is an example
test, not a Hypothesis property.

Every change ends with a sweep of the files it touched: pytest with the regression test, then coverage and mutmut,
read survivor by survivor. A survivor that guards a rule gets a test; an equivalent mutant, a defensive branch, output
text or an unreachable guard is listed with its reason. When a file's tests are tightened, a test whose lines other
tests also run, and whose removal lets no new mutant survive, is proposed for removal. The redundancy check is not run
on every change: over the suite it flagged 24 tests by coverage and confirmed none, at a full mutmut run per test.

**Why:** Tests written after the code tend to mirror the implementation and confirm its mistakes. Written first, from
the owner's rules, they keep the intent with the owner and the edge cases with the agent. The sweep catches what
test-first alone does not, gaps and redundancy, but judgement decides what is worth a test, not the numbers.

**Rejected:**

- Tests after the code by default: they describe what the code does, not what it should.
- Coverage or mutation score as a target: it rewards tests written to move a number.
- The redundancy check by running mutmut once per test left out: slow; reading which tests run each line finds the
  suspects, and mutmut confirms only those.

**Revisit if:** test-first slows design work that is still finding its rules, beyond the ablation exception.
