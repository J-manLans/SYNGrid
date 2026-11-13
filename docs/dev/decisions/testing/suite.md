# Decisions: Testing, the suite

How the suite is built and measured: the regression test, coverage, mutmut and tests first. What CI runs, and the
Hypothesis invariants it runs, are in `ci.md`. The two files are one area, testing, and are read together before a
design opinion on it.

Written in the present tense, in the format of the template in `../../design-decisions.md`. A decision added here
also gets its title in that index.

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

## mutmut forks from a clean server (2026-11-09)

**Decision:** `[tool.mutmut]` sets `process_isolation = "forkserver"`. mutmut starts a server process that imports
the test setup but runs no test, and forks each mutant's worker from it. The default, `fork`, forks each worker from
mutmut's main process.

**Why:** Under `fork`, a mutant that no early test kills hangs in the SB3 runner tests, and mutmut reports it as a
timeout, which it counts as caught. Survivors were hidden as kills.

The cause. Before testing mutants, mutmut runs the targeted suite once in its main process, to time the tests and map
each function to the tests that reach it. The SB3 tests build a PPO model, and its weight init
(`torch.nn.init.orthogonal_`) starts torch's CPU thread pool. A fork copies the process's memory, so the child
inherits the pool's state and its locks, but only the forking thread: the pool's worker threads are not copied. When
the child reaches an SB3 test, torch hands work to a pool whose workers do not exist and waits on a lock no one will
release. mutmut's wall-clock timer then sends `SIGXCPU`, and the exit code reads as a timeout.

Why only survivors. mutmut runs, for each mutant, every test that reaches the mutated function, from any test file.
Every environment builds orbs, droids and a world, so the SB3 tests reach nearly all of `core/`. A killed mutant
fails on an earlier test and stops; a survivor passes everything before the SB3 tests and hangs in them. `only_mutate`
narrows which code is mutated, not which tests run, so it does not help.

How it was found, on `core/orbs/orb_meta.py` (2026-11-09, mutmut 3.8.1, Python 3.12, torch on CPU): under `fork` the
run had not finished after ten minutes. The workers sat at 0 seconds of CPU, one thread each, waiting on a futex, and
`py-spy dump` showed them in `orthogonal_` under `PPO.__init__` in
`tests/unit/runners/agent_runners/sb3/test_sb3_runner.py`. Under `forkserver`, after `rm -rf mutants`, the same file
ran 49 mutants in about 70 seconds: 22 killed, 27 survived, no timeouts. The stats file's map from each function to
its tests was full in both runs.

The problem is known outside the repo: forking a process after torch, OpenMP or BLAS have started their thread pools
deadlocks the child, and the usual fix is to fork from a clean process (`forkserver` or `spawn`). mutmut's own
configuration names torch among the libraries `forkserver` is for.

**Rejected:**

- `fork` with torch limited to one thread (`OMP_NUM_THREADS=1` or `torch.set_num_threads(1)` in a conftest): it
  depends on torch never starting a pool by another path, and it changes the threads the runner tests see.
- Leaving the SB3 tests out of mutmut's test selection: the runner code they guard would lose its mutation check.

**Revisit if:** mutmut changes how it isolates workers, or the SB3 tests stop building torch models.

## Tests come first, from the owner's rules (2026-11-07, revised 2026-11-13)

**Decision:** When new code is added and its rules are decided, the owner states the rules as behaviour, "when X
happens, Y must happen" or "Y must never happen", and the agent adds the edge cases and asks where a rule is unclear.
The agent writes the tests first, and the owner reviews them before any code is written. The code is then written
with the tests as the guide, each test seen red before green. Tests go through the public entry points and describe
behaviour, not structure. Code whose design is still moving, such as a mechanic under ablation, is tried first and
gets its tests once its rule is settled. A "must never" rule outside the world's handful of invariants is an example
test, not a Hypothesis property.

Every change runs pytest, with the regression test. The sweep runs once when a roadmap step is done, over the files
the step touched and before the step leaves the roadmap, and whenever the owner asks to close a test file: coverage
and mutmut, read survivor by survivor. A survivor that guards a rule gets a test; an equivalent mutant, a defensive
branch, output text or an unreachable guard is listed with its reason. When a file's tests are tightened, a test
whose lines other tests also run, and whose removal lets no new mutant survive, is proposed for removal. The
redundancy check is not run on every sweep: over the suite it flagged 24 tests by coverage and confirmed none, at a
full mutmut run per test.

**Why:** Tests written after the code tend to mirror the implementation and confirm its mistakes. Written first, from
the owner's rules, they keep the intent with the owner and the edge cases with the agent. The sweep catches what
test-first alone does not, gaps and redundancy, but judgement decides what is worth a test, not the numbers. The
sweep and the probes are different tools: the regression test and the probes ask whether behaviour changed, after
every change, and the sweep asks whether the tests pin the behaviour down, which is worth asking once a step's code
has stopped moving.

**Rejected:**

- Tests after the code by default: they describe what the code does, not what it should.
- A sweep at the end of every change (decided 2026-11-07 and changed on 2026-11-13): it is a slow, careful job per
  file, and inside a step it reads code the next change moves again.
- Coverage or mutation score as a target: it rewards tests written to move a number.
- The redundancy check by running mutmut once per test left out: slow; reading which tests run each line finds the
  suspects, and mutmut confirms only those.

**Revisit if:** test-first slows design work that is still finding its rules, beyond the ablation exception.
