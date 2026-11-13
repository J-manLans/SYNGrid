# The test sweep

How the sweep that ends a roadmap step is run, over the source files the step touched, before the step is removed
from the roadmap. Tightening an existing test file is the same sweep. When it runs, and what a change runs short of
it, is in `AGENTS.md`, "What a change runs".

What stays out: how tests are written before the code, which is in `writing-tests.md`, and why the suite is
built this way, which is in `decisions/testing/`.

coverage and mutmut are a reading aid, not targets: a file is done when every survivor, every uncovered line and every
rule it decides in a value is either tested or listed with its reason.

## The steps

1. **Run** pytest, the regression test included, so no locked scenario drifted. Then coverage of each touched file
   (`coverage run -m pytest`, then `coverage report -m`), and mutmut on it. mutmut runs only `tests/unit` and
   `tests/integration`, so a survivor means no targeted test guards that code. To scope mutmut to the touched
   files, set `only_mutate` in `[tool.mutmut]` and put the line back after the run. It is the one edit to
   `pyproject.toml` the agent makes without asking.
2. **Read the file for what it decides,** not only for what the tools flag: a value passed as a choice (a `None`, a
   constant, a fixed count) and any comment that says why. Coverage counts the line as run and mutmut never turns a
   `None` into a number, so neither sees a rule that lives in a value. Each such rule joins the list sorted below.
   For a file that takes inputs with limits, such as a config model, check its tests against the input's
   classes, not against the tools: one test of every input valid at once; one test for each invalid class alone,
   since a second invalid input in the same test can hide whether the first was caught; a value below, at and
   above each limit; and a decision table for limits that depend on each other, such as `record_video` needing
   `rgb_array`. A class or a boundary with no test joins the list too.
3. **Sort** each survivor and each uncovered line by one question: would a caller, through the public surface,
   notice if this broke?
   - Yes: a real gap, which gets a test.
   - No behaviour can change: an equivalent mutant, listed and not tested.
   - Nobody decided what should happen: a question for the owner, not a test. A test would decide it.
   - Nothing depends on the code, such as a defensive branch or output text: listed, not propped up with a test.
   - The observation and render packages: skipped, since they are rebuilt.
4. **Test the behaviour, not the mutant.** A test that reads a private attribute only to kill a mutant pins how the
   code works, not what it does.
5. **Confirm it goes red** with its mutant (`mutmut run <name>`), or, where mutmut does not reach, with the code
   broken by hand and restored.
6. **Look for redundant tests,** only when a file's tests are tightened, not on every change.
   `coverage html --show-contexts` lists on each line the tests that ran it. A test whose lines all list others too,
   and which checks the same outcome, is a suspect: comment it out and run mutmut on the file again. If nothing new
   survives, propose removing it.
7. **Report** per file: coverage and mutmut before and after, the tests added or removed, and what is left with its
   reason, briefly.

## Traps

- mutmut does not mutate a decorated method: one marked `@override`, which most strategy methods are, or a pydantic
  validator. A clean mutmut run says nothing about them. Break each one in a touched file by hand, run the tests,
  restore it, and say in the report which were checked that way.
- After a change to `[tool.mutmut]`, `rm -rf mutants` before the next run, or mutmut keeps the old copy and its old
  verdicts.
- mutmut runs with `process_isolation = "forkserver"`. Under the default `fork`, the SB3 tests hang on torch's thread
  pool and every survivor that reaches them shows as a timeout, which mutmut counts as caught. A timeout is read as
  a suspect, not a kill. Why: "mutmut forks from a clean server" in `../design-decisions.md`.
- After restoring a file broken by hand, delete its `__pycache__`. A restore of the same size in the same second
  leaves Python running the broken bytecode.
