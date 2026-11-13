# Writing tests

How tests are written: before the code for a new rule, and with a sweep at the end of every change. When to read
it is in `AGENTS.md`, "Starting a session"; that every code change ships with its tests is in its "How to work with
the owner".

What stays out: where tests go and how they are grouped, which is in `../../code_style.md`, and how the sweep is run,
which is in `test-sweep.md`.

New rules get their tests before their code, and every change ends with a sweep of the files it touched. Why:
"Tests come first, from the owner's rules" in `../design-decisions.md`.

## New code: tests first

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

## The sweep

Every change ends with a sweep over the source files it touched, with coverage and mutmut, and tightening an existing
test file is the same sweep. Read `test-sweep.md` before running one.

Traps when writing a test:

- A test that builds a runner works in `tmp_path` with `monkeypatch.chdir`, so its `output/` lands there.
- The run controls read pygame's queue in one batch and drop what follows Space in it. An event meant for the pause
  screen is posted with `pygame.time.set_timer`.
