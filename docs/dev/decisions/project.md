# Decisions: Project

Decisions about the project as a whole: the Python version, and legacy's role and lifetime.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## Legacy is a reference, not a target (2026-10-25)

**Decision:** The legacy strain and `reproduction_package/` are references. The rebuild is held to learning in the
same general direction as legacy, by comparing curves, not to legacy's rewards or numbers. They are also where to
look up what a behaviour used to be.

One run per setup is enough to judge a curve. The owner has run this code enough to know its stability on sight, and
these scenarios are stable. Running several seeds is an argument for a more chaotic benchmark.

These differ from legacy on purpose:

- A max-tier chain pays what its orbs are worth, not a fixed 10.
- Scaling dense has sparse's ladder and step penalty, not the thesis dense values.
- In delay, eating a tier orb freezes only the tier orbs, and every frozen orb waits the same delay before it can
  return.
- A negative orb does not break a chain.
- Step-wise has no completion bonus, uses `chain_break_penalty`, and charges it for a tier 1 out of order.
- A tier 1 that breaks a chain leaves no chain; legacy restarted the chain with it.
- A threshold break pays the penalty plus everything gathered, not the last tier's reward.
- An ending adds to the step's reward instead of replacing it; a threshold timeout pays `timeout_penalty`; a threshold
  chain is paid what it is owed when energy reaches 0.

**Why:** Holding the rebuild to legacy's numbers would keep behaviour that was decided against.

**Rejected:** an exact per-step match of rewards, or reproducing legacy's results, as the bar.

## SYNGrid moves to Python 3.12, through Pop!_OS 24.04 (2026-11-07)

**Decision:** SYNGrid moves from Python 3.10 to 3.12, the system Python of Pop!_OS 24.04, which the owner's machine
moves to. `requires-python` is `">=3.12,<3.13"`, pinned to one minor version as 3.10 was. A model saved under
3.10 that does not load under 3.12 is retrained from its config and seed; the 22 in `output/` at the move all load,
with their normalization stats. On 3.12 the four regression baselines and the runner probe's digests are the same as
on 3.10.

**Why:** Nothing recorded a reason for the pin to 3.10; it was the system Python of Pop!_OS 22.04. Tools have begun to
drop 3.10: coverage.py's newest line needs 3.11, so every dev tool would freeze at its last 3.10 release. The move
comes between phases, before 6b picks its tools, and with four locked traces and the runner probe to check it by.

**Rejected:**

- uv's standalone Python: it leaves the system alone, but the owner moves the OS to 24.04 anyway.
- The deadsnakes PPA: a system-wide third-party source for what the OS upgrade brings.
- Python in a Docker container: Python is what SYNGrid runs inside, not a service it talks to, so the venv, the GPU,
  the pygame window and the editor would all have to move into the container.
- Another distribution, such as Omarchy: a reinstall, a rolling Python that would still need pinning, and NVIDIA on
  Wayland to solve again.
- 3.13: SB3, torch and pygame have supported 3.12 longest.
- `">=3.12"`, open to newer Pythons: it would allow 3.13, which is turned down above, without anything running on it.
- `">=3.10,<3.13"`, keeping 3.10 as the floor: nothing runs on 3.10 after the move, a `str` enum formats differently
  in an f-string before 3.11, and phase 6b's newest dev tools need 3.11.

**Revisit if:** a dependency SYNGrid needs does not run on 3.12.

## Legacy stays until the perceptions are rewritten (2026-11-07)

**Decision:** `legacy/`, the `USE_LEGACY` switch and `landscape-changes.md` are deleted at the end of phase 7b, once
the five perceptions are rewritten, and not at the close-out. The owner deletes them.

The rebuild merged into `main` before that, as 0.5.0, tagged `v0.5.0`, when phase 6b closed (2026-11-10), with legacy
still in it. From then on each phase is worked on a small branch off `main` and merged back often, so `main` is the
rebuild as far as it has come.

**Why:** Every scenario is locked, and the regression baselines are now what a change is checked against, so nothing at
the scenario level needs legacy. The perceptions are the part the owner works on most closely, and legacy's
perceptions stay runnable beside the rewritten ones until then.

**Rejected:**

- Deleting it now, reading the old perceptions from `cleanup-after-versioning` with `git show`: the source would
  remain, but legacy's perceptions could no longer run next to the new ones.
- Keeping it to the close-out: nothing after the perceptions compares against it.
