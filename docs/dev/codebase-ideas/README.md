# Ideas About the Codebase

Ideas worth returning to that are not problems to solve now. Half-formed is fine. One file per idea, with this
README as the index.

Decisions stay out: once an idea is decided, it moves to its area's file in `decisions/` and leaves here, or keeps
only what is still an idea.

An idea that is dropped is deleted, or moves to `archive/dropped-ideas.md` when `archive/README.md` says it is
worth keeping.

Each idea opens with when it returns: a phase, a scenario or a piece of code that would make it matter, not a guess
at how likely it is. The index below repeats it, so work that reaches that point finds what waits on it.

## Who reads what

- **The index** (this README) can be read at any time. It is small on purpose.
- **An idea file** is opened only at a docs sweep, or when the owner asks. When work reaches an idea's "Returns
  when", the agent says so in one line, and the owner decides whether the file is opened. A pointer to an idea from
  another doc does not open it either.
- **Adding an idea** needs no reading: it is a new file and a line in the index.

The folder has no size limit. It empties itself: an idea that is built moves to `decisions/`, and the docs sweep
checks the rest for ideas the work has moved past (see `docs/dev/procedures/docs-sweep.md`).

## Template

A file named for the idea, in kebab case, and a line in the index below.

```
# Title (date)

**Returns when:**
**Idea:**
**Where it came from:**
**Why it might matter:**
```

## Index

| Idea | Returns when |
|---|---|
| [A GUI on top of the registry](gui-on-the-registry.md) | work on a GUI starts, which no phase of the roadmap holds. Picking a checkpoint returns sooner, when a run continues another, such as a curriculum stage. |
| [A bundle that carries everything its orb kind brings](bundle-carries-its-kind.md) | a second family uses an orb kind that one already uses, or phase 7 or 8b settles what a kind adds to the observation or the HUD. |
| [A held chain that keeps the droid alive](held-chain-keeps-droid-alive.md) | a stochastic goal scenario, with a continuous threshold tier chain beside the goal, is planned. |
| [Effect orbs get their magic from the engine](effect-orbs-magic-from-engine.md) | effect orbs are built. The roadmap lists them under "Not on this roadmap". |
| [Orbs that block without being frozen](blocking-orbs-without-freezing.md) | a scenario needs cells that block, or orbs placed on purpose, such as the vision's wall around a goal orb. |
| [A guide entity that walks the chain](guide-ghost-walks-the-chain.md) | a scenario with a second body on the grid is planned: a guide, a rival or a companion. |
| [Orbs that hurt on touch](orbs-that-hurt-on-touch.md) | a scenario needs an orb that costs the droid for moving into it, such as a fire orb. |
