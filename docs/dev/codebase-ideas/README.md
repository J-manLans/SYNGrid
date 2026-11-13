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
- **An idea file** is opened at a docs sweep, when the owner asks about its subject, or when the owner says to open
  it. Work reaching an idea's "Returns when", or a pointer to it from another doc, does not open it: the agent names
  the idea from the index in one line and asks.
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
| [The GUI's version 2: creating a scenario](gui-version-2.md) | the GUI's version 1 (phase 9) and the sandbox (phase 10) are done. |
| [Orbs that block without being frozen](blocking-orbs-without-freezing.md) | a scenario needs cells that block, or orbs placed on purpose, such as the vision's wall around a goal orb. |
| [A guide entity that walks the chain](guide-ghost-walks-the-chain.md) | a scenario with a second body on the grid is planned: a guide, a rival or a companion. |
| [Orbs that hurt on touch](orbs-that-hurt-on-touch.md) | a scenario needs an orb that costs the droid for moving into it, such as a fire orb. |
