# Archive

What is no longer active but is kept, so that nobody redoes the work or suggests the same thing again without
knowing it was tried. Two kinds of thing live here: investigations that are closed, and codebase ideas that were
dropped.

`git log` already keeps everything that is removed from a doc. The archive is only for what the history would not
lead anyone to.

Stays out:

- A decision, or an alternative turned down in one. It belongs in its area's file in `decisions/`, under
  **Rejected**, for as long as the decision stands. When a decision is reversed and its story still matters, the story
  can move here.
- Anything still open. An idea worth returning to stays in `codebase-ideas/`, and an open question stays in
  `open-questions.md`.
- Run logs, progress notes and what a session did. They go in the commit message.

---

## What is worth keeping

Archive something only when both answers are yes:

1. **Is it likely to come up again?** The idea looks attractive, or the problem can come back: a new machine, a new
   driver, an agent that suggests the same thing.
2. **Did reaching the conclusion cost something?** Runs, debugging, a long design talk, which a few lines can save
   the next person.

Otherwise delete it, and let `git log` hold it.

For example, the pygame and Triton segfault is kept: it can return with the hardware, and it took long to pin down. A
dropped idea whose reason is plain, or that nobody would suggest again, is deleted. A dropped idea that looks elegant
but broke a principle is kept, with a line on why.

## How it is laid out

- **A closed investigation** gets a file of its own, named for what it is about.
- **Dropped ideas** collect in one file, `dropped-ideas.md`, made with the first entry. One entry per idea, by the
  template below.
- **Every file is listed in the index below,** with the date it was archived and one line on what it holds. Each
  dropped idea gets its own row, with its title, the date it was dropped and one line on why, so the index alone
  says whether an idea was tried.

An entry is short. It says what the thing was, why it ended, and what would have to change for the verdict to change.
The last part is not an invitation to revisit: it lets the next reader see whether the ground has moved.

## Entry template for a dropped idea

### Title (date dropped)

**Idea:** what it was.
**Why dropped:** the reason.
**Would change if:** the condition that would make it worth a new look.

---

## Index

| File | Archived | What it holds |
|---|---|---|
| `rppo-regression.md` | 2026-11-10 | Why spatial seemed to learn worse on RPPO after the first refactor: the timeout and chain-break penalties shared one field, so their ratio was pinned |
| `pygame-triton-segfault.md` | 2026-11-07 | Why evaluation with pygame rendering crashed on a hybrid AMD and NVIDIA laptop, and the NVIDIA offload workaround |
