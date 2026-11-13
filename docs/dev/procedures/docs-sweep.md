# The docs sweep

How the pass that ties the docs back together at the end of a session is run. When to offer one is in `AGENTS.md`,
"The docs sweep"; where each kind of fact belongs is its "Where things go" table, which this file checks against.

What stays out: the rules for committing, which are in `AGENTS.md`, "Commits".

## The checks

1. The roadmap and `open-questions.md` against what was done: remove finished steps and answered questions, and update
   the has-it-run cells.
2. Decisions from the session that are missing from `decisions/`, or from the `design-decisions.md` index.
3. New terms that belong in `glossary/`, and that the index names every term in the file it is in.
4. Anything that landed in the wrong file, by the table in `AGENTS.md`. A status fact outside the roadmap's table is
   one of these, `docs/vision.md` included.
5. Size: flag any doc past about 4,000 words (`wc -w`), and any trap in `AGENTS.md` that no longer applies. Flag
   only; how to shrink a doc is the owner's call. `AGENTS.md` weighs most, since every session reads it in full.
   Two exceptions: the files in `codebase-ideas/` are not counted, since the folder has no size limit and check 6
   keeps it short; a `glossary/` file past 1,000 words is cut, by the rule in `glossary/README.md`, without
   asking; and a `decisions/` file past 2,500 words gets a proposed split by topic, made once the owner agrees on the
   files.
7. Line width: run `./scripts/tidy.sh` and rewrap by hand any markdown line it lists past 120 characters. Rewrap only
   those lines, so the diff shows what changed.
6. `codebase-ideas/`, the one time its idea files are read without the owner asking. For each idea:
   - **built:** what was decided goes to its area's file in `decisions/`, and the file and its index line go;
   - **the work has moved past part of it:** edit it to match;
   - **no longer wanted:** delete it, or move it to `archive/` when `archive/README.md` says it is worth keeping.

   Then check that the index matches the folder: one line per file, and the same "Returns when".

Say what changed, and flag borderline calls instead of guessing. Whether something is a decision or still open, and
whether it is an idea or a problem, is the owner's to say.

## When the owner leaves

If the owner says to go ahead and leaves, do only the mechanical parts: removing finished items, moving things to the
right file, adding glossary terms. Leave a borderline call as it is and list it in the commit message, so it waits for
the owner. Then commit the docs.

## Which commit it goes in

A sweep has no subject of its own: it finishes the session's work. So it is almost always amended into the session's
last commit, under the rule in `AGENTS.md`, "Commits", with its borderline calls added to that commit's message. It
is a commit of its own only when that commit is pushed or is the owner's.
