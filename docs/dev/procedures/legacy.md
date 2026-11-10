# Working with legacy

How to run the frozen legacy strain and compare it with the rebuild. What legacy is, and that nothing under it is
edited, is in `AGENTS.md`, "The two strains".

What stays out: the changes that could move what the agent sees or is paid, which are in `../landscape-changes.md`.

## What a comparison means

Legacy is a reference and not a target. Several decisions change rewards on purpose, so the rebuild is held against
legacy's learning curves and not its rewards step for step.

## Traps

- Only one strain is imported per process. Both register the Gymnasium id `syn_grid-v0`.
- The legacy config names its images as `assets/sprites/...` while the folder is `assets_legacy/`, so legacy
  rendering stops with a `FileNotFoundError`.
- The legacy run id has its own scheme. Give legacy runs their own tag or output folder.
- The config schemas differ, so a comparison means something only if both configs describe the same world.
