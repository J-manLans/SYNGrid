# Glossary

What the words mean, in the code and in the other files in `docs/dev/`. A term is defined once here, and a new term
is added when it is coined. Reasons stay out (`design-decisions.md`), and so does the state of the code (the
roadmap's table).

## How it is read

This README is the index: every term, and the file it is in. It can be read at any time. Before writing in an area,
read that area's file and `two-senses.md`, which lists the words used in two senses or not settled, the traps of
every area. A term is looked up through the index, not by reading every file.

## How it grows

A new term goes in its area's file, in the area's order, and its name goes on the area's line below.

An area file stays under 1,000 words (`wc -w`). When one passes it, it is cut into letter ranges at the term counts,
so the parts hold about the same number of terms, named for the area and the range: `observation.md` becomes
`observation-a-l.md` and `observation-m-z.md`. When a part passes the limit again, the area is cut again into one
more range than it had, every part re-cut so they stay even. A cut changes only this index: other docs point to an
area, not to a file of it. `two-senses.md` is never cut.

## Index

- **[Scenarios](scenarios.md)**: Scenario, Goal scenario, Continuous scenario, Family, Single tier chain, Variant,
  Sandbox, Spatial, Scaling sparse, scaling dense, Delay, Silence, Spare steps, Axis, Scenario tag, Locked, Full model,
  Tunables, Base config, Fork.
- **[Building a scenario](building-a-scenario.md)**: Registry, Entry, Entry type, Catalogue, Lock, Block, Rules,
  Strategy, Step result, Watcher, Freeze, frozen, Blocking, Population, Count, Bundle, Recipe, Brick, Reader,
  `build_world`.
- **[The world](world.md)**: World, Droid, Field, Fixed field, Sampled field, Energy, Score, Settle, Terminal reward,
  Reward, Penalty, Step clock, Timeout, Terminated, truncated.
- **[Orbs](orbs.md)**: Orb, Eat, consume, Kind, Category, Direct orb, Fixed reward orb, Negative orb, positive orb,
  Synergy orb, Chain orb, effect orb, Tier, Identity, Radix identity, Dense identity, Lifespan, Despawn, Cool-down,
  Expired, ready.
- **[Digestion](digestion.md)**: Digestion, Digester, Engine, Direct digester, Chain, Progress, Completion, Break, Chain
  worth, Undigested, Scoring mode, Max-tier, Step-wise, Threshold, Reward ladder.
- **[Observation](observation.md)**: Observation, Perception, Visibility, Encoding, Markovian, POMDP, Fog of war, Slot,
  Global feature, Orb feature, View radius, Floodlight, Lantern, Format, HUD, Metric.
- **[Running](running.md)**: Runner, SB3 spec, Run id, Human play, Run controls.
- **[The rebuild](rebuild.md)**: Strain, Legacy, Skeleton, Reproduction package, Phase, Replay probe, Runner probe,
  Tape, Trace, Baseline, Scenario sweep, Rule in a value.
- **[Words with two senses, or not settled](two-senses.md)**: Block, Axis model, Perception, Visibility and encoding,
  Pool, Sweep, Sparse and dense.
