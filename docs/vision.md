# SYNGrid: the vision

*Written 2026-10-27, from a long talk with the owner. It says where the project is heading, not what to build next. The
roadmap does that. Read this first, then the roadmap.*

Picture a small arcade cabinet from a future that never happened. Chunky controls, a few dials, a screen of glowing
numbers that don't tell you what they mean. On the screen is a grid. A little droid moves across it, eating orbs.

What the orbs do is the whole trick. Some pay you. Some cost you. Some pay only if you eat them in the right order.
Later, some will change what other orbs are worth. Pick a handful of orbs, pick what counts as winning, pick how the
droid sees the grid, and you have made a new game. You never build a new world. You combine the same few pieces, the
way you cook different dishes from the same shelf of ingredients.

You can sit at the panel and turn knobs until something clicks. Or you can pick a hard scenario somebody else found and
try to beat your own high score. Either way it should feel like touching a machine, not filling in a form. The look is
retro-futuristic, a little punk, and playful.

Under the paint it is a lab instrument. Every orb and every scenario has to do a real job: help teach or test how a
learning program copes with delayed and structured consequences. The fun is a side effect of that being done well. It
is never the reason something goes in. A small program should be able to learn the easy end of a scenario, with room
above for people and stronger programs. That is why it can serve a student tinkering with settings, someone testing
their own agent, and maybe a lab that needs harder puzzles.

---

## In more detail

### What it is

A grid, a droid that moves in four directions, and orbs. A scenario is what you get by choosing three things: which
orbs are on the grid, how the droid observes it, and what the goal is. The scenario layer exists so that adding an orb
or a scenario never means touching the world or the droid.

The question under it all is how an agent learns that something is worth holding on to before it knows why. The tier
chain was the first measurable form of that question, built for the thesis. Effect orbs are where it grows.

Orbs and observation are free parts. The goal is not, yet. Today it is decided by the scenario's family (tier chain has
its own termination, a continuous family would have another). Whether the goal becomes a part of its own, or stays a
property of the family, is open. A tier-chain goal needs tier orbs, so the coupling may be real and not an accident.

### The feel

- **Retro-futuristic, punk-edged, playful.** The GUI is a control panel you feel you are touching. It should not look
  like a standard research tool, and the graphics will be customised to match.
- **An unlabeled HUD, as a flavour.** The stats don't say what they track. You watch them and work it out, the way an
  agent works out what matters from its rewards. A readable mode can spell it out. This is a fun extra, not a
  requirement.
- **A high score tracker.** For the fun of it, and because it turns the thesis's single human baseline into a spread
  across players and runs.
- **Fun is earned by function.** An idea that is only fun does not go in. The tracker passes because it gives a human
  baseline. The unlabeled HUD passes because it mirrors the benchmark's own idea.

### Two ways in

Tinker at the panel and look for a combination that clicks, or pick a hard scenario and play it. The thesis scenarios
were built to measure, and a human finds them dull. Effect orbs are where a person gets real decisions: synergies,
converters and hazards, where one choice changes the value of another.

### How hard

Every scenario should be learnable by a small agent at its easy end, with headroom above. Turning a dial such as tier
depth makes a task longer for an agent, but not harder for a person, who sees the sequence at once. What makes a person
stop and think is choices that depend on each other. Where that is hard enough without being too hard for an agent is
judged by feel, then checked cheaply: play it yourself, then train a small agent on it. If it is interesting to play
and the agent still shows some learning, the balance is probably right. If the agent flatlines, back it off.

The agents in mind are small ones. Stable Baselines 3 sits at the low-to-middle end of what is out there. A well-funded
lab will use stronger methods, so for them the PPO-family agents are a baseline to beat.

### Who it is for

- **Students**, who tune an agent and watch what changes in the learning curves.
- **Agent builders**, who want to measure one specific capability. What they can measure is whatever the scenario's
  metrics expose, so a new mechanic needs new metrics alongside it.
- **Research labs**, if the scenarios get complex enough. The wording is deliberately careful: the combining design
  leaves room for harder scenarios a lab could use. It does not claim to replace puzzle benchmarks such as ARC-AGI,
  which test something different.
- Someone who just plays is a bonus, there for the feel. It is not a user type the design serves.

It has to be approachable for the first two: a clear README, a scenario that runs in a couple of commands, and curves
that can be read.

### How an agent sees it

The observation adapts to the agent, and the scenario stays the same. The range is flat vectors, a richer composite
layout, pixels (many agents expect images and a convolutional network), and text, so that a language model can play.
Only the vector observations exist today. A text view makes reading the grid its own task, so results should be
compared with care across observation types.

### How it grows

Eventually something like fifteen orbs and twenty scenarios. Every new orb multiplies what can be built, which is why
a big catalogue is worth having. Version one stays small: the thesis scenarios, a few continuous ones, and a first set
of effect orbs.

Every addition has to be clear, focused on one thing, and tested before it counts. An orb that doesn't add a distinct
mechanic is a near-duplicate, and doesn't go in. Combine existing parts first. New things are new self-contained parts,
not edits to old ones.

### What it is not

It is not a place for fun ideas that do nothing for the measuring. It does not promise to cover every kind of RL
benchmark. The long bet, that isolated orb types could train structural priors for something like an active inference
agent, is a possible future, written down in the thesis's future-work file. It is not part of this vision.

### Where it stands (2026-10-27)

The bones are there. One grid, one droid, orbs, and scenarios built from parts. The rebuild is making that cleaner,
and spatial and scaling sparse are built. Delay and dense are not. The face is mostly ahead: the GUI, the look, the
high score tracker, pixel and text observations, effect orbs, and scenarios that are interesting for a human. Being
about a third of the way is a rough guess, not a measurement.
