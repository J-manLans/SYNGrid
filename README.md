# SynergyGrid
> Status: Research prototype. SYNGrid was developed as part of a [bachelor's thesis](https://urn.kb.se/resolve?urn=urn:nbn:se:miun:diva-57794) evaluating configurable benchmarks for Temporal Credit Assignment (TCA) in reinforcement learning. The core mechanics (tier orbs) are implemented and empirically validated; the broader vision (effect orbs, GUI) is in progress. See Roadmap below, and the thesis for full methodology and results.

## What is SYNGrid?
SYNGrid is a lightweight, grid-based benchmark for measuring how well reinforcement learning agents assign credit across long action sequences — the problem of figuring out which past action actually mattered for a reward that arrives much later.

The agent navigates a grid collecting **orbs**. Some orbs (**tier orbs**) must be collected in a specific order — tier 1, then tier 2, then tier 3, and so on. Collect them out of order and the chain breaks. This gives a simple, controllable way to create long-term dependencies: the deeper the tier chain, the further back an agent has to reason to know which early decision set up its later success.

Rather than being a fixed task, SYNGrid is built to be **reconfigured, not rewritten**. A single YAML config file controls things like:
- **Tier depth** — how long the dependency chain is
- **Delay** — a forced waiting period between valid orb consumptions, stretching the temporal gap between action and reward
- **Observability** — full grid view (MDP) vs. a limited agent-centric window (POMDP), adding a spatial memory demand
- **Reward density** — sparse (reward only on full chain completion) vs. dense (reward per correctly consumed orb)

Changing any of these doesn't require touching the codebase — just the config file. This is what makes SYNGrid useful as a diagnostic tool rather than a single fixed task: you can dial difficulty up or down along interpretable axes and watch exactly where an agent's performance breaks down.

A human-playable mode is also included, letting you sanity-check whether a scenario is actually balanced (not trivially easy, not unlearnable) before spending compute on training an agent against it.

![Overview of the three SYNGrid scenarios with training results](/docs/img/scenario_overview.jpg)

## Current State

SYNGrid has been evaluated across three scenarios (spatial memory, tier scaling, and temporal delay) using three PPO-family agents with varying memory capacity (stateless, frame-stacking, recurrent/LSTM). Full results are in the [thesis](https://urn.kb.se/resolve?urn=urn:nbn:se:miun:diva-57794).

**Implemented:**
- Tier orbs — sequential collection mechanic (direct + tier types)
- Negative orbs — gives a consumption penalty
- MDP and POMDP observation modes
- Config-driven scenario construction (YAML, no code changes required)
- Human-runner mode for manual playtesting and balance tuning
- Modular observation space (1D vector, distance-sorted, scalar orb identity encoding)
- Base agent runner infrastructure (SB3 / SB3-contrib compatible)
- Episode logging (CSV) and plotting module for training/evaluation results
- Three validated scenarios: spatial, tier scaling, delay

**Not yet implemented:**
- Effect orbs (transient synergy/hazard/converter mechanics)
- GUI / scenario builder

## Roadmap

- **Effect orbs** — the original motivating mechanic: transient world states where an action's value only becomes clear later, through interaction with something else (synergy, converter, and hazard orbs). This is where SYNGrid's "SYN" (synthesis) becomes literal — see [`SYNGrid_Future_Work.md`](./SYNGrid_Future_Work.md) for the full design reasoning.
- **GUI / scenario builder** — reduce the overhead of hand-editing YAML configs; expose only the relevant variables for the current setup.
- **Curriculum learning support** — using active orb count and tier depth as tunable difficulty axes for staged training.
For the full reasoning behind each direction, see [`SYNGrid_Future_Work.md`](./SYNGrid_Future_Work.md).

---


## Requirements
- Python 3.10 – 3.12
- pip

---

## Installation
### 1. Clone the repository

```bash
git clone https://github.com/J-manLans/SynergyGrid.git
cd synergygrid
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- macOS/Linux:

```bash
source .venv/bin/activate
```

- Windows:

```bash
.venv\Scripts\activate
```

### 3. Install the project

Install in editable mode with development dependencies:

```bash
pip install -e .[dev]
```

## Quick Start

```bash
python3 -m syn_grid
```

This will:

- Train or evaluate an agent or start a human play-testing run depending on the settings in the configs.yaml file. For more detailed information read the [config file](/src/syn_grid/config/configs.yaml) itself

For a more detailed instruction in how to train or reproduce any of the experiments, take a look at the [info](/reproduction_package/info.md) file in the `reproduction_package` folder.
