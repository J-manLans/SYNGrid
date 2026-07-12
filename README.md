# SYNGrid

> A configurable benchmark framework for reinforcement learning.
>
> SYNGrid lets you systematically vary benchmark dimensions — such as temporal delay, dependency depth, observability, and reward density — through configuration rather than creating new environments. While designed around temporal credit assignment, the framework is intended to support controlled experiments within a single configurable grid.
>
> **Status:** Research prototype. Evaluated across three scenarios in a [bachelor's thesis](https://urn.kb.se/resolve?urn=urn:nbn:se:miun:diva-57794).

---

<p align="center">
  <img src="/docs/img/syngrid_scenarios.gif" alt="SyncGrid demonstration">
</p>

---

## The Problem
In reinforcement learning, agents learn by trial and error — but which past actions actually mattered for success? This is the **Temporal Credit Assignment Problem (TCAP)**: determining how credit for an eventual outcome should be assigned across earlier decisions, especially when rewards are delayed.

Since Minsky posed it in 1961, it has remained hard. Most RL benchmarks, both big and focused ones, tend to entangle credit assignment with other challenges, making it hard to isolate where an agent actually struggles.

**The gap:** We lack lightweight, configurable benchmarks that let you dial credit assignment difficulty as a single tunable axis, isolate it from other challenges, and diagnose *how* agents break down.

SYNGrid is one attempt to fill that gap.

---

## What SYNGrid Does
SYNGrid is a configurable grid-based environment for isolating and diagnosing temporal credit assignment in RL agents.

**In its current form:** SYNGrid is a deliberately simple grid world with only two entities: the agent and orbs. The agent moves in the four cardinal directions, and orbs are consumed automatically when stepped on. Rather than adding new game systems, complexity comes from composing orb mechanics by toggling what types of orbs will be active in the run. The goal isn't to build increasingly complex worlds, but to vary one benchmark dimension at a time:

- **Orb tier depth**: How long is the dependency chain? (tier 2 vs. tier 7)
- **Delay**: Decide how many steps after consumption of an orb the next one can spawn, increasing the temporal credit assignment horizon.
- **Observability**: Does the agent see the full grid (MDP), or only a 3×3 window around itself (POMDP)?
- **Reward density**: Sparse (reward only on full chain completion) or dense (reward per correctly consumed orb).
- **Grid size & spawn mechanics**: Adjust world complexity and orb availability.

Change any of these, and the agent faces a different credit assignment challenge, without the need to touch code. A human-playable mode also lets you validate scenarios before training.

**The core mechanic today:** Currently, there are two orb types: tier orbs, which must be collected in strict order (tier 1, then tier 2, then tier 3, and so on — collecting them out of order resets the chain), and negative orbs, which deliver a direct negative reward.

**Where it's headed:** The roadmap intends to extend this foundation with **Effect Orbs** — transient mechanics (nullifier, converter, hazard, etc) that interact and combine into richer scenarios, while keeping the config-driven paradigm intact. This would be where the "SYN" in SYNGrid stops being a stretch and starts earning its name.

### Three Validated Scenarios
The bachelor's thesis established a proof of concept through three scenarios, each isolating a different dimension of temporal credit assignment.

![Overview of the three SYNGrid scenarios with training results](/docs/img/scenario_overview.jpg)

**Spatial Memory (POMDP):** Full grid visibility is removed; the agent sees only a 3×3 agent-centric window. Now memory must do spatial reconstruction *and* sequence tracking.

**Tier Scaling (MDP):** Dependency chains grows under a sparse reward landscape (tier 4 → tier 7). Each additional tier causes the probability of randomly discovering the complete sequence (1/t!) to shrink factorially, making the reward signal increasingly sparse. Once the models fail, reward shaping is introduced at the failure point to test whether the ability to solve the task can be restored.

**Temporal Delay (MDP):** After each valid consumption, the next orb is delayed from spawning. The temporal gap between causally related actions grows (30 → 390 steps), testing how far the learning signal can reach.

---

## What the Research Found
Three agents were trained across these scenarios: stateless PPO (no memory), frame-stacking PPO (fixed observation window), and recurrent PPO (LSTM-based episodic memory).

**Key findings:**

- **Memory helped in partial observability.** In the spatial (POMDP) scenario, agents without memory fumbled in the fog of war, while the LSTM-based agent explored it methodically. In fully observable scenarios, memory made little to no difference.
- **Reward density had a large effect.** When sparse rewards made learning challenging, introducing step-wise feedback restored all three agents ability to solve the task.
- **Temporal distance degrades learning gradually, not catastrophically.** As delay increased, the signal weakened in steps (not a hard cliff). At delay 30, all agents learned; at delay 70+, learning started to become unstable. The exact boundary likely depends on agent architecture and reward structure.

**Bottom line:** SYNGrid is a configurable benchmark framework rather than a fixed one. By systematically varying dimensions within a fixed environment — delay, dependency depth, observability — SYNGrid enables controlled experiments that expose where learning begins to break down and under what conditions. The goal is diagnosis rather than scorekeeping: locating failure points, not simply comparing final performance.

For the full experimental details and analysis, see the [thesis](https://urn.kb.se/resolve?urn=urn:nbn:se:miun:diva-57794).

---

## Quick Start

### Installation

```bash
git clone https://github.com/J-manLans/SYNGrid.git
cd SYNGrid
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .[dev]
```

### Run

```bash
python3 -m syn_grid
```

This will train or evaluate an agent (or start a human play-testing session) based on settings in `config/config.yaml`.

For detailed setup and reproduction instructions, see [`reproduction_package/info.md`](./reproduction_package/info.md).

---

## Configuration
SYNGrid experiments are defined through configuration rather than code. Changing benchmark dimensions such as dependency depth, delay, or observability requires only updating the configuration file, not modifying the environment.

**Example:**

```yaml
world:
  grid_world_conf:
    delay_mode: true
    delay: 30
    max_tier: 3
```

> Changing only `delay` increases the temporal gap between consuming orbs while leaving the rest of the environment unchanged.

See [`config/config.yaml`](/src/syn_grid/config/configs.yaml) for all available options.

---

## Roadmap
**Effect Orbs**: The original vision for SYNGrid. Transient world states where an action's value only becomes clear later through interaction with something else. Examples:
- **Null orbs:** Nullify the next orb; force re-ordering.
- **Converter orbs:** Flip reward signs; change the landscape.
- **Hazard orbs:** Make regions costly to traverse.

These can be mixed, reconfigured, and composed without touching the base environment.

**Scenario Builder**: Hand-editing YAML works but isn't ideal. A visual builder would hide irrelevant settings, let you dial benchmark dimensions, preview scenarios, and save reusable experiments without touching the configuration file.

**Curriculum Learning Support**: A dedicated system for progressively introducing new mechanics and increasing task complexity.

---

## Related Work
[**MiniGrid**](https://minigrid.farama.org/) excels at isolation: each environment probes a specific aspect of learning (memory, instruction following, delayed reward). The trade-off: each task lives in its own environment. Testing a different mechanic means switching environments entirely or writing new code.

[**MiniHack**](https://minihack.readthedocs.io/) is rich and highly configurable, supporting long action sequences and complex reward structures. The trade-off: it entangles credit assignment with navigation, perception, and combat. A comparable task (3-step key-room-staircase) requires ~10M steps in MiniHack but ~800k in SYNGrid—partly because SYNGrid's action space is simpler by design.

**SYNGrid's niche:** A single configurable grid where scenario complexity varies through composed mechanics (not environment switching), while keeping the action and observation spaces minimal. The aim is to scale experimental complexity without scaling environmental complexity.

---

## Citation
If you use SYNGrid in research, please cite:

```bibtex
@thesis{Lansgren2026Evaluating,
  author = {Joel Lansgren},
  title = {Evaluating SYNGrid: A Configurable Benchmark Environment for Temporal Credit Assignment in Reinforcement Learning},
  school = {Mid Sweden University},
  year = {2026},
  url = {https://urn.kb.se/resolve?urn=urn:nbn:se:miun:diva-57794}
}
```

---

## License
[MIT](/LICENSE)

---

