# SYNGrid

> A configurable benchmark for isolating Temporal Credit Assignment in reinforcement learning agents.
> 
> SYNGrid lets you vary temporal credit assignment difficulty through configuration rather than creating new benchmark environments. Instead of switching between dozens of tasks, you modify the settings you desire from a YAML file.
>
> **Status:** Research prototype. Evaluated across three scenarios in a [bachelor's thesis](https://urn.kb.se/resolve?urn=urn:nbn:se:miun:diva-57794).

---

## The Problem
In reinforcement learning, agents learn by trial and error — but which past actions actually mattered for success? This is the **Temporal Credit Assignment Problem (TCAP)**: figuring out which decisions in a long sequence deserve credit for an outcome, especially when the reward arrives much later.

Since Minsky posed it in 1961, it has remained hard. Most RL benchmarks, both big and focused ones, tend to entangle credit assignment with other challenges, making it hard to isolate where an agent actually struggles.

**The gap:** We lack lightweight, configurable benchmarks that let you dial credit assignment difficulty as a single tunable axis, isolate it from other challenges, and diagnose *how* agents break down.

SYNGrid is one attempt to fill that gap.

---

## What SYNGrid Does
SYNGrid is a configurable grid-based environment for isolating and diagnosing temporal credit assignment in RL agents.

**In its current form:** SYNGrid is a simple grid where agents collect **orbs** to earn rewards, and this simplicity is intentional. You don't create new environments for new tasks — instead, you dial configuration knobs in a YAML file to compound the challange instead of writing a totally new one:

- **Tier depth** — How long is the dependency chain? (tier 2 vs. tier 7)
- **Delay** — How many steps between valid orb consumptions? (stretches the temporal gap between action and reward)
- **Observability** — Does the agent see the full grid (MDP), or only a 3×3 window around itself (POMDP)?
- **Reward density** — Sparse (reward only on full chain completion) or dense (reward per correctly consumed orb)?
- **Grid size & spawn mechanics** — Adjust world complexity and orb availability

Change any of these, and the agent faces a different credit assignment challenge, without the need to touch code. A human-playable mode also lets you validate scenarios before training.

**The core mechanic today:** Currently, there are two orb types: tier orbs, which must be collected in strict order (tier 1, then tier 2, then tier 3, and so on — collecting them out of order resets the chain, and only a fully completed chain yields a reward), and negative orbs, which deliver a direct negative reward. 

**Where it's headed:** The roadmap intends to extend this foundation with **Effect Orbs** — transient mechanics (nullifier, converter, hazard, etc) that interact and combine into richer scenarios, while keeping the config-driven paradigm intact. This would be where the "SYN" stops being a stretch and starts earning its name.

### Three Validated Scenarios
These scenarios isolate three different dimensions of temporal credit assignment.

![Overview of the three SYNGrid scenarios with training results](/docs/img/scenario_overview.jpg)

**Spatial Memory (POMDP):** Full grid visibility is removed; the agent sees only a 3×3 agent-centric window. Now memory must do spatial reconstruction *and* sequence tracking.

**Tier Scaling (MDP):** Dependency chains grow (tier 4 → tier 7). Each additional tier causes the probability of randomly discovering the complete sequence (1/t!) to shrink factorially, making the reward signal increasingly sparse.

**Temporal Delay (MDP):** A delay is imposed between valid consumptions. The temporal gap between action and reward grows (30 → 390 steps), testing how far the learning signal can reach.

---

## What the Research Found
Three agents were trained across these scenarios: stateless PPO (no memory), frame-stacking PPO (fixed observation window), and recurrent PPO (LSTM-based episodic memory).

**Key findings:**

- **Memory mattered where it should.** In the spatial scenario, agents without memory fumbled; LSTM-based agents mapped the area methodically. In fully observable scenarios, memory made less difference.
- **Reward density was the real lever.** When sparse rewards made learning impossible, introducing step-wise feedback recovered all three agents.
- **Temporal distance degrades learning gradually, not catastrophically.** As delay increased, the signal weakened in a predictable curve (not a hard cliff). At delay 30, all agents learned; at delay 70+, learning started to become unstable.

**Bottom line:** SYNGrid isolates credit assignment to some extent, but with the caveat that reward density, not just memory architecture, determines learnability. It's a diagnostic tool, not a perfect isolation chamber. And that's useful.

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
All scenario parameters live in a single YAML config file. No code changes needed.

**Example:**

```yaml
world:
  grid_world_conf:    
    grid_rows: &grid_rows 5
    grid_cols: &grid_cols 5
    single_chain_mode: &single_chain_mode true
    delay_mode: false
    delay: 30
    max_tier_scoring: &max_tier_scoring true
    termination_on_max_tier: &termination_on_max_tier true
    curriculum_training: &curriculum_training false
    de_spawn_tiers: &de_spawn_tiers false
    max_tier: &max_tier 4
    max_active_orbs: &max_active_orbs 3
```

> Increasing only max_tier changes the dependency chain while leaving the rest of the environment unchanged.

See [`config/config.yaml`](./src/syn_grid/config/config.yaml) for all available options.

---

## Roadmap
**Effect Orbs**: The original vision for SYNGrid. Transient world states where an action's value only becomes clear later through interaction with something else. Examples:
- **Null orbs:** Nullify the next orb; force re-ordering.
- **Converter orbs:** Flip reward signs; change the landscape.
- **Hazard orbs:** Make regions costly to traverse.

These can be mixed, reconfigured, and composed without touching the base environment.

**GUI / Scenario Builder**: Hand-editing YAML configs works but isn't ideal. A visual builder that hides irrelevant settings would let you dial parameters, see previews, and save scenarios without needing to touch a file.

**Curriculum Learning Support**: A more dedicated system for incrementally increase tiers and orbs into the mix

---

## Design Philosophy
SYNGrid is built on a few convictions:

1. **Isolation matters.** A benchmark only tells you what it isolates. Mixing credit assignment with exploration, perception, and navigation obscures what an agent actually struggles with.
2. **Simplicity is powerful.** Four cardinal directions. Orbs as the only other entity besides the droid. Composable mechanics. No inventory, no combat, no procedural generation. The goal is clean signal, not complex environments.
3. **Configuration over code.** You shouldn't need to rewrite code to test a hypothesis. Scenarios should emerge from toggling mechanics, not rebuilding environments.
4. **Diagnosis over scorekeeping.** A good benchmark tells you *how* agents break down, not just *that* they break down. Where does memory matter? Where does reward density? What's the hard boundary? These questions matter more than a single leaderboard score.

---

## Broader Context
SYNGrid is motivated by a broader question: What can agents learn from reward alone? Unlike humans, RL agents typically begin without built-in knowledge about sequences or causality. SYNGrid provides a controlled environment for studying where reward alone is sufficient—and where it begins to break down.

---

## Related Work
**MiniGrid** ([Chevalier-Boisvert et al.](https://minigrid.farama.org/)) excels at isolation: each environment probes a specific aspect of learning (memory, instruction following, delayed reward). The trade-off: each task lives in its own environment. Testing a different mechanic means switching environments entirely or writing new code.

**MiniHack** ([Samvelyan et al.](https://minihack.readthedocs.io/)) is rich and highly configurable, supporting long action sequences and complex reward structures. The trade-off: it entangles credit assignment with navigation, perception, and combat. A comparable task (3-step key-room-staircase) requires ~10M steps in MiniHack but ~800k in SYNGrid—partly because SYNGrid's action space is simpler by design.

**SYNGrid's niche:** A single configurable grid where scenario complexity varies through composed mechanics (not environment switching), while keeping the action and observation spaces minimal. The bet is that you can build rich, multi-faceted credit assignment challenges without bloating the environment.

---

## Citation
If you use SYNGrid in research, please cite:

```bibtex
@thesis{J-manLans2026,
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

