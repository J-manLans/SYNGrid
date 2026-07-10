# SYNGrid

> A configurable benchmark for isolating Temporal Credit Assignment in reinforcement learning agents.
>
> **Status:** Research prototype. Evaluated across three scenarios in a [bachelor's thesis](https://urn.kb.se/resolve?urn=urn:nbn:se:miun:diva-57794).

---

## The Problem
In reinforcement learning, agents learn by trial and error—but which past actions actually mattered for success? This is the **Temporal Credit Assignment Problem (TCAP)**: figuring out which decisions in a long sequence deserve credit for an outcome, especially when the reward arrives much later.

Since Minsky posed it in 1961, it's remained hard. Most RL benchmarks entangle credit assignment with exploration, perception, and navigation, making it impossible to isolate where an agent actually struggles. The ones that try to isolate it are often heavyweight (requiring millions of steps) or inflexible (fixed tasks, no easy configuration).

**The gap:** We lack lightweight, configurable benchmarks that let you dial credit assignment difficulty as a single tunable axis, isolate it from other challenges, and diagnose *how* agents break down.

SYNGrid is one attempt to fill that gap.

---

## What SYNGrid Does
SYNGrid is a grid-based environment where an agent collects **orbs** to earn rewards. The simplicity is intentional.

**The core mechanic:** Some orbs (**tier orbs**) must be collected in a specific order: tier 1, then tier 2, then tier 3, and so on. Collect them out of order and the chain resets. Only completing a full chain yields reward.

**The power:** You don't create new environments for new tasks. Instead, you dial a few knobs in a config file:
- **Tier depth** — How long is the dependency chain? (tier 2 vs. tier 7)
- **Delay** — How many steps between valid orb consumptions? (stretches the temporal gap between action and reward)
- **Observability** — Does the agent see the full grid (MDP), or only a 3×3 window around itself (POMDP)?
- **Reward density** — Sparse (reward only on full chain completion) or dense (reward per correctly consumed orb)?

Change any of these, and the agent faces a different credit assignment challenge—all without touching code. A human-playable mode lets you validate scenarios before training.

### Three Validated Scenarios
![Overview of the three SYNGrid scenarios with training results](/docs/img/scenario_overview.jpg)

**Spatial Memory (POMDP):** Full grid visibility is removed; the agent sees only a 3×3 window. Now memory must do spatial reconstruction *and* sequence tracking.

**Tier Scaling (MDP):** Dependency chains grow (tier 4 → tier 7). As chains lengthen, the chance of stumbling on a complete sequence drops exponentially (1/6! → 1/7!), thinning the reward signal.

**Temporal Delay (MDP):** A delay is imposed between valid consumptions. The temporal gap between action and reward grows (30 → 390 steps), testing how far the learning signal can reach.

---

## What the Research Found
Three agents were trained across these scenarios: stateless PPO (no memory), frame-stacking PPO (fixed observation window), and recurrent PPO (LSTM-based episodic memory).

**Key findings:**

- **Memory mattered where it should.** In the spatial scenario, agents without memory fumbled; LSTM-based agents mapped the area methodically. In fully observable scenarios, memory made less difference.
- **Reward density was the real lever.** When sparse rewards made learning impossible, introducing step-wise feedback recovered all three agents. This concretely demonstrates the entanglement that existing benchmarks struggle to isolate.
- **Temporal distance degrades learning gradually, not catastrophically.** As delay increased, the signal weakened in a predictable curve (not a hard cliff). At delay 30, all agents learned; at delay 70+, learning became unstable. This suggests a learnable boundary beyond which signal decay overpowers architectural advantages.
- **Config-driven design works.** Changing a single parameter produced the intended behavioral shift. Agents' paths, loop patterns, and exploration strategies all reflected what the scenario was designed to test.

**Bottom line:** SYNGrid isolates credit assignment to some extent—but with the caveat that reward density, not just memory architecture, determines learnability. It's a diagnostic tool, not a perfect isolation chamber. And that's useful.

For the full experimental details and analysis, see the [thesis](https://urn.kb.se/resolve?urn=urn:nbn:se:miun:diva-57794).

---

## Current State
**Implemented:**
- Tier orbs (sequential collection)
- Negative orbs (consumption penalty)
- MDP and POMDP observation modes
- Config-driven scenario construction (YAML-based, no code changes)
- Human-runner mode for playtesting
- Modular observation space (distance-sorted, scalar orb identity)
- Base agent runner infrastructure (SB3/SB3-contrib compatible)
- Episode logging (CSV) and plotting module
- Three validated scenarios (spatial, tier scaling, delay)

**Not yet implemented:**
- Effect orbs (transient mechanics: synergy, converter, hazard)
- GUI / scenario builder

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
  grid_size: 8
  tier_depth: 3
  delay: 30
  observability: "pomdp"  # or "mdp"
  reward_density: "sparse"  # or "dense"

orbs:
  active_count: 3
  types: ["tier", "negative"]

agent:
  architecture: "stateless"  # "stateless", "frame_stacking", "recurrent"
  timesteps: 500_000
```

See [`config/config.yaml`](./src/syn_grid/config/config.yaml) for all available options.

---

## Roadmap
**Effect Orbs** — The original vision for SYNGrid. Transient world states where an action's value only becomes clear later through interaction with something else. Examples:
- **Null orbs:** Nullify the next orb; force re-ordering.
- **Converter orbs:** Flip reward signs; change the landscape.
- **Hazard orbs:** Make regions costly to traverse.

These can be mixed, reconfigured, and composed without touching the base environment. This is where the "SYN" (synergy) in SYNGrid comes alive—not isolated challenges, but *combined* complexity from simple mechanics.

**GUI / Scenario Builder** — Hand-editing YAML configs works but isn't ideal. A visual builder would let you dial parameters, see previews, and save scenarios without needing to touch a file.

**Curriculum Learning Support** — Using active orb count and tier depth as difficulty axes for staged training.

For detailed reasoning and prior experiments, see [`SYNGrid_Future_Work.md`](./SYNGrid_Future_Work.md).

---

## Design Philosophy
SYNGrid is built on a few convictions:

1. **Isolation matters.** A benchmark only tells you what it isolates. Mixing credit assignment with exploration, perception, and navigation obscures what an agent actually struggles with.
2. **Simplicity is powerful.** Four cardinal directions. Two orb types (soon three). Composable mechanics. No inventory, no combat, no procedural generation. The goal is clean signal, not complex environments.
3. **Configuration over code.** You shouldn't need to rewrite code to test a hypothesis. Scenarios should emerge from toggling mechanics, not rebuilding environments.
4. **Diagnosis over scorekeeping.** A good benchmark tells you *how* agents break down, not just *that* they break down. Where does memory matter? Where does reward density? What's the hard boundary? These questions matter more than a single leaderboard score.

---

## Broader Context
The deeper question underlying SYNGrid: **What can tabula rasa agents learn from reward alone?**

Humans arrive with priors—intuitions about causality, sequences, goals. Agents start with nothing. They must learn "do things in order" *and* learn the specific order *simultaneously*, both from a single reward signal. We wouldn't expect a child to figure out ordered tasks with no structure; why do we expect agents to?

Early active inference work (like AXIOM from VERSES AI) suggests that agents equipped with structured priors about dynamics learn orders of magnitude faster than reward-maximizing agents alone. Similarly, neuroscience shows that biological intelligence isn't shaped by reward alone, but by overlapping drives: predictability, effort cost, and outcome.

Whether this narrowness in RL agents is a limitation worth addressing, or simply a different design point, remains open. SYNGrid is a tool for exploring that question empirically.

---

## Related Work
**MiniGrid** ([Chevalier-Boisvert et al.](https://minigrid.farama.org/)) excels at isolation: each environment probes a specific aspect of learning (memory, instruction following, delayed reward). The trade-off: each task lives in its own environment. Testing a different mechanic means switching environments entirely or writing new code.

**MiniHack** ([Samvelyan et al.](https://minihack.readthedocs.io/)) is rich and highly configurable, supporting long action sequences and complex reward structures. The trade-off: it entangles credit assignment with navigation, perception, and combat. A comparable task (3-step key-room-staircase) requires ~10M steps in MiniHack but ~800k in SYNGrid—partly because SYNGrid's action space is simpler by design.

**SYNGrid's niche:** A single configurable grid where scenario complexity varies through composed mechanics (not environment switching), while keeping the action and observation spaces minimal. The bet is that you can build rich, multi-faceted credit assignment challenges without bloating the environment.

---

## Citation
If you use SYNGrid in research, please cite:

```bibtex
@thesis{J-manLans2025,
  author = {J-manLans},
  title = {SYNGrid: A Configurable Benchmark for Temporal Credit Assignment},
  school = {Mid Sweden University},
  year = {2025},
  url = {https://github.com/J-manLans/SYNGrid}
}
```

---

## License

[Add your license here, e.g., MIT, Apache 2.0, etc.]

---

## Acknowledgments

Thanks to my supervisor Rodi Jolak for keeping me on track and ensuring the work stayed clear and readable. Thanks to my wife for patience through late nights, my father for taking care of the dog, and to AI assistants who helped me think through problems and refine ideas while working remotely.
