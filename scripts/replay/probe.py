#!/usr/bin/env python3
"""
Replay a fixed, seeded action tape through a scenario and record what comes back.

Every observation, reward, terminated and truncated flag and episode length is written to
<out>/<label>.npz, with a summary in <label>.json whose digests fingerprint the run. Record two
commits and compare them with diff.py: anything that differs, the code did. This is a change
detector, not a test; the tape is a random walk, not a policy.

The scenario is built from its run YAML in config/yaml/scenarios/, through its registry entry, as a
run builds it. Imports sit inside functions, so PYTHONPATH can point this at another checkout.

Usage:
    probe.py --label NAME [--scenario NAME] [--episodes N] [--out DIR]

Example:
    PYTHONPATH=src python scripts/replay/probe.py --label before
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ENV_SEED = 20260516
ACTION_SEED = 991


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--scenario", default="goal_single_tier_chain_spatial")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", default="output/replay_diff")
    args = ap.parse_args()

    import syn_grid
    from syn_grid.config.config_manager import ConfigManager
    from syn_grid.config.models.global_models import ScenarioName
    from syn_grid.gymnasium.utils.env_factory import make, register_env
    from syn_grid.scenario.registry import SCENARIOS, build_scenario

    name = ScenarioName(args.scenario)
    manager = ConfigManager()
    conf = SCENARIOS[name].full_conf(
        manager.load_raw(manager.scenario_config_path(name))
    )
    max_steps = conf.world_conf.max_steps

    register_env()
    env = make(build_scenario(name, conf), render_mode=None)

    # One tape read by one cursor, so a commit whose episodes end elsewhere still sees the same
    # action at the same index
    budget = args.episodes * max_steps
    actions = np.random.default_rng(ACTION_SEED).integers(
        0, env.action_space.n, size=budget
    )

    obs, rewards, terminated, truncated, ep_lens = [], [], [], [], []
    cursor = 0
    for ep in range(args.episodes):
        o, _ = env.reset(seed=ENV_SEED + ep)
        # The perception returns the same buffer every step, so each row must be a copy
        obs.append(np.array(o, dtype=np.float32).ravel())
        ep_len = 0
        # The environment ends the episode; the budget only guards a scenario that never does
        while cursor < budget:
            o, r, term, trunc, _ = env.step(int(actions[cursor]))
            cursor += 1
            ep_len += 1
            obs.append(np.array(o, dtype=np.float32).ravel())
            rewards.append(r)
            terminated.append(term)
            truncated.append(trunc)
            if term or trunc:
                break
        ep_lens.append(ep_len)
    env.close()

    obs_mat = np.stack(obs)
    rew = np.asarray(rewards, dtype=np.float64)
    term_arr = np.asarray(terminated, dtype=bool)
    trunc_arr = np.asarray(truncated, dtype=bool)
    lens = np.asarray(ep_lens, dtype=np.int64)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{args.label}.npz",
        obs=obs_mat,
        rewards=rew,
        terminated=term_arr,
        truncated=trunc_arr,
        ep_lens=lens,
    )

    summary = {
        "label": args.label,
        "scenario": name.value,
        "syn_grid_path": str(Path(syn_grid.__file__).parent),
        "config_digest": digest(conf.model_dump_json().encode()),
        "episodes": int(lens.size),
        "steps": int(rew.size),
        "obs_dim": int(obs_mat.shape[1]),
        "distinct_obs": len({row.tobytes() for row in obs_mat}),
        "reward_sum": float(rew.sum()),
        "terminated": int(term_arr.sum()),
        "truncated": int(trunc_arr.sum()),
        "reached_max_steps": int((lens >= max_steps).sum()),
        "mean_ep_len": float(lens.mean()),
        "obs_digest": digest(obs_mat.tobytes()),
        "reward_digest": digest(rew.tobytes()),
        "stream_digest": digest(obs_mat.tobytes() + rew.tobytes()),
    }
    (out / f"{args.label}.json").write_text(json.dumps(summary, indent=2))

    print(f"{args.label}: " + " ".join(f"{k}={summary[k]}" for k in list(summary)[4:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
