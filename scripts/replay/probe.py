#!/usr/bin/env python3
"""
Replay a fixed, seeded action tape through a scenario and record what comes back.

Every observation, reward, terminated and truncated flag and episode length is written to
<out>/<label>.npz, with a summary in <label>.json whose digests fingerprint the run. Record two
commits and compare them with diff.py: anything that differs, the code did. This is a change
detector, not a test; the tape is a random walk, not a policy.

A locked scenario is built from its base config, the scenario as it was locked, so a run's tunables
do not move the recording. --run-yaml builds it from its run YAML in config/yaml/scenarios/ instead,
as a run builds it. A scenario that is not locked, or a commit from before locks, is built from its
run YAML. Either way it goes through its registry entry. The summary says which source it used, and
the commit and whether the tree was dirty.

--baseline records the scenario's baseline: it writes to tests/regression/baselines/, named after
the scenario, where the regression test reads it. Imports sit inside functions, so PYTHONPATH can
point this at another checkout.

Usage:
    probe.py --label NAME [--scenario NAME] [--episodes N] [--out DIR] [--run-yaml]
    probe.py --baseline [--scenario NAME]

Example:
    PYTHONPATH=src python scripts/replay/probe.py --label before
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

ENV_SEED = 20260516
ACTION_SEED = 991
BASELINES = Path(__file__).resolve().parents[2] / "tests" / "regression" / "baselines"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--scenario", default="goal_single_tier_chain_spatial")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", default="output/replay_diff")
    ap.add_argument("--run-yaml", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()

    if args.baseline and (args.label or args.run_yaml or args.episodes != 200):
        ap.error("--baseline names, places and sizes the recording itself")
    if not args.baseline and not args.label:
        ap.error("--label is required without --baseline")

    import syn_grid
    from syn_grid.config.config_manager import ConfigManager
    from syn_grid.config.models.global_models import ScenarioName
    from syn_grid.gymnasium.utils.env_factory import make, register_env
    from syn_grid.scenario.registry import SCENARIOS, build_scenario

    name = ScenarioName(args.scenario)
    entry = SCENARIOS[name]
    # A commit from before locks has no lock on its entries
    lock = getattr(entry, "lock", None)
    if lock is not None and not args.run_yaml:
        with lock.base_config_path.open("r") as f:
            conf = entry.conf_class(**yaml.safe_load(f))
        source = "base config"
    else:
        manager = ConfigManager()
        conf = entry.full_conf(manager.load_raw(manager.scenario_config_path(name)))
        source = "run YAML"
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

    out, label = (
        (BASELINES, name.value) if args.baseline else (Path(args.out), args.label)
    )
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{label}.npz",
        obs=obs_mat,
        rewards=rew,
        terminated=term_arr,
        truncated=trunc_arr,
        ep_lens=lens,
    )

    package = Path(syn_grid.__file__).parent
    summary = {
        "label": label,
        "scenario": name.value,
        "source": source,
        "commit": commit_of(package),
        "syn_grid_path": str(package),
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
    (out / f"{label}.json").write_text(json.dumps(summary, indent=2) + "\n")

    shown = list(summary)[list(summary).index("config_digest") :]
    print(f"{label} ({source}): " + " ".join(f"{k}={summary[k]}" for k in shown))
    return 0


def commit_of(package: Path) -> str | None:
    """The commit the package's checkout is at, with "+dirty" if its tree has changes. None
    outside a git checkout."""

    def git(*cmd: str) -> str:
        return subprocess.run(
            ["git", "-C", str(package), *cmd],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--", ".")
    except (OSError, subprocess.CalledProcessError):
        return None

    return commit + ("+dirty" if dirty else "")


if __name__ == "__main__":
    raise SystemExit(main())
