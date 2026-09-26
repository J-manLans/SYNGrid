#!/usr/bin/env python3
"""
Replay a fixed, seeded action sequence through SYNGrid and record what comes back.

The environment is fed a prerecorded tape of random-but-deterministic actions
and every response is written to disk: observation, reward, terminated,
truncated, per episode. Run it against two commits and diff the two recordings.
Anything that differs, the code did it. Seconds per commit instead of GPU-days.

This is a *change detector*, not a test. It answers "does the environment
respond the same way to the same moves?" and nothing else. It cannot tell you
whether an agent learns, or whether a reward landscape is any good -- the tape
is a random walk, not a policy.

Usage:
    replay_probe.py --label <name> [options]

    --label NAME        required. Names the output files (<name>.npz/.json).
    --out DIR           output directory. Default: output/replay_diff
    --config-src PATH   YAML to build the config from.
                        Default: src/syn_grid/config/test_configs.yaml
    --episodes N        how many episodes to run. Default: 200
    --max-steps N       episode cap. Default: 60
    --single-chain      "true" exercises the chain-break path and gives short
                        episodes; "false" exercises the timeout path.
                        Default: false

Examples:
    # baseline for a refactor, both modes
    python scripts/replay_probe.py --label pre-sc   --single-chain true  --episodes 300
    python scripts/replay_probe.py --label pre-cont --single-chain false --episodes 300

    # same config, different commit
    PYTHONPATH=../syngrid-known-good/src \
        python scripts/replay_probe.py --label other --episodes 300

    # a specific scenario's own config
    python scripts/replay_probe.py --label spatial \
        --config-src reproduction_package/spatial_scenario/5x5/rppo.yaml

Output is <out>/<label>.npz (obs, rewards, terminated, truncated, ep_lens) plus
<out>/<label>.json, a human-readable summary whose "stream_digest" is the
fingerprint: equal digests mean equal behaviour.

Three design choices carry the weight, all of them learned the hard way:

* Point PYTHONPATH at the commit you want to probe, so `syn_grid` resolves to
  that checkout while this script and the config source stay pinned to the
  current one. This file is deliberately commit-independent -- all its
  `syn_grid` imports sit inside functions -- so one unchanged script can drive
  any commit. The moment you edit it to accommodate a code change, the
  comparison stops being valid.
* The config is read from --config-src rather than from the target commit's own
  config module. Historical commits ship a test_configs.yaml with an undefined
  YAML anchor, and older schemas require fields the current one dropped, so
  sourcing config per-commit would make them incomparable. Every field that
  matters is pinned explicitly, before validation, so a key a commit's schema
  lacks is ignored by pydantic and recorded in "skipped_pins" rather than
  raising.
* starting_score is pinned high so episodes survive to the horizon. Otherwise
  the droid's score drains on wall contact, every episode ends early, and the
  timeout branch is silently skipped -- the exact path a timeout-penalty change
  lives on. It also means the reward stream here is NOT comparable to a real
  training run's rewards; it is only valid for relative comparison between
  commits.

Pin chain_break_penalty in BOTH config blocks, and hold it constant when
comparing commits, or you will "confirm" nothing. The tape is fixed, the library
versions are whatever interpreter you run, and only the code under test varies.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ENV_SEED = 20260516
ACTION_SEED = 991
N_ACTIONS = 4  # LEFT, DOWN, RIGHT, UP

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_SRC = REPO / "src" / "syn_grid" / "config" / "test_configs.yaml"

# Spatial-scenario parameters. Pinned as raw dotted paths and applied before
# validation, so a key a revision's schema lacks is ignored by pydantic rather
# than raising, and the sidecar reports which pins that happened for.
PINS: dict[str, Any] = {
    "world.grid_world_conf.grid_rows": 5,
    "world.grid_world_conf.grid_cols": 5,
    "world.grid_world_conf.max_tier": 3,
    "world.grid_world_conf.max_tier_scoring": True,
    "world.grid_world_conf.delay_mode": False,
    "world.grid_world_conf.delay": 0,
    "world.grid_world_conf.de_spawn_tiers": False,
    "world.renderer_conf.grid_rows": 5,
    "world.renderer_conf.grid_cols": 5,
    "world.orb_factory_conf.grid_rows": 5,
    "world.orb_factory_conf.grid_cols": 5,
    "world.orb_factory_conf.max_tier": 3,
    "world.droid_conf.grid_rows": 5,
    "world.droid_conf.grid_cols": 5,
    # Kept high so wall contact cannot end the episode before the horizon.
    "world.droid_conf.starting_score": 100000.0,
    # May hardcoded -0.1 in the engine; pinned here so revisions that made it
    # configurable resolve to the same effective value.
    "world.droid_conf.chain_break_penalty": -0.1,
    "obs.observation_handler.perception": "vector_fog_of_war",
}


def deep_set(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = data
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def digest(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def build_conf(args: argparse.Namespace) -> tuple[Any, list[str]]:
    from syn_grid.config.models import FullConf

    data = yaml.safe_load(Path(args.config_src).read_text())
    data = copy.deepcopy(data)

    for k, v in PINS.items():
        deep_set(data, k, v)

    single_chain = args.single_chain == "true"
    mode_pins = {
        "world.grid_world_conf.single_chain_mode": single_chain,
        "world.grid_world_conf.curriculum_training": single_chain,
        "world.grid_world_conf.termination_on_max_tier": single_chain,
        "world.orb_factory_conf.single_chain_mode": single_chain,
        "obs.perception.single_chain_mode": single_chain,
        "obs.perception.curriculum_training": single_chain,
        "obs.observation_handler.max_steps": args.max_steps,
        "obs.perception.max_steps": args.max_steps,
    }
    for k, v in mode_pins.items():
        deep_set(data, k, v)

    try:
        conf = FullConf(**data)
    except Exception as exc:  # surfaced verbatim, per commit
        raise SystemExit(f"CONFIG REJECTED: {type(exc).__name__}: {exc}") from exc

    # Report which pins the resolved schema actually accepted.
    skipped: list[str] = []
    for dotted in list(PINS) + list(mode_pins):
        obj: Any = conf
        try:
            for p in dotted.split("."):
                obj = getattr(obj, p)
        except AttributeError:
            skipped.append(dotted)
    return conf, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="output/replay_diff")
    ap.add_argument("--config-src", default=str(DEFAULT_CONFIG_SRC))
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--single-chain", default="false", choices=["true", "false"])
    args = ap.parse_args()

    from gymnasium import spaces

    from syn_grid.gymnasium.environment import SYNGridEnv

    conf, skipped = build_conf(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = SYNGridEnv(conf.world, conf.obs)
    # Narrow before reading .n: env.action_space carries no annotation, so a type
    # checker sees the gymnasium base class Space, which has no .n. Asserting the
    # concrete type also gives a far better message than an AttributeError.
    action_space = env.action_space
    if not isinstance(action_space, spaces.Discrete):
        raise SystemExit(
            f"expected a Discrete action space, got {type(action_space).__name__}"
        )
    if action_space.n != N_ACTIONS:
        raise SystemExit(
            f"action space has {action_space.n} actions, expected {N_ACTIONS}"
        )

    action_rng = np.random.default_rng(ACTION_SEED)
    budget = args.episodes * args.max_steps
    # Precomputed and consumed by a global step counter, so a revision whose
    # episodes end at different steps still sees the same action at the same
    # index. Divergence in episode length is itself a reported signal.
    actions = action_rng.integers(0, N_ACTIONS, size=budget)

    obs_rows: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    ep_lens: list[int] = []
    cursor = 0

    for ep in range(args.episodes):
        o, _ = env.reset(seed=ENV_SEED + ep)
        obs_rows.append(np.asarray(o, dtype=np.float32).ravel())
        ep_len = 0
        while cursor < budget and ep_len < args.max_steps:
            o, r, term, trunc, _ = env.step(int(actions[cursor]))
            cursor += 1
            ep_len += 1
            obs_rows.append(np.asarray(o, dtype=np.float32).ravel())
            rewards.append(r)
            terminated.append(term)
            truncated.append(trunc)
            if term or trunc:
                break
        ep_lens.append(ep_len)

    # Not every commit defines close() on the env, and where it exists it may
    # raise. Either is fine here -- the recording is already complete.
    closer = getattr(env, "close", None)
    if callable(closer):
        with contextlib.suppress(Exception):
            closer()

    obs_mat = np.stack(obs_rows)
    rew = np.asarray(rewards, dtype=np.float64)
    term_arr = np.asarray(terminated, dtype=bool)
    trunc_arr = np.asarray(truncated, dtype=bool)
    lens = np.asarray(ep_lens, dtype=np.int64)

    np.savez_compressed(
        out_dir / f"{args.label}.npz",
        obs=obs_mat,
        rewards=rew,
        terminated=term_arr,
        truncated=trunc_arr,
        ep_lens=lens,
    )

    sidecar = {
        "label": args.label,
        "syn_grid_path": str(Path(__import__("syn_grid").__file__).parent),
        "single_chain": args.single_chain == "true",
        "max_steps": args.max_steps,
        "episodes": len(ep_lens),
        "steps_recorded": int(rew.size),
        "obs_dim": int(obs_mat.shape[1]),
        "reward_sum": float(rew.sum()),
        "terminated_count": int(term_arr.sum()),
        "truncated_count": int(trunc_arr.sum()),
        "timeouts": int(sum(1 for n in ep_lens if n >= args.max_steps)),
        "mean_ep_len": float(lens.mean()) if lens.size else 0.0,
        "obs_digest": hashlib.sha256(obs_mat.tobytes()).hexdigest()[:16],
        "reward_digest": hashlib.sha256(rew.tobytes()).hexdigest()[:16],
        "stream_digest": hashlib.sha256(obs_mat.tobytes() + rew.tobytes()).hexdigest()[
            :16
        ],
        "skipped_pins": skipped,
        "effective_config_digest": digest(
            {
                "world": conf.world.model_dump(),
                "obs": conf.obs.model_dump(),
            }
        ),
    }
    (out_dir / f"{args.label}.json").write_text(
        json.dumps(sidecar, indent=2, default=str)
    )

    print(
        f"{args.label}: steps={sidecar['steps_recorded']} obs_dim={sidecar['obs_dim']} "
        f"rew_sum={sidecar['reward_sum']:.3f} term={sidecar['terminated_count']} "
        f"trunc={sidecar['truncated_count']} timeouts={sidecar['timeouts']} "
        f"mean_ep={sidecar['mean_ep_len']:.1f} "
        f"stream={sidecar['stream_digest']} skipped={len(skipped)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
