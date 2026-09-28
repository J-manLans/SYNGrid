#!/usr/bin/env python3
"""
Record a behaviour fingerprint for one scenario, for use as a refactor lock.

replay_probe.py is the trusted change detector, but it pins scenario-defining
values (5x5, max_tier_scoring=true, fog-of-war, delay=0) so that two commits
differing only in code stay comparable. Those pins are exactly what you cannot
keep when the point is to compare *different scenarios* before and after a
scenario-selection refactor: they overwrite the very fields that make one
scenario differ from another.

This script keeps replay_probe's hard parts -- fixed seed, fixed action tape,
digest-based comparison, no opinionated pins -- and drops the rest. Scenario
identity comes from the config file you point it at. Parameters that must not
drift are passed explicitly with --override, so what is held constant is
visible in the command line rather than buried in a PINS dict.

Design constraints, same as replay_probe.py:

* This file is commit-independent. Every syn_grid import sits inside a function,
  so one unchanged script can drive any revision. Point PYTHONPATH at the
  revision under test; the config file stays here. The moment you edit this
  script to accommodate a code change, the comparison stops being valid.
* One scenario per process. OrbFactory.__init__ writes class attributes
  (BaseOrb._life_span, TierOrb.max_tier) shared by every GridWorld in the
  process, so locking two scenarios in one run lets the second silently
  rewrite the first's world. The probe has the same constraint.
* starting_score must be raised via --override. Otherwise the droid's score
  drains on wall contact, every episode ends early, and the timeout branch --
  the branch most of the termination refactor touches -- is never executed.
* Comparisons are exact, no epsilon. A float-reassociating refactor shows
  1e-16 differences; read the magnitudes before calling them behavioural.

Usage:
    scenario_lock.py --config PATH --label NAME [options]

    --config PATH     required. YAML to build the scenario from. Scenario
                      identity lives in this file, so post-refactor it is where
                      the new scenario selector goes.
    --label NAME      required. Names the output files.
    --out DIR         default: output/scenario_lock
    --episodes N      default: 120
    --max-steps N     default: 60
    --override JSON   dotted-path overrides applied to the raw YAML *before*
                      validation. '{"world.droid_conf.starting_score": 1e5}'
    --set NAME=VALUE  shorthand for a single --override entry.
    --verify PATH     record every scenario in the manifest and compare each
                      against its recorded digest. Exits non-zero on any
                      difference. This is the mode CI and humans should use.
    --record PATH     record every scenario in the manifest and write the
                      observed digests back to it. Only for deliberately
                      re-baselining, and the diff deserves an explanation.

Examples:
    # baseline
    python scripts/scenario_lock.py --config cfg/spatial.yaml --label pre-spatial \
        --override '{"world.droid_conf.starting_score": 100000.0}'

    # after the refactor, same scenario expressed the new way
    PYTHONPATH=../other/src python scripts/scenario_lock.py \
        --config cfg/spatial.yaml --label post-spatial \
        --override '{"world.droid_conf.starting_score": 100000.0}'

    # compare two recordings
    python scripts/scenario_lock.py --compare pre-spatial post-spatial

    # the check that matters
    python scripts/scenario_lock.py --verify scripts/scenario_lock_baseline.json

Output is <out>/<label>.npz (obs, rewards, terminated, truncated, ep_lens,
scores) plus <out>/<label>.json, whose "stream_digest" is the fingerprint:
equal digests mean equal behaviour.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ENV_SEED = 20260516
ACTION_SEED = 991
N_ACTIONS = 4  # LEFT, DOWN, RIGHT, UP

REPO = Path(__file__).resolve().parent.parent


# ============ #
#  Config prep #
# ============ #


def deep_set(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = data
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def coerce(text: str) -> Any:
    """Parse a --set value as JSON, falling back to the raw string."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def build_conf(config_path: str, overrides: dict[str, Any]):
    from syn_grid.config.models import FullConf

    data = copy.deepcopy(yaml.safe_load(Path(config_path).read_text()))

    for dotted, value in overrides.items():
        deep_set(data, dotted, value)

    try:
        return FullConf(**data)
    except Exception as exc:  # surfaced verbatim, per revision
        raise SystemExit(f"CONFIG REJECTED: {type(exc).__name__}: {exc}") from exc


# ============ #
#  Obs digest  #
# ============ #


def obs_signature(obs: Any) -> str:
    """Structural description of an observation, shape only, no values.

    Catches an observation space that changed shape, which would invalidate
    every reward comparison downstream of it.
    """

    if isinstance(obs, dict):
        inner = {k: list(np.shape(v)) for k, v in sorted(obs.items())}
        return "dict" + json.dumps(inner, sort_keys=True)
    return f"array{np.shape(obs)}:{np.asarray(obs).dtype}"


def obs_bytes(obs: Any) -> bytes:
    """Flatten any observation to bytes, stably, for hashing.

    Dict observations are sorted by key so the hash does not depend on the
    insertion order the perception happened to use.
    """

    if isinstance(obs, dict):
        parts = []
        for k in sorted(obs):
            parts.append(str(k).encode())
            parts.append(np.ascontiguousarray(obs[k], dtype=np.float32).tobytes())
        return b"|".join(parts)
    return np.ascontiguousarray(obs, dtype=np.float32).tobytes()


# ============ #
#    Record    #
# ============ #


def record(args: argparse.Namespace) -> dict[str, Any]:
    from syn_grid.gymnasium.environment import SYNGridEnv

    conf = build_conf(args.config, args.overrides)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scenario selection differs across the refactor: before it the environment
    # took (world, obs) and the scenario was implied by flags in the world
    # config, after it takes (scenario, world, obs). Both are supported so this
    # script can lock a scenario against the pre-refactor revision as well as
    # the current one -- which is the whole point of a lock during a refactor.
    # This is a signature shim only. The tape, the seeds, the episode budget and
    # the digests below are identical either way, so recordings stay comparable.
    scenario = getattr(conf, "scenario", None)
    env = (
        SYNGridEnv(scenario, conf.world, conf.obs)
        if scenario is not None
        else SYNGridEnv(conf.world, conf.obs)
    )

    action_rng = np.random.default_rng(ACTION_SEED)
    budget = args.episodes * args.max_steps
    # Precomputed and consumed by a counter, so a revision whose episodes end
    # at different steps still sees the same action at the same index.
    actions = action_rng.integers(0, N_ACTIONS, size=budget)

    obs_blobs: list[bytes] = []
    obs_sigs: list[str] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    ep_lens: list[int] = []
    scores: list[float] = []
    cursor = 0

    for ep in range(args.episodes):
        o, _ = env.reset(seed=ENV_SEED + ep)
        obs_blobs.append(obs_bytes(o))
        obs_sigs.append(obs_signature(o))
        ep_len = 0
        while cursor < budget and ep_len < args.max_steps:
            o, r, term, trunc, _ = env.step(int(actions[cursor]))
            cursor += 1
            ep_len += 1
            obs_blobs.append(obs_bytes(o))
            obs_sigs.append(obs_signature(o))
            rewards.append(r)
            terminated.append(term)
            truncated.append(trunc)
            if term or trunc:
                scores.append(env.world.droid.score)
                break
        else:
            scores.append(env.world.droid.score)
        ep_lens.append(ep_len)

    closer = getattr(env, "close", None)
    if callable(closer):
        with contextlib.suppress(Exception):
            closer()

    rew = np.asarray(rewards, dtype=np.float64)
    term_arr = np.asarray(terminated, dtype=bool)
    trunc_arr = np.asarray(truncated, dtype=bool)
    lens = np.asarray(ep_lens, dtype=np.int64)

    np.savez_compressed(
        out_dir / f"{args.label}.npz",
        rewards=rew,
        terminated=term_arr,
        truncated=trunc_arr,
        ep_lens=lens,
        scores=np.asarray(scores, dtype=np.float64),
    )
    # Observations are stored as a sha per step rather than raw: a full dump for
    # grid observations is hundreds of MB and nothing reads it. The digest chain
    # still detects the first differing step, which is what a diff needs.
    obs_chain = b"".join(obs_blobs)

    sidecar = {
        "label": args.label,
        "config": str(Path(args.config).resolve()),
        "overrides": args.overrides,
        "syn_grid_path": str(Path(__import__("syn_grid").__file__).parent),
        "episodes": len(ep_lens),
        "steps_recorded": int(rew.size),
        "obs_signature": obs_sigs[0] if obs_sigs else None,
        "obs_signatures_uniform": len(set(obs_sigs)) == 1,
        "reward_sum": float(rew.sum()),
        "terminated_count": int(term_arr.sum()),
        "truncated_count": int(trunc_arr.sum()),
        "timeouts": int(sum(1 for n in ep_lens if n >= args.max_steps)),
        "mean_ep_len": float(lens.mean()) if lens.size else 0.0,
        "obs_digest": hashlib.sha256(obs_chain).hexdigest()[:16],
        "reward_digest": hashlib.sha256(rew.tobytes()).hexdigest()[:16],
        "stream_digest": hashlib.sha256(obs_chain + rew.tobytes()).hexdigest()[:16],
        "effective_config_digest": hashlib.sha256(
            json.dumps(
                {"world": conf.world.model_dump(), "obs": conf.obs.model_dump()},
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:16],
    }
    (out_dir / f"{args.label}.json").write_text(json.dumps(sidecar, indent=2))

    print(
        f"{args.label}: steps={sidecar['steps_recorded']} "
        f"obs={sidecar['obs_signature']} "
        f"rew_sum={sidecar['reward_sum']:.3f} term={sidecar['terminated_count']} "
        f"trunc={sidecar['truncated_count']} timeouts={sidecar['timeouts']} "
        f"mean_ep={sidecar['mean_ep_len']:.1f} "
        f"stream={sidecar['stream_digest']}"
    )
    return sidecar


# ============ #
#    Compare   #
# ============ #


def compare(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    a_json = json.loads((out_dir / f"{args.label_a}.json").read_text())
    b_json = json.loads((out_dir / f"{args.label_b}.json").read_text())

    print(f"A = {args.label_a}\nB = {args.label_b}\n")

    ok = True

    for field in (
        "obs_signature",
        "obs_digest",
        "reward_digest",
        "stream_digest",
        "reward_sum",
        "terminated_count",
        "truncated_count",
        "timeouts",
        "mean_ep_len",
        "effective_config_digest",
    ):
        a, b = a_json.get(field), b_json.get(field)
        same = a == b
        if field in ("reward_sum", "mean_ep_len") and not same:
            try:
                drift = abs(float(b) - float(a))
                same = drift < 1e-9
                extra = f"  (delta={float(b) - float(a):+.3e})"
            except (TypeError, ValueError):
                extra = ""
        else:
            extra = ""
        if field == "effective_config_digest" and not same:
            extra = "  <- resolved config moved; read this before the rest"
        print(f"  {field:<26} {'OK  ' if same else 'DIFF'}  A={a} B={b}{extra}")
        ok = ok and same

    with (
        np.load(out_dir / f"{args.label_a}.npz") as za,
        np.load(out_dir / f"{args.label_b}.npz") as zb,
    ):
        for key in ("rewards", "terminated", "truncated", "ep_lens", "scores"):
            if key not in za or key not in zb:
                continue
            x, y = za[key], zb[key]
            if x.shape != y.shape:
                print(f"  {key:<26} DIFF  SHAPE A{x.shape} B{y.shape}")
                ok = False
                continue
            if not np.array_equal(x, y):
                idx = int(np.argmax(x != y))
                print(f"  {key:<26} DIFF  first at {idx}: A={x[idx]} B={y[idx]}")
                ok = False
            else:
                print(f"  {key:<26} OK    identical over {x.size} entries")

    print()
    if ok:
        print(f"IDENTICAL: {args.label_a} and {args.label_b} behave the same.")
    else:
        print(f"BEHAVIOUR CHANGED between {args.label_a} and {args.label_b}.")
    return 0 if ok else 1


# ============ #
#    Manifest  #
# ============ #


def run_manifest(manifest_path: str, out: str, record_mode: bool) -> int:
    """Record every scenario in a manifest and compare against its digest.

    One scenario per subprocess, because OrbFactory writes class attributes
    shared by every GridWorld in the process -- a single-process sweep would
    let each scenario rewrite the previous one's world.
    """

    import subprocess

    manifest = json.loads(Path(manifest_path).read_text())
    scenarios = manifest["scenarios"]
    failures: list[str] = []
    digests: dict[str, Any] = {}

    for name, spec in scenarios.items():
        label = f"cur-{name}"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(REPO / spec["config"]),
            "--label",
            label,
            "--out",
            out,
            "--episodes",
            str(spec.get("episodes", 120)),
            "--max-steps",
            str(spec.get("max_steps", 60)),
            "--override",
            json.dumps(spec.get("overrides", {})),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(f"  {name:<38} RECORD FAILED")
            print(
                "    "
                + (proc.stderr.strip() or proc.stdout.strip()).replace("\n", "\n    ")
            )
            failures.append(name)
            continue

        got = json.loads((Path(out) / f"{label}.json").read_text())
        digests[name] = {
            "stream_digest": got["stream_digest"],
            "obs_digest": got["obs_digest"],
            "reward_digest": got["reward_digest"],
            "obs_signature": got["obs_signature"],
            "reward_sum": got["reward_sum"],
            "terminated_count": got["terminated_count"],
            "timeouts": got["timeouts"],
            "mean_ep_len": got["mean_ep_len"],
        }
        want = spec.get("expected", {})
        print(proc.stdout.strip())

        if record_mode:
            continue

        diffs = [
            f"{k}: want={want.get(k)} got={digests[name][k]}"
            for k in digests[name]
            if k in want and want[k] != digests[name][k]
        ]
        if diffs:
            print(f"  {name:<38} DIFF")
            for d in diffs:
                print(f"    {d}")
            failures.append(name)
        else:
            print(f"  {name:<38} OK")

    if record_mode:
        manifest["scenarios"] = {
            name: {**specs, "expected": digests.get(name, specs.get("expected", {}))}
            for name, specs in scenarios.items()
        }
        Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nrecorded {len(scenarios)} scenarios to {manifest_path}")
        return 0

    print()
    if failures:
        print(
            f"BEHAVIOUR CHANGED in {len(failures)} scenario(s): {', '.join(failures)}"
        )
        return 1
    print(f"ALL {len(scenarios)} SCENARIOS IDENTICAL to the recorded baseline.")
    return 0


# ============ #
#      CLI     #
# ============ #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--config")
    ap.add_argument("--label")
    ap.add_argument("--out", default=str(REPO / "output" / "scenario_lock"))
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--override", default="{}")
    ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), dest="compare")
    ap.add_argument("--verify", metavar="MANIFEST")
    ap.add_argument("--record", metavar="MANIFEST")
    args = ap.parse_args()

    if args.verify or args.record:
        target = args.verify or args.record
        return run_manifest(target, args.out, record_mode=bool(args.record))

    if args.compare:
        args.label_a, args.label_b = args.compare
        return compare(args)

    if not args.config or not args.label:
        ap.print_help()
        return 2

    overrides: dict[str, Any] = json.loads(args.override)
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects PATH=VALUE, got {item!r}")
        path, value = item.split("=", 1)
        overrides[path] = coerce(value)
    args.overrides = overrides

    record(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
