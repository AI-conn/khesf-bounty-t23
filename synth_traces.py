"""Synthetic trace generator for the t=23 zero-gas bounty.

Zero dependencies, deterministic. Emits normalized 2D trajectory ticks and
discrete symbol-emission bursts across >=2 distinct interaction regimes.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

CHANNELS = ("DIRECT_KEY", "VIRTUAL_GRID", "BATCH_CHUNK")
KINDS = ("normal", "fatigued", "stranger", "mimic")
TRIGGERS = ("POINTER", "DIRECT_KEY", "VIRTUAL_GRID")


@dataclass(frozen=True)
class Profile:
    name: str
    regime: str                 # fast_continuous | slow_chunked
    speed: float                # normalized step size per tick
    speed_cv: float
    turn: float                 # mean turn angle (radians)
    turn_cv: float
    burst_gap: float            # ms between bursts
    burst_cv: float
    batch_share: float          # share of symbols emitted via BATCH_CHUNK
    edit_rate: float            # backspace/replace fraction


PROFILES = (
    Profile("fast_operator", "fast_continuous",
            speed=0.075, speed_cv=0.28, turn=0.42, turn_cv=0.34,
            burst_gap=210, burst_cv=0.30, batch_share=0.06, edit_rate=0.05),
    Profile("slow_operator", "slow_chunked",
            speed=0.021, speed_cv=0.46, turn=0.18, turn_cv=0.62,
            burst_gap=760, burst_cv=0.58, batch_share=0.55, edit_rate=0.14),
)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def generate_session(target: Profile, kind: str, seed: int) -> dict:
    """Return one session: trajectory ticks + symbol bursts."""
    rng = random.Random(seed)

    if kind == "normal":
        p = target
    elif kind == "fatigued":
        # same operator, degraded motor control: slower, more variable, more edits
        p = Profile(target.name + "_fatigued", target.regime,
                    target.speed * 0.85, target.speed_cv * 1.20,
                    target.turn * 1.05, target.turn_cv * 1.15,
                    target.burst_gap * 1.10, target.burst_cv * 1.15,
                    target.batch_share, target.edit_rate * 1.25)
    elif kind == "stranger":
        # An unknown operator: randomized parameters, never a clone of either
        # enrolled profile (a clone would be behaviorally indistinguishable).
        r = random.Random(seed ^ 0xA5A5A5)
        p = Profile("stranger_" + target.name, target.regime,
                    target.speed * r.uniform(0.45, 1.85),
                    target.speed_cv * r.uniform(0.85, 1.55),
                    target.turn * r.uniform(0.6, 1.7),
                    target.turn_cv * r.uniform(0.8, 1.6),
                    target.burst_gap * r.uniform(0.55, 1.9),
                    target.burst_cv * r.uniform(0.8, 1.7),
                    r.uniform(0.02, 0.92), r.uniform(0.01, 0.30))
    elif kind == "mimic":
        # parameter-matched: same mean speed, rate and channel mix as target;
        # higher-order rhythm/turn structure betrays the imitator.
        p = Profile(target.name + "_mimic", target.regime,
                    target.speed, target.speed_cv,
                    target.turn, target.turn_cv * 0.45,
                    target.burst_gap, target.burst_cv * 0.42,
                    target.batch_share, target.edit_rate)
    else:
        raise ValueError(kind)

    fast = p.regime == "fast_continuous"
    n_ticks = 260 if fast else 95
    base_dt = 11 if fast else 52
    x = rng.uniform(0.15, 0.85)
    y = rng.uniform(0.15, 0.85)
    heading = rng.uniform(0, 2 * math.pi)
    ticks = []
    ts = 0
    for i in range(n_ticks):
        dt = max(1.0, base_dt * (1.0 + rng.gauss(0, 0.22)))
        ts += dt
        heading += rng.gauss(0, p.turn * (1.0 + rng.gauss(0, p.turn_cv)))
        step = p.speed * (1.0 + rng.gauss(0, p.speed_cv))
        x = _clamp(x + step * math.cos(heading), 0.001, 0.999)
        y = _clamp(y + step * math.sin(heading), 0.001, 0.999)
        trigger = TRIGGERS[0] if rng.random() < 0.72 else (
            TRIGGERS[1] if rng.random() < 0.5 else TRIGGERS[2])
        ticks.append({"ts_ms": round(ts, 2), "x": round(x, 5),
                      "y": round(y, 5), "trigger_kind": trigger})

    # symbol-emission bursts, including edits
    n_bursts = 48 if fast else 32
    bursts = []
    ts = 0
    for event_id in range(n_bursts):
        ts += max(1.0, p.burst_gap * (1.0 + rng.gauss(0, p.burst_cv)))
        r = rng.random()
        channel = (CHANNELS[2] if r < p.batch_share else
                   (CHANNELS[0] if rng.random() < 0.6 else CHANNELS[1]))
        size = max(1, int(rng.gauss(4 if fast else 9, 1.6)))
        for _ in range(size):
            if rng.random() < p.edit_rate:
                action = "BACKSPACE" if rng.random() < 0.7 else "REPLACE"
            else:
                action = "SYMBOL"
            bursts.append({"ts_ms": round(ts + rng.uniform(0, 8), 2),
                           "channel": channel, "action": action,
                           "event_id": event_id})
    # A mimic is a distinct claimant pretending to be the target, never the
    # enrolled operator itself. Label it with its generated identity.
    identity = p.name if kind == "mimic" else target.name
    return {"profile": identity, "kind": kind, "regime": target.regime,
            "ticks": ticks, "bursts": bursts}


def build_dataset() -> dict:
    """6 calibration replicates and 3 holdout replicates per profile x kind."""
    data = {"calibration": [], "holdout": []}
    for pi, profile in enumerate(PROFILES):
        for ki, kind in enumerate(KINDS):
            base = pi * 1000 + ki * 100
            for rep in range(6):
                data["calibration"].append(generate_session(profile, kind, 1000 + base + rep))
            for rep in range(3):
                data["holdout"].append(generate_session(profile, kind, 7000 + base + rep))
    return data


if __name__ == "__main__":
    d = build_dataset()
    print("calibration sessions:", len(d["calibration"]),
          "holdout sessions:", len(d["holdout"]))
    s = d["holdout"][0]
    print("sample:", s["profile"], s["kind"], s["regime"],
          "ticks:", len(s["ticks"]), "bursts:", len(s["bursts"]))
