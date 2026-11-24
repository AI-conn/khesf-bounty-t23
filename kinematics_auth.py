#!/usr/bin/env python3
"""Channel-invariant continuity verifier + fixed-threshold FAR/FRR evaluation.

Zero dependencies. One-command run:  python3 kinematics_auth.py
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from synth_traces import KINDS, PROFILES, build_dataset


# ---------- channel-invariant feature extraction ----------

def _cv(values):
    m = statistics.fmean(values) if values else 0.0
    if m == 0 or len(values) < 2:
        return 0.0
    return statistics.pstdev(values) / abs(m)


def _autocorr(values):
    """Lag-1 autocorrelation; 0 for degenerate series."""
    if len(values) < 3:
        return 0.0
    m = statistics.fmean(values)
    num = sum((a - m) * (b - m) for a, b in zip(values, values[1:]))
    den = sum((v - m) ** 2 for v in values)
    return num / den if den else 0.0


def extract_features(session: dict) -> dict:
    ticks = session["ticks"]
    bursts = session["bursts"]

    # trajectory: normalize every rate by the session's own median inter-tick
    # interval, so no assumption is made about hardware sampling rate.
    dts = []
    speeds = []
    turns = []
    path = 0.0
    for a, b in zip(ticks, ticks[1:]):
        dt = max(1e-6, b["ts_ms"] - a["ts_ms"])
        dist = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        dts.append(dt)
        speeds.append(dist / dt)
        path += dist
        turns.append(math.atan2(b["y"] - a["y"], b["x"] - a["x"]))

    turn_deltas = []
    turn_signs = []
    for a, b in zip(turns, turns[1:]):
        d = b - a
        turn_deltas.append(abs(math.atan2(math.sin(d), math.cos(d))))
        turn_signs.append(1 if d >= 0 else -1)
    persistence = (statistics.fmean(1.0 if a == b else 0.0
                                    for a, b in zip(turn_signs, turn_signs[1:]))
                   if len(turn_signs) > 1 else 0.0)

    med_dt = statistics.median(dts) if dts else 1.0
    norm_speeds = [s * med_dt for s in speeds]
    speed_autocorr = _autocorr(norm_speeds)
    net = math.hypot(ticks[-1]["x"] - ticks[0]["x"], ticks[-1]["y"] - ticks[0]["y"]) if ticks else 0.0

    # burst events: one timestamp per emission event, not per symbol.
    event_ts = {}
    for b in bursts:
        eid = b.get("event_id", id(b))
        event_ts[eid] = min(b["ts_ms"], event_ts.get(eid, b["ts_ms"]))
    times = sorted(event_ts.values())
    gaps = [max(1e-6, b - a) for a, b in zip(times, times[1:])]
    med_gap = statistics.median(gaps) if gaps else 1.0
    norm_gaps = [g / med_gap for g in gaps]
    jitter = (statistics.fmean(abs(b - a) / med_gap for a, b in zip(gaps, gaps[1:]))
              if len(gaps) > 1 else 0.0)

    per_channel = Counter()
    edits = 0
    for b in bursts:
        per_channel[b["channel"]] += 1
        if b["action"] in ("BACKSPACE", "REPLACE"):
            edits += 1
    total = sum(per_channel.values()) or 1

    # burst sizes per channel (normalized by session total)
    sizes = defaultdict(list)
    last_ts = {}
    for b in bursts:
        c = b["channel"]
        if c in last_ts:
            sizes[c].append(b["ts_ms"] - last_ts[c])
        last_ts[c] = b["ts_ms"]
    burst_size_cv = statistics.fmean([_cv(v) for v in sizes.values() if v]) if sizes else 0.0

    return {
        "tick_dt_cv": _cv(dts),
        "norm_speed_cv": _cv(norm_speeds),
        "turn_mean_abs": statistics.fmean(turn_deltas) if turn_deltas else 0.0,
        "turn_cv": _cv(turn_deltas),
        "turn_persistence": persistence,
        "speed_autocorr": speed_autocorr,
        "path_efficiency": net / path if path else 0.0,
        "burst_gap_cv": _cv(norm_gaps),
        "burst_gap_jitter": jitter,
        "burst_size_cv": burst_size_cv,
        "edit_rate": edits / total,
        "batch_share": per_channel["BATCH_CHUNK"] / total,
        "direct_share": per_channel["DIRECT_KEY"] / total,
        "grid_share": per_channel["VIRTUAL_GRID"] / total,
    }


# ---------- fixed-threshold calibration (NO holdout leakage) ----------

FEATURES = (
    "norm_speed_cv", "turn_cv", "turn_persistence", "speed_autocorr",
    "burst_gap_cv", "burst_gap_jitter", "edit_rate",
    "batch_share", "direct_share", "grid_share",
)

# Fixed, holdout-independent acceptance bounds on per-feature z-scores.
# The lower bound is tighter: rhythm/motor degradation is directional.
Z_LOWER = 2.3
Z_UPPER = 3.5
SD_FLOOR_FRACTION = 0.20


def calibrate(dataset: dict) -> dict:
    models = {}
    for profile in PROFILES:
        rows = [extract_features(s) for s in dataset["calibration"]
                if s["profile"] == profile.name and s["kind"] == "normal"]
        if not rows:
            raise RuntimeError("no calibration rows for " + profile.name)
        mean = {f: statistics.fmean(r[f] for r in rows) for f in FEATURES}
        # variance floor keeps near-constant features from dominating the score
        sd = {f: max(statistics.pstdev([r[f] for r in rows]),
                     abs(mean[f]) * SD_FLOOR_FRACTION, 0.01)
              for f in FEATURES}
        models[profile.name] = {
            "mean": mean, "sd": sd,
            "z_lower": Z_LOWER, "z_upper": Z_UPPER,
            "calib_n": len(rows),
        }
    return models


def score(features, mean, sd) -> float:
    # RMS z-score: bounded influence from any one noisy feature.
    return math.sqrt(statistics.fmean(((features[f] - mean[f]) / sd[f]) ** 2 for f in FEATURES))


def accept(features, model) -> bool:
    """Fixed-interval accept/reject; all features must lie inside calibrated bounds."""
    return all(-model["z_lower"] <= (features[f] - model["mean"][f]) / model["sd"][f]
               <= model["z_upper"] for f in FEATURES)


# ---------- evaluation ----------

def evaluate(dataset: dict, models: dict) -> dict:
    genuine_total = genuine_ok = 0
    impostor_total = impostor_ok = 0
    fat_total = fat_ok = 0
    mimic_total = mimic_ok = 0
    stranger_total = stranger_ok = 0
    matrix = {p.name: {q.name: [0, 0] for q in PROFILES} for p in PROFILES}

    for session in dataset["holdout"]:
        feats = extract_features(session)
        claimant = session["profile"]
        kind = session["kind"]
        for owner, model in models.items():
            accepted = accept(feats, model)
            is_self = claimant == owner
            if is_self and kind in ("normal", "fatigued"):
                genuine_total += 1
                genuine_ok += int(accepted)
                if kind == "fatigued":
                    fat_total += 1
                    fat_ok += int(accepted)
            elif not is_self:
                impostor_total += 1
                impostor_ok += int(accepted)
                if kind == "mimic":
                    mimic_total += 1
                    mimic_ok += int(accepted)
                else:
                    stranger_total += 1
                    stranger_ok += int(accepted)
            if kind == "normal":
                matrix[owner][claimant][1] += 1
                matrix[owner][claimant][0] += int(accepted)

    return {
        "genuine": (genuine_ok, genuine_total),
        "fatigued_self": (fat_ok, fat_total),
        "impostor": (impostor_ok, impostor_total),
        "stranger": (stranger_ok, stranger_total),
        "mimic": (mimic_ok, mimic_total),
        "matrix": matrix,
        "bounds": {k: f"z in [{v['z_lower']}, {v['z_upper']}], sd_floor={SD_FLOOR_FRACTION}"
                   for k, v in models.items()},
    }


def main() -> int:
    data = build_dataset()
    models = calibrate(data)
    r = evaluate(data, models)

    print("=== FIXED-BOUND CALIBRATION (calibration split only; no holdout leakage) ===")
    for k, v in r["bounds"].items():
        print(f"  {k:16s} {v}")

    print("\n=== HOLDOUT FAR / FRR (exact counts) ===")
    for label, key in (("genuine self", "genuine"),
                       ("fatigued self", "fatigued_self"),
                       ("all impostor", "impostor"),
                       ("stranger", "stranger"),
                       ("parameter-matched mimic", "mimic")):
        ok, total = r[key]
        if label.startswith("genuine") or label.startswith("fatigued"):
            rate = (total - ok) / total if total else 0.0
            print(f"  {label:26s} FRR = {rate:.4f}  ({total - ok}/{total} rejected)")
        else:
            rate = ok / total if total else 0.0
            print(f"  {label:26s} FAR = {rate:.4f}  ({ok}/{total} accepted)")

    print("\n=== NxN CROSS-PROFILE SWAP MATRIX (holdout, normal kind) ===")
    cols = [q.name for q in PROFILES]
    print("  enrolled\\claimant " + "".join(f"{c:>18s}" for c in cols))
    for owner in cols:
        cells = []
        for claimant in cols:
            ok, total = r["matrix"][owner][claimant]
            cells.append(f"{ok}/{total} " + ("ACCEPT" if ok == total and total else "REJECT"))
        print(f"  {owner:18s}" + "".join(f"{c:>18s}" for c in cells))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
