"""Synthetic market generator with injected, labeled manipulation events.

Why synthetic exists in a real-data project: the live market has no ground
truth. The standard surveillance-engineering answer is to validate detectors
on a realistic simulator with KNOWN injected events, then deploy on real data
with weak labels. This module is that simulator. Every number it produces is
clearly tagged source="synth" and never mixed into real-data claims.

Injected event types
  pump_dump   coordinated ramp (+20–60%) on elevated volume, then full retrace
  rmt_spike   the RMT signature the wiki FAQ documents: a LOW-volume item
              traded at an extraordinary price for 1–3 buckets
  patch_shock benign negative control: permanent level shift on high volume,
              the kind of legitimate move a game update causes
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config

KIND_PUMP = "pump_dump"
KIND_RMT = "rmt_spike"
KIND_PATCH = "patch_shock"
MANIP_KINDS = {KIND_PUMP, KIND_RMT}

TIERS = ("high", "mid", "low")
TIER_SIGMA = {"high": 0.0015, "mid": 0.0035, "low": 0.0065}   # per-bucket ret vol
TIER_LAMBDA = {"high": 120.0, "mid": 22.0, "low": 1.6}        # volume intensity
TIER_SPREAD = {"high": 0.004, "mid": 0.014, "low": 0.035}     # frac high-low


@dataclass
class TruthEvent:
    item_id: int
    kind: str
    start_ts: int
    end_ts: int
    magnitude: float
    note: str = ""


def _seasonality(ts: np.ndarray) -> np.ndarray:
    tod = (ts % 86400) / 86400.0
    daily = 0.55 + 0.9 * np.clip(np.sin(np.pi * ((tod + 0.15) % 1.0)), 0, None) ** 1.2
    dow = ((ts // 86400) + 4) % 7  # unix epoch was a Thursday
    weekend = np.where((dow == 5) | (dow == 6), 1.18, 1.0)
    return daily * weekend


def generate(n_items: int = 40, days: int = 21, seed: int = 42,
             start_ts: int = 1_771_200_000,
             n_pump: int = 8, n_rmt: int = 6, n_patch: int = 4
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int]]:
    """Returns (items_df, prices_df, truth_df, update_timestamps)."""
    rng = np.random.default_rng(seed)
    period = config.PERIOD_S
    n_t = days * config.BUCKETS_PER_DAY
    ts = start_ts + period * np.arange(n_t)
    season = _seasonality(ts)
    split_idx = int(n_t * 0.7)  # eval split; ≥ half of manip events land after it

    # --- items ---------------------------------------------------------------
    item_ids = 900_000 + np.arange(n_items)  # far above real ID space
    tiers = rng.choice(TIERS, size=n_items, p=[0.25, 0.40, 0.35])
    base_px = 10 ** rng.uniform(2.0, 6.4, size=n_items)
    items_df = pd.DataFrame({
        "id": item_ids, "name": [f"synth_{t}_{i}" for i, t in zip(item_ids, tiers)],
        "members": True, "ge_limit": rng.integers(8, 10_000, n_items),
        "highalch": (base_px * rng.uniform(0.3, 0.8, n_items)).astype(int),
        "source": "synth", "tier": tiers,
    })

    # --- baseline paths --------------------------------------------------------
    mid = np.empty((n_items, n_t))
    vol_hi = np.empty((n_items, n_t), dtype=int)
    vol_lo = np.empty((n_items, n_t), dtype=int)
    for i in range(n_items):
        t = tiers[i]
        rets = rng.normal(0, TIER_SIGMA[t], n_t)
        rets[0] = 0
        mid[i] = base_px[i] * np.exp(np.cumsum(rets))
        lam = TIER_LAMBDA[t] * season
        vol_hi[i] = rng.poisson(lam * 0.5)
        vol_lo[i] = rng.poisson(lam * 0.5)

    truth: list[TruthEvent] = []

    def _pick_window(length: int, prefer_test: bool) -> int:
        lo = split_idx + 5 if prefer_test else 5
        hi = n_t - length - 5
        return int(rng.integers(lo, hi))

    # --- pump & dumps -----------------------------------------------------------
    pumpable = np.flatnonzero(np.isin(tiers, ["mid", "low"]))
    for k, i in enumerate(rng.choice(pumpable, size=n_pump, replace=False)):
        L = int(rng.integers(18, 60))
        s = _pick_window(2 * L, prefer_test=(k >= n_pump // 3))
        f = rng.uniform(1.20, 1.60)
        ramp = np.linspace(1.0, f, L)
        decay = np.linspace(f, 1.0, L)
        mid[i, s:s + L] *= ramp
        mid[i, s + L:s + 2 * L] *= decay
        vmult = rng.uniform(8, 20)
        vol_hi[i, s:s + 2 * L] = np.maximum(vol_hi[i, s:s + 2 * L], 1) * vmult
        vol_lo[i, s:s + 2 * L] = np.maximum(vol_lo[i, s:s + 2 * L], 1) * vmult
        truth.append(TruthEvent(int(item_ids[i]), KIND_PUMP,
                                int(ts[s]), int(ts[s + 2 * L - 1]), float(f),
                                f"ramp x{f:.2f} over {L} buckets, vol x{vmult:.0f}"))

    # --- RMT spikes (low-liquidity, extraordinary price, tiny volume) -----------
    low_items = np.flatnonzero(tiers == "low")
    rmt_items = rng.choice(low_items, size=n_rmt, replace=False)
    rmt_marks: list[tuple[int, int, int, float]] = []
    for k, i in enumerate(rmt_items):
        w = int(rng.integers(1, 4))
        s = _pick_window(w + 2, prefer_test=(k >= n_rmt // 3))
        f = rng.uniform(8, 18)
        rmt_marks.append((i, s, w, f))
        truth.append(TruthEvent(int(item_ids[i]), KIND_RMT,
                                int(ts[s]), int(ts[s + w - 1]), float(f),
                                f"low-vol trade at x{f:.1f} median price"))

    # --- patch shocks (benign negative controls) ---------------------------------
    update_idx = sorted(rng.choice(np.arange(int(n_t * 0.15), n_t - 30, dtype=int),
                                   size=n_patch, replace=False))
    update_ts = [int(ts[u]) for u in update_idx]
    for u in update_idx:
        affected = rng.choice(n_items, size=5, replace=False)
        for i in affected:
            shift = rng.uniform(0.75, 1.35)
            mid[i, u:] *= shift
            boost = rng.uniform(6, 12)
            vol_hi[i, u:u + 12] = np.maximum(vol_hi[i, u:u + 12], 1) * boost
            vol_lo[i, u:u + 12] = np.maximum(vol_lo[i, u:u + 12], 1) * boost
            truth.append(TruthEvent(int(item_ids[i]), KIND_PATCH,
                                    int(ts[u]), int(ts[u + 11]), float(shift),
                                    "game-update level shift (benign)"))

    # --- assemble long-form prices ------------------------------------------------
    frames = []
    for i in range(n_items):
        spread = TIER_SPREAD[tiers[i]] * rng.uniform(0.7, 1.3, n_t)
        hi = mid[i] * (1 + spread / 2)
        lo = mid[i] * (1 - spread / 2)
        hv, lv = vol_hi[i].copy(), vol_lo[i].copy()
        hi = np.where(hv == 0, np.nan, hi)
        lo = np.where(lv == 0, np.nan, lo)
        frames.append(pd.DataFrame({
            "item_id": int(item_ids[i]), "ts": ts,
            "avg_high": hi, "avg_low": lo, "high_vol": hv, "low_vol": lv,
            "source": "synth",
        }))
    prices = pd.concat(frames, ignore_index=True)

    # Overlay RMT signatures after assembly: one-sided extraordinary low-side trade.
    for i, s, w, f in rmt_marks:
        med = float(np.median(mid[i, max(0, s - 200):s]))
        m = (prices["item_id"] == int(item_ids[i])) & prices["ts"].isin(ts[s:s + w])
        prices.loc[m, "avg_low"] = med * f
        prices.loc[m, "avg_high"] = np.nan
        prices.loc[m, "low_vol"] = int(rng.integers(1, 4))
        prices.loc[m, "high_vol"] = 0

    truth_df = pd.DataFrame([asdict(e) for e in truth])
    split_ts = int(ts[split_idx])
    truth_df["in_test"] = truth_df["start_ts"] >= split_ts
    items_df.attrs["split_ts"] = split_ts
    prices.attrs["split_ts"] = split_ts
    return items_df, prices, truth_df, update_ts
