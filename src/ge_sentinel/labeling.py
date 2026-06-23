"""Weak supervision over unlabeled market data.

Real fraud teams rarely have ground truth; they have heuristics, community
reports, and announcements. We encode those as labeling functions (LFs) that
vote {+1 manipulation, -1 benign, 0 abstain} and combine them with a weighted
label model into probabilistic weak labels for the supervised layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

DEFAULT_WEIGHTS = {"lf_pump": 1.0, "lf_rmt": 1.3, "lf_quiet": 0.6, "lf_patch": 1.5}


def lf_pump(f: pd.DataFrame) -> np.ndarray:
    """Coordinated buy pressure: volume blowout + sustained 30-min ramp."""
    return np.where((f["vol_z"] > 5) & (f["ret6"] > 0.08), 1, 0)


def lf_rmt(f: pd.DataFrame) -> np.ndarray:
    """The wiki-documented RMT signature: low-liquidity item, extraordinary
    price, near-zero volume, often one-sided."""
    hit = (f["px_ratio"] > 5) & (f["vol"] <= 3) & (f["liq_tier_code"] == 0)
    return np.where(hit, 1, 0)


def lf_quiet(f: pd.DataFrame) -> np.ndarray:
    """Plainly normal microstructure votes benign."""
    return np.where((f["vol_z"].abs() < 2) & (f["ret_z"].abs() < 2)
                    & (f["px_ratio"].between(0.8, 1.25)), -1, 0)


def lf_patch(f: pd.DataFrame, update_ts: list[int]) -> np.ndarray:
    """Buckets near an announced game update vote benign — moves there are
    expected repricing, the classic false-positive source."""
    if not update_ts:
        return np.zeros(len(f), dtype=int)
    ts = f["ts"].to_numpy()
    w = config.PATCH_SUPPRESS_W * config.PERIOD_S
    near = np.zeros(len(f), dtype=bool)
    for u in update_ts:
        near |= (ts >= u - w) & (ts <= u + 2 * w)
    return np.where(near, -1, 0)


def apply_lfs(features: pd.DataFrame, update_ts: list[int]) -> pd.DataFrame:
    out = features[["item_id", "ts"]].copy()
    out["lf_pump"] = lf_pump(features)
    out["lf_rmt"] = lf_rmt(features)
    out["lf_quiet"] = lf_quiet(features)
    out["lf_patch"] = lf_patch(features, update_ts)
    return out


def label_model(votes: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Weighted vote → probabilistic weak label (Snorkel-style, dependency-free)."""
    w = weights or DEFAULT_WEIGHTS
    score = sum(votes[c] * w[c] for c in w)
    out = votes[["item_id", "ts"]].copy()
    out["weak_score"] = score.astype(float)
    out["weak_prob"] = 1.0 / (1.0 + np.exp(-2.0 * out["weak_score"]))
    out["weak_label"] = (out["weak_score"] > 0).astype(int)
    return out
