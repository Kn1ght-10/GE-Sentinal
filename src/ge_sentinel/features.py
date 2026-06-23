"""Market-microstructure feature engineering.

Per (item, 5-minute bucket): robust volume/return z-scores against a rolling
1-day baseline, a bid-ask-style spread proxy from the high/low gap, a
price-vs-history ratio (the RMT signature), and liquidity tiering. Robust
statistics (median/MAD) are used throughout because manipulation is exactly
the tail we must not let contaminate the baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

MAD_K = 1.4826
EPS = 1e-9

FEATURES = ["vol", "vol_z", "ret1", "ret6", "ret_z", "spread", "spread_z",
            "px_ratio", "log_px_ratio", "zero_side", "liq_tier_code"]


def _roll_robust_z(s: pd.Series, w: int, minp: int, abs_floor: float = 1e-9) -> pd.Series:
    """Median/MAD z-score with a floor on the denominator. Zero-inflated series
    (illiquid items: long runs of identical/zero values) collapse the MAD to 0,
    which would turn any ordinary tick into an astronomical z — the single
    biggest false-positive source in illiquid-market surveillance. The floor is
    max(MAD, 0.5·rolling σ, abs_floor)."""
    med = s.rolling(w, min_periods=minp).median()
    mad = (s - med).abs().rolling(w, min_periods=minp).median()
    sd = s.rolling(w, min_periods=minp).std()
    denom = np.maximum(MAD_K * mad, 0.5 * sd.fillna(0.0)).clip(lower=abs_floor)
    return (s - med) / denom


def build_features(prices: pd.DataFrame, baseline: int | None = None) -> pd.DataFrame:
    """Input: long-form prices_5m rows. Output: feature frame keyed (item_id, ts)."""
    w = baseline or config.BASELINE_W
    minp = max(36, w // 6)
    df = prices.sort_values(["item_id", "ts"]).copy()

    g = df.groupby("item_id", group_keys=False)
    mid = df[["avg_high", "avg_low"]].mean(axis=1, skipna=True)
    df["mid"] = g.apply(lambda x: x.assign(_m=x[["avg_high", "avg_low"]]
                        .mean(axis=1, skipna=True))["_m"].ffill(limit=6))
    df["mid"] = df["mid"].fillna(mid)
    df["vol"] = df["high_vol"].fillna(0) + df["low_vol"].fillna(0)
    df["zero_side"] = ((df["high_vol"].fillna(0) == 0) ^ (df["low_vol"].fillna(0) == 0)).astype(int)

    df["logmid"] = np.log(df["mid"].clip(lower=EPS))
    df["ret1"] = g["logmid"].diff()
    df["ret6"] = g["logmid"].diff(6)

    df["vol_z"] = g["vol"].transform(lambda s: _roll_robust_z(s, w, minp, abs_floor=1.0)).clip(-5, 60)
    df["ret_z"] = g["ret1"].transform(lambda s: _roll_robust_z(s, w, minp, abs_floor=1e-3)).clip(-60, 60)

    both = df["avg_high"].notna() & df["avg_low"].notna()
    df["spread"] = np.where(both, (df["avg_high"] - df["avg_low"]) / df["mid"], np.nan)
    df["spread_z"] = g["spread"].transform(lambda s: _roll_robust_z(s, w, minp, abs_floor=1e-3)).clip(-30, 30)

    long_w = min(w * 7, max(w * 2, 1))
    px_med = g["mid"].transform(lambda s: s.rolling(long_w, min_periods=w // 2).median())
    df["px_ratio"] = (df["mid"] / (px_med + EPS)).clip(upper=200)
    df["log_px_ratio"] = np.log(df["px_ratio"].clip(lower=1e-3))

    # Liquidity tiers from each item's median volume (terciles across items).
    med_vol = df.groupby("item_id")["vol"].median()
    q1, q2 = med_vol.quantile([1 / 3, 2 / 3])
    tier = med_vol.apply(lambda v: 0 if v <= q1 else (1 if v <= q2 else 2))
    df["liq_tier_code"] = df["item_id"].map(tier).astype(int)  # 0 low / 1 mid / 2 high

    for c in ["ret1", "ret6", "vol_z", "ret_z", "spread_z", "log_px_ratio"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["px_ratio"] = df["px_ratio"].fillna(1.0)
    df["spread"] = df["spread"].fillna(0.0)
    return df
