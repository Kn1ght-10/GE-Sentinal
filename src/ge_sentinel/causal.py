"""Causal layer: event studies on market shocks.

Difference-style event study: cumulative abnormal log-return of a treated
basket vs a control basket around an event timestamp, with a bootstrap CI
over treated items. Demonstrated on synthetic patch shocks (known truth);
the identical function applies to real natural experiments once daily
history is backfilled (e.g. the March 2019 Venezuela blackout, which press
coverage links to supply disruption on farmed items).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _logmid_panel(prices: pd.DataFrame, item_ids: list[int]) -> pd.DataFrame:
    df = prices[prices["item_id"].isin(item_ids)].copy()
    df["mid"] = df[["avg_high", "avg_low"]].mean(axis=1, skipna=True)
    df = df.dropna(subset=["mid"])
    pan = df.pivot_table(index="ts", columns="item_id", values="mid", aggfunc="mean")
    return np.log(pan).sort_index()


def event_study(prices: pd.DataFrame, treated: list[int], control: list[int],
                event_ts: int, pre: int = 2 * 288, post: int = 288,
                n_boot: int = 300, seed: int = 0) -> dict:
    p = config.PERIOD_S
    lo, hi = event_ts - pre * p, event_ts + post * p
    pan_t = _logmid_panel(prices, treated).loc[lo:hi]
    pan_c = _logmid_panel(prices, control).loc[lo:hi]
    if pan_t.empty or pan_c.empty:
        raise ValueError("insufficient data around event window")

    # Normalize each series to its own pre-event mean (market-model-lite).
    pre_t = pan_t.loc[:event_ts - p].mean()
    pre_c = pan_c.loc[:event_ts - p].mean()
    rel_t = pan_t - pre_t
    rel_c = (pan_c - pre_c).mean(axis=1)
    abnormal = rel_t.sub(rel_c, axis=0)          # per treated item
    car_post = abnormal.loc[event_ts:].mean()    # mean abnormal level post

    rng = np.random.default_rng(seed)
    cols = list(car_post.index)
    boots = [car_post[rng.choice(cols, size=len(cols), replace=True)].mean()
             for _ in range(n_boot)]
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])

    path = abnormal.mean(axis=1)
    return {
        "event_ts": int(event_ts),
        "n_treated": len(cols),
        "n_control": pan_c.shape[1],
        "avg_abnormal_post_pct": round(float(car_post.mean()) * 100, 2),
        "ci95_pct": [round(float(lo_ci) * 100, 2), round(float(hi_ci) * 100, 2)],
        "path": path,  # pd.Series for plotting
    }
