"""Detection ensemble.

Layered simplest-first, each layer scored on a comparable z-like scale:
  1. rules        — the weak-supervision heuristics, kept in production
                    (interpretable benchmark every ML layer must beat)
  2. stats        — robust volume / return / spread z-scores
  3. iforest      — Isolation Forest over the joint feature space
  4. forecast     — seasonal-naive counterfactual; large residual = anomaly,
                    and the residual doubles as economic-impact evidence
  5. supervised   — gradient boosting trained on WEAK labels only (never truth)

Ensemble = weighted max, then patch-window suppression (announced game updates
are expected repricing). Contiguous high-score buckets merge into episodes —
the unit a human investigator reviews.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from . import config
from .labeling import apply_lfs

IF_FEATURES = ["vol_z", "ret_z", "ret6", "spread_z", "log_px_ratio", "zero_side", "liq_tier_code"]
DEFAULT_ENSEMBLE_W = {"rules": 1.0, "stats": 0.8, "iforest": 0.9, "forecast": 0.9}


def score_rules(f: pd.DataFrame, update_ts: list[int]) -> pd.Series:
    v = apply_lfs(f, update_ts)
    s = v["lf_pump"] * (3.0 + f["vol_z"].clip(0) / 4.0) + v["lf_rmt"] * 4.5
    return s.clip(lower=0).rename("rules")


def score_stats(f: pd.DataFrame) -> pd.Series:
    s = np.maximum.reduce([
        (f["vol_z"].clip(0) / 3.0).to_numpy(),
        (f["ret_z"].abs() / 3.5).to_numpy(),
        (f["spread_z"].clip(0) / 4.0).to_numpy(),
        (np.maximum(f["log_px_ratio"], 0) / np.log(3.0)).to_numpy(),
    ])
    return pd.Series(s, index=f.index, name="stats").clip(0, 20)


def score_iforest(f: pd.DataFrame, train_mask: np.ndarray, seed: int = 0) -> pd.Series:
    X = f[IF_FEATURES].to_numpy()
    model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(X[train_mask])
    raw = -model.score_samples(X)
    q50, q99 = np.quantile(raw[train_mask], [0.5, 0.99])
    z = (raw - q50) / (q99 - q50 + 1e-9) * 2.0
    return pd.Series(z, index=f.index, name="iforest").clip(0, 20)


def score_forecast(f: pd.DataFrame, train_mask: np.ndarray) -> pd.Series:
    """Seasonal-naive counterfactual: ŷ_t = 0.6·median(y around t-1day) + 0.4·y_{t-1}.

    The daily-lag reference is a 13-bucket centered MEDIAN, not the raw lag:
    a short spike yesterday (an RMT print, possibly forward-filled across a few
    no-trade buckets on an illiquid item) would otherwise poison today's
    baseline and fire a phantom alert exactly one day after the real event.
    Thirteen buckets outvotes any spike+ffill run (≤9) while genuine multi-hour
    patterns (pumps run ≥18 buckets) pass through intact."""
    df = f[["item_id", "ts", "logmid"]].copy()
    g = df.groupby("item_id")["logmid"]
    d = config.BUCKETS_PER_DAY
    lag_med = pd.concat([g.shift(k) for k in range(d - 6, d + 7)], axis=1).median(axis=1)
    pred = 0.6 * lag_med + 0.4 * g.shift(1)
    pred = pred.fillna(g.shift(1))
    resid = (df["logmid"] - pred).fillna(0.0)
    df["resid"] = resid
    sd = (df[train_mask].groupby("item_id")["resid"].std().clip(lower=1e-4))
    z = (resid.abs() / df["item_id"].map(sd).fillna(resid.std() + 1e-4)) / 2.0
    # Recency gate: a big residual against yesterday with a flat price NOW is a
    # stale baseline (e.g. the day-after echo of a benign level shift), not an
    # ongoing anomaly. Damp it hard unless the price is actually moving.
    moving = (f["ret6"].abs() > 0.01) | (f["ret1"].abs() > 0.004)
    z = z * np.where(moving.to_numpy(), 1.0, 0.15)
    return z.rename("forecast").clip(0, 20)


def score_supervised(f: pd.DataFrame, weak: pd.DataFrame, train_mask: np.ndarray,
                     seed: int = 0) -> tuple[pd.Series, HistGradientBoostingClassifier | None]:
    """Boosted trees on weak labels from the TRAIN window only. Truth is never
    seen by any model — it is reserved for evaluation."""
    X = f[IF_FEATURES].to_numpy()
    y = weak["weak_label"].to_numpy()
    Xtr, ytr = X[train_mask], y[train_mask]
    if ytr.sum() < 10:  # not enough weak positives to learn from
        return pd.Series(0.0, index=f.index, name="supervised"), None
    # Downsample the huge negative class for speed/balance.
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(ytr == 1)
    neg = rng.choice(np.flatnonzero(ytr == 0), size=min(len(pos) * 25, (ytr == 0).sum()),
                     replace=False)
    idx = np.concatenate([pos, neg])
    clf = HistGradientBoostingClassifier(max_depth=6, max_iter=150, random_state=seed)
    clf.fit(Xtr[idx], ytr[idx])
    proba = clf.predict_proba(X)[:, 1]
    # Returned on a 0..1 probability scale: the ensemble uses this as a
    # re-ranker, never as a standalone alert source (saturated tree probas
    # carry no ranking information at the top of the scale).
    return pd.Series(proba, index=f.index, name="supervised"), clf


def suppress_patches(score: pd.Series, ts: pd.Series, update_ts: list[int]) -> pd.Series:
    if not update_ts:
        return score
    w = config.PATCH_SUPPRESS_W * config.PERIOD_S
    t = ts.to_numpy()
    near = np.zeros(len(t), dtype=bool)
    for u in update_ts:
        near |= (t >= u - w) & (t <= u + 2 * w)
    return pd.Series(np.where(near, score * config.PATCH_SUPPRESS_FACTOR, score),
                     index=score.index)


def ensemble(f: pd.DataFrame, update_ts: list[int], weak: pd.DataFrame,
             train_mask: np.ndarray, weights: dict | None = None, seed: int = 0
             ) -> tuple[pd.DataFrame, dict]:
    w = weights or DEFAULT_ENSEMBLE_W
    parts = {
        "rules": score_rules(f, update_ts),
        "stats": score_stats(f),
        "iforest": score_iforest(f, train_mask, seed),
        "forecast": score_forecast(f, train_mask),
    }
    sup, clf = score_supervised(f, weak, train_mask, seed)
    base_keys = list(parts)
    stacked = np.stack([parts[k].to_numpy() * w[k] for k in base_keys])
    base = stacked.max(axis=0)
    # The weak-label classifier RE-RANKS: corroborated signals get up to +50%,
    # uncorroborated borderline pops get damped to 75% — but the classifier can
    # never originate an episode by itself (its probabilities saturate and
    # carry no ranking signal at the top of the scale).
    raw = base * (0.75 + 0.5 * sup.to_numpy())
    out = f[["item_id", "ts"]].copy()
    for k, s in parts.items():
        out[k] = s.to_numpy()
    out["supervised"] = sup.to_numpy()
    out["score_raw"] = raw
    out["score"] = suppress_patches(pd.Series(raw, index=f.index), f["ts"], update_ts).to_numpy()
    out["top_detector"] = [base_keys[i] for i in stacked.argmax(axis=0)]
    return out, {"clf": clf}


def episodes(scores: pd.DataFrame, tau: float | None = None,
             gap: int | None = None) -> list[dict]:
    """Merge contiguous above-threshold buckets (allowing small gaps) into
    reviewable alert episodes, one per item per burst."""
    tau = config.ALERT_TAU if tau is None else tau
    gap = config.EPISODE_GAP if gap is None else gap
    eps: list[dict] = []
    for iid, g in scores.groupby("item_id"):
        g = g.sort_values("ts")
        hot = g[g["score"] >= tau]
        if hot.empty:
            continue
        cur = None
        for _, r in hot.iterrows():
            if cur and r["ts"] - cur["ts_end"] <= gap * config.PERIOD_S:
                cur["ts_end"] = int(r["ts"])
                if r["score"] > cur["score"]:
                    cur.update(score=float(r["score"]), peak_ts=int(r["ts"]),
                               top_detector=r["top_detector"])
            else:
                if cur:
                    eps.append(cur)
                cur = dict(item_id=int(iid), ts_start=int(r["ts"]), ts_end=int(r["ts"]),
                           peak_ts=int(r["ts"]), score=float(r["score"]),
                           top_detector=r["top_detector"])
        if cur:
            eps.append(cur)
    eps.sort(key=lambda e: -e["score"])
    return eps
