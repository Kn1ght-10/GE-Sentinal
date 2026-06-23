"""Evaluation against injected ground truth — walk-forward, leakage-free.

Models train on the first 70% of time using ONLY weak labels; everything here
runs on the held-out 30%. Metrics mirror how surveillance teams are judged:
precision@K (alert quality at review capacity), event recall, median
time-to-detection, and a negative control (benign patch shocks flagged).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config
from .synthetic import KIND_PATCH, MANIP_KINDS

GRACE_PRE = 2     # buckets an episode may start before the true onset
GRACE_POST = 12   # buckets after the true end still credited


def _match(ep: dict, truth: pd.DataFrame) -> pd.Series | None:
    p = config.PERIOD_S
    cand = truth[(truth["item_id"] == ep["item_id"])
                 & (ep["ts_start"] >= truth["start_ts"] - GRACE_PRE * p)
                 & (ep["ts_start"] <= truth["end_ts"] + GRACE_POST * p)]
    return None if cand.empty else cand.iloc[0]


def truth_row_mask(f: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    """Bucket-level manipulation mask for AUC computation."""
    mask = np.zeros(len(f), dtype=int)
    ts = f["ts"].to_numpy()
    iid = f["item_id"].to_numpy()
    p = config.PERIOD_S
    for _, e in truth[truth["kind"].isin(MANIP_KINDS)].iterrows():
        mask |= ((iid == e["item_id"]) & (ts >= e["start_ts"])
                 & (ts <= e["end_ts"] + GRACE_POST * p)).astype(int)
    return mask


def evaluate(eps: list[dict], truth: pd.DataFrame, scores: pd.DataFrame,
             split_ts: int, k: int = 25) -> dict:
    p = config.PERIOD_S
    test_eps = [e for e in eps if e["ts_start"] >= split_ts]
    truth_test = truth[truth["end_ts"] >= split_ts]
    manip_test = truth_test[truth_test["kind"].isin(MANIP_KINDS)]

    top = sorted(test_eps, key=lambda e: -e["score"])[:k]
    matches = [( e, _match(e, truth_test)) for e in top]
    tp = [(e, m) for e, m in matches if m is not None and m["kind"] in MANIP_KINDS]
    patch_hits = [(e, m) for e, m in matches if m is not None and m["kind"] == KIND_PATCH]

    matched_ids = set()
    delays = []
    for e, m in tp:
        key = (int(m["item_id"]), int(m["start_ts"]))
        if key not in matched_ids:
            matched_ids.add(key)
            delays.append(max(0, (e["ts_start"] - int(m["start_ts"])) / p))

    recall_keys = {(int(r["item_id"]), int(r["start_ts"])) for _, r in manip_test.iterrows()}
    detected = set()
    for e in test_eps:  # recall over ALL test episodes, not just top-K
        m = _match(e, manip_test)
        if m is not None:
            detected.add((int(m["item_id"]), int(m["start_ts"])))

    test_mask = (scores["ts"] >= split_ts).to_numpy()
    y_true = truth_row_mask(scores[test_mask], truth)
    y_score = scores.loc[test_mask, "score"].to_numpy()
    auc = float(roc_auc_score(y_true, y_score)) if 0 < y_true.sum() < len(y_true) else float("nan")

    # Precision at several review budgets. With few injected events, large K
    # has a hard ceiling (matched-episode count / K) — reported, not hidden.
    curve = {}
    for kk in sorted({5, 10, 15, k}):
        top_kk = sorted(test_eps, key=lambda e: -e["score"])[:kk]
        hits = sum(1 for e in top_kk
                   if (m := _match(e, truth_test)) is not None and m["kind"] in MANIP_KINDS)
        curve[str(kk)] = round(hits / max(1, len(top_kk)), 4)

    report = {
        "k": k,
        "n_test_episodes": len(test_eps),
        "precision_at_k": round(len(tp) / max(1, len(top)), 4),
        "precision_curve": curve,
        "recall_manip_events": round(len(detected & recall_keys) / max(1, len(recall_keys)), 4),
        "n_manip_events_test": int(len(recall_keys)),
        "median_detection_delay_min": (round(float(np.median(delays)) * p / 60, 1)
                                       if delays else None),
        "patch_false_alarms_in_topk": len(patch_hits),
        "row_level_auc": round(auc, 4) if auc == auc else None,
        "alerts_per_day_test": round(len(test_eps) /
                                     max(1e-9, (scores.loc[test_mask, 'ts'].max()
                                                - split_ts) / 86400), 2),
    }
    return report


def save_report(report: dict, extra: dict | None = None) -> str:
    out = dict(report)
    if extra:
        out.update(extra)
    path = config.OUTPUT_DIR / "eval_report.json"
    path.write_text(json.dumps(out, indent=2))
    return str(path)
