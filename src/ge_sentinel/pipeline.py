"""End-to-end pipeline orchestration.

run_demo() executes the full system on one machine with zero external
dependencies: real seed data + the synthetic market with injected ground
truth → DB → features → weak labels → detection ensemble → walk-forward
evaluation → persisted alerts → analyst memos → figures → eval report.

Honesty contract: models never see truth labels (truth is evaluation-only);
every synthetic row is tagged source="synth"; real rows are tagged
source="seed_real"/"live"; reported metrics are validation on synthetic
injected events, stated as such everywhere they appear.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config, db  # noqa: E402
from .causal import event_study  # noqa: E402
from .detectors import ensemble, episodes  # noqa: E402
from .evaluate import evaluate, save_report  # noqa: E402
from .features import build_features  # noqa: E402
from .ingest import SEED_ITEM_NAMES, gap_report, load_prices, load_real_seed  # noqa: E402
from .labeling import apply_lfs, label_model  # noqa: E402
from .memo import TfidfRetriever, write_memos  # noqa: E402
from .synthetic import KIND_PATCH, KIND_PUMP, KIND_RMT, generate  # noqa: E402

SYNTH_ID_MIN = 900_000


# --------------------------------------------------------------------------
def _fresh_db():
    """Demo always starts from a clean database file (SQLite default)."""
    if config.DB_URL.startswith("sqlite:///"):
        p = Path(config.DB_URL.replace("sqlite:///", "", 1))
        if p.exists():
            p.unlink()
        p.parent.mkdir(parents=True, exist_ok=True)
    return db.init_db(db.get_engine(config.DB_URL))


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """DataFrame → DB-safe dicts (numpy scalars and NaN don't bind cleanly)."""
    recs = df[cols].to_dict("records")
    for r in recs:
        for k, v in r.items():
            if isinstance(v, float) and np.isnan(v):
                r[k] = None
            elif isinstance(v, np.integer):
                r[k] = int(v)
            elif isinstance(v, np.floating):
                r[k] = float(v)
            elif isinstance(v, np.bool_):
                r[k] = bool(v)
    return recs


def _near_update(ts: int, update_ts: list[int]) -> bool:
    w = config.PATCH_SUPPRESS_W * config.PERIOD_S
    return any(u - w <= ts <= u + 2 * w for u in update_ts)


def _episode_evidence(e: dict, feats: pd.DataFrame, votes: pd.DataFrame,
                      update_ts: list[int]) -> tuple[dict, str]:
    p = config.PERIOD_S
    m = ((feats["item_id"] == e["item_id"])
         & (feats["ts"] >= e["ts_start"] - 2 * p)
         & (feats["ts"] <= e["ts_end"] + 2 * p))
    w, vw = feats[m], votes[m]
    if w.empty:
        return {}, "unclassified anomaly"
    near = _near_update(int(e["peak_ts"]), update_ts)
    if bool(vw["lf_rmt"].any()):
        kind = KIND_RMT
    elif bool(vw["lf_pump"].any()) or (w["vol_z"].max() > 5 and w["ret6"].max() > 0.05):
        kind = KIND_PUMP
    elif near:
        kind = KIND_PATCH
    else:
        kind = "unclassified anomaly"
    mid0 = float(w["mid"].iloc[0]) or 1.0
    turnover = float((w["vol"] * w["mid"]).sum())
    peak_exc = (float(w["mid"].max()) / mid0 - 1) * 100
    net_move = (float(w["mid"].iloc[-1]) / mid0 - 1) * 100
    ev = {
        "vol_z_peak": float(w["vol_z"].max()),
        "ret6_peak": float(w["ret6"].max()),
        "px_ratio_peak": float(w["px_ratio"].max()),
        "spread_z_peak": float(w["spread_z"].max()),
        "near_update": near,
        "impact_note": (f"~{turnover:,.0f} gp turnover during a {peak_exc:+.1f}% peak "
                        f"excursion ({net_move:+.1f}% net over the window) versus the "
                        f"pre-episode price."),
    }
    return ev, kind


# --------------------------------------------------------------------------
def _fig_real_whip(all_prices: pd.DataFrame, path: Path) -> None:
    wp = all_prices[all_prices["item_id"] == 4151].sort_values("ts")
    if wp.empty:
        return
    mid = wp[["avg_high", "avg_low"]].mean(axis=1, skipna=True)
    t = pd.to_datetime(wp["ts"], unit="s")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, mid, lw=1.2, color="#1f4e79", label="mid price (gp)")
    ax2 = ax.twinx()
    ax2.bar(t, wp["high_vol"] + wp["low_vol"], width=300 / 86400, alpha=0.3,
            color="#888", label="volume")
    # The genuine missing bucket captured in the seed pull.
    ax.axvspan(pd.to_datetime(1774937700, unit="s"), pd.to_datetime(1774938000, unit="s"),
               color="orange", alpha=0.35)
    ax.set_title("Abyssal whip (4151) — REAL OSRS GE 5-minute data (seed excerpt); "
                 "orange = genuine missing bucket")
    ax.set_ylabel("price (gp)")
    ax2.set_ylabel("units traded / 5 min")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_anatomy(feats: pd.DataFrame, truth: pd.DataFrame, eps: list[dict],
                 path_price: Path, path_score: Path, scores: pd.DataFrame) -> int | None:
    pumps = truth[(truth["kind"] == KIND_PUMP) & truth["in_test"]]
    if pumps.empty:
        pumps = truth[truth["kind"] == KIND_PUMP]
    if pumps.empty:
        return None
    ev = pumps.sort_values("magnitude", ascending=False).iloc[0]
    iid = int(ev["item_id"])
    p = config.PERIOD_S
    lo, hi = int(ev["start_ts"]) - 80 * p, int(ev["end_ts"]) + 80 * p
    w = feats[(feats["item_id"] == iid) & feats["ts"].between(lo, hi)]
    s = scores[(scores["item_id"] == iid) & scores["ts"].between(lo, hi)]
    t = pd.to_datetime(w["ts"], unit="s")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, w["mid"], lw=1.2, color="#1f4e79")
    ax.axvspan(pd.to_datetime(int(ev["start_ts"]), unit="s"),
               pd.to_datetime(int(ev["end_ts"]), unit="s"),
               color="green", alpha=0.14, label="injected pump_dump (ground truth)")
    for e in (x for x in eps if x["item_id"] == iid and lo <= x["ts_start"] <= hi):
        ax.axvspan(pd.to_datetime(e["ts_start"], unit="s"),
                   pd.to_datetime(e["ts_end"] + p, unit="s"),
                   color="red", alpha=0.22)
    ax.set_title(f"Injected pump & dump anatomy — synthetic item {iid} "
                 "(green = truth, red = GE-Sentinel alert)")
    ax.set_ylabel("price (gp)")
    ax2 = ax.twinx()
    ax2.bar(t, w["vol"], width=300 / 86400, alpha=0.25, color="#888")
    ax2.set_ylabel("volume")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_price, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(pd.to_datetime(s["ts"], unit="s"), s["score"], lw=1.0, color="#b30000")
    ax.axhline(config.ALERT_TAU, ls="--", color="#444", lw=1,
               label=f"alert threshold τ={config.ALERT_TAU}")
    ax.axvspan(pd.to_datetime(int(ev["start_ts"]), unit="s"),
               pd.to_datetime(int(ev["end_ts"]), unit="s"), color="green", alpha=0.14)
    ax.set_title(f"Ensemble anomaly score — item {iid}")
    ax.set_ylabel("score")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_score, dpi=130)
    plt.close(fig)
    return iid


def _fig_eval(report: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    curve = report.get("precision_curve") or {str(report["k"]): report["precision_at_k"]}
    kk_sorted = sorted(curve, key=int)
    names = [f"precision@{kk}" for kk in kk_sorted] + ["event recall", "row-level AUC"]
    vals = [curve[kk] for kk in kk_sorted] + [report["recall_manip_events"],
                                              report.get("row_level_auc") or 0]
    colors = ["#1f4e79"] * len(kk_sorted) + ["#2e7d32", "#6a1b9a"]
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xlim(0, 1.05)
    delay = report.get("median_detection_delay_min")
    ax.set_title("Validation on synthetic injected events (held-out 30%)\n"
                 f"median detection delay: {delay} min · "
                 f"patch false alarms in top-{report['k']}: "
                 f"{report['patch_false_alarms_in_topk']} · "
                 f"alerts/day: {report['alerts_per_day_test']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def run_demo(fast: bool = False, seed: int = 42, with_memos: bool = True,
             k: int | None = None) -> dict:
    t0 = time.time()
    engine = _fresh_db()

    # 1) Real seed data (committed excerpt of the live API).
    n_real = load_real_seed(engine)

    # 2) Synthetic market with injected, labeled events.
    if fast:
        items_df, prices, truth_df, update_ts = generate(
            n_items=12, days=10, seed=seed, n_pump=4, n_rmt=3, n_patch=2)
    else:
        items_df, prices, truth_df, update_ts = generate(seed=seed)
    split_ts = int(prices.attrs["split_ts"])

    db.upsert(engine, db.items, _records(
        items_df, ["id", "name", "members", "ge_limit", "highalch", "source"]))
    db.upsert(engine, db.prices_5m, _records(
        prices, ["item_id", "ts", "avg_high", "avg_low", "high_vol", "low_vol", "source"]))
    with engine.begin() as cx:
        cx.execute(db.truth_events.insert(), _records(
            truth_df, ["item_id", "kind", "start_ts", "end_ts", "magnitude", "note"]))

    # 3) Round-trip through the DB (proves storage layer), then features.
    all_prices = load_prices(engine)
    feats = build_features(all_prices)
    train_mask = (feats["ts"] < split_ts).to_numpy()

    # 4) Weak supervision (no truth labels anywhere near the models).
    votes = apply_lfs(feats, update_ts)
    weak = label_model(votes)

    # 5) Detection ensemble + episodes.
    scores, _aux = ensemble(feats, update_ts, weak, train_mask, seed=seed)
    eps_all = episodes(scores)
    eps_synth = [e for e in eps_all if e["item_id"] >= SYNTH_ID_MIN]
    synth_scores = scores[scores["item_id"] >= SYNTH_ID_MIN]

    # 6) Walk-forward evaluation on the held-out window (synthetic truth only;
    #    real items are scored but never counted — no truth exists for them).
    report = evaluate(eps_synth, truth_df, synth_scores, split_ts,
                      k=(k or (10 if fast else 25)))

    # 7) Persist alerts with evidence + classification.
    top_alerts = eps_all[:50]
    evidences: dict[int, dict] = {}
    for i, e in enumerate(top_alerts):
        ev, kind = _episode_evidence(e, feats, votes, update_ts)
        e["kind_guess"] = kind
        e["evidence"] = ev
        evidences[i] = ev
    db.record_alerts(engine, top_alerts)

    # 8) Analyst memos (RAG over the corpus; LLM polish only if key present).
    memo_paths: list[str] = []
    for stale in config.MEMO_DIR.glob("alert_*"):
        stale.unlink()  # demo is a fresh run; don't mix memos across runs
    if with_memos:
        item_names = {**SEED_ITEM_NAMES,
                      **dict(zip(items_df["id"].astype(int), items_df["name"]))}
        memo_paths = write_memos(top_alerts, evidences, item_names,
                                 retriever=TfidfRetriever(), top_n=5)

    # 9) Causal event study on one known benign patch shock (method demo).
    causal_out = None
    try:
        u = update_ts[0]
        treated = truth_df[(truth_df["kind"] == KIND_PATCH)
                           & (truth_df["start_ts"] == u)]["item_id"].astype(int).tolist()
        quiet = sorted(set(items_df["id"].astype(int))
                       - set(truth_df["item_id"].astype(int)))[:8]
        n_pre = int(min(2 * 288, (u - int(prices["ts"].min())) / config.PERIOD_S - 4))
        n_post = int(min(144, (int(prices["ts"].max()) - u) / config.PERIOD_S - 2))
        if treated and quiet and n_pre > 24 and n_post > 12:
            cs = event_study(prices, treated, quiet, u, pre=n_pre, post=n_post,
                             n_boot=200, seed=seed)
            causal_out = {kk: vv for kk, vv in cs.items() if kk != "path"}
    except Exception as exc:  # demo resilience; causal layer is additive
        causal_out = {"error": str(exc)}

    # 10) Figures.
    a = config.ASSETS_DIR
    _fig_real_whip(all_prices, a / "fig_real_whip.png")
    showcase = _fig_anatomy(feats, truth_df, eps_all,
                            a / "fig_event_anatomy.png",
                            a / "fig_score_timeline.png", scores)
    _fig_eval(report, a / "fig_eval_summary.png")

    gaps = gap_report(engine, 4151)
    extra = {
        "runtime_s": round(time.time() - t0, 1),
        "fast_mode": fast,
        "split_ts": split_ts,
        "rows_total": int(len(all_prices)),
        "rows_real_seed": int(n_real),
        "items_synth": int(len(items_df)),
        "real_whip_missing_buckets": int(gaps["missing_buckets"].sum()) if len(gaps) else 0,
        "n_alerts_recorded": len(top_alerts),
        "n_memos": len(memo_paths),
        "showcase_item": showcase,
        "causal_patch_event_study": causal_out,
        "validation_data": "synthetic injected events (models trained on weak labels only)",
    }
    report_path = save_report(report, extra)

    print("\n=== GE-Sentinel demo complete ===")
    print(f"rows: {extra['rows_total']:,} (real seed: {n_real}) | "
          f"runtime: {extra['runtime_s']}s")
    print(f"precision@{report['k']}: {report['precision_at_k']} | "
          f"recall: {report['recall_manip_events']} "
          f"({report['n_manip_events_test']} manip events in test) | "
          f"AUC: {report['row_level_auc']}")
    print(f"median detection delay: {report['median_detection_delay_min']} min | "
          f"patch FPs in top-{report['k']}: {report['patch_false_alarms_in_topk']} | "
          f"alerts/day: {report['alerts_per_day_test']}")
    if causal_out and "error" not in causal_out:
        print(f"causal (benign patch study): {causal_out['avg_abnormal_post_pct']}% "
              f"abnormal move, 95% CI {causal_out['ci95_pct']}")
    print(f"report: {report_path}\nmemos: {len(memo_paths)} in {config.MEMO_DIR}")
    return {"report": report, "extra": extra, "episodes": eps_synth[:10],
            "memo_paths": memo_paths}
