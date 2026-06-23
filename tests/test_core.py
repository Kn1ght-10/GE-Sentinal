"""Unit tests: synthetic market, features, weak labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ge_sentinel import synthetic
from ge_sentinel.features import FEATURES, build_features
from ge_sentinel.labeling import apply_lfs, label_model

GEN = dict(n_items=8, days=6, seed=7, n_pump=3, n_rmt=2, n_patch=1)


def test_synthetic_shapes_and_reproducibility():
    items_a, prices_a, truth_a, upd_a = synthetic.generate(**GEN)
    items_b, prices_b, truth_b, upd_b = synthetic.generate(**GEN)
    pd.testing.assert_frame_equal(prices_a, prices_b)
    pd.testing.assert_frame_equal(truth_a, truth_b)
    assert upd_a == upd_b

    assert len(items_a) == 8
    assert len(prices_a) == 8 * 6 * 288
    assert set(truth_a["kind"]) == {"pump_dump", "rmt_spike", "patch_shock"}
    assert truth_a["in_test"].any(), "eval needs events in the held-out window"

    one = prices_a[prices_a["item_id"] == int(prices_a["item_id"].iloc[0])]
    assert (one["ts"].diff().dropna() == 300).all()
    assert (prices_a["source"] == "synth").all()


def test_features_finite_and_spike_visible():
    _, prices, truth, _ = synthetic.generate(**GEN)
    f = build_features(prices)
    for c in FEATURES:
        assert c in f.columns
        assert np.isfinite(f[c].to_numpy()).all(), f"non-finite values in {c}"

    # The latest injected pump (well past the rolling-baseline warm-up) must
    # light up the volume z-score inside its true window.
    pump = truth[truth["kind"] == "pump_dump"].sort_values("start_ts").iloc[-1]
    w = f[(f["item_id"] == pump["item_id"])
          & f["ts"].between(pump["start_ts"], pump["end_ts"])]
    assert w["vol_z"].max() > 5

    # RMT spike must register as an extraordinary price ratio.
    rmt = truth[truth["kind"] == "rmt_spike"].sort_values("start_ts").iloc[-1]
    r = f[(f["item_id"] == rmt["item_id"])
          & f["ts"].between(rmt["start_ts"], rmt["end_ts"])]
    assert r["px_ratio"].max() > 4


def test_labeling_functions_vote():
    _, prices, _, upd = synthetic.generate(**GEN)
    f = build_features(prices)
    votes = apply_lfs(f, upd)
    weak = label_model(votes)

    assert (votes["lf_patch"] == -1).any(), "patch window LF should fire"
    assert weak["weak_label"].sum() > 0, "some weak positives expected"
    # Weak positives should be rare (events are rare).
    assert weak["weak_label"].mean() < 0.05
    assert weak["weak_prob"].between(0, 1).all()
