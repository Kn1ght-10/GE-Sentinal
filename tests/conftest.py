"""Shared fixtures.

demo_env runs the FAST end-to-end pipeline exactly once per test session into
an isolated temp sandbox (own DB, outputs, memos, assets) and is shared by the
pipeline and API tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # for `api.main`


@pytest.fixture(scope="session")
def demo_env(tmp_path_factory):
    from ge_sentinel import config

    tmp = tmp_path_factory.mktemp("ge_sentinel")
    config.DB_URL = f"sqlite:///{tmp / 'test.db'}"
    config.OUTPUT_DIR = tmp / "outputs"
    config.MEMO_DIR = config.OUTPUT_DIR / "memos"
    config.ASSETS_DIR = tmp / "assets"
    for d in (config.OUTPUT_DIR, config.MEMO_DIR, config.ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    from ge_sentinel.pipeline import run_demo

    result = run_demo(fast=True, with_memos=True)
    return {"config": config, "result": result, "tmp": tmp}
