"""Refresh the committed real seed excerpts from the live wiki API.

Run on a machine with internet access and GE_SENTINEL_UA set to a descriptive
User-Agent with your contact info (the API blocks default client UAs):

    GE_SENTINEL_UA="ge-sentinel - you@example.com" python scripts/fetch_real_seed.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ge_sentinel import config  # noqa: E402
from ge_sentinel.api_client import WikiPricesClient  # noqa: E402

ITEMS = {4151: "Abyssal whip", 561: "Nature rune", 13190: "Old school bond"}


def main() -> None:
    out = config.SEED_REAL_DIR
    out.mkdir(parents=True, exist_ok=True)
    with WikiPricesClient() as client:
        for iid, name in ITEMS.items():
            try:
                payload = client.timeseries(iid, "5m")
            except Exception as exc:
                print(f"skip {iid} ({name}): {exc}")
                continue
            doc = {
                "itemId": iid,
                "timestep": "5m",
                "source": f"{config.API_BASE}/timeseries?timestep=5m&id={iid}",
                "fetched_note": f"Live refresh for {name}.",
                "data": payload.get("data", []),
            }
            (out / f"timeseries_5m_{iid}.json").write_text(json.dumps(doc))
            print(f"wrote {iid} {name}: {len(doc['data'])} points")


if __name__ == "__main__":
    main()
