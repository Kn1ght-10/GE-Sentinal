"""Optional Prefect orchestration for the live collector.

    pip install prefect
    python flows/prefect_flow.py            # one sweep
    prefect deploy / serve for schedules    # see Prefect docs

The default deployment path (GitHub Actions cron) has zero infra; this flow
is the step-up when you want retries, observability, and backfill runs.
"""
try:
    from prefect import flow, task
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Prefect not installed — pip install prefect") from exc

from ge_sentinel import db
from ge_sentinel.api_client import WikiPricesClient
from ge_sentinel.ingest import collect_once


@task(retries=3, retry_delay_seconds=30)
def sweep() -> int:
    engine = db.init_db()
    with WikiPricesClient() as client:
        return collect_once(engine, client)


@flow(name="ge-sentinel-collect")
def collect_flow() -> None:
    n = sweep()
    print(f"upserted {n} rows")


if __name__ == "__main__":
    collect_flow()
