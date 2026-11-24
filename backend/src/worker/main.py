"""Standalone worker runner CLI supporting Temporal and ARQ profiles."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from config import Configuration  # noqa: E402
from worker.temporal_workflow import (  # noqa: E402
    build_temporal_worker_definitions,
    run_research_arq_job,
)

logger = logging.getLogger("deepresearch.worker")


async def _run_temporal_worker(config: Configuration) -> None:  # pragma: no cover - runtime daemon
    try:
        from temporalio.client import Client
        from temporalio.worker import Worker
    except ImportError as exc:
        raise RuntimeError("temporalio package is required for the temporal worker profile") from exc

    wf_cls, activities = build_temporal_worker_definitions()
    client = await Client.connect(
        config.temporal_address,
        namespace=config.temporal_namespace,
    )
    worker = Worker(
        client,
        task_queue=config.temporal_task_queue,
        workflows=[wf_cls],
        activities=activities,
    )
    logger.info("Starting Temporal worker on task_queue=%s", config.temporal_task_queue)
    await worker.run()


def _run_arq_worker(config: Configuration) -> None:  # pragma: no cover - runtime daemon
    try:
        from arq.connections import RedisSettings
        from arq.worker import Worker
    except ImportError as exc:
        raise RuntimeError("arq package is required for the arq worker profile") from exc

    dsn = config.arq_redis_dsn or config.redis_url or "redis://127.0.0.1:6379/0"

    class WorkerSettings:
        functions = [run_research_arq_job]
        redis_settings = RedisSettings.from_dsn(dsn)

    logger.info("Starting ARQ worker on redis=%s", dsn)
    Worker(
        functions=WorkerSettings.functions,
        redis_settings=WorkerSettings.redis_settings,
    ).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepResearch durable worker runner.")
    parser.add_argument("--backend", choices=["temporal", "arq"], default=None)
    args = parser.parse_args(argv)

    config = Configuration.from_env()
    selected = args.backend or config.workflow_runner
    if selected == "temporal":
        asyncio.run(_run_temporal_worker(config))
        return 0
    if selected == "arq":
        _run_arq_worker(config)
        return 0
    logger.info("workflow_runner=%s requires no external worker process.", selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
