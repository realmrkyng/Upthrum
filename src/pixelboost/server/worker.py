"""Job queue for long-running enhancements.

Deliberately in-process rather than Redis/Celery. A single GPU box serving one
tenant does not need a broker, and adding one triples the deployment surface for
zero benefit. If you outgrow this, the interface here is small enough to swap:
``submit`` returns an id, ``get`` returns a status dict, results are bytes.

Concurrency model: one ``ThreadPoolExecutor``, one semaphore around the actual
inference call. The semaphore is what keeps a 4-worker pool from thrashing a
single GPU -- ONNX Runtime will happily run four sessions concurrently and be
three times slower than running them one at a time.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("pixelboost.server.worker")


@dataclass
class Job:
    id: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    filename: str = "input.png"
    output_format: str = "png"
    options: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, int] = field(default_factory=lambda: {"done": 0, "total": 0})
    result: bytes | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_type: str | None = None
    future: Future | None = None

    def as_dict(self, include_result: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "filename": self.filename,
            "output_format": self.output_format,
            "options": self.options,
            "progress": dict(self.progress),
            "meta": dict(self.meta),
        }
        if self.elapsed_ms is not None:
            data["elapsed_ms"] = self.elapsed_ms
        if self.error:
            data["error"] = self.error
            data["error_type"] = self.error_type
        if include_result:
            data["result_bytes"] = len(self.result or b"")
        return data

    @property
    def elapsed_ms(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return round((end - self.started_at) * 1000.0, 2)

    @property
    def done(self) -> bool:
        return self.status in ("succeeded", "failed")


class JobManager:
    def __init__(
        self,
        handler: Callable[[bytes, dict[str, Any], Callable[[int, int], None]], Any],
        workers: int = 2,
        queue_size: int = 64,
        ttl_seconds: int = 3600,
        max_queued: int | None = None,
    ) -> None:
        self.handler = handler
        self.workers = max(1, int(workers))
        self.queue_size = int(queue_size)
        self.max_queued = int(max_queued if max_queued is not None else queue_size)
        self.ttl_seconds = int(ttl_seconds)
        self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="pb-worker")
        self._gate = threading.Semaphore(self.workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    def stats(self) -> dict[str, int]:
        with self._lock:
            active = sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))
        return {
            "workers": self.workers,
            "queued": active,
            "capacity": self.max_queued,
            "retained": len(self._jobs),
        }

    def submit(
        self,
        payload: bytes,
        options: dict[str, Any],
        filename: str = "input.png",
        output_format: str = "png",
    ) -> Job:
        self._evict()
        with self._lock:
            active = sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))
            if active >= self.max_queued:
                raise RuntimeError(
                    f"queue full ({active}/{self.max_queued}); retry shortly or "
                    f"raise server.queue_size"
                )
            job = Job(
                id=uuid.uuid4().hex[:16],
                filename=filename,
                output_format=output_format,
                options=dict(options),
            )
            self._jobs[job.id] = job
            job.future = self._pool.submit(self._run, job, payload)
            return job

    def _run(self, job: Job, payload: bytes) -> None:
        with self._gate:
            job.status = "running"
            job.started_at = time.time()

            def progress(done: int, total: int) -> None:
                job.progress = {"done": int(done), "total": int(total)}

            try:
                blob, meta = self.handler(payload, job.options, progress)
                job.result = blob
                job.meta = meta
                job.status = "succeeded"
            except Exception as exc:  # noqa: BLE001 - reported to the client verbatim
                job.status = "failed"
                job.error = str(exc)
                job.error_type = type(exc).__name__
                log.exception("job %s failed", job.id)
            finally:
                job.finished_at = time.time()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.future is None:
            return False
        if job.status != "queued":
            return False
        cancelled = job.future.cancel()
        if cancelled:
            job.status = "cancelled"
            job.finished_at = time.time()
        return cancelled

    def _evict(self) -> None:
        if self.ttl_seconds <= 0:
            return
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            stale = [
                jid
                for jid, job in self._jobs.items()
                if job.done and (job.finished_at or 0) < cutoff
            ]
            for jid in stale:
                self._jobs.pop(jid, None)

    def shutdown(self, wait: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)


def default_worker_count() -> int:
    from pixelboost.backends.registry import has_gpu

    if has_gpu():
        return max(1, int(os.environ.get("PIXELBOOST_GPU_WORKERS", 1)))
    return max(1, int(os.environ.get("PIXELBOOST_CPU_WORKERS", (os.cpu_count() or 4) // 2)))
