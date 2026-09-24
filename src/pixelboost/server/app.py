"""FastAPI application.

Two request shapes, because they solve different problems:

* ``POST /v1/enhance`` -- synchronous, returns the image in the response body.
  Right for thumbnails, avatars and anything under ~2 MP. Uses a thread pool so
  several CPU requests run concurrently.
* ``POST /v1/jobs`` -- asynchronous, returns an id you poll. Right for posters,
  prints and 4x-scale photos, where a 30-90 s request would otherwise sit on a
  proxy's 60 s read timeout and fail for reasons unrelated to the work.

Both paths share one warm :class:`~pixelboost.engine.Engine`, so the model is
loaded once at start-up rather than per request.
"""

from __future__ import annotations

import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple

from pixelboost import __version__, imageio
from pixelboost.backends.registry import backend_capabilities, describe_environment
from pixelboost.config import Config
from pixelboost.errors import PixelBoostError
from pixelboost.types import EnhanceOptions

# These must live at module scope. With ``from __future__ import annotations``
# every annotation is a string, and FastAPI resolves it through
# ``get_type_hints`` using this module's globals. A locally imported ``Header``
# or ``Request`` is invisible there, the annotation fails to resolve, and the
# symptom is a confusing 422 on every route rather than an import error.
try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    FASTAPI_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - depends on install extra
    FASTAPI_IMPORT_ERROR = _exc

    Depends = FastAPI = Header = HTTPException = Request = Response = None  # type: ignore
    CORSMiddleware = JSONResponse = None  # type: ignore

log = logging.getLogger("pixelboost.server")

IMAGE_MEDIA = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
    "avif": "image/avif",
}


def _options_from(payload: Dict[str, Any], base: EnhanceOptions) -> EnhanceOptions:
    """Merge request parameters over the configured defaults.

    Unknown keys are ignored rather than rejected, so adding a field in a new
    release never breaks an existing client.
    """
    opts = EnhanceOptions.from_dict(base.to_dict())
    for key, value in payload.items():
        if not hasattr(opts, key) or key in ("extra",):
            continue
        current = getattr(opts, key)
        if value is None:
            continue
        try:
            if isinstance(current, bool):
                coerced = str(value).strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int) and not isinstance(current, bool):
                coerced = int(float(value))
            elif isinstance(current, float):
                coerced = float(value)
            else:
                coerced = value
        except (TypeError, ValueError):
            continue
        setattr(opts, key, coerced)
    return opts


def _coerce_form(form: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in form:
        if key == "file":
            continue
        value = form[key]
        if isinstance(value, str) and value.strip().startswith("{"):
            try:
                loaded = json.loads(value)
                if isinstance(loaded, dict):
                    out.update(loaded)
                    continue
            except json.JSONDecodeError:
                pass
        out[key] = value
    return out


def create_app(config: Optional[Config] = None) -> Any:
    """Build the ASGI app. Raises a clear error if FastAPI is missing."""
    if FASTAPI_IMPORT_ERROR is not None:
        raise PixelBoostError(
            "the HTTP server needs extra packages: pip install 'pixelboost[server]' "
            f"(import failed with: {FASTAPI_IMPORT_ERROR})"
        ) from FASTAPI_IMPORT_ERROR

    from pixelboost.engine import Engine
    from pixelboost.server.worker import JobManager, default_worker_count

    cfg = config or Config()
    state: Dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: Any):
        engine = Engine(cfg)
        state["engine"] = engine

        if cfg.server.workers:
            workers = cfg.server.workers
        else:
            workers = default_worker_count()
            cfg.server.workers = workers

        def handler(payload: bytes, options: Dict[str, Any], progress: Any) -> Tuple[bytes, Dict[str, Any]]:
            opts = _options_from(options, cfg.defaults)
            blob, result = engine.enhance_bytes(payload, opts)
            return blob, result.summary()

        state["jobs"] = JobManager(
            handler,
            workers=workers,
            queue_size=cfg.server.queue_size,
            ttl_seconds=cfg.server.job_ttl_seconds,
        )
        log.info(
            "pixelboost %s ready (workers=%d, models_dir=%s)",
            __version__,
            workers,
            cfg.models_dir,
        )
        yield
        state["jobs"].shutdown(wait=False)
        engine.close()

    app = FastAPI(
        title="PixelBoost",
        version=__version__,
        description="CPU/GPU accelerated image quality enhancement and super-resolution.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-PixelBoost-Backend", "X-PixelBoost-Tiles", "X-PixelBoost-Ms"],
    )

    async def auth(request: Request, x_api_key: Optional[str] = Header(default=None)) -> None:
        keys = cfg.server.api_keys
        if not keys:
            return
        provided = x_api_key or request.query_params.get("api_key")
        if provided not in keys:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    def engine() -> Any:
        inst = state.get("engine")
        if inst is None:
            raise HTTPException(status_code=503, detail="engine not ready")
        return inst

    async def read_request(request: Request) -> Tuple[bytes, Dict[str, Any]]:
        limit = cfg.server.max_upload_mb * 1024 * 1024
        ctype = (request.headers.get("content-type") or "").lower()

        if ctype.startswith("multipart/form-data"):
            try:
                form = await request.form()
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"cannot parse multipart body (is python-multipart installed?): {exc}",
                ) from exc
            upload = form.get("file")
            if upload is None:
                raise HTTPException(status_code=400, detail="no 'file' field in the form")
            data = await upload.read()
            fields = _coerce_form(form)
            filename = getattr(upload, "filename", None) or "input.png"
        else:
            data = await request.body()
            fields = {k: v for k, v in request.query_params.items()}
            filename = request.query_params.get("filename", "input.png")

        if not data:
            raise HTTPException(status_code=400, detail="empty request body")
        if len(data) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"upload is {len(data) / 1e6:.1f} MB, limit is "
                f"{cfg.server.max_upload_mb} MB (raise server.max_upload_mb)",
            )
        fields.pop("api_key", None)
        if "format" in fields and "output_format" not in fields:
            fields["output_format"] = fields.pop("format")
        return data, {"fields": fields, "filename": filename}

    @app.get("/healthz", tags=["meta"])
    async def healthz() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "engine": bool(state.get("engine")),
            "jobs": state["jobs"].stats() if "jobs" in state else {},
        }

    @app.get("/v1/capabilities", tags=["meta"])
    async def capabilities(_: None = Depends(auth)) -> Dict[str, Any]:
        data = backend_capabilities(cfg)
        data["version"] = __version__
        data["environment"] = describe_environment().splitlines()
        data["models_dir"] = cfg.models_dir
        data["server"] = {
            "workers": cfg.server.workers,
            "max_upload_mb": cfg.server.max_upload_mb,
            "auth_required": bool(cfg.server.api_keys),
        }
        return data

    @app.get("/v1/stats", tags=["meta"])
    async def stats(_: None = Depends(auth)) -> Dict[str, Any]:
        inst = state.get("engine")
        loaded = []
        if inst is not None:
            loaded = [{"key": list(k), "provider": v.provider} for k, v in inst._backends.items()]
        return {"jobs": state["jobs"].stats(), "loaded_backends": loaded}

    @app.post("/v1/enhance", tags=["enhance"])
    async def enhance_sync(request: Request, _: None = Depends(auth)) -> Any:
        data, ctx = await read_request(request)
        inst = engine()
        opts = _options_from(ctx["fields"], cfg.defaults)

        try:
            loaded = imageio.probe(data)
        except PixelBoostError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc

        want_json = (
            "application/json" in (request.headers.get("accept") or "")
            or str(ctx["fields"].get("response", "")).lower() == "json"
        )

        try:
            blob, result = inst.enhance_bytes(data, opts)
        except PixelBoostError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MemoryError as exc:
            raise HTTPException(status_code=507, detail=f"out of memory: {exc}") from exc

        summary = result.summary()
        summary["input"] = loaded
        headers = {
            "X-PixelBoost-Backend": result.backend,
            "X-PixelBoost-Provider": result.provider,
            "X-PixelBoost-Model": str(result.model or ""),
            "X-PixelBoost-Tiles": str(result.tiles),
            "X-PixelBoost-Ms": f"{result.elapsed_ms:.1f}",
        }

        fmt = (opts.output_format or loaded["format"] or "png").lower()
        if want_json:
            return JSONResponse(
                {
                    "image_base64": base64.b64encode(blob).decode("ascii"),
                    "format": fmt,
                    "meta": summary,
                },
                headers=headers,
            )
        return Response(
            content=blob,
            media_type=IMAGE_MEDIA.get(fmt, "application/octet-stream"),
            headers=headers,
        )

    @app.post("/v1/jobs", tags=["jobs"], status_code=202)
    async def submit_job(request: Request, _: None = Depends(auth)) -> Any:
        data, ctx = await read_request(request)
        fields = ctx["fields"]

        # Reject a bad upload now rather than letting it fail asynchronously --
        # a client that has to poll to discover its file was not an image is a
        # bad client experience.
        try:
            imageio.probe(data)
        except PixelBoostError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc

        fmt = str(fields.get("output_format") or fields.get("format") or cfg.server.result_format)
        try:
            job = state["jobs"].submit(
                data,
                fields,
                filename=ctx["filename"],
                output_format=fmt,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return JSONResponse(
            {"id": job.id, "status": job.status, "poll": f"/v1/jobs/{job.id}"},
            status_code=202,
        )

    @app.get("/v1/jobs/{job_id}", tags=["jobs"])
    async def job_status(job_id: str, _: None = Depends(auth)) -> Dict[str, Any]:
        job = state["jobs"].get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job id")
        return job.as_dict()

    @app.get("/v1/jobs/{job_id}/result", tags=["jobs"])
    async def job_result(job_id: str, _: None = Depends(auth)) -> Any:
        job = state["jobs"].get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job id")
        if job.status == "failed":
            raise HTTPException(status_code=422, detail=job.error or "job failed")
        if not job.done:
            raise HTTPException(status_code=409, detail=f"job is {job.status}")

        fmt = (job.output_format or "png").lower()
        headers = {
            "Content-Disposition": f'attachment; filename="pb_{job.id}.{fmt}"',
            "X-PixelBoost-Ms": str(job.elapsed_ms or 0),
        }
        return Response(
            content=job.result or b"",
            media_type=IMAGE_MEDIA.get(fmt, "application/octet-stream"),
            headers=headers,
        )

    @app.delete("/v1/jobs/{job_id}", tags=["jobs"])
    async def cancel_job(job_id: str, _: None = Depends(auth)) -> Dict[str, Any]:
        cancelled = state["jobs"].cancel(job_id)
        if not cancelled:
            raise HTTPException(status_code=409, detail="job is not cancellable")
        return {"id": job_id, "status": "cancelled"}

    return app
