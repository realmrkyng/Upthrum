"""Model download and ONNX export.

Kept in the package rather than in ``scripts/`` so the CLI, the HTTP server and
the standalone script all share one implementation.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from typing import Any, Callable

from pixelboost.config import MODEL_REGISTRY, Config, ModelSpec
from pixelboost.errors import ModelNotFound, PixelBoostError

ProgressFn = Callable[[int, int], None]


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: str, progress: ProgressFn | None = None) -> str:
    """Stream to a temp file, then atomically move into place.

    A half-written 64 MB checkpoint that looks present is worse than no
    checkpoint at all -- it fails at inference time with an opaque tensor-shape
    error. Atomic replace removes that failure mode.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "pixelboost/0.1"})
    fd, tmp = tempfile.mkstemp(prefix=".pixelboost-dl-", dir=os.path.dirname(os.path.abspath(dest)))
    os.close(fd)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                block = response.read(1 << 16)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if progress:
                    progress(done, total)
        os.replace(tmp, dest)
    except urllib.error.URLError as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise PixelBoostError(
            f"download failed for {url}: {exc}. If the host is blocked, fetch the "
            f"file manually and drop it into the models directory."
        ) from exc
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return dest


def ensure_model(
    cfg: Config,
    spec: ModelSpec,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> str:
    if spec.kind == "none":
        raise ModelNotFound(f"{spec.name} is a model-free backend; nothing to download")
    if not spec.url:
        raise ModelNotFound(f"{spec.name} has no download URL; supply the file manually")

    dest = cfg.model_path(spec)
    if os.path.isfile(dest) and not force:
        if spec.sha256 and file_sha256(dest) != spec.sha256:
            raise PixelBoostError(
                f"{dest} exists but its checksum does not match the registry; "
                f"re-run with --force"
            )
        return dest

    _download(spec.url, dest, progress)
    if spec.sha256:
        actual = file_sha256(dest)
        if actual != spec.sha256:
            os.unlink(dest)
            raise PixelBoostError(
                f"checksum mismatch for {spec.name}: expected {spec.sha256}, got {actual}"
            )
    return dest


def list_models(cfg: Config | None = None) -> list[dict[str, Any]]:
    from pixelboost.backends.registry import onnx_path_for, torch_path_for

    rows = []
    for spec in MODEL_REGISTRY.values():
        pth = torch_path_for(cfg, spec) if cfg else None
        onnx = onnx_path_for(cfg, spec) if cfg else None
        rows.append(
            {
                "name": spec.name,
                "scale": spec.scale,
                "arch": spec.arch,
                "tags": list(spec.tags),
                "pth": pth,
                "onnx": onnx,
                "present": bool(pth or onnx),
                "description": spec.description,
                "url": spec.url,
            }
        )
    return rows


def export_onnx(
    cfg: Config,
    spec: ModelSpec,
    output: str | None = None,
    opset: int = 17,
    fp16_weights: bool = False,
    dynamic: bool = True,
    simplify: bool = False,
    check: int = 64,
) -> str:
    """Convert a ``.pth`` checkpoint to ONNX.

    Dynamic axes are on by default. That is what lets the tiler feed arbitrary
    tile sizes; a static export would force every tile to the exact training
    resolution and waste most of the GPU on padding.
    """
    try:
        import torch  # noqa: WPS433
    except ImportError as exc:
        raise PixelBoostError(
            "exporting to ONNX needs PyTorch. `pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu` is enough for conversion."
        ) from exc

    from pixelboost.backends.torch_backend import build_model, load_state_dict

    pth = cfg.model_path(spec)
    if not os.path.isfile(pth):
        raise ModelNotFound(f"{pth} not found; download it first with `models download`")

    if output is None:
        suffix = "_fp16.onnx" if fp16_weights else ".onnx"
        output = os.path.join(cfg.models_dir, os.path.splitext(spec.filename)[0] + suffix)

    net = build_model(
        spec.arch,
        scale=spec.scale,
        num_feat=spec.num_feat,
        num_block=spec.num_block,
        num_conv=spec.num_conv,
    )
    missing, _ = net.load_state_dict(load_state_dict(pth), strict=False)
    if missing:
        raise ModelNotFound(
            f"checkpoint does not match arch={spec.arch}: {len(missing)} tensors missing"
        )
    net.eval()

    dummy = torch.zeros(1, 3, check, check, dtype=torch.float32)
    if fp16_weights:
        net = net.half()
        dummy = dummy.half()

    axes = (
        {0: "batch", 2: "height", 3: "width"}
        if dynamic
        else {0: "batch"}
    )
    with torch.inference_mode():
        torch.onnx.export(
            net,
            dummy,
            output,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": axes, "output": axes},
            opset_version=int(opset),
            do_constant_folding=True,
            dynamo=False,
        )

    if simplify:
        try:
            import onnx  # noqa: WPS433
            from onnxsim import simplify as onnx_simplify  # noqa: WPS433

            model = onnx.load(output)
            simplified, ok = onnx_simplify(model)
            if ok:
                onnx.save(simplified, output)
        except ImportError:
            pass

    try:
        import onnxruntime as ort  # noqa: WPS433

        sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"input": dummy.cpu().numpy()})[0]
        want = (1, 3, check * spec.scale, check * spec.scale)
        if tuple(got.shape) != want:
            raise PixelBoostError(
                f"exported model produced {got.shape}, expected {want}"
            )
    except ImportError:
        pass

    return output


def remove_model(cfg: Config, name: str, onnx_too: bool = True) -> list[str]:
    spec = cfg.resolve_model(name)
    removed: list[str] = []
    candidates = [cfg.model_path(spec)] if spec.filename else []
    if onnx_too and spec.filename:
        stem = os.path.splitext(spec.filename)[0]
        candidates += [
            os.path.join(cfg.models_dir, stem + ".onnx"),
            os.path.join(cfg.models_dir, stem + "_fp16.onnx"),
        ]
    for path in candidates:
        if path and os.path.isfile(path):
            os.remove(path)
            removed.append(path)
    return removed


def disk_usage(models_dir: str) -> dict[str, Any]:
    if not os.path.isdir(models_dir):
        return {"files": 0, "bytes": 0, "path": models_dir}
    files = 0
    total = 0
    for name in os.listdir(models_dir):
        path = os.path.join(models_dir, name)
        if os.path.isfile(path):
            files += 1
            total += os.path.getsize(path)
    return {"files": files, "bytes": total, "path": models_dir, "human": _human(total)}


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def available_space(path: str) -> int:
    probe = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path)) or "."
    return shutil.disk_usage(probe).free
