"""Command line interface.

Every enhancement flag defaults to :data:`argparse.SUPPRESS`, which means the
namespace only contains flags the user actually typed. That is what makes the
three-layer precedence work (defaults < config file < CLI) without a pile of
``if args.x is not None`` checks -- an unset flag simply leaves the config value
in place.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import logging
import os
import statistics
import sys
import time
from typing import Any

from pixelboost import __version__
from pixelboost.backends.registry import BACKEND_NAMES, backend_capabilities, describe_environment
from pixelboost.config import Config, load_config
from pixelboost.errors import PixelBoostError
from pixelboost.types import EnhanceOptions

LOG_FORMAT = "%(levelname)-7s %(name)s: %(message)s"

UPTHRUM_FLAGS = {
    "upthrum_device": "device",
    "upthrum_bands": "bands",
    "upthrum_top_frequency": "top_frequency",
    "upthrum_phase_gain": "phase_gain",
    "upthrum_persistence": "persistence_relative",
    "upthrum_coherence_power": "coherence_power",
    "upthrum_anisotropy": "anisotropy",
    "upthrum_detail": "detail",
    "upthrum_topology": "topology",
    "upthrum_chroma": "chroma",
}

ENHANCE_KEYS = (
    "scale",
    "width",
    "height",
    "longest_side",
    "keep_aspect",
    "provider",
    "fp16",
    "threads",
    "tile",
    "tile_overlap",
    "tile_pad",
    "max_pixels",
    "pre_downscale",
    "denoise",
    "denoise_radius",
    "chroma_denoise",
    "detail",
    "detail_radius",
    "detail_eps",
    "sharpen",
    "sharpen_radius",
    "sharpen_threshold",
    "gamma",
    "contrast",
    "saturation",
    "auto_levels",
    "alpha_mode",
    "output_format",
    "quality",
    "preserve_metadata",
)


def setup_logging(verbosity: int, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)


def S(**kwargs: Any) -> dict[str, Any]:
    """argparse kwargs with SUPPRESS default, so unset flags stay absent."""
    kwargs.setdefault("default", argparse.SUPPRESS)
    return kwargs


def _add_common(p: argparse.ArgumentParser) -> None:
    """Repeat the global flags on a subparser.

    Users write ``pixelboost serve --port 8000 --quiet`` far more often than they
    put the flag before the subcommand, and argparse rejects the trailing form
    unless the subparser knows about it. Because every default is SUPPRESS, an
    omitted flag simply leaves the top-level value in place -- and ``-v`` still
    accumulates across both positions.
    """
    p.add_argument("-c", "--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument(
        "-v", "--verbose", action="count", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)


def _add_target_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("output size")
    g.add_argument("--scale", type=float, **S(help="upscale factor, e.g. 2 / 3 / 4 (default 4)"))
    g.add_argument("--width", type=int, **S(help="exact output width in pixels"))
    g.add_argument("--height", type=int, **S(help="exact output height in pixels"))
    g.add_argument("--longest-side", type=int, **S(help="scale so the longest edge equals this"))
    g.add_argument("--no-keep-aspect", dest="keep_aspect", action="store_false", **S(help="allow non-uniform scaling"))
    g.add_argument("--max-pixels", type=int, **S(help="refuse/clamp outputs above this pixel count"))


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("backend")
    g.add_argument("--backend", choices=BACKEND_NAMES, **S(help="inference backend (default auto)"))
    g.add_argument("--model", **S(help="registry model name, or a path to a .pth/.onnx file"))
    g.add_argument("--model-path", **S(help="explicit model file path, overrides --model lookup"))
    g.add_argument("--models-dir", **S(help="where model files live"))
    g.add_argument("--provider", **S(help="execution provider: auto|cpu|cuda|tensorrt|directml|coreml|openvino|rocm"))
    g.add_argument("--fp16", action="store_true", **S(help="half precision (CUDA/TensorRT only)"))
    g.add_argument("--threads", type=int, **S(help="intra-op threads for the CPU provider (0 = let ORT decide)"))
    g.add_argument("--tile", type=int, **S(help="tile size in source pixels, 0 disables tiling (default 512)"))
    g.add_argument("--tile-overlap", type=int, **S(help="overlap between tiles, source pixels (default 16)"))
    g.add_argument("--tile-pad", type=int, **S(help="reflect padding around the image before tiling (default 16)"))
    g.add_argument("--pre-downscale", dest="pre_downscale", action="store_true", **S(help="downscale before inference to save VRAM"))
    g.add_argument("--no-pre-downscale", dest="pre_downscale", action="store_false", **S(help="never pre-downscale"))


def _add_upthrum_args(p: argparse.ArgumentParser) -> None:
    """Flags for the phase-reconstruction backend.

    Only meaningful with ``--backend upthrum``, but registered on every
    enhancement subcommand so that the flags exist wherever a backend can be
    chosen. They travel into ``EnhanceOptions.extra["upthrum"]`` rather than
    becoming fields on the options dataclass, which keeps that dataclass a
    description of the pipeline and lets the algorithm's parameters evolve
    without touching the shared type.
    """
    g = p.add_argument_group("upthrum (phase reconstruction)")
    g.add_argument(
        "--upthrum-device",
        choices=("auto", "cpu", "cuda"),
        **S(help="run the FFT-bound analysis on cpu or cuda (default auto)"),
    )
    g.add_argument("--upthrum-bands", type=int, **S(help="log-Gabor bands in the hierarchy (default 3)"))
    g.add_argument(
        "--upthrum-top-frequency",
        type=float,
        **S(help="centre of the highest band, cycles/px, must stay under 0.5 (default 0.22)"),
    )
    g.add_argument(
        "--upthrum-phase-gain",
        type=float,
        **S(help="phase extrapolation multiplier; 1.0 is the exact linearisation (default 1.0)"),
    )
    g.add_argument(
        "--upthrum-persistence",
        type=float,
        **S(help="persistence threshold as a fraction of the p1-p99 range (default 0.18)"),
    )
    g.add_argument(
        "--upthrum-coherence-power",
        type=float,
        **S(help="exponent on the phase-coherence gate (default 0.5)"),
    )
    g.add_argument(
        "--upthrum-anisotropy",
        type=float,
        **S(help="structure-aligned amplitude kernel, 0..1 (default 0.55)"),
    )
    g.add_argument(
        "--upthrum-detail",
        type=float,
        **S(help="micro-contrast built into the backend, 0..1.5 (default 0)"),
    )
    g.add_argument(
        "--no-upthrum-topology",
        dest="upthrum_topology",
        action="store_false",
        **S(help="disable the persistent-homology constraint (strictly worse; for ablation)"),
    )
    g.add_argument(
        "--no-upthrum-chroma",
        dest="upthrum_chroma",
        action="store_false",
        **S(help="leave chroma on the smooth path instead of reconstructing it"),
    )


def _add_enhance_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("quality")
    g.add_argument("--denoise", type=float, **S(help="pre-upscale edge-preserving denoise strength, 0..1 (default 0)"))
    g.add_argument("--denoise-radius", type=float, **S(help="denoise radius in source pixels (default 3)"))
    g.add_argument("--chroma-denoise", type=float, **S(help="post-upscale chroma smoothing sigma (default 0)"))
    g.add_argument("--detail", type=float, **S(help="guided-filter detail boost, 0..1.5 (default 0.35)"))
    g.add_argument("--detail-radius", type=int, **S(help="detail extraction radius (default 4)"))
    g.add_argument("--detail-eps", type=float, **S(help="guided filter epsilon; lower = stronger edge lock (default 1e-3)"))
    g.add_argument("--sharpen", type=float, **S(help="final unsharp amount, 0..1 (default 0)"))
    g.add_argument("--sharpen-radius", type=float, **S(help="unsharp radius (default 1.0)"))
    g.add_argument("--sharpen-threshold", type=float, **S(help="unsharp noise floor (default 0.004)"))
    g.add_argument("--auto-levels", type=float, **S(help="pre-upscale luma stretch, 0..1 (default 0)"))
    g.add_argument("--gamma", type=float, **S(help="gamma correction (default 1.0)"))
    g.add_argument("--contrast", type=float, **S(help="-1..1 luma contrast around mid-grey (default 0)"))
    g.add_argument("--saturation", type=float, **S(help="1.0 = unchanged (default 1.0)"))

    g = p.add_argument_group("output")
    g.add_argument("--alpha-mode", choices=("lanczos", "nearest", "backend"), **S(help="how the alpha plane is scaled"))
    g.add_argument("--format", dest="output_format", **S(help="png|jpg|webp|tiff|bmp|avif (default: follow the output extension)"))
    g.add_argument("--quality", type=int, **S(help="lossy quality 1..100 (default 95)"))
    g.add_argument("--strip-metadata", dest="preserve_metadata", action="store_false", **S(help="drop ICC/EXIF from the output"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixelboost",
        description="CPU/GPU accelerated image quality enhancement and super-resolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  pixelboost upscale photo.jpg -o photo_4x.png --scale 4\n"
            "  pixelboost upscale *.jpg -o out/ --model realesrgan-x4plus-anime --detail 0.5\n"
            "  pixelboost upscale in.png -o out.png --backend classical        # no model needed\n"
            "  pixelboost upscale in.png -o out.png --backend upthrum --scale 4  # phase reconstruction\n"
            "  pixelboost upscale in.png -o out.png --provider cuda --fp16 --tile 768\n"
            "  pixelboost batch ./photos -o ./enhanced --recursive --scale 4\n"
            "  pixelboost serve --host 0.0.0.0 --port 8000 --workers 4\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"pixelboost {__version__}")
    parser.add_argument("-c", "--config", help="path to pixelboost.yaml / .json / .toml")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v for info, -vv for debug")
    parser.add_argument("-q", "--quiet", action="store_true", help="errors only")

    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    up = sub.add_parser("upscale", help="enhance one image or a list of images")
    _add_common(up)
    up.add_argument("inputs", nargs="+", help="input files, globs or directories")
    up.add_argument("-o", "--output", required=True, help="output file or directory")
    up.add_argument("--suffix", default="_upscaled", help="suffix when writing into a directory")
    up.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    up.add_argument("--json", action="store_true", help="print a machine-readable report")
    _add_target_args(up)
    _add_backend_args(up)
    _add_upthrum_args(up)
    _add_enhance_args(up)

    ba = sub.add_parser("batch", help="walk a directory tree and enhance everything")
    _add_common(ba)
    ba.add_argument("input", help="directory to scan")
    ba.add_argument("-o", "--output", required=True, help="output directory")
    ba.add_argument("--recursive", action="store_true", help="descend into subdirectories")
    ba.add_argument("--ext", default="jpg,jpeg,png,webp,bmp,tif,tiff", help="comma-separated extensions")
    ba.add_argument("--suffix", default="_upscaled", help="filename suffix for outputs")
    ba.add_argument("--overwrite", action="store_true", help="reprocess files that already exist")
    ba.add_argument("--json", action="store_true", help="print a machine-readable report")
    _add_target_args(ba)
    _add_backend_args(ba)
    _add_upthrum_args(ba)
    _add_enhance_args(ba)

    sv = sub.add_parser("serve", help="run the HTTP API")
    _add_common(sv)
    sv.add_argument("--host", help="bind address (default 0.0.0.0)")
    sv.add_argument("--port", type=int, help="bind port (default 8000)")
    sv.add_argument("--workers", type=int, help="parallel request slots (default 2)")
    sv.add_argument("--reload", action="store_true", help="dev only: reload on source change")
    sv.add_argument("--proxy-headers", action="store_true", help="trust X-Forwarded-* from a reverse proxy")

    sub.add_parser("capabilities", help="show available backends, providers and models")

    mo = sub.add_parser("models", help="manage model files")
    _add_common(mo)
    mo.add_argument("action", choices=("list", "download", "remove", "export", "usage"))
    mo.add_argument("--model", help="model name (default: the configured one)")
    mo.add_argument("--all", action="store_true", help="apply to every registry entry")
    mo.add_argument("--force", action="store_true", help="re-download even if present")
    mo.add_argument("--output", help="output path for export")
    mo.add_argument("--fp16", action="store_true", help="export half-precision ONNX")
    mo.add_argument("--opset", type=int, default=17, help="ONNX opset for export")
    mo.add_argument("--static", action="store_true", help="export with a fixed input size")

    be = sub.add_parser("benchmark", help="measure throughput on a synthetic image")
    _add_common(be)
    be.add_argument("--size", type=int, default=512, help="synthetic input edge length (default 512)")
    be.add_argument("--repeat", type=int, default=3, help="iterations (default 3)")
    be.add_argument("--warmup", type=int, default=1, help="warmup iterations (default 1)")
    _add_target_args(be)
    _add_backend_args(be)
    _add_upthrum_args(be)
    _add_enhance_args(be)

    return parser


def _apply_config_overrides(cfg: Config, ns: argparse.Namespace) -> Config:
    for attr in ("backend", "model", "provider", "models_dir"):
        if hasattr(ns, attr):
            setattr(cfg, attr, getattr(ns, attr))
    if hasattr(ns, "fp16"):
        cfg.fp16 = ns.fp16
    if hasattr(ns, "threads"):
        cfg.threads = ns.threads
    if hasattr(ns, "tile"):
        cfg.tile = ns.tile
    return cfg


def _build_options(cfg: Config, ns: argparse.Namespace) -> EnhanceOptions:
    opts = dataclasses.replace(cfg.defaults)
    for key in ENHANCE_KEYS:
        if hasattr(ns, key):
            setattr(opts, key, getattr(ns, key))
    if hasattr(ns, "backend"):
        opts.backend = ns.backend
    if hasattr(ns, "model"):
        opts.model = ns.model
    if hasattr(ns, "model_path"):
        opts.model_path = ns.model_path
    if not hasattr(ns, "tile") and cfg.tile:
        opts.tile = cfg.tile
    overrides = {
        param: getattr(ns, flag) for flag, param in UPTHRUM_FLAGS.items() if hasattr(ns, flag)
    }
    if overrides:
        opts.extra = {**(opts.extra or {}), "upthrum": overrides}
    return opts


def _expand_inputs(patterns: list[str], exts: list[str] | None = None) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            found.extend(_walk(pattern, recursive=True, exts=exts))
            continue
        if any(ch in pattern for ch in "*?["):
            matched = sorted(glob.glob(pattern, recursive=True))
            found.extend(p for p in matched if os.path.isfile(p))
            continue
        if os.path.isfile(pattern):
            found.append(pattern)
            continue
        raise PixelBoostError(f"input not found: {pattern}")
    seen = set()
    unique = []
    for path in found:
        real = os.path.abspath(path)
        if real not in seen:
            seen.add(real)
            unique.append(path)
    return unique


def _walk(root: str, recursive: bool, exts: list[str] | None) -> list[str]:
    exts = [e.lower().lstrip(".") for e in (exts or [])]
    out: list[str] = []
    if recursive:
        for base, _dirs, files in os.walk(root):
            for name in sorted(files):
                if not exts or name.rsplit(".", 1)[-1].lower() in exts:
                    out.append(os.path.join(base, name))
    else:
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isfile(path) and (not exts or name.rsplit(".", 1)[-1].lower() in exts):
                out.append(path)
    return out


def _resolve_outputs(inputs: list[str], output: str, suffix: str, overwrite: bool) -> list[str | None]:
    multi = len(inputs) > 1 or os.path.isdir(output) or not os.path.splitext(output)[1]
    targets: list[str | None] = []
    for path in inputs:
        if multi:
            stem, ext = os.path.splitext(os.path.basename(path))
            targets.append(os.path.join(output, f"{stem}{suffix}{ext}"))
        else:
            targets.append(output)
    if not overwrite:
        blocked = [t for t in targets if t and os.path.exists(t)]
        if blocked and len(blocked) == len(targets):
            raise PixelBoostError(
                f"all {len(blocked)} outputs already exist (first: {blocked[0]}); "
                f"pass --overwrite to replace them"
            )
        targets = [None if (t and os.path.exists(t)) else t for t in targets]
    return targets


def _progress_printer():
    state = {"last": -1}

    def report(done: int, total: int) -> None:
        if total <= 1:
            return
        pct = int(done * 100 / total)
        if pct == state["last"]:
            return
        state["last"] = pct
        sys.stderr.write(f"\r  tiles {done}/{total} ({pct}%)")
        sys.stderr.flush()
        if done >= total:
            sys.stderr.write("\r" + " " * 40 + "\r")

    return report


def cmd_upscale(cfg: Config, ns: argparse.Namespace) -> int:
    from pixelboost.engine import Engine

    opts = _build_options(cfg, ns)
    inputs = _expand_inputs(ns.inputs)
    targets = _resolve_outputs(inputs, ns.output, ns.suffix, ns.overwrite)
    report = _progress_printer()

    results = []
    failures = 0
    with Engine(cfg) as engine:
        for src, dst in zip(inputs, targets):
            if dst is None:
                results.append({"input": src, "status": "skipped"})
                continue
            try:
                result, written = engine.enhance_file(src, dst, opts, on_progress=report)
                entry = {"input": src, "output": written, "status": "ok", **result.summary()}
                if not ns.json:
                    print(
                        f"{src} -> {written}  "
                        f"{result.summary()['src_size']} -> {result.summary()['dst_size']}  "
                        f"[{result.backend}/{result.provider}] {result.elapsed_ms:.0f} ms"
                    )
                    for note in result.notes:
                        print(f"  note: {note}")
                results.append(entry)
            except PixelBoostError as exc:
                failures += 1
                results.append({"input": src, "status": "error", "error": str(exc)})
                print(f"{src}: {exc}", file=sys.stderr)

    if ns.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    return 1 if failures else 0


def cmd_batch(cfg: Config, ns: argparse.Namespace) -> int:
    from pixelboost.engine import Engine

    opts = _build_options(cfg, ns)
    exts = [e for e in ns.ext.split(",") if e.strip()]
    inputs = _walk(ns.input, ns.recursive, exts)
    if not inputs:
        print("no matching files", file=sys.stderr)
        return 1

    with Engine(cfg) as engine:
        reports = engine.batch(
            inputs,
            ns.output,
            opts,
            suffix=ns.suffix,
            skip_existing=not ns.overwrite,
        )

    ok = sum(1 for r in reports if r.get("status") == "ok")
    skipped = sum(1 for r in reports if r.get("status") == "skipped")
    errors = [r for r in reports if r.get("status") == "error"]

    if ns.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
    else:
        print(f"done: {ok} enhanced, {skipped} skipped, {len(errors)} failed")
        for entry in errors[:20]:
            print(f"  {entry['input']}: {entry.get('error')}", file=sys.stderr)
    return 1 if errors else 0


def cmd_serve(cfg: Config, ns: argparse.Namespace) -> int:
    try:
        import uvicorn  # noqa: WPS433
    except ImportError:
        print(
            "the HTTP server needs extra packages:\n"
            "  pip install 'pixelboost[server]'",
            file=sys.stderr,
        )
        return 2

    from pixelboost.server.app import create_app

    host = ns.host or cfg.server.host
    port = ns.port or cfg.server.port
    workers = ns.workers or cfg.server.workers
    cfg.server.workers = workers

    app = create_app(cfg)
    print(f"pixelboost {__version__} serving on http://{host}:{port}  (workers={workers})")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        proxy_headers=ns.proxy_headers,
        timeout_keep_alive=cfg.server.keep_alive,
        reload=ns.reload,
    )
    return 0


def cmd_capabilities(cfg: Config, ns: argparse.Namespace) -> int:
    from pixelboost.models import disk_usage, list_models

    caps = backend_capabilities(cfg)
    caps["environment"] = describe_environment().splitlines()
    caps["models_dir"] = cfg.models_dir
    caps["models_on_disk"] = disk_usage(cfg.models_dir)
    print(json.dumps(caps, indent=2, ensure_ascii=False))
    print("\nmodels:", file=sys.stderr)
    for row in list_models(cfg):
        flag = "present" if row["present"] else "missing"
        print(f"  {row['name']:<28} x{row['scale']}  {row['arch']:<6} {flag}", file=sys.stderr)
    return 0


def cmd_models(cfg: Config, ns: argparse.Namespace) -> int:
    from pixelboost.config import MODEL_REGISTRY
    from pixelboost.models import disk_usage, ensure_model, export_onnx, list_models, remove_model

    action = ns.action
    if action == "list":
        rows = list_models(cfg)
        if getattr(ns, "json", False):
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
            for row in rows:
                mark = "[x]" if row["present"] else "[ ]"
                print(f"{mark} {row['name']:<28} x{row['scale']}  {row['arch']:<6} {row['description']}")
        return 0

    if action == "usage":
        print(json.dumps(disk_usage(cfg.models_dir), indent=2))
        return 0

    names = list(MODEL_REGISTRY) if ns.all else [ns.model or cfg.model]

    if action == "remove":
        for name in names:
            removed = remove_model(cfg, name)
            print(f"{name}: removed {len(removed)} file(s)")
        return 0

    if action == "download":
        failures = 0
        for name in names:
            spec = cfg.resolve_model(name)
            try:
                path = ensure_model(cfg, spec, force=ns.force, progress=_byte_progress(name))
                print(f"{name}: {path}")
            except PixelBoostError as exc:
                failures += 1
                print(f"{name}: {exc}", file=sys.stderr)
        return 1 if failures else 0

    if action == "export":
        failures = 0
        for name in names:
            spec = cfg.resolve_model(name)
            try:
                if not os.path.isfile(cfg.model_path(spec)):
                    ensure_model(cfg, spec, progress=_byte_progress(name))
                out = export_onnx(
                    cfg,
                    spec,
                    output=ns.output,
                    opset=ns.opset,
                    fp16_weights=ns.fp16,
                    dynamic=not ns.static,
                )
                print(f"{name}: {out}")
            except PixelBoostError as exc:
                failures += 1
                print(f"{name}: {exc}", file=sys.stderr)
        return 1 if failures else 0
    return 2


def _byte_progress(name: str):
    def report(done: int, total: int) -> None:
        if total <= 0:
            sys.stderr.write(f"\r  {name}: {done / 1e6:.1f} MB")
        else:
            sys.stderr.write(f"\r  {name}: {done * 100 / total:5.1f}%  ({done / 1e6:.1f} MB)")
        sys.stderr.flush()

    return report


def cmd_benchmark(cfg: Config, ns: argparse.Namespace) -> int:

    from pixelboost.backends.registry import create_backend
    from pixelboost.pipeline import Pipeline

    opts = _build_options(cfg, ns)
    opts.width = None
    opts.height = None
    size = int(ns.size)
    image = _synthetic(size)
    backend = create_backend(cfg, opts)

    try:
        for _ in range(max(0, ns.warmup)):
            Pipeline(backend, opts, size, size).run(image.copy())
    except PixelBoostError as exc:
        print(f"warmup failed: {exc}", file=sys.stderr)

    timings: list[float] = []
    tiles = 0
    out_size = (0, 0)
    for _ in range(max(1, ns.repeat)):
        pipeline = Pipeline(backend, opts, size, size)
        started = time.perf_counter()
        out, _alpha = pipeline.run(image.copy())
        timings.append((time.perf_counter() - started) * 1000.0)
        tiles = pipeline.tile_count
        out_size = (out.shape[1], out.shape[0])

    src_mp = size * size / 1e6
    dst_mp = out_size[0] * out_size[1] / 1e6
    median = statistics.median(timings)
    payload: dict[str, Any] = {
        "backend": backend.name,
        "provider": backend.provider,
        "model": backend.model,
        "input": [size, size],
        "output": list(out_size),
        "tiles": tiles,
        "scale": opts.scale if not (opts.width or opts.height or opts.longest_side) else None,
        "runs_ms": [round(t, 1) for t in timings],
        "median_ms": round(median, 1),
        "min_ms": round(min(timings), 1),
        "src_mpix_per_s": round(src_mp / (median / 1000.0), 3),
        "dst_mpix_per_s": round(dst_mp / (median / 1000.0), 3),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _synthetic(size: int):
    """A test pattern with the properties that matter for timing.

    Flat gradients alone would under-report cost on real photos and would hide
    texture artefacts. Mixing high-frequency detail with smooth ramps gives a
    number that tracks reality closely enough to size a server with.
    """
    import numpy as np

    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    r = x / max(1, size - 1)
    g = y / max(1, size - 1)
    b = 0.5 + 0.5 * np.sin(x / 7.0) * np.cos(y / 9.0)
    img = np.stack([r, g, b], axis=-1)
    checker = ((x // 5 + y // 5) % 2).astype(np.float32) * 0.12
    img = np.clip(img * 0.8 + checker[..., None] + 0.05, 0.0, 1.0)
    return img.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    setup_logging(ns.verbose, ns.quiet)

    try:
        cfg = load_config(ns.config)
    except PixelBoostError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if hasattr(ns, "models_dir") and ns.models_dir:
        cfg.models_dir = ns.models_dir
    if ns.command in ("upscale", "batch", "benchmark"):
        cfg = _apply_config_overrides(cfg, ns)

    handlers = {
        "upscale": cmd_upscale,
        "batch": cmd_batch,
        "serve": cmd_serve,
        "capabilities": cmd_capabilities,
        "models": cmd_models,
        "benchmark": cmd_benchmark,
    }
    try:
        return handlers[ns.command](cfg, ns)
    except PixelBoostError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
