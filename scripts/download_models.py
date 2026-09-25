#!/usr/bin/env python3
"""Download model checkpoints.

    python scripts/download_models.py --list
    python scripts/download_models.py --model realesrgan-x4plus
    python scripts/download_models.py --all
    python scripts/download_models.py --all --export-onnx

The ONNX export step needs PyTorch; the plain download does not.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pixelboost.config import MODEL_REGISTRY, load_config  # noqa: E402
from pixelboost.errors import PixelBoostError  # noqa: E402
from pixelboost.models import (  # noqa: E402
    available_space,
    disk_usage,
    ensure_model,
    export_onnx,
    list_models,
)


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def progress(name: str):
    last = [-1]

    def report(done: int, total: int) -> None:
        pct = int(done * 100 / total) if total else 0
        if pct == last[0]:
            return
        last[0] = pct
        bar = "#" * (pct // 3)
        sys.stdout.write(f"\r  {name:<28} [{bar:<33}] {pct:3d}%")
        sys.stdout.flush()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", help="model name (repeatable)")
    parser.add_argument("--all", action="store_true", help="every entry in the registry")
    parser.add_argument("--list", action="store_true", help="show the registry and exit")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--export-onnx", action="store_true", help="also convert to ONNX")
    parser.add_argument("--fp16", action="store_true", help="export half precision ONNX")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--static", action="store_true", help="export with fixed input size")
    parser.add_argument("--models-dir", help="override the models directory")
    parser.add_argument("--config", help="path to a pixelboost config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.models_dir:
        cfg.models_dir = args.models_dir
    os.makedirs(cfg.models_dir, exist_ok=True)

    if args.list or not (args.model or args.all):
        print(f"models directory: {cfg.models_dir}")
        print(f"free space      : {human(available_space(cfg.models_dir))}\n")
        for row in list_models(cfg):
            mark = "[x]" if row["present"] else "[ ]"
            print(f"{mark} {row['name']:<28} x{row['scale']}  {row['arch']:<6} {row['description']}")
        if not args.list:
            print("\npass --model NAME or --all to download")
        return 0

    names = list(MODEL_REGISTRY) if args.all else list(args.model or [])

    free = available_space(cfg.models_dir)
    needed = sum(
        60 * 1024 * 1024 for name in names if name in MODEL_REGISTRY
    )
    if free < needed * 1.2:
        print(
            f"warning: {human(free)} free, roughly {human(needed)} needed. "
            f"Set PIXELBOOST_HOME or --models-dir to a larger volume.",
            file=sys.stderr,
        )

    failures = 0
    for name in names:
        try:
            spec = cfg.resolve_model(name)
        except PixelBoostError as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            failures += 1
            continue

        try:
            path = ensure_model(cfg, spec, force=args.force, progress=progress(name))
            print(f"\r  {name:<28} {human(os.path.getsize(path))}  {path}")
        except PixelBoostError as exc:
            print(f"\r  {name:<28} FAILED: {exc}", file=sys.stderr)
            failures += 1
            continue

        if args.export_onnx:
            try:
                out = export_onnx(
                    cfg,
                    spec,
                    opset=args.opset,
                    fp16_weights=args.fp16,
                    dynamic=not args.static,
                )
                print(f"  {'':<28} onnx -> {out}")
            except PixelBoostError as exc:
                print(f"  {'':<28} onnx export failed: {exc}", file=sys.stderr)
                failures += 1

    usage = disk_usage(cfg.models_dir)
    print(f"\nmodels directory: {usage['path']} ({usage['files']} files, {usage['human']})")
    print("next: pixelboost upscale input.jpg -o output.png --scale 4")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
