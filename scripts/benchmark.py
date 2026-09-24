#!/usr/bin/env python3
"""Tile-size sweep and throughput report.

    python scripts/benchmark.py --size 1024 --scale 4
    python scripts/benchmark.py --size 1024 --model realesr-classical
    python scripts/benchmark.py --size 512 --tiles 0,256,512,768 --json out.json

The useful output is the *knee*: throughput rises with tile size until the GPU
saturates, then flattens while memory keeps climbing. Pick the smallest tile on
the flat part -- that leaves headroom for concurrent requests.

A tile of 0 means "no tiling", i.e. one inference over the whole image. Useful
as an upper bound on single-request speed, and as a way to find the size at
which the GPU actually runs out of memory.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from pixelboost.config import load_config  # noqa: E402
from pixelboost.errors import PixelBoostError  # noqa: E402
from pixelboost.pipeline import Pipeline  # noqa: E402
from pixelboost.types import EnhanceOptions  # noqa: E402


def test_image(size: int, seed: int = 7) -> np.ndarray:
    """Deterministic pattern with both smooth ramps and high-frequency detail.

    A flat gradient would under-report cost (most convolutions are data
    independent, but the pipeline's post-processing is not) and would hide
    texture artefacts. Mixing both gives a number that tracks real photos.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    r = x / max(1, size - 1)
    g = y / max(1, size - 1)
    b = 0.5 + 0.25 * np.sin(x / 7.0) * np.cos(y / 11.0)
    img = np.stack([r, g, b], axis=-1)
    detail = rng.random((size, size, 3), dtype=np.float32) * 0.06 - 0.03
    checker = ((x // 9 + y // 9) % 2).astype(np.float32) * 0.10
    return np.clip(img * 0.8 + detail + checker[..., None] + 0.06, 0.0, 1.0).astype(np.float32)


def bench(backend, opts, image, repeat: int, warmup: int) -> dict:
    h, w = image.shape[:2]
    for _ in range(max(0, warmup)):
        try:
            Pipeline(backend, opts, w, h).run(image.copy())
        except PixelBoostError:
            pass

    runs = []
    tiles = 0
    out_size = (0, 0)
    error = None
    for _ in range(max(1, repeat)):
        pipeline = Pipeline(backend, opts, w, h)
        started = time.perf_counter()
        try:
            out, _ = pipeline.run(image.copy())
        except (PixelBoostError, MemoryError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        runs.append((time.perf_counter() - started) * 1000.0)
        tiles = pipeline.tile_count
        out_size = (out.shape[1], out.shape[0])

    if not runs:
        return {"tile": opts.tile, "ok": False, "error": error}

    median = statistics.median(runs)
    return {
        "tile": opts.tile,
        "ok": True,
        "tiles_per_image": tiles,
        "output": list(out_size),
        "median_ms": round(median, 1),
        "min_ms": round(min(runs), 1),
        "max_ms": round(max(runs), 1),
        "src_mpix_per_s": round(h * w / 1e6 / (median / 1000.0), 3),
        "dst_mpix_per_s": round(out_size[0] * out_size[1] / 1e6 / (median / 1000.0), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=512, help="synthetic input edge length")
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tiles", default="256,512,768,0", help="comma-separated tile sizes; 0 = no tiling")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--model")
    parser.add_argument("--provider")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--json", help="write the full report to this path")
    parser.add_argument("--config")
    parser.add_argument("--models-dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.models_dir:
        cfg.models_dir = args.models_dir

    from pixelboost.backends.registry import create_backend, describe_environment

    print(describe_environment())
    print()

    image = test_image(args.size)
    base = EnhanceOptions(
        scale=args.scale,
        backend=args.backend,
        model=args.model or cfg.model,
        provider=args.provider,
        fp16=args.fp16,
        threads=args.threads,
        pre_downscale=False,
    )

    try:
        backend = create_backend(cfg, base)
    except PixelBoostError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"backend={backend.name} provider={backend.provider} model={backend.model} "
        f"input={args.size}x{args.size} scale={args.scale}x\n"
    )

    header = f"{'tile':>6} {'tiles':>6} {'median ms':>10} {'src MP/s':>9} {'dst MP/s':>9}"
    print(header)
    print("-" * len(header))

    report = {
        "environment": describe_environment().splitlines(),
        "backend": backend.name,
        "provider": backend.provider,
        "model": backend.model,
        "input": [args.size, args.size],
        "scale": args.scale,
        "results": [],
    }

    best = None
    for raw in args.tiles.split(","):
        tile = int(raw.strip() or 0)
        opts = EnhanceOptions.from_dict(base.to_dict())
        opts.tile = tile
        row = bench(backend, opts, image, args.repeat, args.warmup)
        report["results"].append(row)

        if not row["ok"]:
            print(f"{tile:>6} {'':>6} {'FAILED':>10}   {row.get('error', '')[:60]}")
            continue

        print(
            f"{tile:>6} {row['tiles_per_image']:>6} {row['median_ms']:>10.1f} "
            f"{row['src_mpix_per_s']:>9.3f} {row['dst_mpix_per_s']:>9.3f}"
        )
        if best is None or row["dst_mpix_per_s"] > best["dst_mpix_per_s"]:
            best = row

    if best:
        print(
            f"\nbest: tile={best['tile']} at {best['dst_mpix_per_s']} output MP/s "
            f"({best['median_ms']} ms per {args.size}x{args.size} image)"
        )
        print(
            "guidance: pick the smallest tile within ~5 % of the best throughput -- "
            "that leaves VRAM headroom for concurrent requests."
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"report written to {args.json}")

    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
