#!/usr/bin/env python3
"""Convert a PyTorch checkpoint to ONNX.

    python scripts/export_onnx.py --model realesrgan-x4plus
    python scripts/export_onnx.py --model realesrgan-x4plus --fp16
    python scripts/export_onnx.py --model realesrgan-x4plus --static --check 192

Why dynamic axes matter: a static export pins the graph to one input size, and
the tiler then has to pad every tile up to that size. For a 512px tile on a
1024x1024 photo, a static 512 export is fine; for a 300px tile it wastes 65 % of
every inference. Dynamic axes let the tiler pick the tile it actually wants.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pixelboost.config import load_config  # noqa: E402
from pixelboost.errors import PixelBoostError  # noqa: E402
from pixelboost.models import ensure_model, export_onnx  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="realesrgan-x4plus", help="registry model name or a .pth path")
    parser.add_argument("--output", help="destination .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="half-precision weights")
    parser.add_argument("--static", action="store_true", help="fixed input size instead of dynamic axes")
    parser.add_argument("--check", type=int, default=64, help="edge length used for the shape assertion")
    parser.add_argument("--simplify", action="store_true", help="run onnx-simplifier afterwards")
    parser.add_argument("--download", action="store_true", help="fetch the .pth first if missing")
    parser.add_argument("--models-dir")
    parser.add_argument("--config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.models_dir:
        cfg.models_dir = args.models_dir

    try:
        spec = cfg.resolve_model(args.model)
    except PixelBoostError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pth = cfg.model_path(spec)
    if not os.path.isfile(pth) and args.download:
        print(f"downloading {spec.name} ...")
        pth = ensure_model(cfg, spec)
    if not os.path.isfile(pth):
        print(f"error: {pth} not found. Run scripts/download_models.py first, or pass --download.", file=sys.stderr)
        return 2

    print(f"arch={spec.arch} scale={spec.scale} block={spec.num_block} feat={spec.num_feat} conv={spec.num_conv}")
    try:
        out = export_onnx(
            cfg,
            spec,
            output=args.output,
            opset=args.opset,
            fp16_weights=args.fp16,
            dynamic=not args.static,
            simplify=args.simplify,
            check=args.check,
        )
    except PixelBoostError as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1

    size = os.path.getsize(out)
    print(f"wrote {out} ({size / 1e6:.1f} MB)")
    print("verify: python -c \"from pixelboost.backends.onnx_backend import available_providers; "
          "print(available_providers())\"")
    print(f"run   : pixelboost upscale in.jpg -o out.png --model-path {out} --scale {spec.scale}")
    if not args.static:
        print("note  : dynamic axes exported -- the tiler may use any tile size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
