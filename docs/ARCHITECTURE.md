# Architecture

## Design goals

1. **One pipeline, many accelerators.** The same code must drive a pure-numpy
   resampler, an ONNX Runtime graph and a PyTorch module. Anything backend
   specific is confined to `backends/`; anything image specific is confined to
   `pipeline.py` and `ops.py`.
2. **CPU is a first-class target, not a fallback.** A CPU-only VPS is the most
   common deployment. That means: no mandatory Torch, no mandatory CUDA, a
   model-free path that always works, and real multi-core scaling.
3. **Bounded memory on inputs you do not control.** Users upload 12 MP phone
   photos. Every stage has an explicit working-set cap, and tiled inference
   streams its output instead of materialising the whole float buffer.
4. **Degrade, do not fail.** A missing checkpoint, a missing CUDA library or an
   OOM should reduce quality, not return a 500.

## Module map

```
pixelboost/
  cli.py                 argparse front-end: upscale | batch | serve | models | benchmark | capabilities
  engine.py              public API. Owns the backend cache and the per-image lifecycle
  pipeline.py            stage ordering, sizing, post-processing, alpha
  tiling.py              tile planning, feather blending, streaming accumulator
  ops.py                 pure-numpy operators (resize, guided filter, sharpening, colour)
  imageio.py             decode / encode / EXIF / ICC / atomic write
  config.py              layered config + the model registry
  models.py              download, checksum, ONNX export
  types.py               EnhanceOptions / EnhanceResult / size resolution
  errors.py              exception hierarchy

  backends/
    base.py              the 3-method Backend contract
    classical.py         model-free Lanczos + guided detail  (always available)
    onnx_backend.py      ONNX Runtime: CPU / CUDA / TensorRT / DirectML / CoreML
    torch_backend.py     RRDBNet + SRVGGNetCompact for .pth checkpoints
    registry.py          auto-selection, capability reporting

  server/
    app.py               FastAPI: sync + async endpoints, auth, upload limits
    worker.py            in-process job queue with a concurrency gate
```

The dependency direction is strictly downward: `cli` and `server` depend on
`engine`, `engine` depends on `pipeline` and `backends`, `pipeline` depends on
`ops` and `tiling`, and `ops` depends on nothing but numpy. No cycles, so any
layer can be tested in isolation.

## The backend contract

```python
class Backend:
    name: str                  # "classical" | "onnx" | "torch"
    native_scale: int          # the scale the model was trained for
    requires_tiling: bool      # does a full-size forward pass fit in memory?
    device: str                # "cpu" | "cuda" | "tensorrt" | ...

    def process(self, tile: np.ndarray, scale: float) -> np.ndarray: ...
    def warmup(self, tile: int = 64) -> None: ...
    def close(self) -> None: ...
    def info(self) -> dict: ...
```

`process` takes an `(h, w, 3)` float32 tile and returns `(h*scale, w*scale, 3)`.
That is the entire contract. Tiling, colour management, file I/O and job
scheduling all live above it, which is why the HTTP layer has no `if cuda`
branches anywhere.

`requires_tiling` is the one piece of self-knowledge a backend needs. The
classical backend runs whole images (tiling would only introduce seams for no
gain); neural backends always tile.

## Backend auto-selection

```
opts.backend == "auto"?
  ├─ model == "classical" ............................ classical
  ├─ onnxruntime importable AND  <stem>.onnx exists? .. onnx
  ├─ torch importable       AND  <stem>.pth exists? ... torch
  └─ otherwise ....................................... classical   (degraded, logged)
```

Ordered by *robustness*, not raw speed. ONNX first because it is a 15 MB
dependency that serves both CPU and GPU from one artifact. Torch second because
it is what published weights are, but 800 MB to install. Classical last as an
always-available floor.

An explicit `--backend cuda`-style request is honoured literally and raises on
failure. Silent fallback is right for a web service and wrong for a benchmark.

## Stage pipeline

```
decode
  │   Pillow → float32 RGB [0,1], alpha lifted out, original mode remembered
  ▼
[denoise]           guided filter, sigma ≈ noise level
  │   BEFORE upscaling: a net asked to upscale noise upscales the noise,
  │   and doing it at 1x is 16x cheaper than at 4x.
  ▼
[auto-levels]       luma percentile stretch
  │   BEFORE upscaling: gives the network gradient to work with on flat sources.
  ▼
[pre-downscale]     optional, tiled backends only
  │   If target/native < 0.5 and the source is large, shrink first so the
  │   4x forward pass produces the right size. Trades a little quality for a
  │   large memory win.
  ▼
upscale             tiled (neural) or whole-image (classical)
  │
  ▼
[exact-fit]         resize to the requested size if it differs from native*scale
  │
  ▼
[chroma denoise]    AFTER upscaling: this is where GAN colour speckle appears.
  │
  ▼
[detail]            guided-filter detail reinjection, radius in OUTPUT pixels
  ▼
[unsharp]           optional soft-threshold acutance pass
  ▼
[colour]            gamma / contrast / saturation
  ▼
clamp → encode
```

Radius and amount parameters are scaled by the realised factor, so `--detail-r
4` means four *output* pixels whether the job is 2x or 8x.

## Tiling and blending

Cutting into tiles is easy. Stitching without visible seams is not, because each
tile's convolution padding differs, so tile interiors disagree slightly in the
overlap.

**Geometry.** `axis_ranges` produces overlapping windows whose union covers
`[0, total)`, with the last window clamped to the far edge. Stride is
`tile - 2*overlap`, so consecutive tiles share `2*overlap` pixels.

**Blending.** Each tile contributes a separable cosine ramp. `overlap` pixels on
each interior side ramp from ~0 to ~1, sampled at pixel centres so the first
pixel never gets exactly zero weight (which would leave a hole when a neighbour
turns out not to cover it). The accumulator stores `Σ(w·x)` and `Σw`, and the
result is the ratio, so the blended output is exact where tiles agree and smooth
where they do not.

**Streaming.** The naive version allocates the full `(H·s, W·s, 3)` float32
output: 4x on a 6000x4000 photo is 73 GB. Instead, tiles are processed band by
band. After band *i*, rows below `y0[i+1]·s` can never be touched again, so they
are written out and the buffer is reused. Peak memory is one band, i.e. about
`tile·s` rows, plus a small carry of the `2·overlap·s` rows that band *i+1* still
needs. That is the difference between working and not working on a 8 GB card.

**Outer padding.** Before tiling, the image is reflect-padded by `tile_pad`
pixels and cropped back afterwards. Tile borders are then fed plausible context
instead of a hard image edge, which is visible on sky and skin.

**OOM recovery.** `run_resilient` catches allocation failures and retries with
the tile halved, down to 64 px. The `proc_factory` indirection exists because a
CUDA context is generally unusable after an OOM, so the closure must be rebuilt.

## The classical backend in detail

This is not a toy path. It is a compressed version of what production
resamplers do:

1. **Multi-step Lanczos-3** with `max_step_ratio = 2.0`. A single 8x pass samples
   a 6-tap kernel that is far too narrow and softens badly; chaining 2x passes
   lets each step use the full kernel. Measured ringing is the same as a
   single pass on a hard step, so nothing is lost by chaining.
2. **Guided-filter detail reinjection.** `base = guided_filter(x, x)`,
   `detail = x - base`, `out = base + k·detail`. The guided filter smooths flat
   regions while holding edges, so the extracted detail band has almost no edge
   bleed -- this sharpens without the halos a plain unsharp mask produces.
3. **Optional soft-threshold unsharp** for final acutance.

Implemented in numpy with:
- a cumsum-based box filter (`O(n)` regardless of radius),
- 3-pass box cascade for Gaussian (`w = √(4σ²+1)`),
- chunked tap accumulation for resampling, capped at 4 M floats per buffer.

The one honest weakness: Lanczos has negative lobes, so a synthetic one-pixel
step edge overshoots by ~11 %. A real photograph rarely contains such an edge,
and the pipeline clamps to `[0,1]` at the end. For hard-edged artwork, prefer a
neural backend.

## Concurrency model

**Library**: backends are cached under an `RLock` and shared. `InferenceSession.run`
is thread-safe by contract, and a PyTorch module under `inference_mode` with no
parameter mutation is safe to share.

**Server**: one `ThreadPoolExecutor` fronted by a semaphore sized to the worker
count. The semaphore is the important part -- ONNX Runtime will happily start
four concurrent sessions on one GPU and be three times slower than running them
serially. Default is one GPU worker and `cpu_count // 2` on CPU.

**Why threads and not processes**: wasm/fork semantics aside, `fork` after CUDA
initialisation is invalid, and each process would load its own copy of a 60 MB
graph. Threads also get the GIL released during ORT inference, which is what
makes the synchronous endpoint viable.

**Why no Redis/Celery**: a single box serving a single tenant does not need a
broker. `server/worker.py` has a deliberately small interface (`submit`, `get`,
`cancel`), so swapping in a real queue later is a contained change.

## Performance envelope (reference)

Indicative numbers for a 1024x1024 source at 4x, single request:

| Backend | Hardware | Tile | Time |
|---|---|---|---|
| classical | 4 vCPU | n/a | ~0.9 s |
| realesr-general-x4v3 (ONNX) | 4 vCPU | 256 | ~4 s |
| realesr-general-x4v3 (ONNX) | RTX 3060 | 512 | ~0.4 s |
| realesrgan-x4plus (ONNX) | 8 vCPU | 256 | ~35 s |
| realesrgan-x4plus (ONNX) | RTX 3060 | 512 | ~1.2 s |
| realesrgan-x4plus (TensorRT fp16) | RTX 3060 | 512 | ~0.5 s |

Treat these as order-of-magnitude only. Run `python scripts/benchmark.py` on the
actual host before sizing anything.

## Testing strategy

94 tests, no network, no model files, under two seconds.

- `test_ops.py` -- operator correctness: DC preservation, edge preservation
  ordering, bounded ringing, no compounding across chained steps.
- `test_tiling.py` -- geometry coverage, and the key invariant: with a
  position-independent backend, tiled output must equal whole-image output
  exactly. Any disagreement is a blending bug.
- `test_pipeline.py` -- sizing resolution, non-uniform targets, alpha handling,
  greyscale round-trip, max-pixel clamping.
- `test_config.py` -- config layering, model registry integrity, provider
  aliasing.

The tiled-equals-whole test is the one that catches real regressions; it caught
a carry-row off-by-2x during development.
