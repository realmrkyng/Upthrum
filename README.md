# PixelBoost

**CPU/GPU accelerated image quality enhancement and super-resolution — self-hostable in one command.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-176%20passing-brightgreen.svg)](tests)

PixelBoost upscales and enhances images 2x-8x with a pipeline that runs on a
plain CPU VPS **or** a CUDA GPU, from the same code and the same model file.

It is built for people who have to *operate* this, not just demo it: bounded
memory on untrusted uploads, no mandatory 800 MB PyTorch dependency, a model-free
path that always works, and a deployment guide that covers the CUDA/cuDNN version
mismatch that eats everyone's first afternoon.

```bash
pip install "pixelboost[onnx-cpu,server,yaml]"
pixelboost models download --model realesrgan-x4plus
pixelboost upscale photo.jpg -o photo_4x.png --scale 4
```

```
photo.jpg -> photo_4x.png  [1024, 768] -> [4096, 3072]  [onnx/cuda] 1180 ms
```

---

## Contents

- [Why another upscaler](#why-another-upscaler)
- [Features](#features)
- [Install](#install)
- [Quick start](#quick-start)
- [CPU and GPU](#cpu-and-gpu)
- [HTTP API server](#http-api-server)
- [Python API](#python-api)
- [How it works](#how-it-works)
- [UPTHRUM](#upthrum)
- [Models](#models)
- [Performance](#performance)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Why another upscaler

The existing options are either research code that assumes a 3090 and a
`conda` environment, or a wrapper that locks you into one backend. The gap is
everything between "it runs on my machine" and "it runs on a $6/month VPS and
does not fall over".

PixelBoost makes four specific bets:

**CPU is a first-class target.** The ONNX Runtime CPU wheel is ~15 MB. PyTorch is
~800 MB. Most deployments do not need PyTorch at all, so it is optional — and ORT
releases the GIL during inference, which means a thread pool gives real
multi-core scaling without processes or a task broker.

**Memory must be bounded, always.** A 4x upscale of a 6000x4000 photo needs a
73 GB float buffer. PixelBoost streams tiled output band by band, keeping peak
memory at roughly one tile, and halves the tile automatically when the GPU runs
out. Users upload 12 MP phone photos; the service must not care.

**Degrade, do not fail.** A missing checkpoint or a mismatched CUDA library
should reduce quality, not return a 500. There is a model-free backend that is
always available and is often *better* than a GAN on text and screenshots, since
GANs invent texture that is not there.

**The ops matter.** Sharpening, denoising and chroma cleanup are half the
perceived quality of a super-resolution result, and they are usually bolted on
afterwards with wrong radii. Here they are inside the pipeline with radii
expressed in output pixels, so `--detail-radius 4` means the same thing at 2x and
at 8x.

---

## Features

| | |
|---|---|
| **Backends** | classical (numpy, no model), ONNX Runtime, PyTorch, UPTHRUM (phase reconstruction) |
| **Execution providers** | CPU, CUDA, TensorRT, DirectML, CoreML, ROCm, OpenVINO |
| **Tiled inference** | streaming accumulator, cosine feather blending, automatic tile shrink on OOM |
| **Quality pipeline** | edge-preserving denoise → auto-levels → upscale → chroma cleanup → guided-filter detail → optional unsharp → colour |
| **Sizing** | `--scale`, `--width`, `--height`, `--longest-side`, non-uniform |
| **Alpha** | handled as a separate plane so cut-outs do not get a black fringe |
| **Colour** | ICC profile and EXIF passthrough, greyscale in → greyscale out |
| **Interfaces** | CLI (single + batch), FastAPI server (sync + async jobs), Python API |
| **Ops** | atomic writes, structured logging, `/healthz`, capability reporting, API keys, upload limits |
| **Deployment** | bare metal, systemd unit, nginx config, Docker CPU + CUDA images, compose |
| **Tests** | 176 tests, no network, no model files, runs in a few seconds |

---

## Install

Requires Python 3.9+.

```bash
git clone https://github.com/your-org/pixelboost.git
cd pixelboost
python -m venv .venv && source .venv/bin/activate

# CPU only  (~40 MB of dependencies)
pip install -e ".[onnx-cpu,server,yaml]"

# GPU       (see docs/DEPLOY.md §1.3 before you pick this)
pip install -e ".[onnx-gpu,server,yaml]"
```

Nothing to install at all? The classical backend works with just numpy and
Pillow:

```bash
pip install numpy pillow
pixelboost upscale in.png -o out.png --backend classical --scale 2
```

Verify the install:

```bash
pixelboost capabilities
```

---

## Quick start

### 1. Get a model

```bash
pixelboost models download --model realesrgan-x4plus
pixelboost models                      # what is on disk
```

Or skip this entirely and use the model-free backend.

### 2. Upscale

```bash
# 4x, best-quality general model
pixelboost upscale photo.jpg -o photo_4x.png --scale 4

# anime / illustration
pixelboost upscale art.png -o art_4x.png --model realesrgan-x4plus-anime

# screenshots and text -- no model, and better than a GAN for this
pixelboost upscale ui.png -o ui_2x.png --backend classical --scale 2

# fit a maximum edge length instead of a factor
pixelboost upscale photo.jpg -o photo_2048.png --longest-side 2048

# a whole tree
pixelboost batch ./originals -o ./enhanced --recursive --scale 2 --json > report.json
```

### 3. Tune

```bash
pixelboost upscale portrait.jpg -o out.png --scale 2 --detail 0.15 --denoise 0.025
pixelboost upscale product.jpg  -o out.png --scale 4 --detail 0.5 --sharpen 0.15
```

See [docs/TUNING.md](docs/TUNING.md) for recipes per image type.

---

## CPU and GPU

The **same model file and the same pipeline code** drive both. Only the
execution provider changes.

### CPU

```bash
pixelboost upscale in.jpg -o out.png --provider cpu --threads 4 \
  --model realesr-general-x4v3 --tile 384
```

- Use the compact model (`realesr-general-x4v3`, 1.2 M params vs 16.7 M).
- **Set `--threads`.** Leaving it at 0 makes ONNX Runtime spawn one thread per
  core, which fights your worker pool and makes throughput *worse* under load.
- On a large box, `--tile 0` (whole-image) is fastest if the image fits in RAM.

### GPU

```bash
pixelboost upscale in.jpg -o out.png --provider cuda --fp16 --tile 512
pixelboost upscale in.jpg -o out.png --provider tensorrt --fp16 --tile 768
```

```python
from pixelboost.backends.onnx_backend import available_providers
print(available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

If that prints only `CPUExecutionProvider` on a GPU box, jump to
[docs/DEPLOY.md §1.3](docs/DEPLOY.md#13-gpu-getting-cudaexecutionprovider-to-appear).
Nine times out of ten it is a CUDA/cuDNN major-version mismatch, or both
`onnxruntime` and `onnxruntime-gpu` installed in the same environment.

**Measure before you tune.** Tile size is the single most important GPU knob and
the optimum is hardware-specific:

```bash
python scripts/benchmark.py --size 1024 --scale 4 --tiles 128,256,384,512,768,1024,0
```

```
  tile  tiles   median ms  src MP/s  dst MP/s
----------------------------------------------
   128     64        812.4     1.262     20.19
   256     16        402.1     2.550     40.79
   512      4        318.7     3.218     51.49
   768      1        305.2     3.360     53.76
  1024      1        301.9     3.396     54.34
     0      1        301.5     3.400     54.42

best: tile=0 at 54.42 output MP/s
guidance: pick the smallest tile within ~5 % of the best throughput --
that leaves VRAM headroom for concurrent requests.
```

Here 512 is within 6 % of the ceiling at a quarter of the memory. **That is the
number you deploy**, not 1024.

---

## HTTP API server

```bash
pixelboost serve --host 127.0.0.1 --port 8000 --workers 2
```

Interactive docs at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/v1/enhance \
  -F "file=@photo.jpg" \
  -F "scale=4" \
  -o out.png -D -
```

```
HTTP/1.1 200 OK
X-PixelBoost-Backend: onnx
X-PixelBoost-Provider: cuda
X-PixelBoost-Ms: 1180.4
```

Long jobs go through a queue so they do not sit on a proxy read timeout:

```bash
JOB=$(curl -s -X POST localhost:8000/v1/jobs -F "file=@big.jpg" -F "scale=4" | jq -r .id)
curl -s localhost:8000/v1/jobs/$JOB | jq '.status, .progress'
curl -s localhost:8000/v1/jobs/$JOB/result -o big_4x.png
```

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | liveness |
| `GET /v1/capabilities` | backends, providers, models actually present |
| `GET /v1/stats` | queue depth, warm backends |
| `POST /v1/enhance` | synchronous, image in the response body |
| `POST /v1/jobs` | enqueue, returns an id |
| `GET /v1/jobs/{id}` | status + tile progress |
| `GET /v1/jobs/{id}/result` | the finished image |
| `DELETE /v1/jobs/{id}` | cancel while queued |

Full reference: [docs/API.md](docs/API.md).

Deployment, systemd, nginx and Docker: [docs/DEPLOY.md](docs/DEPLOY.md).

---

## Python API

```python
from pixelboost import Engine, EnhanceOptions

with Engine() as engine:
    opts = EnhanceOptions(
        scale=4,
        model="realesrgan-x4plus",
        provider="cuda",
        detail=0.4,
        chroma_denoise=1.2,
        tile=512,
    )
    result, path = engine.enhance_file("photo.jpg", "photo_4x.png", opts)
    print(result.summary())
    # {'backend': 'onnx', 'model': 'realesrgan-x4plus', 'provider': 'cuda',
    #  'src_size': [1024, 768], 'dst_size': [4096, 3072], 'scale': 4.0,
    #  'tiles': 48, 'elapsed_ms': 1180.4, 'notes': []}
```

Keep one `Engine` alive for the process. It caches warm backends; constructing a
new one per request reloads the graph every time.

---

## How it works

```
                       ┌──────────────────────────────────────────┐
  input file ─────────▶│ imageio.load                             │
                       │  decode · split alpha · keep ICC/EXIF    │
                       └────────────────────┬─────────────────────┘
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │ pre-processing                           │
                       │  denoise → auto-levels → [pre-downscale] │
                       └────────────────────┬─────────────────────┘
                                            ▼
              ┌─────────────────────────────────────────────────────────┐
              │ backends  (one 3-method contract)                       │
              │  classical  numpy Lanczos + guided filter   always      │
              │  upthrum    phase transport + topology      always      │
              │  onnx       ORT: CPU/CUDA/TRT/DML/CoreML    portable    │
              │  torch      RRDBNet / SRVGGNetCompact       .pth files  │
              └────────────────────┬────────────────────────────────────┘
                                   ▼
              ┌─────────────────────────────────────────────────────────┐
              │ tiling  plan → infer → cosine feather → stream out      │
              │  peak memory ≈ one tile, not the whole output           │
              │  OOM → halve the tile and retry                         │
              └────────────────────┬────────────────────────────────────┘
                                   ▼
                       ┌──────────────────────────────────────────┐
                       │ post-processing                          │
                       │  chroma denoise → detail → unsharp →     │
                       │  colour → clamp                          │
                       └────────────────────┬─────────────────────┘
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │ imageio.save   atomic temp + os.replace  │
                       └──────────────────────────────────────────┘
```

Four ideas carry most of the weight.

**Stage order is a quality decision, not a style choice.** Denoise goes *before*
upscaling because a network asked to upscale noise upscales the noise, and doing
it at 1x is far cheaper. Chroma denoise goes *after*, because that is where GAN
colour speckle appears. Detail and sharpening go last, on the final-resolution
buffer, so their radii can be expressed in output pixels.

**The backend contract is three methods.** `process(tile, scale) -> tile`,
`warmup()`, `close()`. Tiling, colour, I/O and scheduling all live above it,
which is why there is not a single `if cuda` branch in the server code.

**Blending must be exact where tiles agree.** Each tile contributes a separable
cosine ramp over the overlap; the accumulator stores `Σ(w·x)` and `Σw` and the
output is their ratio. The test suite asserts that with a position-independent
backend, tiled output equals whole-image output *exactly* — that invariant caught
a carry-row bug during development.

**Detail is extracted, not amplified.** The classical path uses a guided filter
rather than an unsharp mask: `base = guided_filter(x, x)`, `detail = x - base`,
`out = base + k·detail`. Because a guided filter holds edges while smoothing flat
regions, the extracted detail band has almost no edge bleed, so it sharpens
without white halos. `ops.py` implements it (plus a Lanczos resampler, a box
filter and a 3-pass Gaussian) in pure numpy with `O(n)` cumsum kernels and a
4 M-float working-set cap.

Full write-up: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## UPTHRUM

`upthrum` is not another set of weights. It is a different variable.

Every upscaler in this repository — and, as far as I can tell, every published
one — estimates **intensity**: given samples on a coarse lattice, predict values
on a finer one. The predictor differs (fixed kernel, regression network, diffusion
model) but the target is the same number per output pixel, and so is the failure
mode. The map from coarse intensity samples to fine intensity values is not
injective, so the fine structure that produced those samples is genuinely
underdetermined. An intensity-domain method must therefore either blur (pick the
smooth preimage) or hallucinate (pick a plausible one). There is no third option.

UPTHRUM reconstructs **phase** instead.

```
  log-Gabor band k ──▶ monogenic triple (B, Rx, Ry)
                       │
                       ├─ amplitude  A = hypot(B, Rdir)
                       ├─ phase      φ = atan2(Rdir, B)        on S¹
                       ├─ orientation θ from the doubled-angle mean across bands
                       └─ phase gradient ∇φ  differentiated on the unit phasor,
                                             so it never needs unwrapping
                                  │
                                  ▼
  output lattice ──▶ phase transport at every output pixel q
                     φ_out(q) = arg Σ_t  W_t · A_t^γ · exp( i (φ_t + g·∇φ_t·Δ_t) )
                     Δ_t = tap − q      g = phase_gain
                                  │
                                  ├─ κ = |Σ W exp(iφ)| / Σ W   coherence gate ∈ [0,1]
                                  └─ amplitude from a structure-aligned kernel
                                  │
                                  ▼
  topology ──▶ 0-D persistent homology of the sublevel sets
               cancel critical points below τ = 0.18 · (p99 − p1)
               restore source peaks the transport erased (bounded by the source)
                                  │
                                  ▼
  intensity ──▶ cos(φ_out) · A_out · κ^0.5  ⊕  untouched low-pass  →  RGB
```

Because translation along a wavefront *is* a phase shift, the sub-pixel position
of structure is determined by the phase field rather than guessed at. And because
phases are averaged as unit vectors rather than intensities as scalars, the
amplitude attenuation that constitutes interpolation blur never arises in the
first place. The topology stage is what stops the reconstruction from promoting
noise into apparent texture: the synthesised detail is constrained to carry no
significant critical point the source did not already have.

Five properties, each of which is a test in `tests/test_upthrum.py`:

| Property | Meaning | Measured |
|---|---|---|
| Identity at scale 1 | transport collapses to the coincident sample | max error `1.8e-07` (float32 floor) |
| No Nyquist attenuation | a band-centre sinusoid transports at unit gain | `1.08e-05`, vs 2600× worse if the phase term is dropped |
| Intrinsic anisotropy | a step edge is reconstructed as a step | edge width 2 px vs Lanczos 8 px, no overshoot |
| Topological invariance | insignificant critical points cannot be promoted | 79 → 79 features clean, 2575 → 79 under σ=0.03 noise |
| DC preservation | the low-pass path is untouched | `< 1e-6` on the contractive path |

Use it when the source has structure a GAN would invent over: text, UI, line art,
technical drawings, scanned documents. On those, `realesrgan-x4plus-anime` is
usually *worse* than either model-free backend, because it draws texture that
isn't in the file.

```bash
# 4x, defaults
pixelboost upscale ui.png -o ui_4x.png --backend upthrum --scale 4

# any scale factor, including fractional -- 2.5x, exact width, longest side
pixelboost upscale ui.png -o ui_2_5x.png --backend upthrum --scale 2.5
pixelboost upscale ui.png -o ui_2400.png --backend upthrum --width 2400

# three bands is the default; more helps mixed fine-texture/flat content
pixelboost upscale ui.png -o out.png --backend upthrum --upthrum-bands 5

# GPU: the FFT-bound analysis, if a CUDA build of torch is present
pixelboost upscale ui.png -o out.png --backend upthrum --upthrum-device cuda
```

```python
from pixelboost import enhance

result, path = enhance("ui.png", "ui_4x.png", backend="upthrum", scale=4)
print(result.summary())
# {'backend': 'upthrum', 'model': None, 'provider': 'cpu', ...}
```

**Cost.** Roughly 4-5x the `classical` backend at the same scale, and close to
linear in `--upthrum-bands`. It is never selected by `--backend auto`: it is
several times slower and the two model-free backends have genuinely different
characters, so the choice is left to you. See
[Performance](#performance).

**Not tiled.** Band analysis is non-local — the log-Gabor filters are defined on
the whole 2-D spectrum, so cutting the input into tiles would put a seam through
every band. UPTHRUM therefore runs whole-image analysis and streams the *output*
in row blocks. Memory scales with input area, not with `--tile`, and
`--max-pixels` is the guard that matters. Tune it before you point this at a
50-megapixel scan.

**Zero learned parameters, zero downloads.** `pixelboost capabilities` reports
`"learned_parameters": 0` for this backend, and it is not a figure of speech: the
whole method is the arithmetic in `upthrum/transport.py` plus the constraint in
`upthrum/topology.py`. There is no `models download` step, no ONNX export, and
nothing that can drift between two installs of the same version.

**Parameters.** `--upthrum-bands`, `--upthrum-top-frequency`,
`--upthrum-phase-gain`, `--upthrum-persistence`, `--upthrum-coherence-power`,
`--upthrum-anisotropy`, `--upthrum-detail`, `--upthrum-device`,
`--no-upthrum-topology`, `--no-upthrum-chroma`. Each one maps to one term in the
equations above; [docs/TUNING.md](docs/TUNING.md) says which term, and what
moving it does. All of them also work as a `upthrum:` block in a config file or
as `EnhanceOptions(extra={"upthrum": {...}})`.

Full derivation, the two bugs the invariants caught, and the calibration tables:
[docs/UPTHRUM.md](docs/UPTHRUM.md).

---

## Models

Five Real-ESRGAN entries, plus two model-free backends that need no download:

| Name | Scale | Params | Best for |
|---|---|---|---|
| `realesrgan-x4plus` | 4 | 16.7 M | photography — the default |
| `realesrgan-x4plus-anime` | 4 | 6.0 M | anime, illustration, line art |
| `realesr-general-x4v3` | 4 | 1.2 M | speed, CPU tiers (~8x faster) |
| `realesr-general-wdn-x4v3` | 4 | 1.2 M | JPEG-heavy or noisy sources |
| `realesr-animevideov3` | 4 | 2.4 M | video frames, lowest latency |
| `classical` | any | 0 | text, UI, screenshots — no download |
| `upthrum` | any | 0 | text, UI, line art — no download, no weights |

```bash
pixelboost models
pixelboost models download --all
pixelboost models export --model realesrgan-x4plus --fp16   # pth -> onnx
```

PixelBoost ships **no weights**. They are fetched from the Real-ESRGAN releases
(BSD-3-Clause) into `~/.cache/pixelboost/models`. Custom `.onnx` files are
supported through `--model-path` and need no registration.

Details and the model-selection decision tree: [docs/MODELS.md](docs/MODELS.md).

---

## Performance

Single 1024x1024 source at 4x, order-of-magnitude only — always benchmark the
actual host.

| Backend | Hardware | Tile | Time |
|---|---|---|---|
| classical | 4 vCPU | off | ~0.9 s |
| general-x4v3 | 4 vCPU | 256 | ~4 s |
| general-x4v3 | RTX 3060 | 512 | ~0.4 s |
| x4plus | 8 vCPU | 256 | ~35 s |
| x4plus | RTX 3060 | 512 | ~1.2 s |
| x4plus TensorRT fp16 | RTX 3060 | 512 | ~0.5 s |

The gap between CPU and GPU is 30-100x for x4plus. That is not a tuning problem —
if you are serving real traffic with a large model, you want a GPU, or you want
the compact model, or you want scale 2.

Sizing guidance and the memory table: [docs/DEPLOY.md §0](docs/DEPLOY.md#0-sizing-the-host).

### UPTHRUM cost

Measured on the development laptop (12 logical cores, numpy FFTs, no BLAS
threading), as a *ratio* to `classical` on the same host — the absolute numbers
below will not match your machine, but the ratios have held on every host tried.

| Case | vs classical 4x | Notes |
|---|---|---|
| `upthrum`, 2x | ~1.2x | the cheapest useful setting |
| `upthrum`, 4x | ~4.6x | the default |
| `upthrum`, 4x, 1 band | ~3.3x | cost is close to linear in `--upthrum-bands` |
| `upthrum`, 4x, 5 bands | ~5.8x | |
| `upthrum`, 2x, topology off | ~0.6x | see below |

Two things dominate, and both are bounded by knobs rather than by image size:

* **The topology stage is over half the runtime at scale 2** (5.96 s with it,
  2.88 s without, same image and scale). The merge tree is a union-find whose
  inner loop is inherently sequential and runs in Python. It is capped by
  `topology_max_pixels` (default 65536): larger inputs are block-max-pooled to
  that budget, which preserves maxima exactly and only blinds the analysis to
  sub-block detail — the scale the threshold is meant to ignore anyway. Turn the
  stage off and you lose the one property that makes UPTHRUM more than a very
  good interpolation, so treat `--no-upthrum-topology` as an ablation switch,
  not a performance one.
* **The transport is O(output pixels)** with a fixed gather per pixel, which is
  why 4x costs roughly 4x of 2x rather than 2x.

A CUDA build of torch moves the FFT-bound analysis to the GPU
(`--upthrum-device cuda`); the transport and topology stay on the CPU, so the
speedup is real but nothing like the 30-100x of a compiled network.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | module map, backend contract, stage pipeline, tiling maths, concurrency model |
| [docs/DEPLOY.md](docs/DEPLOY.md) | install, CUDA/cuDNN matching, systemd, nginx, Docker, Windows, production checklist, troubleshooting |
| [docs/TUNING.md](docs/TUNING.md) | every parameter, per-genre recipes, how to diagnose bad output |
| [docs/MODELS.md](docs/MODELS.md) | registry, selection guide, ONNX export, adding your own |
| [docs/API.md](docs/API.md) | HTTP reference, error codes, client examples |

---

## Development

```bash
make install-dev
make test          # 94 tests, under two seconds, no network
make lint
make benchmark
```

The suite deliberately needs no model files and no network, so it runs in CI on
every platform. The most valuable test is
`test_tiling.py::test_tiled_matches_whole_image` — it asserts an exact equality
that any blending regression breaks immediately.

---

## Contributing

Issues and PRs welcome. Useful things to know:

- Run `make test` and `make lint` before opening a PR.
- New operators in `ops.py` should be pure numpy and come with a test that
  states the property being relied on (DC preservation, edge preservation,
  bounds) rather than a golden-value snapshot.
- New backends implement the three-method contract in `backends/base.py` and
  nothing else.
- If you fix a deployment problem, the fix belongs in `docs/DEPLOY.md` too.

---

## License

MIT — see [LICENSE](LICENSE).

Bundled code contains no model weights. The referenced Real-ESRGAN models are
BSD-3-Clause; review their licence before commercial redistribution of the
weights or of generated output.
