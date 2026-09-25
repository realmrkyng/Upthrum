# Models

## Registry

Defined in `src/pixelboost/config.py`. Adding an entry is enough -- the download
script, CLI, HTTP capability endpoint and backend factory all read from it.

| Name | Scale | Params | Arch | Best for |
|---|---|---|---|---|
| `realesrgan-x4plus` | 4 | 16.7 M | RRDB x23 | photography, general |
| `realesrgan-x4plus-anime` | 4 | 6.0 M | RRDB x6 | anime, illustration, line art |
| `realesr-general-x4v3` | 4 | 1.2 M | SRVGG x32 | speed, CPU tiers |
| `realesr-general-wdn-x4v3` | 4 | 1.2 M | SRVGG x32 | JPEG-compressed / noisy sources |
| `realesr-animevideov3` | 4 | 2.4 M | SRVGG x16 | video frames, lowest latency |
| `classical` | any | 0 | none | text, screenshots, always-available fallback |
| `upthrum` | any | 0 | none | text, UI, line art — phase reconstruction, no download |

All Real-ESRGAN weights are BSD-3-Clause. PixelBoost ships none of them; they are
downloaded on demand into `~/.cache/pixelboost/models`.

## Choosing

```
Is the source text, a screenshot, UI or flat line art?
  └─ yes -> upthrum if you can afford 4-6x the classical runtime
         -> classical otherwise
         (a GAN will invent texture, thicken strokes and melt 12pt glyphs;
          both model-free backends will not, and upthrum keeps edges hard
          rather than merely smooth)

Is the source anime / illustration?
  └─ yes -> realesrgan-x4plus-anime

Is it a photograph with visible JPEG blocking or sensor noise?
  └─ yes -> realesr-general-wdn-x4v3
         (upthrum also tolerates noise, but by refusing to amplify it,
          not by removing it — pair it with --denoise if the source is dirty)

Do you need throughput on CPU, or sub-second latency?
  └─ yes -> realesr-general-x4v3

Anything else (the default)
  └─ realesrgan-x4plus
```

`classical` vs `upthrum` on the same text source: classical is a very good
resampler, so edges land where the kernel puts them and are softened by the
kernel's width. UPTHRUM transports the phase, so a step edge stays a step and
the sub-pixel position is determined rather than kernel-interpolated — measured
edge width 2 px against Lanczos' 8 px, with no overshoot. It costs 4–6x the
runtime, and it is never chosen by `auto`. See
[README.md § UPTHRUM](../README.md#upthrum).

## Downloading

```bash
pixelboost models download --model realesrgan-x4plus
pixelboost models download --all --force
python scripts/download_models.py --list
```

Downloads stream to a temp file and are moved into place atomically. A truncated
checkpoint is worse than a missing one -- it fails at inference time with an
opaque tensor-shape error -- so an interrupted download never leaves a file that
looks present.

Models land in `PIXELBOOST_HOME/models`, default `~/.cache/pixelboost/models`.
`PIXELBOOST_HOME` is respected by both the CLI and the Docker images (which set
it to `/models` so it can be a volume).

## ONNX

PyTorch is not needed to *serve* an ONNX model, only to *produce* one.

```bash
pip install "pixelboost[export,torch-cpu]"
python scripts/export_onnx.py --model realesrgan-x4plus --download
# or
pixelboost models export --model realesrgan-x4plus --fp16
```

The exporter:

- builds the architecture from the registry entry's `arch` field,
- loads the checkpoint, failing loudly on a tensor-count mismatch,
- exports with **dynamic H/W axes** unless `--static` is passed,
- runs a shape assertion through ONNX Runtime if it is installed.

### Dynamic vs static axes

A static export pins the graph to one input size, so the tiler must pad every
tile up to it. For a 512px tile at 4x that is fine; for a 300px tile you waste
65 % of every inference. Dynamic axes are the default for that reason. The one
case for `--static` is a TensorRT build where a fixed shape lets it optimise more
aggressively -- benchmark both before committing.

### fp16

```bash
python scripts/export_onnx.py --model realesrgan-x4plus --fp16
```

Halves the file and roughly 1.7x the throughput on a modern GPU. Quality loss is
usually imperceptible for 4x super-resolution, but it is not zero: if your users
are editing the output at 100 %, compare first.

### Where the file goes

The auto-selection logic looks for `<models_dir>/<stem>.onnx`, so
`RealESRGAN_x4plus.pth` is found by `RealESRGAN_x4plus.onnx`. Point anywhere with
`--model-path`.

## Architecture notes

**RRDBNet** (`arch: rrdb`) is the Real-ESRGAN generator: `conv_first`, 23
residual-in-residual dense blocks, then `conv_body`, two nearest-neighbour
upsample + conv stages, and a final head. Residual scaling `* 0.2` at both the
dense-block and RRDB level.

`src/pixelboost/backends/torch_backend.py` carries the definition inline rather
than depending on BasicSR. The nesting and attribute names must match the
checkpoint exactly -- that is why `conv_body` sits in `RRDBNet` and not in
`RRDB`, and why `ResidualDenseBlock` concatenates rather than adds. If you
change any of it, `load_state_dict` will start reporting missing tensors.

**SRVGGNetCompact** (`arch: srvgg`) is the fast one: a plain VGG-style stack with
PReLU, a `PixelShuffle` upsampler, and a nearest-neighbour bicubic-free residual
skip. 1.2 M parameters against 16.7 M.

## Adding your own model

**ONNX is the easy path.** Drop the file in the models directory and either name
it after a registry entry or point at it directly:

```bash
pixelboost upscale in.jpg -o out.png --model-path /path/to/mine.onnx \
  --scale 4 --tile 256
```

The backend reads the input name, dtype (`float32` / `float16`) and whether the
spatial dimensions are dynamic straight from the graph. Output range is
auto-detected (`[0,1]`, `[-1,1]` or `[0,255]`); override with
`normalize_output` if you have an unusual model.

Two things to check when a custom model looks wrong:

1. **Scale mismatch.** The backend asserts `output == input x native_scale`. A
   model that does 2x when you told it 4 will raise rather than silently produce
   a half-size image. Set `--scale 2` and `native_scale` accordingly.
2. **Fixed input size.** If the graph has a hardcoded 256x256 input, tiles must
   be ≤ 256 and will be zero-padded up. Re-export with dynamic axes if you can.

**PyTorch checkpoints** require an architecture implementation. Add the class to
`torch_backend.py` and register an `arch` tag in `build_model`. Anything not
covered by `rrdb` or `srvgg` should be converted to ONNX instead.

## Licensing

Real-ESRGAN is BSD-3-Clause, as is BasicSR. That permits commercial use. Verify
before you ship:

- the exact licence of any model **you** add,
- whether your jurisdiction treats AI-upscaled output as a derivative work,
- that you are not redistributing weights whose training data you cannot
  account for.

PixelBoost itself is MIT and contains no weights.
