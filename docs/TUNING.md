# Tuning guide

Everything here is about one trade-off triangle: **quality**, **speed**, and
**memory**. You get to pick two.

Start from the shipped preset closest to your goal and adjust one thing at a
time. Changing three parameters at once is how people end up with an image that
is soft, crunchy and colour-shifted simultaneously and no idea which knob did it.

```bash
pixelboost upscale in.jpg -o out.png --config configs/fast.yaml
pixelboost upscale in.jpg -o out.png --config configs/quality.yaml
```

---

## Presets

| | fast | default | quality |
|---|---|---|---|
| model | general-x4v3 | x4plus | x4plus |
| scale | 2 | 4 | 4 |
| denoise | 0 | 0 | 0.035 |
| chroma_denoise | 0 | 0 | 1.2 |
| detail | 0.25 | 0.35 | 0.40 |
| auto_levels | 0 | 0 | 0.25 |
| tile | off | 512 | 512 |
| relative time | 1x | ~12x | ~20x |

---

## The parameters, in order of impact

### `scale` / `width` / `longest_side`

The single biggest lever. Cost scales with the *output* pixel count, so 2x costs
a quarter of 4x. If you are serving a web gallery, the browser is going to
downscale it anyway -- ask for the size you will actually display.

`--longest-side 2048` is usually what you want rather than `--scale 4`, because
it behaves the same for both a landscape and a portrait upload.

### `model`

16.7 M vs 1.2 M parameters. ~8-10x throughput difference, and the fast model is
genuinely good on clean sources -- it is mostly worse on fine texture and on
heavily compressed input.

### `tile`

Affects speed and memory, not quality (beyond `overlap` effects).

Sweep it on the real hardware:

```bash
python scripts/benchmark.py --size 1024 --scale 4 --tiles 128,256,384,512,768,1024,0
```

Read the table for the **knee**: throughput climbs with tile size, then flattens
while memory keeps rising. Pick the smallest tile within ~5 % of peak. That
leaves VRAM headroom for concurrent requests, which matters more than the last
few percent of single-request speed.

Rules of thumb: RTX 3060 12 GB → 512-768. 4 GB card → 256-384. T4 16 GB → 768.
A CPU box should not use a tile at all if the image fits in RAM.

`--tile 0` disables tiling. Use it to find the memory wall, not in production.

### `tile_overlap`

Raise it if you can see a faint grid in flat areas (sky, skin, studio
backdrops). x4plus has a large receptive field and behaves better at 32 or even
48. Cost is roughly linear in overlap.

### `tile_pad`

Reflect padding around the whole image before tiling. Affects only the outermost
tile ring, but that is exactly where sky and vignettes live. 16 is fine; 24 if
you see a soft border.

### `detail`

The guided-filter detail reinjection. This is the main quality knob on the
classical backend and a useful finisher on the neural ones.

| Value | Look |
|---|---|
| 0.0 | untouched model output |
| 0.15 | subtle; safest for portraits and skin |
| 0.25-0.35 | natural "sharper" reading; good default |
| 0.5-0.7 | punchy; good for architecture, product, screenshots |
| > 1.0 | watercolour: texture invented, skin waxy |

Because it is edge-preserving rather than a plain unsharp mask, it does not
produce white halos along high-contrast edges -- but it *will* amplify noise, so
pair high values with `denoise`.

`detail_radius` is in output pixels. Raise it to 6-8 to bring out mid-frequency
texture (foliage, fabric) instead of micro-contrast. `detail_eps` controls how
hard the filter locks onto edges: lower = sharper transitions preserved, higher
= more smoothing before extraction. Leave it alone unless you know why you are
changing it.

### `denoise`

Runs *before* upscaling, which is the point: a network asked to upscale noise
faithfully upscales the noise, and removing it at 1x is 16x cheaper.

| Source | Value |
|---|---|
| clean RAW / PNG export | 0.0 |
| good JPEG, q >= 90 | 0.02-0.03 |
| social media re-compressed | 0.04-0.06 |
| heavy JPEG artefacts, old scans | 0.06-0.10 |

Above ~0.12 you start destroying legitimate texture along with the noise, and no
amount of `detail` brings it back.

### `chroma_denoise`

Blurs only the chroma planes, after upscaling. GANs commonly hallucinate colour
speckle in smooth gradients and dark areas. The eye tolerates blurred chroma far
better than blurred luma, so this is nearly free perceptually.

1.0-2.0 removes most speckle on photographs. Set 0 for anime and line art, where
flat colour regions are intentional and any smoothing is visible as banding.

### `sharpen`

A conventional unsharp mask with a soft noise threshold. **Off by default on
purpose** -- stacking it on top of `detail` is the fastest route to a crunchy
image. Turn it on only when `detail` alone is not enough, and then keep it at
0.15-0.3.

### `auto_levels`

Luma percentile stretch before upscaling. Helps a lot on washed-out sources --
social media exports, screenshots of screenshots -- because a flat, low-contrast
input gives the network little gradient to work with.

0.25-0.5 for mildly flat sources, 1.0 for genuinely washed out. Leave at 0 when
the source is already well exposed; over-stretching clips highlights.

### `alpha_mode`

`lanczos` (default) scales the alpha plane with a plain high-quality resampler.
Do **not** change it to route alpha through the neural backend: a GAN will paint
opaque colour into the transparent halo around a cut-out subject, producing a
dark fringe. `nearest` is for pixel art with hard transparency.

---

## Recipes

**Product photos, white background**

```bash
pixelboost upscale product.jpg -o out.png \
  --model realesrgan-x4plus --scale 4 \
  --detail 0.5 --detail-radius 5 --sharpen 0.15 --chroma-denoise 0.8
```

**Portraits** -- keep skin soft, do not chase texture

```bash
pixelboost upscale portrait.jpg -o out.png \
  --model realesrgan-x4plus --scale 2 \
  --detail 0.15 --denoise 0.025 --saturation 0.98
```

**Anime / illustration**

```bash
pixelboost upscale art.png -o out.png \
  --model realesrgan-x4plus-anime --scale 4 \
  --detail 0.2 --chroma-denoise 0 --auto-levels 0
```

**Screenshots, UI, text** -- do not use a GAN

```bash
pixelboost upscale ui.png -o out.png \
  --backend classical --scale 2 --detail 0.6 --sharpen 0.2
```

**Old scanned photos**

```bash
pixelboost upscale scan.jpg -o out.png \
  --model realesr-general-wdn-x4v3 --scale 4 \
  --denoise 0.08 --auto-levels 0.6 --chroma-denoise 1.5 --detail 0.4
```

**Bulk processing overnight on CPU**

```bash
pixelboost batch ./originals -o ./enhanced --recursive --scale 2 \
  --model realesr-general-x4v3 --tile 0 --threads 4 --json > report.json
```

**Maximum throughput API** -- latency over quality

```yaml
model: realesr-general-x4v3
provider: cuda
fp16: true
tile: 256
defaults:
  scale: 2
  detail: 0.2
  chroma_denoise: 0
```

---

## Diagnosing output

| What you see | What it is | Fix |
|---|---|---|
| Soft / no improvement over bicubic | scale below the model's native, then post-downscaled; or tile too large for the model | use 4x and downscale in the browser, or raise `overlap` |
| White halos on high-contrast edges | `sharpen` too high | drop `sharpen` to 0, rely on `detail` |
| Crunchy, gritty texture | `detail` + `sharpen` stacked | `--detail 0.2 --sharpen 0` |
| Waxy skin, smeared foliage | `detail` too high | `--detail 0.15`; consider `--denoise 0.02` |
| Colour speckle in shadows and sky | GAN colour hallucination | `--chroma-denoise 1.5` |
| Faint grid in flat areas | overlap too small | `--tile-overlap 32` |
| Soft or dark border ring | `tile_pad` too small | `--tile-pad 24` |
| Noise amplified into texture | `denoise` too low | raise `--denoise`, do not lower `--detail` |
| Grey halo around a cut-out subject | alpha routed through the net | `--alpha-mode lanczos` |
| Text strokes thickened or melted | wrong model for the content | `--backend classical` |
| Washed out or clipped after processing | `auto_levels` too strong | lower to 0.25 or 0 |

## Measuring instead of guessing

Quality changes are easy to imagine and hard to see. Build a reference set of
5-10 representative images, run the candidate settings over all of them, and
compare at 100 % side by side. If you want a number, PSNR and SSIM against the
original are only meaningful for *downscaled* comparisons; for perceptual quality
on 4x output, a human eyeball on a matched crop beats any metric.

For throughput, always measure on the target host:

```bash
python scripts/benchmark.py --size 1024 --scale 4 --tiles 256,512,768 --json bench.json
```
