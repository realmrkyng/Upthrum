# Deployment guide

Three supported ways to run PixelBoost in production, in order of how often they
are the right answer:

| Route | Use when | Jump to |
|---|---|---|
| Bare metal + systemd | you have a normal Linux VPS or GPU box | [§2](#2-bare-metal-linux) |
| Docker | you want reproducibility or already run containers | [§3](#3-docker) |
| Library only | you are embedding it in your own service | [§5](#5-using-it-as-a-library) |

---

## 0. Sizing the host

Rule of thumb for **realesrgan-x4plus at 4x**:

| Source | Peak RAM (CPU path) | Peak VRAM (GPU path) |
|---|---|---|
| 1 MP | ~600 MB | ~1.2 GB |
| 4 MP | ~2.0 GB | ~2.5 GB |
| 12 MP | ~5.5 GB | ~4.0 GB at tile 512 |

Add ~700 MB baseline for the Python process and the loaded graph.

**Minimum viable CPU box:** 2 vCPU / 2 GB RAM, `realesr-general-x4v3`, scale 2.
Serves a few thousand images a day.

**Minimum viable GPU box:** any 4 GB card (GTX 1650, T4, RTX 3050),
`realesrgan-x4plus`, tile 384. Roughly 50-100x the CPU throughput.

The three things that actually decide your bill:
- **Model choice.** x4plus is 16.7 M params; x4v3 is 1.2 M and ~8x faster. Start
  with x4v3 and move up only if users complain.
- **Scale factor.** 2x costs a quarter of 4x. Most web use cases do not need 4x.
- **Pre-downscale.** For large sources it cuts the forward pass cost roughly
  quadratically. See [TUNING.md](TUNING.md).

---

## 1. Install the Python package

### 1.1 Linux / macOS

```bash
git clone https://github.com/your-org/pixelboost.git
cd pixelboost

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU only
pip install -e ".[onnx-cpu,server,yaml]"

# or GPU (read §1.3 first -- the wheel pins a CUDA version)
pip install -e ".[onnx-gpu,server,yaml]"
```

Verify:

```bash
pixelboost --version
pixelboost capabilities
```

`capabilities` prints which backends and providers were detected. On a CPU box
you want to see `"providers": ["CPUExecutionProvider"]`. On a GPU box with a
working install you want `CUDAExecutionProvider` (and ideally
`TensorrtExecutionProvider`) in that list.

### 1.2 Download the models

Nothing works until at least one checkpoint is on disk.

```bash
pixelboost models download --model realesrgan-x4plus
pixelboost models download --model realesr-general-x4v3     # the fast one
pixelboost models                    # list what you have
pixelboost models usage              # disk consumption
```

Default location is `~/.cache/pixelboost/models`. Move it with
`PIXELBOOST_HOME=/srv/pixelboost/data` or `models_dir:` in the config file.

> **Do this before your first request.** Loading a 64 MB checkpoint takes ~1.5 s
> the first time and a TensorRT engine build takes 30-90 s. `warmup: true` in the
> config front-loads that at start-up instead of during a user request.

### 1.3 GPU: getting `CUDAExecutionProvider` to appear

This is where most deployments stall. `pip install onnxruntime-gpu` is not
sufficient -- the wheel links against a specific CUDA major and cuDNN major
version, and if they do not match, ONNX Runtime silently reports only
`CPUExecutionProvider`.

| onnxruntime-gpu | CUDA | cuDNN |
|---|---|---|
| 1.16.x | 11.8 | 8 |
| 1.17.x | 11.8 | 8 |
| 1.18.x | 12.x | 9 |
| 1.19.x - 1.20.x | 12.x | 9 |

**The path that works reliably**: install the CUDA and cuDNN runtime wheels from
PyPI instead of chasing system packages. They are self-contained and land inside
your venv, so nothing else on the host is touched.

```bash
pip install onnxruntime-gpu
# cuDNN 9 for ORT 1.18+; use nvidia-cudnn-cu11==8.9.2.26 for ORT 1.17
pip install nvidia-cudnn-cu12
pip install nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
```

Then check:

```bash
python - <<'EOF'
import onnxruntime as ort
print("ORT", ort.__version__)
print("providers:", ort.get_available_providers())
EOF
```

If `CUDAExecutionProvider` is missing, work through this list in order:

1. **Driver too old.** `nvidia-smi` shows the max CUDA the driver supports in
   the top-right corner. CUDA 12.x needs driver >= 525.
2. **CUDA libraries not on the link path.** The nvidia-* wheels install into
   `site-packages/nvidia/*/lib`. Either `pip install nvidia-cuda-runtime-cu12`
   etc., or export:
   ```bash
   export LD_LIBRARY_PATH=$(python -c \
     "import os,nvidia,glob;print(':'.join(glob.glob(os.path.dirname(nvidia.__file__)+'/*/lib')))"):$LD_LIBRARY_PATH
   ```
3. **Both runtimes installed.** `onnxruntime` and `onnxruntime-gpu` share the
   module name `onnxruntime`. Installing both leaves a broken mix. Fix:
   ```bash
   pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu
   ```
4. **cuDNN major mismatch.** ORT 1.18+ wants cuDNN 9. `libcudnn.so.8` on
   `LD_LIBRARY_PATH` will not satisfy it.
5. **Container without the driver mounted.** Pass `--gpus all` and confirm
   `nvidia-smi` works *inside* the container.

A quick end-to-end check:

```bash
pixelboost models download --model realesr-general-x4v3
pixelboost benchmark --size 512 --scale 4 --provider cuda --tile 512
```

You should see `provider: cuda` and a `dst_mpix_per_s` in the tens. If you see
`provider: cpu`, go back to step 1.

### 1.4 TensorRT (optional, biggest single win)

TensorRT typically takes another 2-3x over CUDA for these networks. It only pays
off if you serve enough traffic to keep the engine cache warm, because building
an engine takes 30-90 s.

```bash
pip install tensorrt
pixelboost upscale in.jpg -o out.png --provider tensorrt --fp16
```

The engine and timing caches are written to `PIXELBOOST_TRT_CACHE`
(default `~/.cache/pixelboost/trt`). **Persist that directory across restarts** --
if it is inside a container's writable layer you rebuild the engine after every
deploy, and a rolling restart turns into a 90-second outage per replica.

---

## 2. Bare metal (Linux)

### 2.1 Layout

```
/srv/pixelboost/
  .venv/                 the virtualenv
  pixelboost.yaml        config
  data/
    models/              checkpoints + TensorRT cache
```

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin pixelboost
sudo mkdir -p /srv/pixelboost && sudo chown -R pixelboost:pixelboost /srv/pixelboost

sudo -u pixelboost python3 -m venv /srv/pixelboost/.venv
sudo -u pixelboost /srv/pixelboost/.venv/bin/pip install \
  "pixelboost[onnx-cpu,server,yaml] @ git+https://github.com/your-org/pixelboost.git"

sudo -u pixelboost env PIXELBOOST_HOME=/srv/pixelboost/data \
  /srv/pixelboost/.venv/bin/pixelboost models download --all
```

### 2.2 Configuration

Start from the shipped default and edit:

```bash
sudo cp configs/default.yaml /srv/pixelboost/pixelboost.yaml
sudo chown pixelboost:pixelboost /srv/pixelboost/pixelboost.yaml
sudo chmod 640 /srv/pixelboost/pixelboost.yaml
```

Minimum you must change:

```yaml
models_dir: /srv/pixelboost/data/models

backend: onnx                 # pin it in production; "auto" can surprise you
model: realesr-general-x4v3   # or realesrgan-x4plus for quality
provider: cpu                 # or cuda
threads: 4

tile: 512
max_pixels: 32000000          # 32 MP -- protects the box from a 50 MP upload

server:
  host: 127.0.0.1             # nginx terminates TLS, so bind localhost only
  port: 8000
  workers: 2
  api_keys: ["sk-live-CHANGE-ME"]
  max_upload_mb: 32
```

> `provider: cpu` with `threads: 4` on a 4 vCPU box is the single most reliable
> configuration. `threads: 0` lets ONNX Runtime spawn one thread per core, which
> combined with `workers: 2` causes oversubscription and *slower* throughput.

### 2.3 systemd

```bash
sudo cp deploy/pixelboost.service /etc/systemd/system/
sudo cp deploy/pixelboost.env.example /etc/pixelboost/pixelboost.env
sudo chmod 600 /etc/pixelboost/pixelboost.env
sudo nano /etc/systemd/system/pixelboost.service   # fix User / WorkingDirectory / ExecStart
sudo systemctl daemon-reload
sudo systemctl enable --now pixelboost
sudo systemctl status pixelboost
journalctl -u pixelboost -f
```

### 2.4 nginx + TLS

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/pixelboost
sudo ln -s /etc/nginx/sites-available/pixelboost /etc/nginx/sites-enabled/
sudo nano /etc/nginx/sites-available/pixelboost   # set server_name
sudo nginx -t && sudo systemctl reload nginx

sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d upscale.example.com
```

Two settings that bite if you skip them, both already in `deploy/nginx.conf`:

- `client_max_body_size 32m` -- nginx defaults to **1 MB** and will reject a
  normal phone photo with a 413 that looks like an application bug.
- `proxy_read_timeout 300s` -- a CPU 4x upscale of a 4 MP photo takes 20-40 s;
  the 60 s default is fine until one day it is not, and then you get an
  intermittent 504 under load.

Since `proxy_request_buffering off` is set, the request streams to the app rather
than being spooled to disk first.

### 2.5 Verify

```bash
curl -s localhost:8000/healthz | jq

curl -s -X POST localhost:8000/v1/enhance \
  -H "X-API-Key: sk-live-CHANGE-ME" \
  -F "file=@photo.jpg" \
  -F "scale=4" -F "output_format=png" \
  -o out.png -D -

# telemetry comes back in the headers
# X-PixelBoost-Backend: onnx
# X-PixelBoost-Provider: cpu
# X-PixelBoost-Ms: 4213.7
```

---

## 3. Docker

### 3.1 CPU

```bash
docker build -f docker/Dockerfile.cpu -t pixelboost:cpu .

docker volume create pb-models
docker run --rm -v pb-models:/models pixelboost:cpu models download --all

docker run -d --name pixelboost \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v pb-models:/models \
  -v "$PWD/configs:/configs:ro" \
  -e PIXELBOOST_CONFIG=/configs/default.yaml \
  -e PIXELBOOST_CPU_WORKERS=2 \
  --memory 4g \
  pixelboost:cpu
```

### 3.2 GPU

```bash
# host prerequisite
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

docker build -f docker/Dockerfile.gpu -t pixelboost:gpu .
docker run --rm --gpus all pixelboost:gpu capabilities     # must list CUDAExecutionProvider

docker run -d --name pixelboost-gpu \
  --restart unless-stopped --gpus all \
  -p 127.0.0.1:8000:8000 \
  -v pb-models:/models \
  -v "$PWD/configs:/configs:ro" \
  -e PIXELBOOST_CONFIG=/configs/default.yaml \
  -e PIXELBOOST_GPU_WORKERS=1 \
  --shm-size 2g \
  pixelboost:gpu
```

`--shm-size 2g` is not optional at 4x: the default 64 MB `/dev/shm` causes
opaque crashes inside cuDNN's workspace allocation.

The GPU image pins `nvidia/cuda:12.4.1-cudnn-runtime`. If you change the
onnxruntime-gpu version, change the base tag to match, or `capabilities` will
report CPU-only.

### 3.3 compose

```bash
docker compose -f docker/docker-compose.yml --profile cpu up -d
docker compose -f docker/docker-compose.yml --profile gpu up -d
```

---

## 4. Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install "pixelboost[onnx-cpu,server,yaml]"

# DirectML is the practical GPU path on Windows -- no CUDA toolkit needed,
# it works on AMD, Intel and NVIDIA alike.
pip install onnxruntime-directml

pixelboost models download --model realesr-general-x4v3
pixelboost upscale photo.jpg -o photo_4x.png --provider directml --scale 4
pixelboost serve --host 127.0.0.1 --port 8000
```

Use the **Windows** backend `directml` rather than `cuda` unless you already have
a matched CUDA + cuDNN install; the DirectML provider is a single wheel and has a
much higher success rate on workstations.

For a service, run it under NSSM or as a scheduled task at start-up. Note that
`torch` and file paths with non-ASCII characters are a known source of grief on
Windows -- prefer the ONNX backend.

---

## 5. Using it as a library

```python
from pixelboost import Engine, EnhanceOptions

with Engine() as engine:
    opts = EnhanceOptions(scale=4, model="realesrgan-x4plus", tile=512)
    result, path = engine.enhance_file("in.jpg", "out.png", opts)
    print(result.summary())
    # {'backend': 'onnx', 'provider': 'cuda', 'src_size': [1024, 768],
    #  'dst_size': [4096, 3072], 'scale': 4.0, 'tiles': 48, 'elapsed_ms': 1180.4}
```

One-shot:

```python
from pixelboost import enhance
result, path = enhance("in.jpg", "out.png", scale=2, backend="classical")
```

Batch:

```python
reports = engine.batch(glob.glob("photos/*.jpg"), "out/", opts, suffix="_4x")
```

**Keep one `Engine` for the process lifetime.** It caches warm backends; building
a new one per request reloads a 64 MB graph each time.

Thread pool:

```python
from concurrent.futures import ThreadPoolExecutor

with Engine() as engine, ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(engine.enhance_file, p, f"out/{i}.png", opts)
               for i, p in enumerate(paths)]
```

---

## 6. Production checklist

- [ ] `pixelboost capabilities` shows the provider you expect
- [ ] Models downloaded **and** `warmup: true` in the config
- [ ] `pixelboost.yaml` has an explicit `backend`, `model`, `provider`
- [ ] `threads` set, not left at 0, whenever `workers > 1`
- [ ] `max_pixels` set to something a 50 MP upload cannot blow past
- [ ] `api_keys` non-empty if the port is reachable from anywhere
- [ ] nginx `client_max_body_size` ≥ the app's `max_upload_mb`
- [ ] nginx `proxy_read_timeout` ≥ your worst-case job time
- [ ] TensorRT cache directory on a persistent volume
- [ ] Service runs as a non-root user
- [ ] `journalctl -u pixelboost` shows no repeated OOM or provider warnings
- [ ] Disk has room for a temp file per concurrent upload (uploads spill to disk
      with `proxy_request_buffering off`)

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `providers: ["CPUExecutionProvider"]` on a GPU box | CUDA/cuDNN mismatch, or both ORT wheels installed | §1.3 |
| First request takes 60+ s, later ones are fast | TensorRT engine build, or cold model load | `warmup: true`; persist `PIXELBOOST_TRT_CACHE` |
| `RuntimeError: CUDA out of memory` | tile too large | `--tile 256`, or leave `--tile 0` off and let `run_resilient` halve it |
| Visible grid / seams in the output | overlap too small for the model's receptive field | `--tile-overlap 32` (or 48 for x4plus) |
| Output is crunchy / oversharpened | detail and sharpen stacked | `--detail 0.2 --sharpen 0`; see TUNING.md |
| Skin looks waxy, foliage like watercolour | detail too high | `--detail 0.15` |
| Grey/noisy borders along the image edge | reflective padding interacting with a small `tile_pad` | `--tile-pad 24` |
| First tile row is dark or blurred | missing padding on a fixed-input-size ONNX model | re-export with dynamic axes (`scripts/export_onnx.py`) |
| `413 Request Entity Too Large` from nginx, not the app | nginx 1 MB default | `client_max_body_size 32m` |
| `504` on large images only | nginx read timeout | `proxy_read_timeout 300s` |
| Service dies under load with no traceback | `/dev/shm` too small in Docker | `--shm-size 2g` |
| Throughput *drops* when workers is raised | GPU context contention | `workers: 1` on GPU, always |
| Everything is slow after deploy, fine locally | `threads: 0` with multiple workers | set `threads` explicitly |
| `ModuleNotFoundError: pixelboost` under systemd | venv path wrong in the unit | absolute `/srv/.../.venv/bin/pixelboost` |
| Alpha channel becomes a black halo | alpha was upscaled with the RGB net | keep `alpha_mode: lanczos` |
| EXIF orientation looks wrong after processing | Pillow does not auto-rotate | call `ImageOps.exif_transpose` before `Engine.enhance` |
