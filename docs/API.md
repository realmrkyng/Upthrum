# HTTP API

Base URL is whatever you expose, e.g. `https://upscale.example.com`.

Interactive docs are at `/docs` (Swagger) and `/redoc` while the service runs.
Everything below is also in the OpenAPI schema at `/openapi.json`.

## Authentication

If `server.api_keys` is non-empty, every endpoint except `/healthz` requires:

```
X-API-Key: sk-live-...
```

or `?api_key=sk-live-...`. A wrong or missing key returns `401`.

Keys are compared with `==`, not in constant time, and stored in plaintext in the
config. That is appropriate for a single-tenant internal service. If you are
exposing this to the public internet, put it behind an auth proxy and treat
these keys as an additional layer, not the only one.

## Meta

### `GET /healthz`

No auth. Used by the container healthcheck and by load balancers.

```json
{
  "status": "ok",
  "version": "0.1.0",
  "engine": true,
  "jobs": { "workers": 2, "queued": 0, "capacity": 64, "retained": 3 }
}
```

### `GET /v1/capabilities`

What this instance can actually do. Call it before writing a client so you do
not hardcode a model that is not installed.

```json
{
  "version": "0.1.0",
  "backends": ["classical", "nearest", "upthrum", "onnx"],
  "onnxruntime": true,
  "torch": false,
  "upthrum": true,
  "providers": ["CPUExecutionProvider"],
  "gpu": false,
  "methods": [
    { "backend": "classical", "family": "resampling", "variable": "intensity",
      "learned_parameters": 0, "needs_model_file": false },
    { "backend": "upthrum", "family": "phase reconstruction",
      "variable": "local phase (S^1), not intensity",
      "representation": "monogenic signal, per log-Gabor band",
      "transport": "amplitude-weighted coherent mean of unit phasors",
      "constraint": "0-dimensional persistent homology of sublevel sets",
      "learned_parameters": 0, "trained_on": null,
      "scales": "any >= 1", "identity_at_scale_one": true,
      "needs_model_file": false }
  ],
  "environment": ["onnxruntime : yes", "upthrum     : yes (device=cpu)",
                  "providers   : CPUExecutionProvider"],
  "models_dir": "/models",
  "models": [
    { "name": "realesrgan-x4plus", "scale": 4, "arch": "rrdb",
      "license": "BSD-3-Clause", "tags": ["photo", "general", "default"],
      "pth_present": true, "onnx_present": true }
  ],
  "server": { "workers": 2, "max_upload_mb": 32, "auth_required": true }
}
```

The `methods` array is the machine-readable answer to "what is the difference
between these backends": `classical` resamples intensity, `upthrum` reconstructs
phase under a topological constraint, and both carry `learned_parameters: 0`.
A client that wants to expose the choice to its own users should read this
rather than hardcoding the comparison.

Requesting `backend: "upthrum"` needs no model and returns
`model: null` in the result — model-free backends report no model rather than
the configured default.

### `GET /v1/stats`

Job queue state plus which backends are currently loaded (i.e. warm).

---

## Enhancing

### `POST /v1/enhance` — synchronous

Best for images under ~2 MP. Anything larger will likely exceed a proxy read
timeout; use the job endpoints instead.

**Request** — `multipart/form-data` with a `file` field, or a raw body with query
parameters (no `python-multipart` needed for the raw form).

```bash
curl -X POST https://upscale.example.com/v1/enhance \
  -H "X-API-Key: sk-live-..." \
  -F "file=@photo.jpg" \
  -F "scale=4" \
  -F "model=realesrgan-x4plus" \
  -F "detail=0.4" \
  -F "output_format=png" \
  -o out.png
```

Raw body variant, convenient from a shell or a signed-URL flow:

```bash
curl -X POST "https://upscale.example.com/v1/enhance?scale=2&format=webp&quality=90" \
  -H "X-API-Key: sk-live-..." \
  -H "Content-Type: image/jpeg" \
  --data-binary @photo.jpg \
  -o out.webp
```

**Response** — the image bytes by default. Useful metadata is in the headers so
you do not have to parse the body to get it:

```
X-PixelBoost-Backend: onnx
X-PixelBoost-Provider: cuda
X-PixelBoost-Model: realesrgan-x4plus
X-PixelBoost-Tiles: 48
X-PixelBoost-Ms: 1180.4
```

Return a JSON envelope instead — for clients that cannot read response headers,
or when you want the input probe too — with `Accept: application/json` or
`response=json`:

```json
{
  "image_base64": "iVBORw0KGgo...",
  "format": "png",
  "meta": {
    "backend": "onnx", "provider": "cuda", "model": "realesrgan-x4plus",
    "src_size": [1024, 768], "dst_size": [4096, 3072], "scale": 4.0,
    "tiles": 48, "elapsed_ms": 1180.4, "notes": [],
    "input": { "width": 1024, "height": 768, "format": "JPEG",
               "mode": "RGB", "has_alpha": false, "animated": false }
  }
}
```

### Request parameters

Any field of `EnhanceOptions` may be passed as a form field or query parameter.
Values are coerced to the declared type; unrecognised keys are ignored rather
than rejected, so adding a parameter in a new release never breaks an old client.

Naming: in the HTTP API you may use `format` as an alias for `output_format`.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `scale` | float | 4 | upscale factor |
| `width`, `height` | int | — | exact output size |
| `longest_side` | int | — | scale so the long edge equals this |
| `keep_aspect` | bool | true | with only one of width/height given, false leaves the other axis alone |
| `model` | str | config | registry name or file path |
| `provider` | str | auto | `cpu` / `cuda` / `tensorrt` / `directml` / `coreml` / `openvino` |
| `fp16` | bool | false | CUDA / TensorRT only |
| `tile` | int | 512 | 0 disables tiling |
| `tile_overlap` | int | 16 | raise to 32 if seams appear |
| `tile_pad` | int | 16 | border quality |
| `max_pixels` | int | 64M | output ceiling; larger requests are clamped, with a note |
| `denoise` | float | 0 | pre-upscale, 0-0.1 typical |
| `chroma_denoise` | float | 0 | post-upscale, 1-2 typical |
| `detail` | float | 0.35 | 0.15 portraits, 0.5 product |
| `detail_radius` | int | 4 | output pixels |
| `sharpen` | float | 0 | leave off unless needed |
| `auto_levels` | float | 0 | for washed-out sources |
| `gamma`, `contrast`, `saturation` | float | 1.0 / 0 / 1.0 | |
| `alpha_mode` | str | lanczos | never route alpha through the net |
| `output_format` | str | inferred | png / jpeg / webp / tiff / bmp |
| `quality` | int | 95 | lossy formats |
| `preserve_metadata` | bool | true | ICC + EXIF passthrough |
| `response` | str | — | set to `json` for the JSON envelope |

### Errors

| Status | Meaning |
|---|---|
| 400 | empty body, no `file` field, malformed multipart |
| 401 | bad or missing API key |
| 413 | upload exceeds `server.max_upload_mb` |
| 415 | body is not a decodable image |
| 422 | enhancement failed — bad parameters, model missing, scale mismatch |
| 429 | job queue full |
| 500 | unexpected; check the service log |
| 503 | engine still starting; retry shortly |
| 507 | out of memory |

Errors are `{"detail": "..."}`.

---

## Async jobs

### `POST /v1/jobs` → `202`

Same body as `/v1/enhance`.

```json
{ "id": "9f2c41ab7d3e0051", "status": "queued", "poll": "/v1/jobs/9f2c41ab7d3e0051" }
```

### `GET /v1/jobs/{id}`

```json
{
  "id": "9f2c41ab7d3e0051",
  "status": "running",
  "created_at": 1789000000.12,
  "started_at": 1789000000.44,
  "finished_at": null,
  "filename": "photo.jpg",
  "output_format": "png",
  "progress": { "done": 18, "total": 48 },
  "meta": {},
  "elapsed_ms": 2140.5
}
```

`status` is one of `queued`, `running`, `succeeded`, `failed`, `cancelled`.

### `GET /v1/jobs/{id}/result`

Returns the image as an attachment. `409` if the job has not finished, `422` if
it failed.

### `DELETE /v1/jobs/{id}`

Cancels a job that is still queued. `409` if it has already started — enhancing
is not preemptible.

### Polling

```python
import io, time, requests
from PIL import Image

job = requests.post(
    "https://upscale.example.com/v1/jobs",
    headers={"X-API-Key": KEY},
    files={"file": ("p.jpg", open("p.jpg", "rb"))},
    data={"scale": "4"},
).json()

while True:
    state = requests.get(f"https://upscale.example.com/v1/jobs/{job['id']}",
                         headers={"X-API-Key": KEY}).json()
    if state["status"] in ("succeeded", "failed", "cancelled"):
        break
    print(f"\r{state['progress']['done']}/{state['progress']['total']}", end="")
    time.sleep(1)

blob = requests.get(f"https://upscale.example.com/v1/jobs/{job['id']}/result",
                    headers={"X-API-Key": KEY}).content
Image.open(io.BytesIO(blob)).save("out.png")
```

Jobs and their results live in memory and are evicted `server.job_ttl_seconds`
after completion (default 1 hour). **A restart loses everything queued or
completed.** If you need durable jobs, put a real queue in front and keep the
image in object storage.

---

## Operational notes

**Concurrency.** The synchronous endpoint runs on a thread pool sized by
`server.workers`. On GPU keep it at 1: concurrent CUDA sessions on one device are
slower in aggregate than running them serially. On CPU `cpu_count // 2` is a
reasonable start, with `threads` set explicitly so ONNX Runtime does not
oversubscribe.

**Upload limits.** `server.max_upload_mb` is enforced after the body is read, so
it protects memory but not bandwidth. Also set `client_max_body_size` in nginx.

**Memory.** Peak use is roughly `tile² × scale² × 3 × 4` bytes for the inference
buffer plus the output image. `tile: 512` at 4x is ~12 MB of inference buffer and
much more for the output; `max_pixels` is the real protection.

**Timeouts.** For a CPU box, expect 20-40 s for a 4 MP 4x job. Set
`proxy_read_timeout 300s` and prefer the job endpoints for anything above 2 MP.
