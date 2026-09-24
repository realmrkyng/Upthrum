# PixelBoost

**CPU / GPU 双路加速的图像画质增强与超分辨率引擎 —— 一条命令即可自部署。**

[![CI](https://github.com/your-org/pixelboost/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/pixelboost/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-94%20passing-brightgreen.svg)](tests)

PixelBoost 能把图像放大并增强 2~8 倍。**同一份代码、同一个模型文件**，既能在几
十块钱一个月的纯 CPU 小服务器上跑，也能跑在 CUDA GPU 上。

它不是给论文复现用的，是给要**长期运维**的人用的：上传量不可控但内存必须有界、
不强制安装 800 MB 的 PyTorch、内置一条永远可用的免模型通路，以及一份把你第一次
踩坑的 CUDA / cuDNN 版本错配问题直接讲清楚的部署文档。

```bash
pip install "pixelboost[onnx-cpu,server,yaml]"
pixelboost models download --model realesrgan-x4plus
pixelboost upscale photo.jpg -o photo_4x.png --scale 4
```

```
photo.jpg -> photo_4x.png  [1024, 768] -> [4096, 3072]  [onnx/cuda] 1180 ms
```

---

## 为什么再造一个轮子

现有的选择要么是假设你有 3090 和 conda 环境的研究代码，要么是把某一种后端焊死
的封装。缺的是「在我机器上能跑」和「在 6 美元/月的小机器上不崩」之间的那一段。

PixelBoost 押了四个具体的判断：

**CPU 是一等公民，不是备胎。** ONNX Runtime 的 CPU wheel 只有约 15 MB，PyTorch
是约 800 MB。绝大多数部署根本不需要 PyTorch，所以它是可选的；而且 ORT 在做推理
时会释放 GIL，意味着线程池就能拿到真实的多核并行，不需要多进程，也不太需要任务
队列中间件。

**内存必须有界。** 6000×4000 的照片做 4 倍放大，理论上需要 73 GB 的浮点缓冲。
PixelBoost 按行带流式输出分块结果，峰值内存约为「一个 tile」而不是「整张图」；
显存爆了会自动把 tile 减半重试。用户上传的是 1200 万像素的手机照片，服务不能
对此有意见。

**降级，而不是失败。** 模型文件缺失或 CUDA 库版本对不上，应该表现为画质下降，
而不是 500 报错。所以有一条永不可用的免模型后端——而且在截图、文字、UI 这类内容
上，它常常**比 GAN 更好**，因为 GAN 会凭空织出不存在的纹理。

**后处理才是观感的大头。** 去噪、锐化、色度清理占了一张超分图「看起来好不好」的
一半，而它们通常是被事后随便贴上去的，半径还是错的。在 PixelBoost 里它们是流水
线的正式环节，且半径统一以**输出像素**为单位，所以 `--detail-radius 4` 在 2 倍和
8 倍下含义一致。

---

## 特性

| | |
|---|---|
| **后端** | classical（纯 numpy，无需模型）、ONNX Runtime、PyTorch |
| **执行提供者** | CPU、CUDA、TensorRT、DirectML、CoreML、ROCm、OpenVINO |
| **分块推理** | 流式累加器、余弦羽化拼接、显存不足自动缩小 tile |
| **画质流水线** | 保边去噪 → 自动色阶 → 放大 → 色度降噪 → 引导滤波细节增强 → 可选 USM → 色彩调整 |
| **尺寸控制** | `--scale` / `--width` / `--height` / `--longest-side`，支持非等比 |
| **Alpha** | 独立通道处理，抠图主体不会出现黑边 |
| **色彩** | ICC / EXIF 透传，灰度进 → 灰度出 |
| **接口** | CLI（单图 + 批量）、FastAPI 服务（同步 + 异步任务）、Python API |
| **运维** | 原子写文件、结构化日志、`/healthz`、能力探测、API Key、上传体积限制 |
| **部署** | 裸机、systemd 单元、nginx 配置、Docker（CPU + CUDA）、compose |
| **测试** | 94 个测试，不联网、不需要模型文件，2 秒内跑完 |

---

## 安装

需要 Python 3.9+。

```bash
git clone https://github.com/your-org/pixelboost.git
cd pixelboost
python -m venv .venv && source .venv/bin/activate

# 仅 CPU（依赖约 40 MB）
pip install -e ".[onnx-cpu,server,yaml]"

# GPU（选之前请先看 docs/DEPLOY.md 第 1.3 节）
pip install -e ".[onnx-gpu,server,yaml]"
```

一个包都不想多装？免模型后端只要 numpy + Pillow：

```bash
pip install numpy pillow
pixelboost upscale in.png -o out.png --backend classical --scale 2
```

验证安装：

```bash
pixelboost capabilities
```

---

## 快速开始

### 1. 获取模型

```bash
pixelboost models download --model realesrgan-x4plus
pixelboost models                      # 看看本地有什么
```

也可以完全跳过这一步，直接用免模型后端。

### 2. 放大

```bash
# 4 倍，通用最佳画质模型
pixelboost upscale photo.jpg -o photo_4x.png --scale 4

# 动漫 / 插画
pixelboost upscale art.png -o art_4x.png --model realesrgan-x4plus-anime

# 截图和文字 —— 不用模型，比 GAN 更适合这类内容
pixelboost upscale ui.png -o ui_2x.png --backend classical --scale 2

# 按目标边长而不是倍率
pixelboost upscale photo.jpg -o photo_2048.png --longest-side 2048

# 整个目录
pixelboost batch ./originals -o ./enhanced --recursive --scale 2 --json > report.json
```

### 3. 调参

```bash
pixelboost upscale portrait.jpg -o out.png --scale 2 --detail 0.15 --denoise 0.025
pixelboost upscale product.jpg  -o out.png --scale 4 --detail 0.5 --sharpen 0.15
```

各类图像的推荐配方见 [docs/TUNING.md](docs/TUNING.md)。

---

## CPU 与 GPU

**同一个模型文件、同一套流水线代码**，区别只在执行提供者。

### CPU

```bash
pixelboost upscale in.jpg -o out.png --provider cpu --threads 4 \
  --model realesr-general-x4v3 --tile 384
```

- 用紧凑模型（`realesr-general-x4v3`，120 万参数 vs 1670 万参数）。
- **一定要显式设 `--threads`。** 留 0 会让 ONNX Runtime 按核数起线程，和你的
  worker 池互相抢，负载一上来吞吐反而更差。
- 内存够大的话，`--tile 0`（整图推理）最快。

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

如果在有显卡的机器上只打印出 `CPUExecutionProvider`，直接看
[docs/DEPLOY.md 1.3 节](docs/DEPLOY.md#13-gpu-getting-cudaexecutionprovider-to-appear)。
十次里有九次是 CUDA / cuDNN 主版本错配，或者 `onnxruntime` 和 `onnxruntime-gpu`
两个包装在了同一个环境里。

**先测再调。** tile 大小是 GPU 上最重要的参数，而最优值和硬件强相关：

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
```

这里 512 只用四分之一的显存就拿到峰值的 94%。**上线要用的是 512，不是 1024** ——
省下来的显存是留给并发请求的。

---

## HTTP 服务

```bash
pixelboost serve --host 127.0.0.1 --port 8000 --workers 2
```

在线文档：`http://localhost:8000/docs`

```bash
curl -X POST http://localhost:8000/v1/enhance \
  -F "file=@photo.jpg" -F "scale=4" \
  -o out.png -D -
```

```
HTTP/1.1 200 OK
X-PixelBoost-Backend: onnx
X-PixelBoost-Provider: cuda
X-PixelBoost-Ms: 1180.4
```

耗时的任务走队列，避免卡在反向代理的读超时上：

```bash
JOB=$(curl -s -X POST localhost:8000/v1/jobs -F "file=@big.jpg" -F "scale=4" | jq -r .id)
curl -s localhost:8000/v1/jobs/$JOB | jq '.status, .progress'
curl -s localhost:8000/v1/jobs/$JOB/result -o big_4x.png
```

| 接口 | 用途 |
|---|---|
| `GET /healthz` | 存活探针 |
| `GET /v1/capabilities` | 当前可用的后端、执行提供者、已下载模型 |
| `GET /v1/stats` | 队列深度、已加载后端 |
| `POST /v1/enhance` | 同步，图片直接在响应体里返回 |
| `POST /v1/jobs` | 入队，返回任务 id |
| `GET /v1/jobs/{id}` | 状态 + 分块进度 |
| `GET /v1/jobs/{id}/result` | 取回结果图 |
| `DELETE /v1/jobs/{id}` | 排队中可取消 |

完整接口文档：[docs/API.md](docs/API.md)
部署（systemd / nginx / Docker / Windows）：[docs/DEPLOY.md](docs/DEPLOY.md)

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

**整个进程共用一个 `Engine` 实例。** 它会缓存已经热起来的后端；每个请求新建一个
等于每次都重新加载一次模型。

---

## 工作原理

```
                       ┌──────────────────────────────────────────┐
  输入文件 ───────────▶│ imageio.load                             │
                       │  解码 · 拆出 alpha · 保留 ICC/EXIF        │
                       └────────────────────┬─────────────────────┘
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │ 预处理                                    │
                       │  去噪 → 自动色阶 → [预降采样]              │
                       └────────────────────┬─────────────────────┘
                                            ▼
              ┌─────────────────────────────────────────────────────────┐
              │ 后端（统一的 3 方法契约）                                  │
              │  classical  numpy Lanczos + 引导滤波      永远可用         │
              │  onnx       ORT: CPU/CUDA/TRT/DML/CoreML  一处导出多处跑   │
              │  torch      RRDBNet / SRVGGNetCompact     直接吃 .pth     │
              └────────────────────┬────────────────────────────────────┘
                                   ▼
              ┌─────────────────────────────────────────────────────────┐
              │ 分块 规划 → 推理 → 余弦羽化 → 流式写出                     │
              │  峰值内存 ≈ 一个 tile，而非整张输出                        │
              │  显存不足 → tile 减半重试                                 │
              └────────────────────┬────────────────────────────────────┘
                                   ▼
                       ┌──────────────────────────────────────────┐
                       │ 后处理                                    │
                       │  色度降噪 → 细节增强 → USM → 色彩 → 钳位    │
                       └────────────────────┬─────────────────────┘
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │ imageio.save   临时文件 + os.replace 原子写 │
                       └──────────────────────────────────────────┘
```

真正撑起这套东西的是四个判断。

**阶段顺序是画质决策，不是风格选择。** 去噪放在放大之**前**：让网络去放大噪声，
它会非常忠实地把噪声也放大，而且在 1 倍下做这件事便宜得多。色度降噪放在放大之
**后**：GAN 的彩色噪点正是产生在那一层。细节增强和锐化放在最后，在最终分辨率
上做，半径才可以用输出像素来表达。

**后端契约只有三个方法。** `process(tile, scale) -> tile`、`warmup()`、`close()`。
分块、色彩、读写、调度全在它之上，所以服务端代码里找不到一个 `if cuda` 分支。

**拼接必须在分块一致处严格相等。** 每个 tile 在重叠区贡献一条可分离的余弦斜坡，
累加器保存 `Σ(w·x)` 和 `Σw`，输出取两者之比。测试里有一条断言：在一个与位置无关
的后端下，分块输出必须与整图输出**完全相等**。开发期间正是这条断言抓出了行带
carry 的错位 bug。

**细节是「提取」出来的，不是「放大」出来的。** classical 通路用引导滤波而不是
USM：`base = guided_filter(x, x)`，`detail = x - base`，`out = base + k·detail`。
引导滤波在平滑平坦区域的同时能锁住边缘，所以提取出的细节带几乎没有边缘渗出——
这就是它能锐化而不出现白边的原因。`ops.py` 用纯 numpy 实现了它，外加 Lanczos
重采样、盒式滤波和三遍高斯，全部基于 `O(n)` 的 cumsum 核，并把单缓冲区限制在
400 万浮点数以内。

完整设计说明：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 模型

内置 5 个条目，实际只需要下载一个就能开始：

| 名称 | 倍率 | 参数量 | 适用 |
|---|---|---|---|
| `realesrgan-x4plus` | 4 | 16.7 M | 摄影照片 —— 默认 |
| `realesrgan-x4plus-anime` | 4 | 6.0 M | 动漫、插画、线稿 |
| `realesr-general-x4v3` | 4 | 1.2 M | 速度优先、CPU 机器（快约 8 倍） |
| `realesr-general-wdn-x4v3` | 4 | 1.2 M | 高压缩 / 噪点源图 |
| `realesr-animevideov3` | 4 | 2.4 M | 视频帧，延迟最低 |
| `classical` | 任意 | 0 | 文字、UI、截图 —— 零下载 |

```bash
pixelboost models
pixelboost models download --all
pixelboost models export --model realesrgan-x4plus --fp16   # pth -> onnx
```

PixelBoost **不打包任何权重**。模型从 Real-ESRGAN 的 release 下载
（BSD-3-Clause）到 `~/.cache/pixelboost/models`。自己训练的 `.onnx` 直接用
`--model-path` 指定，不需要注册。

模型选择决策树和 ONNX 导出的细节：[docs/MODELS.md](docs/MODELS.md)

---

## 性能参考

1024×1024 原图放大 4 倍，数量级参考，**务必在目标机器上实测**。

| 后端 | 硬件 | Tile | 耗时 |
|---|---|---|---|
| classical | 4 vCPU | 关 | ~0.9 s |
| general-x4v3 | 4 vCPU | 256 | ~4 s |
| general-x4v3 | RTX 3060 | 512 | ~0.4 s |
| x4plus | 8 vCPU | 256 | ~35 s |
| x4plus | RTX 3060 | 512 | ~1.2 s |
| x4plus TensorRT fp16 | RTX 3060 | 512 | ~0.5 s |

x4plus 上 CPU 和 GPU 差 30~100 倍。这不是调参能解决的：要么上 GPU，要么换紧凑
模型，要么把倍率降到 2。

选型与内存占用表：[docs/DEPLOY.md 第 0 节](docs/DEPLOY.md#0-sizing-the-host)

---

## 文档

| 文档 | 内容 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 模块划分、后端契约、阶段流水线、分块数学、并发模型 |
| [docs/DEPLOY.md](docs/DEPLOY.md) | 安装、CUDA/cuDNN 匹配、systemd、nginx、Docker、Windows、上线检查清单、故障排查 |
| [docs/TUNING.md](docs/TUNING.md) | 逐个参数的取舍、分题材配方、画质问题诊断 |
| [docs/MODELS.md](docs/MODELS.md) | 模型清单、选择指南、ONNX 导出、接入自己的模型 |
| [docs/API.md](docs/API.md) | HTTP 接口参考、错误码、客户端示例 |

---

## 开发

```bash
make install-dev
make test          # 94 个测试，2 秒内，不联网
make lint
make benchmark
```

测试刻意做到不需要模型文件、不需要网络，所以能在所有平台的 CI 上跑。最有价值的
一条是 `test_tiling.py::test_tiled_matches_whole_image`——它断言的是一个精确相等，
任何拼接逻辑的回退都会立刻被它抓到。

---

## 参与贡献

欢迎提 Issue 和 PR。几个约定：

- 提交 PR 前请跑 `make test` 和 `make lint`。
- `ops.py` 里的新算子必须是纯 numpy，并且测试要断言**依赖的性质**（直流保持、
  保边性、值域边界），而不是打一个快照金值。
- 新后端只需实现 `backends/base.py` 里的三方法契约，别做多余的事。
- 如果你修掉了一个部署问题，请同时把结论补进 `docs/DEPLOY.md`。

---

## 许可证

MIT，见 [LICENSE](LICENSE)。

代码中不含任何模型权重。所引用的 Real-ESRGAN 模型为 BSD-3-Clause；如需商业化
分发权重或生成结果，请先自行确认其许可条款。
