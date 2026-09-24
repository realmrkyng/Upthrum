"""PyTorch backend for Real-ESRGAN ``.pth`` checkpoints.

ONNX is the better production target, but most published weights ship as
``.pth``. Rather than tell users to go convert them, this module carries the
RRDBNet definition inline -- about 60 lines -- and loads the checkpoint
directly. That removes an install step that is otherwise the single most common
reason someone gives up on a super-resolution repo.

Set ``PIXELBOOST_TORCH=0`` to skip importing torch entirely on hosts that only
use the ONNX path; the import is lazy so a missing torch never breaks anything
else.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pixelboost.backends.base import Backend
from pixelboost.errors import BackendUnavailable, ModelNotFound


def _torch():
    try:
        import torch  # noqa: WPS433
        import torch.nn as nn  # noqa: WPS433
        import torch.nn.functional as F  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - depends on install extra
        raise BackendUnavailable(
            "PyTorch is not installed. Install the CPU build with "
            "`pip install torch --index-url https://download.pytorch.org/whl/cpu` "
            "or a CUDA build from https://pytorch.org/get-started/locally/ . "
            "If you only need inference, prefer the ONNX backend: it is a far "
            "smaller dependency."
        ) from exc
    return torch, nn, F


def build_rrdbnet(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    num_feat: int = 64,
    num_block: int = 23,
    num_grow_ch: int = 32,
    scale: int = 4,
):
    """Factory that constructs RRDBNet. Kept as a function so torch is optional.

    Architecture follows ``basicsr.archs.rrdbnet_arch`` exactly, including the
    ``* 0.2`` residual scaling. The nesting and attribute names must match or
    the checkpoint will not load -- that is why this is written out rather than
    reimplemented from the paper.
    """
    torch, nn, F = _torch()

    class ResidualDenseBlock(nn.Module):
        def __init__(self, nf: int = 64, gc: int = 32) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf: int, gc: int = 32) -> None:
            super().__init__()
            self.rdb1 = ResidualDenseBlock(nf, gc)
            self.rdb2 = ResidualDenseBlock(nf, gc)
            self.rdb3 = ResidualDenseBlock(nf, gc)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = int(scale)
            in_ch = num_in_ch
            if self.scale == 2:
                in_ch = num_in_ch * 4
            elif self.scale == 1:
                in_ch = num_in_ch * 16
            self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            if self.scale == 2:
                feat = F.pixel_unshuffle(x, 2)
            elif self.scale == 1:
                feat = F.pixel_unshuffle(x, 4)
            else:
                feat = x
            feat = self.conv_first(feat)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet()


def build_srvggnet(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    num_feat: int = 64,
    num_conv: int = 32,
    scale: int = 4,
):
    """Compact VGG-style super-res net (``realesr-general-x4v3``).

    ~1.2 M parameters against RRDBNet's 16.7 M. It is roughly 8x faster for
    maybe 10 % less perceptual quality, and it also ships a denoise variant,
    which makes it the right default for a busy CPU tier.
    """
    torch, nn, F = _torch()

    class SRVGGNetCompact(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = int(scale)
            self.body = nn.ModuleList()
            self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
            for _ in range(num_conv):
                self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                self.body.append(nn.PReLU(num_parameters=num_feat))
            self.body.append(
                nn.Conv2d(num_feat, num_out_ch * self.scale * self.scale, 3, 1, 1)
            )
            self.upsampler = nn.PixelShuffle(self.scale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            base = F.interpolate(x, scale_factor=self.scale, mode="nearest")
            return out + base

    return SRVGGNetCompact()


def build_model(arch: str, scale: int = 4, num_block: int = 23, num_feat: int = 64, num_conv: int = 32):
    """Dispatch on the architecture tag carried by a :class:`ModelSpec`."""
    arch = (arch or "rrdb").lower()
    if arch in ("rrdb", "rrdbnet"):
        return build_rrdbnet(num_feat=num_feat, num_block=num_block, scale=scale)
    if arch in ("srvgg", "compact", "srvggnetcompact"):
        return build_srvggnet(num_feat=num_feat, num_conv=num_conv, scale=scale)
    raise BackendUnavailable(
        f"unsupported architecture {arch!r}; supported: rrdb, srvgg. "
        f"Export exotic models to ONNX instead."
    )


def load_state_dict(path: str) -> Dict[str, Any]:
    torch, _, _ = _torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("params_ema", "params", "state_dict", "model"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    if not isinstance(ckpt, dict):
        raise ModelNotFound(f"unrecognised checkpoint layout in {path}")
    clean = {}
    for k, v in ckpt.items():
        clean[k[7:] if k.startswith("module.") else k] = v
    return clean


class TorchBackend(Backend):
    name = "torch"

    def __init__(
        self,
        model_path: str,
        provider: Optional[str] = None,
        *,
        native_scale: int = 4,
        num_block: int = 23,
        num_feat: int = 64,
        num_grow_ch: int = 32,
        fp16: bool = False,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        align: int = 8,
        arch: str = "rrdb",
        num_conv: int = 32,
    ) -> None:
        super().__init__(model=model_name or os.path.basename(model_path), provider=provider)
        if not os.path.isfile(model_path):
            raise ModelNotFound(f"model file not found: {model_path}")

        torch, _, _ = _torch()
        requested = (provider or device or os.environ.get("PIXELBOOST_DEVICE") or "auto").lower()
        if requested in ("auto", ""):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested in ("gpu",):
            device = "cuda"
        elif requested in ("dml", "directml", "mps", "cpu", "cuda"):
            device = "mps" if requested == "mps" and torch.backends.mps.is_available() else requested
        else:
            device = "cpu"
        if device != "cpu" and not torch.cuda.is_available():
            raise BackendUnavailable(
                f"CUDA was requested but torch.cuda.is_available() is False. "
                f"Check the driver (`nvidia-smi`) and that you installed a "
                f"CUDA-enabled torch wheel, not the default CPU one."
            )

        self.model_path = model_path
        self.native_scale = int(native_scale)
        self.requires_tiling = True
        self.align = int(align)
        self.device = "cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")
        self.fp16 = bool(fp16) and self.device == "cuda"
        self.dtype = torch.float16 if self.fp16 else torch.float32

        net = build_model(
            arch,
            scale=self.native_scale,
            num_feat=int(num_feat),
            num_block=int(num_block),
            num_conv=int(num_conv),
        )
        state = load_state_dict(model_path)
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            raise ModelNotFound(
                f"checkpoint {model_path} is missing {len(missing)} tensors "
                f"(first: {missing[0]}). Check that --arch and the block counts "
                f"match the file: x4plus is rrdb/num_block=23, "
                f"x4plus-anime is rrdb/num_block=6, "
                f"general-x4v3 is srvgg/num_conv=32."
            )
        net.eval()
        net.to(device=self.device, dtype=self.dtype)
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
        self.net = net
        self._torch = torch

    @property
    def provider(self) -> str:
        return self.device

    def process(self, tile: np.ndarray, scale: float) -> np.ndarray:
        torch = self._torch
        h, w = tile.shape[0], tile.shape[1]
        x = torch.from_numpy(np.ascontiguousarray(np.transpose(tile, (2, 0, 1)))).unsqueeze(0)
        x = x.to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            y = self.net(x)
        out = y.squeeze(0).float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        return out[: h * self.native_scale, : w * self.native_scale]

    def warmup(self, tile: int = 64) -> None:
        side = max(8, int(tile))
        self.process(np.zeros((side, side, 3), np.float32), float(self.native_scale))
        if self.device == "cuda":
            self._torch.cuda.synchronize()

    def info(self) -> Dict[str, Any]:
        data = super().info()
        torch = self._torch
        data.update(
            {
                "model_path": self.model_path,
                "fp16": self.fp16,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device_name": (
                    torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu"
                ),
            }
        )
        return data

    def close(self) -> None:
        try:
            self.net.cpu()
        except Exception:
            pass
        self.net = None  # type: ignore[assignment]
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
