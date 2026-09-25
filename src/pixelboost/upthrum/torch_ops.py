"""Optional CUDA path for the FFT-bound part of UPTHRUM.

The analysis stage is where the wall-clock goes: for three bands it is eighteen
transforms, all dense and all separable. That is the profile a GPU is built for,
so when a CUDA build of PyTorch is present the decomposition runs there and the
rest of the pipeline stays on the CPU host, where its per-pixel gather is faster
than a kernel launch would be at these sizes.

The numpy implementation in ``transform.py`` remains the reference. This module
must agree with it to within float32 rounding, and that agreement is asserted by
``test_torch_matches_numpy`` when a GPU is available -- a silent divergence
between two implementations of the same estimator is the kind of bug that only
shows up as a slightly different report on someone else's machine.

Importing this module is cheap and safe on a machine without PyTorch: only
:func:`available` is ever called before the import of ``torch`` itself.
"""

from __future__ import annotations

import numpy as np

from .filters import FilterBank
from .transform import BandAnalysis

_TORCH_OK: int = 0


def available() -> bool:
    """Whether a CUDA-capable PyTorch is importable. Cached after the first call."""
    global _TORCH_OK
    if _TORCH_OK != 0:
        return _TORCH_OK > 0
    try:
        import torch  # noqa: F401

        _TORCH_OK = 1 if torch.cuda.is_available() else -1
    except Exception:
        _TORCH_OK = -1
    return _TORCH_OK > 0


def _grids(shape: tuple[int, int], device, dtype):
    import torch

    h, w = shape
    fy = torch.fft.fftfreq(h, d=1.0, device=device, dtype=dtype)[:, None]
    fx = torch.fft.rfftfreq(w, d=1.0, device=device, dtype=dtype)[None, :]
    return fx, fy


def _riesz(shape: tuple[int, int], device, dtype):
    import torch

    fx, fy = _grids(shape, device, dtype)
    r = torch.sqrt(fx * fx + fy * fy)
    r = r.clone()
    r[0, 0] = float("inf")
    rx = (-1j * fx / r).to(torch.complex64)
    ry = (-1j * fy / r).to(torch.complex64)
    rx[0, 0] = 0.0
    ry[0, 0] = 0.0
    return rx, ry


def _derivatives(shape: tuple[int, int], device, dtype):
    import torch

    fx, fy = _grids(shape, device, dtype)
    two_pi_j = 2j * np.pi
    return (
        (two_pi_j * fx).to(torch.complex64),
        (two_pi_j * fy).to(torch.complex64),
    )


def analyse(luminance: np.ndarray, bank: FilterBank) -> list[BandAnalysis]:
    """GPU implementation of :func:`pixelboost.upthrum.transform.analyse`."""
    import torch

    device = torch.device("cuda")
    dtype = torch.float32
    shape = (int(luminance.shape[0]), int(luminance.shape[1]))

    src = torch.from_numpy(np.ascontiguousarray(luminance, dtype=np.float32)).to(device)
    spectrum = torch.fft.rfft2(src)

    rx_m, ry_m = _riesz(shape, device, dtype)
    dx_m, dy_m = _derivatives(shape, device, dtype)

    results: list[BandAnalysis] = []
    eps = 1e-8

    for index, transfer in enumerate(bank.bands):
        tf = torch.from_numpy(np.ascontiguousarray(transfer, dtype=np.float32)).to(device)
        band = torch.fft.irfft2(spectrum * tf, s=shape)

        band_spec = torch.fft.rfft2(band)
        rx = torch.fft.irfft2(band_spec * rx_m, s=shape)
        ry = torch.fft.irfft2(band_spec * ry_m, s=shape)

        odd = torch.sqrt(rx * rx + ry * ry)
        amplitude = torch.sqrt(band * band + odd * odd)
        safe = torch.clamp(amplitude, min=eps)
        phase = torch.atan2(odd, band)
        orientation = torch.atan2(ry, rx)

        z = torch.complex(band / safe, odd / safe)
        stacked = torch.stack([z.real, z.imag], dim=0)
        z_spec = torch.fft.rfft2(stacked, dim=(-2, -1))
        gx = torch.fft.irfft2(z_spec * dx_m, s=shape, dim=(-2, -1))
        gy = torch.fft.irfft2(z_spec * dy_m, s=shape, dim=(-2, -1))
        grad_x = torch.imag(torch.conj(z) * torch.complex(gx[0], gx[1]))
        grad_y = torch.imag(torch.conj(z) * torch.complex(gy[0], gy[1]))

        results.append(
            BandAnalysis(
                index=index,
                centre=float(bank.centres[index]),
                band=band.detach().cpu().numpy().astype(np.float32),
                amplitude=amplitude.detach().cpu().numpy().astype(np.float32),
                phase=phase.detach().cpu().numpy().astype(np.float32),
                orientation=orientation.detach().cpu().numpy().astype(np.float32),
                grad_x=grad_x.detach().cpu().numpy().astype(np.float32),
                grad_y=grad_y.detach().cpu().numpy().astype(np.float32),
                energy=float(amplitude.mean().item()),
            )
        )

    del spectrum
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


__all__ = ["analyse", "available"]
