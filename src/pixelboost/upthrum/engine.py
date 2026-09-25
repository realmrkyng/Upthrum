"""UPTHRUM -- Unified Phase-Topology Hierarchical Reconstruction Under the Monogenic manifold.

A super-resolution method built on a different variable than every other one.

    from pixelboost.upthrum import Upthrum
    engine = Upthrum()
    image, report = engine.enhance(photo, scale=4)

Conventional super-resolution estimates intensity: given samples on a coarse
lattice, predict values on a finer one. Whether the predictor is a fixed kernel, a
convolutional network or a diffusion model, the target is the same -- a number per
output pixel -- and so is the failure mode, because the mapping from coarse
intensity samples to fine intensity values is not injective. The fine structure
that produced those samples is genuinely underdetermined, so any intensity-domain
method must either blur (choose the smooth preimage) or hallucinate (choose a
plausible one).

UPTHRUM does not estimate intensity. It reconstructs **phase**. For each band of a
log-Gabor hierarchy it decomposes the signal into a monogenic triple (band-pass
component plus two Riesz components), expresses the local structure as an
amplitude and a phase on the circle, transports the phase to the output lattice by
coherent averaging of phasors, and only then re-derives intensity. Because
translation along a wavefront is exactly a phase shift, the sub-pixel position of
structure is *determined* by the phase field rather than guessed -- and the
amplitude attenuation that constitutes interpolation blur never arises, because
phases are averaged as unit vectors rather than intensities as scalars.

The name is the claim: **U**nified (one operator set, no per-scale special cases)
**P**hase-**T**opology (phase transport constrained by persistent homology)
**H**ierarchical (per-band, octave-spaced) **R**econstruction **U**nder the
**M**onogenic manifold (the representation that supplies phase at every
orientation without an explicit orientation search).

Five properties, each of which is a test
----------------------------------------

* **Exact identity at scale 1.** The transport collapses to the coincident sample,
  because the bilinear cell degenerates at integer query points. Verified by
  ``test_identity_at_scale_one``; it catches coordinate-convention errors that are
  otherwise invisible and that no visual inspection would reveal.
* **No attenuation at Nyquist.** A sinusoid at the band centre is transported with
  unit gain at any sub-pixel offset. Verified by ``test_no_attenuation_at_band_centre``.
* **Intrinsic anisotropy.** The phase gradient is a vector field; a step edge is
  reconstructed as a step, not as a smoothed ramp. Verified by
  ``test_edge_is_not_softened`` against a Lanczos baseline.
* **Topological invariance.** Critical points below the persistence threshold
  cannot be promoted into apparent structure. Verified by
  ``test_topology_suppresses_noise_peak``.
* **DC preservation.** The low-pass path is untouched, so mean brightness is
  carried across exactly. Verified by ``test_dc_is_preserved``.

Cost and acceleration
---------------------

No learned parameters, no training data, no model file. The cost is a fixed number
of FFTs per band plus a small gather per output pixel, both of which are dense
pointwise or separable operations -- the shapes a GPU is actually good at. The
numpy implementation here is the reference; ``device="auto"`` dispatches the
FFT-bound analysis to CUDA through :mod:`pixelboost.upthrum.torch_ops` when a GPU
build of PyTorch is present, and falls back silently when it is not, so the same
code path runs on a laptop and on a server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import OutOfMemory, UnsupportedScale
from ..ops import rgb_to_ycbcr, ycbcr_to_rgb
from .filters import FilterBank, build_filter_bank
from .params import UpthrumParams
from .reconstruct import inject_chroma, reconstruct_luminance
from .transform import analyse, coherence, dominant_orientation
from .transport import transport

EPS = 1e-8


@dataclass
class UpthrumReport:
    """What the run did, in numbers. Cheap to produce, expensive to guess at."""

    scale: float
    shape_in: tuple[int, int]
    shape_out: tuple[int, int]
    device: str
    bands: int
    colour: bool
    band_summary: dict[str, Any]
    mean_coherence: float
    identity_error: float | None
    topology: Any
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": round(self.scale, 4),
            "shape_in": list(self.shape_in),
            "shape_out": list(self.shape_out),
            "device": self.device,
            "bands": self.bands,
            "colour": self.colour,
            "band_summary": self.band_summary,
            "mean_coherence": round(self.mean_coherence, 6),
            "identity_error": None if self.identity_error is None else float(self.identity_error),
            "topology": self.topology,
            "seconds": round(self.seconds, 4),
        }

    def summary(self) -> str:
        lines = [
            f"UPTHRUM {self.shape_in[1]}x{self.shape_in[0]} -> {self.shape_out[1]}x{self.shape_out[0]}"
            f"  (scale {self.scale:g}, {self.bands} bands, {self.device})",
            f"  mean coherence {self.mean_coherence:.4f}",
        ]
        if self.identity_error is not None:
            lines.append(f"  scale-1 identity error {self.identity_error:.3e}")
        if isinstance(self.topology, dict):
            lines.append(
                "  topology tau={tau} peaks={p} pits={q} match={m}".format(
                    tau=self.topology.get("tau"),
                    p=self.topology["candidate"]["peaks"] if "candidate" in self.topology else "-",
                    q=self.topology["candidate"]["pits"] if "candidate" in self.topology else "-",
                    m=self.topology.get("matches", "-"),
                )
            )
        lines.append(f"  {self.seconds:.2f}s")
        return "\n".join(lines)


def resolve_device(device: str) -> str:
    """Map a device request onto what this machine can actually do.

    ``"cpu"`` is honoured literally. Anything else -- ``"auto"``, ``"cuda"``,
    ``"gpu"`` -- means "use the GPU if there is one", and resolves to ``"cpu"``
    when :mod:`torch_ops` is unavailable. That fallback is deliberate: the
    numpy path is the reference implementation and produces the same numbers,
    so a missing CUDA install costs time and never changes the result.
    """
    if device in ("cpu", ""):
        return "cpu"
    from . import torch_ops

    if torch_ops.available():
        return "cuda"
    return "cpu"


class Upthrum:
    """The reconstruction engine.

    Stateless apart from configuration: one instance can be reused across images
    and across threads, since nothing is cached between calls.
    """

    def __init__(self, params: UpthrumParams | None = None, device: str = "cpu") -> None:
        self.params = params or UpthrumParams()
        self.device = device

    def build_bank(self, shape: tuple[int, int]) -> FilterBank:
        p = self.params
        return build_filter_bank(shape, bands=p.bands, top_frequency=p.top_frequency, sigma=p.band_sigma)

    def plan(self, shape: tuple[int, int], scale: float) -> tuple[int, int]:
        if scale <= 0:
            raise UnsupportedScale("scale must be positive")
        if scale < 1.0:
            raise UnsupportedScale(f"UPTHRUM reconstructs upward only; got scale={scale}")
        out = (int(round(shape[0] * scale)), int(round(shape[1] * scale)))
        if self.params.max_pixels > 0 and out[0] * out[1] > self.params.max_pixels:
            raise OutOfMemory(
                f"output {out[1]}x{out[0]} exceeds max_pixels={self.params.max_pixels}"
            )
        return out

    def enhance(
        self,
        image: np.ndarray,
        scale: float = 2.0,
        alpha: np.ndarray | None = None,
    ) -> tuple[np.ndarray, UpthrumReport]:
        """Reconstruct ``image`` at ``scale``.

        Accepts ``(H, W)`` grayscale or ``(H, W, 3)`` colour in ``[0, 1]``. Returns
        the result on the same scale, plus a report of what actually happened.
        """
        import time

        started = time.perf_counter()
        params = self.params
        src = np.asarray(image, dtype=np.float32)
        if src.ndim == 2:
            colour = False
            luma_in = src
        elif src.ndim == 3 and src.shape[2] >= 3:
            colour = True
            luma_in, cb, cr = rgb_to_ycbcr(src)
            luma_in = luma_in.astype(np.float32)
        else:
            raise ValueError(f"unsupported image shape {src.shape}")

        in_shape = (int(luma_in.shape[0]), int(luma_in.shape[1]))
        out_shape = self.plan(in_shape, scale)

        device = resolve_device(self.device)
        bank = self.build_bank(in_shape)

        if device == "cuda":
            from . import torch_ops

            analyses = torch_ops.analyse(luma_in, bank)
        else:
            analyses = analyse(luma_in, bank)
        theta = dominant_orientation(analyses, in_shape)
        kappa = coherence(analyses, in_shape)

        transported = transport(analyses, theta, out_shape, scale, params)
        luma_out, residual, info = reconstruct_luminance(
            luma_in, analyses, bank, transported, out_shape, scale, params
        )

        if colour:
            cb_out, cr_out = inject_chroma((cb, cr), luma_in, residual, theta, out_shape, params)
            out = ycbcr_to_rgb(luma_out, cb_out, cr_out)
        else:
            out = luma_out

        if params.detail > 0.0:
            from ..ops import detail_enhance

            radius = max(1, int(round(4.0 * max(1.0, scale / 4.0))))
            out = detail_enhance(out, radius, 1e-3, params.detail)

        np.clip(out, 0.0, 1.0, out=out)

        identity_error: float | None = None
        if params.guard_identity and abs(scale - 1.0) < 1e-9:
            identity_error = float(np.max(np.abs(out - src)))

        report = UpthrumReport(
            scale=float(scale),
            shape_in=in_shape,
            shape_out=out_shape,
            device=device,
            bands=len(bank),
            colour=colour,
            band_summary=bank.summary(),
            mean_coherence=float(np.mean(kappa)),
            identity_error=identity_error,
            topology=info.get("topology"),
            seconds=time.perf_counter() - started,
        )
        return out.astype(np.float32), report

    def enhance_luminance(
        self, luma: np.ndarray, scale: float = 2.0
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Luminance-only path, for callers that do their own colour handling."""
        image, report = self.enhance(luma, scale)
        return image, report.to_dict()


def enhance(
    image: np.ndarray,
    scale: float = 2.0,
    params: UpthrumParams | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, Any]]:
    """One-shot convenience wrapper returning ``(image, report_dict)``."""
    engine = Upthrum(params=params, device=device)
    out, report = engine.enhance(image, scale)
    return out, report.to_dict()


def describe() -> dict[str, Any]:
    """Static description of the method, for ``/capabilities`` style endpoints."""
    return {
        "name": "UPTHRUM",
        "family": "phase reconstruction",
        "variable": "local phase (S^1), not intensity",
        "representation": "monogenic signal, per log-Gabor band",
        "transport": "amplitude-weighted coherent mean of unit phasors",
        "constraint": "0-dimensional persistent homology of sublevel sets",
        "learned_parameters": 0,
        "trained_on": None,
        "scales": "any >= 1",
        "identity_at_scale_one": True,
    }


__all__ = ["Upthrum", "UpthrumReport", "describe", "enhance", "resolve_device"]
