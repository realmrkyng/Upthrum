"""UPTHRUM backend -- phase reconstruction as the upscaling stage.

Every other backend in this package answers the same question in the same
variable: *what intensity should this output pixel have?* A fixed kernel
(``classical``), a regression network (``torch``), a compiled graph
(``onnx``) -- the target is always a number, and the failure mode is always the
same. The map from coarse intensity samples to fine intensity values is not
injective, so the fine structure that produced those samples is genuinely
undetermined and the method must either blur or invent.

This backend changes the variable. :mod:`pixelboost.upthrum` reconstructs the
**phase** of each log-Gabor band of the monogenic signal, transports it to the
output lattice as a coherent mean of unit phasors, and re-derives intensity
afterwards. Translation along a wavefront is a phase shift, so sub-pixel
position is *determined* rather than guessed, and the amplitude attenuation
that constitutes interpolation blur cannot arise because phases are averaged as
unit vectors instead of intensities as scalars. A one-dimensional persistent
homology of the reconstructed luma then removes critical points the source did
not have, which is what keeps noise from being promoted into texture.

Practical properties, which are the reason this is a separate backend rather
than another entry in ``classical``:

* **No model file, no training, no download.** The backend is constructible on
  a bare server, exactly like ``classical``.
* **Any scale factor.** ``native_scale`` is 1 because the transport is defined
  for an arbitrary output lattice, not for an integer multiplier. The pipeline
  therefore calls :meth:`process` once with the true fractional scale, and
  ``--width``/``--height``/``--longest-side`` need no separate resampling step.
* **Not tiled.** Band analysis is non-local: tiling the input would break the
  FFT into blocks and put a seam through every band. Whole-image analysis is
  the correct mode, so memory scales with input area rather than with ``--tile``.
* **Opt-in.** ``auto`` never selects it. UPTHRUM is markedly slower than
  ``classical`` per megapixel and the two have different characters on
  different content; the choice belongs to the operator, not to a heuristic.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pixelboost.backends.base import Backend
from pixelboost.errors import UnsupportedScale
from pixelboost.upthrum import Upthrum, UpthrumParams, describe, resolve_device

log = logging.getLogger("pixelboost.backends.upthrum")

DEFAULT_WARMUP_TILE = 64


class UpthrumBackend(Backend):
    """Phase-domain super-resolution under a topological constraint."""

    name = "upthrum"
    native_scale = 1
    requires_tiling = False
    handles_detail = True

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        *,
        params: UpthrumParams | None = None,
        device: str = "auto",
        **overrides: Any,
    ) -> None:
        base = params or UpthrumParams()
        if overrides:
            merged = base.to_dict()
            merged.update({k: v for k, v in overrides.items() if k in merged})
            base = UpthrumParams.from_dict(merged)

        self.params = base
        self.device_request = device
        resolved = resolve_device(device)

        super().__init__(model=None, provider=resolved)
        self.device = resolved
        self._engine = Upthrum(params=self.params, device=device)
        self.last_report: dict[str, Any] | None = None

    def process(self, tile: np.ndarray, scale: float) -> np.ndarray:
        """Reconstruct ``tile`` at ``scale``.

        ``tile`` is ``(h, w, 3)`` float32 in ``[0, 1]``; the return value is
        ``(round(h * scale), round(w * scale), 3)`` float32 in ``[0, 1]``. The
        pipeline is responsible for the alpha plane and for any exact-size fit,
        so nothing here touches either.
        """
        if scale < 1.0:
            raise UnsupportedScale(
                f"UPTHRUM reconstructs upward only; got scale={scale:g}. "
                f"Downscaling is a resampling problem, not a reconstruction one."
            )
        out, report = self._engine.enhance(tile, scale)
        self.last_report = report.to_dict()
        return out

    def warmup(self, tile: int = DEFAULT_WARMUP_TILE) -> None:
        """Run one throwaway reconstruction at scale 1.

        There is no CUDA context to build here unless torch is present, but the
        first call still pays for the numpy FFT plans and for building the
        filter bank twice per band, and a scale-1 pass is also a free assertion:
        the transport must reproduce its input exactly, so a coordinate bug
        surfaces at start-up instead of in a user's photograph.
        """
        probe = np.zeros((tile, tile, 3), dtype=np.float32)
        self.process(probe, 1.0)

    def info(self) -> dict[str, Any]:
        data = super().info()
        data.update(
            {
                "device_request": self.device_request,
                "params": self.params.to_dict(),
                "method": describe(),
                "note": "phase reconstruction; no learned parameters, no model file",
            }
        )
        if self.last_report is not None:
            data["last_report"] = self.last_report
        return data

    def close(self) -> None:
        """Nothing to release.

        Unlike the onnx and torch backends there is no session, no device
        allocation and no mapped weight file -- the engine is stateless numpy
        plus an optional temporary CUDA buffer that goes out of scope with each
        call. Overridden explicitly so that the contract is answered rather
        than merely inherited by accident.
        """


__all__ = ["UpthrumBackend"]
