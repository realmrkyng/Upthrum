"""UPTHRUM -- phase-domain super-resolution under a topological constraint.

The sub-package is laid out in the order the algorithm runs, so reading the files
in this order is reading the method:

    params.py       every knob, tied to the equation it belongs to
    filters.py      frequency-domain operators (Riesz, exact derivative, band bank)
    transform.py    monogenic decomposition -> amplitude, phase, orientation, grad(phi)
    transport.py    sub-pixel phase transport   <- the core idea
    topology.py     0-D persistent homology     <- the constraint
    reconstruct.py  reassembly, topological repair, chroma
    engine.py       public API
    torch_ops.py    optional CUDA path for the FFT-bound analysis

Nothing here is trained, nothing is downloaded, and there is no model file. The
full method is the arithmetic in ``transport.py`` plus the constraint in
``topology.py``.

    from pixelboost.upthrum import Upthrum
    engine = Upthrum()
    image, report = engine.enhance(photo, scale=4)
    print(report.summary())
"""

from __future__ import annotations

from typing import Any

from .engine import Upthrum, UpthrumReport, describe, enhance, resolve_device
from .params import UpthrumParams

__all__ = [
    "Upthrum",
    "UpthrumParams",
    "UpthrumReport",
    "describe",
    "enhance",
    "resolve_device",
]


def __getattr__(name: str) -> Any:
    """Lazy access to the internal stages, for tests and for interactive work."""
    stages = {
        "filters": "filters",
        "transform": "transform",
        "transport": "transport",
        "topology": "topology",
        "reconstruct": "reconstruct",
        "torch_ops": "torch_ops",
    }
    if name in stages:
        import importlib

        module = importlib.import_module(f"pixelboost.upthrum.{stages[name]}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'pixelboost.upthrum' has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | {"filters", "transform", "transport", "topology", "reconstruct", "torch_ops"})
