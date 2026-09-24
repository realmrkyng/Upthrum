"""Exception hierarchy for PixelBoost.

Every error raised by the library derives from :class:`PixelBoostError`, so a
server integration can catch one type and return a clean HTTP response.
"""

from __future__ import annotations


class PixelBoostError(Exception):
    """Base class for all PixelBoost failures."""


class ConfigError(PixelBoostError):
    """Raised when a configuration file or option value is invalid."""


class ImageReadError(PixelBoostError):
    """Raised when an input image cannot be decoded."""


class ImageWriteError(PixelBoostError):
    """Raised when an output image cannot be encoded or written."""


class BackendUnavailable(PixelBoostError):
    """Raised when a backend cannot be constructed.

    Typical causes: ``onnxruntime`` not installed, model file missing, or the
    requested execution provider is not present in the installed runtime.
    """


class ModelNotFound(BackendUnavailable):
    """Raised when a model file referenced by name or path does not exist."""


class ProviderUnavailable(BackendUnavailable):
    """Raised when a requested ONNX Runtime execution provider is missing."""


class OutOfMemory(PixelBoostError):
    """Raised when a tile or a whole-image buffer exceeds the memory budget."""


class UnsupportedScale(PixelBoostError):
    """Raised when a backend cannot produce the requested scale factor."""
