"""Geometric centerline and sampling-line package for segmented leaf masks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from leafsampling.pipeline import CenterlineSamplingConfig

__all__ = ["CenterlineSamplingConfig"]


def __getattr__(name: str) -> Any:
    """Load image-pipeline exports only when they are requested."""
    if name == "CenterlineSamplingConfig":
        from leafsampling.pipeline import CenterlineSamplingConfig

        return CenterlineSamplingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
