"""Lightweight copy of the gazelle gaze estimation package."""

from . import gaze_prediction
from .model import get_gazelle_model

__all__ = [
    "gaze_prediction",
    "get_gazelle_model",
]
