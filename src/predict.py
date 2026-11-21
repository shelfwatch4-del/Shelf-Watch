"""Prediction service and custom series forecasting interface.

SRS: FR-M-14 | Contract: C7
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.train import (
    MODELS_DIR,
    predict_custom,
    select_and_save_best_model,
)

__all__ = [
    "MODELS_DIR",
    "predict_custom",
    "select_and_save_best_model",
]
