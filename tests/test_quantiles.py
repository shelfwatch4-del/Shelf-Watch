"""Unit tests for src/quantiles.py (Task M10)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import pytest

from src.quantiles import (
    compute_conformal_correction,
    empirical_coverage,
    pinball_loss,
    sort_and_clip_intervals,
)


def test_sort_and_clip_intervals_ordering_holds_for_every_row() -> None:
    """Ordering 0 <= p10 <= point <= p90 must hold for every row, including inverted and negative inputs."""
    # Deliberately pathological cases:
    # row 0: regular order
    # row 1: p10 > point > p90 (inverted)
    # row 2: negative values
    # row 3: point smaller than p10
    # row 4: point larger than p90
    p10_raw = np.array([10.0, 50.0, -20.0, 30.0, 10.0])
    point_raw = np.array([20.0, 30.0, -5.0, 15.0, 70.0])
    p90_raw = np.array([30.0, 10.0, 5.0, 40.0, 50.0])

    p10, pt, p90 = sort_and_clip_intervals(p10_raw, point_raw, p90_raw)

    assert len(p10) == len(pt) == len(p90) == 5

    # Check non-negativity
    assert np.all(p10 >= 0.0)
    assert np.all(pt >= 0.0)
    assert np.all(p90 >= 0.0)

    # Check ordering
    assert np.all(p10 <= pt)
    assert np.all(pt <= p90)

    # Specific row checks
    # Row 0: [10, 20, 30] -> [10, 20, 30]
    np.testing.assert_allclose([p10[0], pt[0], p90[0]], [10.0, 20.0, 30.0])
    # Row 1: [50, 30, 10] sorted -> [10, 30, 50]
    np.testing.assert_allclose([p10[1], pt[1], p90[1]], [10.0, 30.0, 50.0])
    # Row 2: [-20, -5, 5] sorted -> [-20, -5, 5] clipped at 0 -> [0, 0, 5]
    np.testing.assert_allclose([p10[2], pt[2], p90[2]], [0.0, 0.0, 5.0])


def test_empirical_coverage_matches_hand_computed_example() -> None:
    """Coverage function must match exact hand-computed example.

    y   = [10, 20, 30, 40, 50]
    p10 = [ 8, 25, 30, 35, 45]
    p90 = [12, 30, 35, 42, 48]
    Row 0: 10 in [8, 12]   -> True
    Row 1: 20 in [25, 30]  -> False (below lower)
    Row 2: 30 in [30, 35]  -> True  (exact boundary)
    Row 3: 40 in [35, 42]  -> True
    Row 4: 50 in [45, 48]  -> False (above upper)
    Inside = 3 of 5 -> Coverage = 0.60 (60.0%)
    """
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    p10 = np.array([8.0, 25.0, 30.0, 35.0, 45.0])
    p90 = np.array([12.0, 30.0, 35.0, 42.0, 48.0])

    cov = empirical_coverage(y, p10, p90)
    assert cov == pytest.approx(0.60)


def test_pinball_loss_matches_hand_computed_example() -> None:
    """Pinball loss must match exact hand-computed math for alpha = 0.1 and 0.9.

    For alpha = 0.1:
      y = [10.0, 20.0], q = [12.0, 16.0]
      Row 0: e = y - q = -2.0 (< 0). Loss = (0.1 - 1) * (-2.0) = 0.9 * 2.0 = 1.8
      Row 1: e = y - q = +4.0 (>= 0). Loss = 0.1 * 4.0 = 0.4
      Mean loss = (1.8 + 0.4) / 2 = 1.1

    For alpha = 0.9:
      y = [10.0, 20.0], q = [12.0, 16.0]
      Row 0: e = -2.0 (< 0). Loss = (0.9 - 1) * (-2.0) = 0.1 * 2.0 = 0.2
      Row 1: e = +4.0 (>= 0). Loss = 0.9 * 4.0 = 3.6
      Mean loss = (0.2 + 3.6) / 2 = 1.9
    """
    y = np.array([10.0, 20.0])
    q = np.array([12.0, 16.0])

    loss_10 = pinball_loss(y, q, alpha=0.1)
    loss_90 = pinball_loss(y, q, alpha=0.9)

    assert loss_10 == pytest.approx(1.1)
    assert loss_90 == pytest.approx(1.9)


def test_conformal_correction_achieves_target_coverage() -> None:
    """Conformal residual correction shifts empirical coverage to target ~80%."""
    rng = np.random.RandomState(42)
    y_val = rng.normal(loc=100.0, scale=20.0, size=1000)
    # Intentionally narrow intervals centered around mean that under-cover (e.g. ~30% coverage)
    p10_val = np.full(1000, 92.0)
    p90_val = np.full(1000, 108.0)

    initial_cov = empirical_coverage(y_val, p10_val, p90_val)
    assert initial_cov < 0.40

    c = compute_conformal_correction(y_val, p10_val, p90_val, target_coverage=0.80)
    calibrated_cov = empirical_coverage(y_val, p10_val - c, p90_val + c)

    assert calibrated_cov == pytest.approx(0.80, abs=0.02)


def test_saved_models_reload_and_reproduce_predictions(tmp_path: Path) -> None:
    """Saved models reload in fresh objects and reproduce predictions bit-for-bit."""
    X_train = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [10.0, 20.0, 15.0, 25.0, 30.0, 35.0],
        }
    )
    y_train = np.array([12.0, 24.0, 22.0, 33.0, 41.0, 48.0])

    q10 = LGBMRegressor(objective="quantile", alpha=0.1, random_state=42, verbosity=-1, deterministic=True, n_jobs=1)
    q90 = LGBMRegressor(objective="quantile", alpha=0.9, random_state=42, verbosity=-1, deterministic=True, n_jobs=1)

    q10.fit(X_train, y_train)
    q90.fit(X_train, y_train)

    X_test = pd.DataFrame(
        {
            "f1": [2.5, 4.5],
            "f2": [18.0, 32.0],
        }
    )
    preds_q10_orig = q10.predict(X_test)
    preds_q90_orig = q90.predict(X_test)

    # Save
    q10_file = tmp_path / "q10.joblib"
    q90_file = tmp_path / "q90.joblib"
    joblib.dump(q10, q10_file)
    joblib.dump(q90, q90_file)

    # Reload in fresh variables
    reloaded_q10 = joblib.load(q10_file)
    reloaded_q90 = joblib.load(q90_file)

    preds_q10_reloaded = reloaded_q10.predict(X_test)
    preds_q90_reloaded = reloaded_q90.predict(X_test)

    np.testing.assert_allclose(preds_q10_orig, preds_q10_reloaded)
    np.testing.assert_allclose(preds_q90_orig, preds_q90_reloaded)
