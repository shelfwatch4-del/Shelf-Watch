"""Tests for the risk engine (src/risk_engine.py).

Implements test cases from SRS Appendix A, T-R-1 to T-R-6, and Task B4 specs:
1. Appendix A worked example (forecast_weekly 100, sigma 20, avg_units 90, stock_weeks 3,
   L 2, service_level 0.95, noise 1.1, 0.8, 0.6 -> forecast_lt 200, safety_stock ~46.5,
   reorder_point ~246.5, on_hand 297, 216, 162, classes LOW, MEDIUM, HIGH, doc ~20.8, 15.1, 11.3).
2. Advice on the same example (order_up_to ~356.99, recommended_order_qty 195, drops by 100
   when on_order is 100, revenue_at_risk = max(0, 200 - 162) x price).
3. stockout_probability: 0.5 when on_hand == forecast_lead_time; increases as on_hand
   decreases; sigma 0 gives 1 or 0.
4. Overrides: stock override replaces simulated stock, sets source "user", and changes class;
   lead time override applies only to that product.
5. Edge cases: forecast 0 (doc 999, no crash), sigma 0, lead time 6, service 0.99,
   threshold equality (not HIGH when equal to flt; LOW when equal to rop), empty input.
6. Determinism, and on_hand independent of other parameters.
7. Invalid params raise ValueError (lead 0, 7, 1.5; service 0.5, 1.0; stock 0, 9; unknown key; bool).
8. Output column order equals SRS list (ending with risk_level, action).
9. Action: inactive gives REVIEW_STOP with qty 0; HIGH gives ORDER_NOW; days_to_reorder 3
   with qty > 0 gives ORDER_SOON; ample stock gives HOLD; lifecycle=None works.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.risk_engine import (
    DEFAULTS,
    OUTPUT_COLUMNS,
    compute_risk,
    sensitivity_table,
    stock_noise,
    summarize,
    validate_params,
)


def _make_sample_df(rows: list[dict]) -> pd.DataFrame:
    """Helper to create a well-formed forecast DataFrame for tests."""
    df = pd.DataFrame(rows)
    if "description" not in df.columns:
        df["description"] = [f"Item {r['product_id']}" for r in rows]
    if "forecast_p10" not in df.columns:
        df["forecast_p10"] = df["forecast_weekly"] * 0.8
    if "forecast_p90" not in df.columns:
        df["forecast_p90"] = df["forecast_weekly"] * 1.2
    if "last_avg_price" not in df.columns:
        df["last_avg_price"] = 2.50
    return df


class TestValidateParams:
    """Tests for validate_params and invalid parameter handling (Item 7, T-R-4)."""

    def test_default_values(self) -> None:
        params = validate_params()
        assert params == DEFAULTS

    def test_valid_custom_params(self) -> None:
        custom = {
            "lead_time_weeks": 4,
            "service_level": 0.90,
            "initial_stock_weeks": 5.5,
            "seed": 123,
        }
        res = validate_params(custom)
        assert res == custom

    def test_boundary_values(self) -> None:
        # Min boundaries
        res_min = validate_params({
            "lead_time_weeks": 1,
            "service_level": 0.80,
            "initial_stock_weeks": 0.5,
        })
        assert res_min["lead_time_weeks"] == 1
        assert res_min["service_level"] == 0.80
        assert res_min["initial_stock_weeks"] == 0.5

        # Max boundaries
        res_max = validate_params({
            "lead_time_weeks": 6,
            "service_level": 0.99,
            "initial_stock_weeks": 8.0,
        })
        assert res_max["lead_time_weeks"] == 6
        assert res_max["service_level"] == 0.99
        assert res_max["initial_stock_weeks"] == 8.0

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown parameter: foo"):
            validate_params({"foo": 1})

    def test_bool_rejected(self) -> None:
        # Booleans must not pass as int or float
        with pytest.raises(ValueError, match="lead_time_weeks"):
            validate_params({"lead_time_weeks": True})
        with pytest.raises(ValueError, match="service_level"):
            validate_params({"service_level": False})
        with pytest.raises(ValueError, match="initial_stock_weeks"):
            validate_params({"initial_stock_weeks": True})
        with pytest.raises(ValueError, match="seed"):
            validate_params({"seed": False})

    def test_out_of_range_lead_time(self) -> None:
        # lead 0, 7 and 1.5 must raise ValueError
        with pytest.raises(ValueError, match="lead_time_weeks"):
            validate_params({"lead_time_weeks": 0})
        with pytest.raises(ValueError, match="lead_time_weeks"):
            validate_params({"lead_time_weeks": 7})
        with pytest.raises(ValueError, match="lead_time_weeks"):
            validate_params({"lead_time_weeks": 1.5})  # type: ignore

    def test_out_of_range_service_level(self) -> None:
        # service 0.5 and 1.0 must raise ValueError
        with pytest.raises(ValueError, match="service_level"):
            validate_params({"service_level": 0.5})
        with pytest.raises(ValueError, match="service_level"):
            validate_params({"service_level": 0.79})
        with pytest.raises(ValueError, match="service_level"):
            validate_params({"service_level": 1.0})

    def test_out_of_range_initial_stock(self) -> None:
        # initial stock 0 and 9 must raise ValueError
        with pytest.raises(ValueError, match="initial_stock_weeks"):
            validate_params({"initial_stock_weeks": 0})
        with pytest.raises(ValueError, match="initial_stock_weeks"):
            validate_params({"initial_stock_weeks": 0.4})
        with pytest.raises(ValueError, match="initial_stock_weeks"):
            validate_params({"initial_stock_weeks": 8.1})
        with pytest.raises(ValueError, match="initial_stock_weeks"):
            validate_params({"initial_stock_weeks": 9})

    def test_compute_risk_raises_on_invalid_params(self) -> None:
        df = _make_sample_df([{
            "product_id": "P1",
            "forecast_weekly": 100.0,
            "sigma_error": 10.0,
            "avg_weekly_units_train": 100.0,
            "last_avg_price": 2.0,
        }])
        with pytest.raises(ValueError, match="lead_time_weeks"):
            compute_risk(df, params={"lead_time_weeks": 0})


class TestAppendixAExample:
    """Tests based on SRS Appendix A worked example (Item 1, T-R-1)."""

    def test_appendix_a_worked_example(self) -> None:
        """Given: forecast_weekly=100, sigma_error=20, avg_weekly_units_train=90,

        initial_stock_weeks=3, L=2, service_level=0.95 (z ~ 1.645).
        Noise = [1.1, 0.8, 0.6] -> on_hand = [297, 216, 162].
        forecast_lead_time = 200, safety_stock about 46.5, reorder_point about 246.5.
        Risk classes: LOW, MEDIUM, HIGH.
        Days of cover: about 20.8, 15.1, 11.3.
        """
        rows = [
            {
                "product_id": f"P{i}",
                "forecast_weekly": 100.0,
                "sigma_error": 20.0,
                "avg_weekly_units_train": 90.0,
                "last_avg_price": 2.0,
            }
            for i in range(1, 4)
        ]
        df = _make_sample_df(rows)
        params = {
            "lead_time_weeks": 2,
            "service_level": 0.95,
            "initial_stock_weeks": 3.0,
        }
        noises = [1.1, 0.8, 0.6]

        res = compute_risk(df, params=params, _noise=noises)

        # 1. Expected forecast_lead_time = 100 x 2 = 200
        assert np.allclose(res["forecast_lead_time"], [200.0, 200.0, 200.0])

        # 2. safety_stock = 1.644853 x 20 x sqrt(2) ≈ 46.523 (about 46.5)
        assert np.allclose(res["safety_stock"], [46.523, 46.523, 46.523], atol=0.1)
        assert res.loc[0, "safety_stock"] == pytest.approx(46.5, abs=0.1)

        # 3. reorder_point = 200 + safety_stock ≈ 246.523 (about 246.5)
        assert np.allclose(res["reorder_point"], [246.523, 246.523, 246.523], atol=0.1)
        assert res.loc[0, "reorder_point"] == pytest.approx(246.5, abs=0.1)

        # 4. on_hand: 3 x 90 x noise = [297, 216, 162]
        assert np.allclose(res["on_hand"], [297.0, 216.0, 162.0])

        # 5. risk_level: LOW, MEDIUM, HIGH
        assert list(res["risk_level"]) == ["LOW", "MEDIUM", "HIGH"]

        # 6. days_of_cover: on_hand / (100 / 7) ≈ [20.79, 15.12, 11.34] (about 20.8, 15.1, 11.3)
        expected_doc = [297.0 / (100.0 / 7.0), 216.0 / (100.0 / 7.0), 162.0 / (100.0 / 7.0)]
        assert np.allclose(res["days_of_cover"], expected_doc, atol=0.1)
        assert res.loc[0, "days_of_cover"] == pytest.approx(20.8, abs=0.1)
        assert res.loc[1, "days_of_cover"] == pytest.approx(15.1, abs=0.1)
        assert res.loc[2, "days_of_cover"] == pytest.approx(11.3, abs=0.1)


class TestReplenishmentAdvice:
    """Tests for advice formulas: order_up_to, recommended_order_qty, revenue_at_risk (Item 2, T-R-5)."""

    def test_advice_calculations(self) -> None:
        """Advice on Appendix A example:

        order_up_to = 100 x 3 + 1.645 x 20 x sqrt(3) (about 356.99).
        With on_order 0 and noise 0.6 (on_hand 162):
        recommended_order_qty = ceil(order_up_to - 162) = 195.
        With on_order 100 it drops by 100 to 95.
        revenue_at_risk = max(0, 200 - 162) x last_avg_price.
        """
        rows = [
            {
                "product_id": "P_ZERO_ORDER",
                "forecast_weekly": 100.0,
                "sigma_error": 20.0,
                "avg_weekly_units_train": 90.0,
                "last_avg_price": 5.0,
            },
            {
                "product_id": "P_HUNDRED_ORDER",
                "forecast_weekly": 100.0,
                "sigma_error": 20.0,
                "avg_weekly_units_train": 90.0,
                "last_avg_price": 5.0,
            },
        ]
        df = _make_sample_df(rows)
        params = {
            "lead_time_weeks": 2,
            "service_level": 0.95,
            "initial_stock_weeks": 3.0,
        }
        noises = [0.6, 0.6]  # on_hand = 162 for both
        open_orders = {"P_ZERO_ORDER": 0, "P_HUNDRED_ORDER": 100}

        res = compute_risk(df, params=params, open_orders=open_orders, _noise=noises)

        # Expected order_up_to = 100 * (2 + 1) + norm.ppf(0.95) * 20 * sqrt(3) ≈ 356.9818 (about 356.99)
        z = norm.ppf(0.95)
        expected_order_up_to = 100.0 * 3.0 + z * 20.0 * math.sqrt(3.0)
        assert expected_order_up_to == pytest.approx(356.99, abs=0.02)

        # P_ZERO_ORDER: ceil(order_up_to - 162) = ceil(356.9818 - 162) = 195
        assert res.loc[0, "recommended_order_qty"] == math.ceil(expected_order_up_to - 162.0)
        assert res.loc[0, "recommended_order_qty"] == 195

        # P_HUNDRED_ORDER: with on_order 100, drops by 100 to 95
        assert res.loc[1, "recommended_order_qty"] == 95
        assert res.loc[0, "recommended_order_qty"] - res.loc[1, "recommended_order_qty"] == 100

        # revenue_at_risk = max(0, 200 - 162) * 5.0 = 38 * 5.0 = 190.0
        assert res.loc[0, "revenue_at_risk"] == max(0.0, 200.0 - 162.0) * 5.0
        assert res.loc[0, "revenue_at_risk"] == 190.0
        assert res.loc[1, "revenue_at_risk"] == 190.0

        # days_to_reorder: on_hand (162) < reorder_point (246.5) so max(0, ...) = 0.0
        assert res.loc[0, "days_to_reorder"] == 0.0


class TestStockoutProbability:
    """Tests for stockout_probability under normal assumption and edge cases (Item 3, T-R-5)."""

    def test_stockout_probability_properties(self) -> None:
        """- equals 0.5 when on_hand == forecast_lead_time

        - increases as on_hand decreases.
        """
        rows = [
            {
                "product_id": "P_EQ",
                "forecast_weekly": 100.0,
                "sigma_error": 20.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            },
            {
                "product_id": "P_LOW",
                "forecast_weekly": 100.0,
                "sigma_error": 20.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            },
            {
                "product_id": "P_HIGH",
                "forecast_weekly": 100.0,
                "sigma_error": 20.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            },
        ]
        df = _make_sample_df(rows)
        # L = 1 -> forecast_lead_time = 100
        # Overrides: P_EQ on_hand = 100, P_LOW on_hand = 80, P_HIGH on_hand = 120
        overrides = {
            "P_EQ": {"on_hand": 100.0},
            "P_LOW": {"on_hand": 80.0},
            "P_HIGH": {"on_hand": 120.0},
        }
        res = compute_risk(df, stock_overrides=overrides)

        # 1. equals 0.5 when on_hand == forecast_lead_time
        assert pytest.approx(res.loc[0, "stockout_probability"], 1e-4) == 0.5

        # 2. Increases as on_hand decreases: prob(80) > prob(100) > prob(120)
        assert res.loc[1, "stockout_probability"] > 0.5
        assert res.loc[2, "stockout_probability"] < 0.5
        assert (
            res.loc[1, "stockout_probability"]
            > res.loc[0, "stockout_probability"]
            > res.loc[2, "stockout_probability"]
        )

    def test_sigma_zero_stockout_probability(self) -> None:
        """When sigma_error is 0: gives 1 when on_hand < forecast_lead_time, else 0."""
        rows = [
            {
                "product_id": "P_BELOW",
                "forecast_weekly": 50.0,
                "sigma_error": 0.0,
                "avg_weekly_units_train": 50.0,
                "last_avg_price": 1.0,
            },
            {
                "product_id": "P_ABOVE",
                "forecast_weekly": 50.0,
                "sigma_error": 0.0,
                "avg_weekly_units_train": 50.0,
                "last_avg_price": 1.0,
            },
        ]
        df = _make_sample_df(rows)
        overrides = {
            "P_BELOW": {"on_hand": 49.9},
            "P_ABOVE": {"on_hand": 50.0},  # Exactly equal is NOT below
        }
        res = compute_risk(df, stock_overrides=overrides)
        assert res.loc[0, "stockout_probability"] == 1.0
        assert res.loc[1, "stockout_probability"] == 0.0


class TestOverrides:
    """Tests for per-product stock, lead time, and service level overrides (Item 4, T-R-6)."""

    def test_user_stock_override_changes_class_and_source(self) -> None:
        """A stock override replaces simulated stock, sets on_hand_source "user"

        and can change the class (e.g. from HIGH to LOW).
        """
        rows = [
            {
                "product_id": "P1",
                "forecast_weekly": 100.0,
                "sigma_error": 10.0,
                "avg_weekly_units_train": 10.0,  # Simulated stock: 3 * 10 * noise ~ 30 (< 100 -> HIGH)
                "last_avg_price": 1.0,
            },
            {
                "product_id": "P2",
                "forecast_weekly": 100.0,
                "sigma_error": 10.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            },
        ]
        df = _make_sample_df(rows)

        # Baseline without override
        res_baseline = compute_risk(df)
        assert res_baseline.loc[0, "on_hand_source"] == "simulated"
        assert res_baseline.loc[0, "risk_level"] == "HIGH"

        # Apply override on P1: set on_hand = 500 (>> rop ~116 -> LOW)
        overrides = {
            "P1": {"on_hand": 500.0},
        }
        res_override = compute_risk(df, stock_overrides=overrides)

        assert res_override.loc[0, "on_hand"] == 500.0
        assert res_override.loc[0, "on_hand_source"] == "user"
        assert res_override.loc[0, "risk_level"] == "LOW"
        # P1 class changed from HIGH to LOW
        assert res_override.loc[0, "risk_level"] != res_baseline.loc[0, "risk_level"]

        # P2 remains simulated
        assert res_override.loc[1, "on_hand_source"] == "simulated"

    def test_per_product_lead_time_override_isolated(self) -> None:
        """A per-product lead time override is applied only to that product."""
        rows = [
            {
                "product_id": "P1",
                "forecast_weekly": 100.0,
                "sigma_error": 10.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            },
            {
                "product_id": "P2",
                "forecast_weekly": 100.0,
                "sigma_error": 10.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            },
        ]
        df = _make_sample_df(rows)
        params = {"lead_time_weeks": 1, "service_level": 0.95}
        overrides = {
            "P1": {"lead_time_weeks": 3, "service_level": 0.85},
        }
        res = compute_risk(df, params=params, stock_overrides=overrides)

        # P1 has L=3 -> forecast_lead_time = 300
        assert res.loc[0, "forecast_lead_time"] == 300.0
        # P2 uses global L=1 -> forecast_lead_time = 100
        assert res.loc[1, "forecast_lead_time"] == 100.0


class TestThresholdEqualityAndEdgeCases:
    """Tests for threshold equality rules, zero forecast, empty input (Item 5, T-R-2)."""

    def test_threshold_equality_safer_class(self) -> None:
        """On-hand exactly equal to a threshold:

        - on_hand == forecast_lead_time -> MEDIUM (not HIGH)
        - on_hand == reorder_point -> LOW (not MEDIUM)
        """
        rows = [
            {
                "product_id": "P_EQ_FLT",
                "forecast_weekly": 100.0,
                "sigma_error": 10.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            },
            {
                "product_id": "P_EQ_ROP",
                "forecast_weekly": 100.0,
                "sigma_error": 10.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            },
        ]
        df = _make_sample_df(rows)
        # L=1, sl=0.95 -> flt=100, ss=1.644853*10*1 ≈ 16.44853, rop ≈ 116.44853
        flt = 100.0
        rop = 100.0 + norm.ppf(0.95) * 10.0
        overrides = {
            "P_EQ_FLT": {"on_hand": flt},
            "P_EQ_ROP": {"on_hand": rop},
        }
        res = compute_risk(df, stock_overrides=overrides)

        # Equality is NOT below threshold, falls in safer class
        assert res.loc[0, "risk_level"] == "MEDIUM"
        assert res.loc[0, "risk_level"] != "HIGH"
        assert res.loc[1, "risk_level"] == "LOW"
        assert res.loc[1, "risk_level"] != "MEDIUM"

    def test_zero_forecast_edge_case(self) -> None:
        """Forecast weekly = 0 gives days_of_cover = 999 and does not crash."""
        rows = [
            {
                "product_id": "P_ZERO",
                "forecast_weekly": 0.0,
                "sigma_error": 0.0,
                "avg_weekly_units_train": 50.0,
                "last_avg_price": 2.0,
            }
        ]
        df = _make_sample_df(rows)
        res = compute_risk(df)
        assert res.loc[0, "days_of_cover"] == 999.0
        assert res.loc[0, "risk_level"] == "LOW"
        assert res.loc[0, "stockout_probability"] == 0.0
        assert res.loc[0, "action"] == "HOLD"

    def test_sigma_zero_edge_case(self) -> None:
        """Sigma = 0 runs through compute_risk without error."""
        rows = [
            {
                "product_id": "P_SIGMA0",
                "forecast_weekly": 50.0,
                "sigma_error": 0.0,
                "avg_weekly_units_train": 50.0,
                "last_avg_price": 2.0,
            }
        ]
        df = _make_sample_df(rows)
        res = compute_risk(df)
        assert res.loc[0, "safety_stock"] == 0.0
        assert res.loc[0, "reorder_point"] == res.loc[0, "forecast_lead_time"]

    def test_lead_time_six_and_service_ninety_nine(self) -> None:
        """Lead time 6 and service level 0.99 execute cleanly."""
        rows = [
            {
                "product_id": "P1",
                "forecast_weekly": 50.0,
                "sigma_error": 10.0,
                "avg_weekly_units_train": 50.0,
                "last_avg_price": 2.0,
            }
        ]
        df = _make_sample_df(rows)
        params = {"lead_time_weeks": 6, "service_level": 0.99}
        res = compute_risk(df, params=params)
        assert res.loc[0, "forecast_lead_time"] == 300.0
        assert res.loc[0, "safety_stock"] == pytest.approx(norm.ppf(0.99) * 10.0 * math.sqrt(6), abs=0.01)

    def test_empty_forecast_df(self) -> None:
        """Empty input DataFrame returns empty DataFrame with all 18 output columns."""
        empty_df = pd.DataFrame(
            columns=[
                "product_id",
                "forecast_weekly",
                "forecast_p10",
                "forecast_p90",
                "sigma_error",
                "avg_weekly_units_train",
                "last_avg_price",
            ]
        )
        res = compute_risk(empty_df)
        assert res.empty
        assert list(res.columns) == OUTPUT_COLUMNS


class TestDeterminismAndNoise:
    """Tests for deterministic noise generation and parameter independence (Item 6, T-R-3)."""

    def test_stock_noise_deterministic(self) -> None:
        pids = ["85123A", "22423", "47566", "20725"]
        n1 = stock_noise(pids, seed=42)
        n2 = stock_noise(pids, seed=42)
        pd.testing.assert_series_equal(n1, n2)

        # Different seed differs
        n3 = stock_noise(pids, seed=99)
        assert not np.allclose(n1, n3)

    def test_compute_risk_deterministic_same_seed(self) -> None:
        rows = [
            {
                "product_id": f"P{i}",
                "forecast_weekly": 50.0,
                "sigma_error": 5.0,
                "avg_weekly_units_train": 40.0,
                "last_avg_price": 1.0,
            }
            for i in range(5)
        ]
        df = _make_sample_df(rows)
        res1 = compute_risk(df, params={"seed": 42})
        res2 = compute_risk(df, params={"seed": 42})
        pd.testing.assert_frame_equal(res1, res2)

    def test_on_hand_independent_of_lead_time_or_service_level(self) -> None:
        """Simulated on_hand depends only on initial_stock_weeks, avg_train, and noise.

        Changing lead_time_weeks or service_level must not change simulated on_hand.
        """
        rows = [
            {
                "product_id": f"P{i}",
                "forecast_weekly": 50.0,
                "sigma_error": 5.0,
                "avg_weekly_units_train": 40.0,
                "last_avg_price": 1.0,
            }
            for i in range(10)
        ]
        df = _make_sample_df(rows)

        res1 = compute_risk(df, params={"lead_time_weeks": 1, "service_level": 0.90})
        res2 = compute_risk(df, params={"lead_time_weeks": 4, "service_level": 0.99})

        np.testing.assert_array_almost_equal(res1["on_hand"], res2["on_hand"])


class TestOutputColumnOrder:
    """Tests that output column order matches SRS list (Item 8)."""

    def test_output_column_order(self) -> None:
        """Output column order equals the SRS list (ending with risk_level, action)."""
        expected_srs_order = [
            "product_id",
            "description",
            "forecast_weekly",
            "forecast_p10",
            "forecast_p90",
            "forecast_lead_time",
            "on_hand",
            "on_hand_source",
            "safety_stock",
            "reorder_point",
            "days_of_cover",
            "stockout_probability",
            "on_order",
            "recommended_order_qty",
            "days_to_reorder",
            "revenue_at_risk",
            "risk_level",
            "action",
        ]
        df = _make_sample_df([{
            "product_id": "P1",
            "forecast_weekly": 50.0,
            "sigma_error": 5.0,
            "avg_weekly_units_train": 40.0,
            "last_avg_price": 1.0,
        }])
        res = compute_risk(df)

        assert list(res.columns) == expected_srs_order
        assert list(res.columns)[-2:] == ["risk_level", "action"]


class TestActionRules:
    """Tests for action label determination (Item 9, SRS 6.2 item 10)."""

    def test_action_logic(self) -> None:
        """Action rules:

        - an inactive product gives REVIEW_STOP with quantity 0
        - a HIGH product gives ORDER_NOW
        - a product with days_to_reorder 3 and quantity > 0 gives ORDER_SOON
        - a product with ample stock gives HOLD
        """
        rows = [
            {
                "product_id": "P_INACTIVE",
                "forecast_weekly": 100.0,
                "sigma_error": 10.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            },
            {
                "product_id": "P_HIGH",
                "forecast_weekly": 100.0,
                "sigma_error": 10.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            },
            {
                "product_id": "P_SOON",
                "forecast_weekly": 70.0,  # daily demand = 10
                "sigma_error": 0.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            },
            {
                "product_id": "P_HOLD",
                "forecast_weekly": 70.0,
                "sigma_error": 0.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            },
        ]
        df = _make_sample_df(rows)
        # For P_SOON: flt=70, rop=70. on_hand=100. (on_hand - rop)/(70/7) = 30 / 10 = 3 days <= 7
        # order_up_to = 70 * 2 = 140. recommended_order_qty = 140 - 100 = 40 > 0.
        # For P_HOLD: flt=70, rop=70. on_hand=300. (on_hand - rop)/10 = 23 days > 7.
        overrides = {
            "P_INACTIVE": {"on_hand": 10.0},  # would be HIGH, but inactive
            "P_HIGH": {"on_hand": 10.0},      # on_hand < flt (100) -> HIGH
            "P_SOON": {"on_hand": 100.0},
            "P_HOLD": {"on_hand": 300.0},
        }
        lifecycle = {"P_INACTIVE": "inactive", "P_HIGH": "mature", "P_SOON": "mature", "P_HOLD": "mature"}

        res = compute_risk(df, stock_overrides=overrides, lifecycle=lifecycle)

        # 1. Inactive -> REVIEW_STOP with quantity 0
        assert res.loc[0, "action"] == "REVIEW_STOP"
        assert res.loc[0, "recommended_order_qty"] == 0

        # 2. HIGH -> ORDER_NOW
        assert res.loc[1, "risk_level"] == "HIGH"
        assert res.loc[1, "action"] == "ORDER_NOW"

        # 3. days_to_reorder 3 and quantity > 0 -> ORDER_SOON
        assert res.loc[2, "days_to_reorder"] == 3.0
        assert res.loc[2, "recommended_order_qty"] > 0
        assert res.loc[2, "action"] == "ORDER_SOON"

        # 4. Ample stock -> HOLD
        assert res.loc[3, "days_to_reorder"] > 7.0
        assert res.loc[3, "action"] == "HOLD"

    def test_without_lifecycle_data(self) -> None:
        """With lifecycle=None the rules still work smoothly."""
        rows = [
            {
                "product_id": "P1",
                "forecast_weekly": 100.0,
                "sigma_error": 10.0,
                "avg_weekly_units_train": 100.0,
                "last_avg_price": 1.0,
            }
        ]
        df = _make_sample_df(rows)
        res = compute_risk(df, lifecycle=None)
        assert res.loc[0, "action"] in {"ORDER_NOW", "ORDER_SOON", "HOLD"}


class TestMockDataIntegration:
    """Test compute_risk on actual mock next_week_forecast.csv."""

    def test_run_on_mock_c4(self) -> None:
        mock_path = Path("data/mock/next_week_forecast.csv")
        if not mock_path.exists():
            pytest.skip("data/mock/next_week_forecast.csv not found")

        df = pd.read_csv(mock_path, dtype={"product_id": str})
        res = compute_risk(df)

        assert len(res) == len(df)
        assert list(res.columns) == OUTPUT_COLUMNS
        assert not res.isna().any().any()
        assert set(res["risk_level"].unique()).issubset({"HIGH", "MEDIUM", "LOW"})
        assert set(res["action"].unique()).issubset({"ORDER_NOW", "ORDER_SOON", "REVIEW_STOP", "HOLD"})
        assert (res["days_of_cover"] >= 0).all()
        assert (res["stockout_probability"] >= 0).all()
        assert (res["stockout_probability"] <= 1.0).all()


class TestSummarize:
    """Tests for summarize function."""

    def test_empty_input_returns_zeros(self) -> None:
        empty_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        summary = summarize(empty_df)
        expected = {
            "n_high": 0,
            "n_medium": 0,
            "n_low": 0,
            "n_order_now": 0,
            "n_order_soon": 0,
            "n_review_stop": 0,
            "n_hold": 0,
            "share_at_risk": 0.0,
            "avg_days_of_cover": 0.0,
            "total_revenue_at_risk": 0.0,
        }
        assert summary == expected

    def test_summarize_matches_hand_count(self) -> None:
        # 5 products:
        # P1: HIGH, ORDER_NOW, days_of_cover=5.0, rev_at_risk=100.0
        # P2: MEDIUM, ORDER_SOON, days_of_cover=15.0, rev_at_risk=50.0
        # P3: MEDIUM, HOLD, days_of_cover=25.0, rev_at_risk=0.0
        # P4: LOW, HOLD, days_of_cover=35.0, rev_at_risk=0.0
        # P5: LOW, REVIEW_STOP, days_of_cover=999.0, rev_at_risk=0.0
        risk_df = pd.DataFrame(
            {
                "product_id": ["P1", "P2", "P3", "P4", "P5"],
                "risk_level": ["HIGH", "MEDIUM", "MEDIUM", "LOW", "LOW"],
                "action": ["ORDER_NOW", "ORDER_SOON", "HOLD", "HOLD", "REVIEW_STOP"],
                "days_of_cover": [5.0, 15.0, 25.0, 35.0, 999.0],
                "revenue_at_risk": [100.0, 50.0, 0.0, 0.0, 0.0],
            }
        )
        s = summarize(risk_df)

        assert s["n_high"] == 1
        assert s["n_medium"] == 2
        assert s["n_low"] == 2

        assert s["n_order_now"] == 1
        assert s["n_order_soon"] == 1
        assert s["n_review_stop"] == 1
        assert s["n_hold"] == 2

        # share_at_risk: (1 + 2) / 5 = 3 / 5 = 0.6
        assert s["share_at_risk"] == 0.6

        # avg_days_of_cover ignores 999.0: mean(5, 15, 25, 35) = 80 / 4 = 20.0
        assert s["avg_days_of_cover"] == 20.0

        # total_revenue_at_risk: 100 + 50 = 150.0
        assert s["total_revenue_at_risk"] == 150.0

    def test_summarize_all_zero_forecast_ignored(self) -> None:
        # All items have days_of_cover = 999.0
        risk_df = pd.DataFrame(
            {
                "product_id": ["P1", "P2"],
                "risk_level": ["LOW", "LOW"],
                "action": ["HOLD", "HOLD"],
                "days_of_cover": [999.0, 999.0],
                "revenue_at_risk": [0.0, 0.0],
            }
        )
        s = summarize(risk_df)
        assert s["avg_days_of_cover"] == 0.0
        assert s["share_at_risk"] == 0.0
        assert s["n_low"] == 2

    def test_summarize_on_mock_c4(self) -> None:
        mock_path = Path("data/mock/next_week_forecast.csv")
        df = pd.read_csv(mock_path, dtype={"product_id": str})
        risk_df = compute_risk(df)
        s = summarize(risk_df)

        assert s["n_high"] + s["n_medium"] + s["n_low"] == len(df)
        assert (
            s["n_order_now"] + s["n_order_soon"] + s["n_review_stop"] + s["n_hold"]
            == len(df)
        )
        assert 0.0 <= s["share_at_risk"] <= 1.0
        assert s["avg_days_of_cover"] > 0
        assert s["total_revenue_at_risk"] >= 0


class TestSensitivityTable:
    """Tests for sensitivity_table function."""

    def test_default_shape_and_columns(self) -> None:
        mock_path = Path("data/mock/next_week_forecast.csv")
        df = pd.read_csv(mock_path, dtype={"product_id": str})
        res = sensitivity_table(df)

        # 4 lead times x 3 service levels = 12 rows
        assert len(res) == 12
        expected_cols = ["lead_time_weeks", "service_level", "n_high", "n_medium", "n_low"]
        assert list(res.columns) == expected_cols

        # Check lead times and service levels
        assert sorted(res["lead_time_weeks"].unique()) == [1, 2, 3, 4]
        assert sorted(res["service_level"].unique()) == [0.90, 0.95, 0.99]

    def test_row_equals_compute_risk_defaults(self) -> None:
        mock_path = Path("data/mock/next_week_forecast.csv")
        df = pd.read_csv(mock_path, dtype={"product_id": str})
        res = sensitivity_table(df)

        default_row = res.loc[
            (res["lead_time_weeks"] == 1) & (res["service_level"] == 0.95)
        ]
        assert len(default_row) == 1

        # Compare against direct compute_risk with defaults
        risk_df = compute_risk(df, params=DEFAULTS)
        summary = summarize(risk_df)

        assert default_row.iloc[0]["n_high"] == summary["n_high"]
        assert default_row.iloc[0]["n_medium"] == summary["n_medium"]
        assert default_row.iloc[0]["n_low"] == summary["n_low"]

    def test_counts_sum_to_product_count_in_every_row(self) -> None:
        mock_path = Path("data/mock/next_week_forecast.csv")
        df = pd.read_csv(mock_path, dtype={"product_id": str})
        res = sensitivity_table(df)

        row_sums = res["n_high"] + res["n_medium"] + res["n_low"]
        assert (row_sums == len(df)).all()

    def test_high_counts_monotonic_in_lead_time(self) -> None:
        """HIGH counts must not decrease as lead time grows for fixed service level.

        Explanation:
        forecast_lead_time = forecast_weekly * L. Since forecast_weekly >= 0,
        forecast_lead_time is non-decreasing with L. Furthermore, simulated on-hand
        depends only on initial_stock_weeks, training average, and seed (independent of L).
        Therefore, any product with on_hand < forecast_weekly * L1 will strictly also
        have on_hand < forecast_weekly * L2 for L2 > L1.
        """
        mock_path = Path("data/mock/next_week_forecast.csv")
        df = pd.read_csv(mock_path, dtype={"product_id": str})
        res = sensitivity_table(df)

        for sl in [0.90, 0.95, 0.99]:
            sl_rows = res[res["service_level"] == sl].sort_values("lead_time_weeks")
            high_counts = sl_rows["n_high"].tolist()
            # Assert non-decreasing: high_counts[i] <= high_counts[i+1]
            for i in range(len(high_counts) - 1):
                assert high_counts[i] <= high_counts[i + 1], (
                    f"n_high decreased from L={sl_rows.iloc[i]['lead_time_weeks']} ({high_counts[i]}) "
                    f"to L={sl_rows.iloc[i+1]['lead_time_weeks']} ({high_counts[i+1]}) at service_level={sl}"
                )

    def test_empty_forecast_df(self) -> None:
        empty_df = pd.DataFrame(
            columns=[
                "product_id",
                "forecast_weekly",
                "forecast_p10",
                "forecast_p90",
                "sigma_error",
                "avg_weekly_units_train",
                "last_avg_price",
            ]
        )
        res = sensitivity_table(empty_df)
        assert len(res) == 12
        assert (res["n_high"] == 0).all()
        assert (res["n_medium"] == 0).all()
        assert (res["n_low"] == 0).all()

    def test_custom_lead_times_and_service_levels(self) -> None:
        mock_path = Path("data/mock/next_week_forecast.csv")
        df = pd.read_csv(mock_path, dtype={"product_id": str})
        res = sensitivity_table(
            df,
            lead_times=(2, 3),
            service_levels=(0.85, 0.95),
        )
        assert len(res) == 4
        assert sorted(res["lead_time_weeks"].unique()) == [2, 3]
        assert sorted(res["service_level"].unique()) == [0.85, 0.95]
