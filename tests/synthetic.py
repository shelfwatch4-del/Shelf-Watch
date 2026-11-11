"""Deterministic synthetic raw transaction generator for pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_raw(n_products: int = 150, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic UCI-shaped raw transaction DataFrame.

    Mimics the UCI Online Retail schema and date range (2010-12-01 to
    2011-12-09), with no Saturday trading, an empty Christmas closure week
    (2010-12-27), ~25% missing CustomerIDs, high-volume top-100 products,
    low-volume runner-ups/rare products, and injected junk/outlier rows.
    """
    rng = np.random.default_rng(seed)

    all_days = pd.date_range("2010-12-01", "2011-12-09", freq="D")
    # Exclude Saturdays (weekday == 5) and shop-closed week starting 2010-12-27
    closed_start = pd.Timestamp("2010-12-27")
    closed_end = pd.Timestamp("2011-01-02")
    open_days = all_days[
        (all_days.weekday != 5)
        & ~((all_days >= closed_start) & (all_days <= closed_end))
    ]

    # Map each open day to its Monday week_start for structured sampling
    day_week = (open_days - pd.to_timedelta(open_days.weekday, unit="D")).normalize()
    full_weeks = pd.date_range("2010-12-06", "2011-11-28", freq="W-MON")
    train_weeks = full_weeks[:44]
    test_weeks = full_weeks[44:]
    open_train_weeks = [w for w in train_weeks if w != closed_start]

    days_by_week: dict[pd.Timestamp, pd.DatetimeIndex] = {
        w: open_days[day_week == w] for w in day_week.unique()
    }
    partial_days = open_days[
        (open_days < pd.Timestamp("2010-12-06"))
        | (open_days > pd.Timestamp("2011-12-04"))
    ]

    rows: list[dict[str, object]] = []
    invoice_counter = 536365

    for idx in range(n_products):
        stock_code = f"PROD{idx:03d}"
        description = f"SYNTHETIC PRODUCT {idx:03d}"
        base_price = round(float(rng.uniform(0.85, 12.50)), 2)

        if idx < 100:
            # High-volume products: >= 35 active training weeks, skip a few for zero-demand gaps
            n_train_active = int(rng.integers(36, len(open_train_weeks)))
            chosen_train = set(
                rng.choice(open_train_weeks, size=n_train_active, replace=False)
            )
            # Ensure first open week and last test week are active so gaps lie inside range
            chosen_train.add(open_train_weeks[0])
            chosen_test = set(
                rng.choice(test_weeks, size=int(rng.integers(6, len(test_weeks) + 1)), replace=False)
            )
            chosen_test.add(test_weeks[-1])
            active_weeks = sorted(chosen_train | chosen_test)

            # Also emit occasional partial-week rows on the first few products
            if idx < 10 and len(partial_days) > 0:
                for p_day in partial_days[:2]:
                    invoice_counter += 1
                    rows.append(
                        {
                            "InvoiceNo": str(invoice_counter),
                            "StockCode": stock_code,
                            "Description": description,
                            "Quantity": int(rng.integers(5, 20)),
                            "InvoiceDate": p_day + pd.Timedelta(hours=10),
                            "UnitPrice": base_price,
                            "CustomerID": (
                                np.nan
                                if rng.random() < 0.25
                                else float(rng.integers(12000, 18000))
                            ),
                            "Country": "United Kingdom",
                        }
                    )

            for w in active_weeks:
                w_days = days_by_week[pd.Timestamp(w)]
                tx_day = pd.Timestamp(rng.choice(w_days))
                invoice_counter += 1
                rows.append(
                    {
                        "InvoiceNo": str(invoice_counter),
                        "StockCode": stock_code,
                        "Description": description,
                        "Quantity": int(rng.integers(8, 35)),
                        "InvoiceDate": tx_day + pd.Timedelta(hours=int(rng.integers(8, 17))),
                        "UnitPrice": base_price,
                        "CustomerID": (
                            np.nan
                            if rng.random() < 0.25
                            else float(rng.integers(12000, 18000))
                        ),
                        "Country": "United Kingdom",
                    }
                )

        elif idx < 110:
            # Eligible runner-ups (>= 30 active train weeks) with very low volume
            chosen_train = rng.choice(open_train_weeks, size=34, replace=False)
            chosen_test = rng.choice(test_weeks, size=4, replace=False)
            for w in sorted(list(chosen_train) + list(chosen_test)):
                w_days = days_by_week[pd.Timestamp(w)]
                tx_day = pd.Timestamp(rng.choice(w_days))
                invoice_counter += 1
                rows.append(
                    {
                        "InvoiceNo": str(invoice_counter),
                        "StockCode": stock_code,
                        "Description": description,
                        "Quantity": 1,
                        "InvoiceDate": tx_day + pd.Timedelta(hours=11),
                        "UnitPrice": base_price,
                        "CustomerID": (
                            np.nan
                            if rng.random() < 0.25
                            else float(rng.integers(12000, 18000))
                        ),
                        "Country": "United Kingdom",
                    }
                )
        else:
            # Rare tail products
            n_rare = int(rng.integers(3, 12))
            chosen_weeks = rng.choice(open_train_weeks, size=n_rare, replace=False)
            for w in chosen_weeks:
                w_days = days_by_week[pd.Timestamp(w)]
                tx_day = pd.Timestamp(rng.choice(w_days))
                invoice_counter += 1
                rows.append(
                    {
                        "InvoiceNo": str(invoice_counter),
                        "StockCode": stock_code,
                        "Description": description,
                        "Quantity": int(rng.integers(1, 4)),
                        "InvoiceDate": tx_day + pd.Timedelta(hours=12),
                        "UnitPrice": base_price,
                        "CustomerID": (
                            np.nan
                            if rng.random() < 0.25
                            else float(rng.integers(12000, 18000))
                        ),
                        "Country": "United Kingdom",
                    }
                )

    # Inject required known junk & outlier rows
    junk_rows: list[dict[str, object]] = [
        {
            "InvoiceNo": "C999001",
            "StockCode": "PROD000",
            "Description": "SYNTHETIC PRODUCT 000",
            "Quantity": -5,
            "InvoiceDate": pd.Timestamp("2011-03-14 09:30:00"),
            "UnitPrice": 2.50,
            "CustomerID": 15000.0,
            "Country": "United Kingdom",
        },
        {
            "InvoiceNo": "590001",
            "StockCode": "POST",
            "Description": "POSTAGE",
            "Quantity": 1,
            "InvoiceDate": pd.Timestamp("2011-03-15 10:00:00"),
            "UnitPrice": 18.00,
            "CustomerID": 15001.0,
            "Country": "United Kingdom",
        },
        {
            "InvoiceNo": "590002",
            "StockCode": "DOT",
            "Description": "DOTCOM POSTAGE",
            "Quantity": 1,
            "InvoiceDate": pd.Timestamp("2011-03-16 10:15:00"),
            "UnitPrice": 12.00,
            "CustomerID": np.nan,
            "Country": "United Kingdom",
        },
        {
            "InvoiceNo": "590003",
            "StockCode": "99999Z",
            "Description": "ZERO PRICE JUNK",
            "Quantity": 10,
            "InvoiceDate": pd.Timestamp("2011-03-17 11:00:00"),
            "UnitPrice": 0.0,
            "CustomerID": 15002.0,
            "Country": "United Kingdom",
        },
        {
            "InvoiceNo": "590004",
            "StockCode": "PROD000",
            "Description": "SYNTHETIC PRODUCT 000",
            "Quantity": 80995,
            "InvoiceDate": pd.Timestamp("2011-04-04 14:00:00"),
            "UnitPrice": 2.50,
            "CustomerID": 15003.0,
            "Country": "United Kingdom",
        },
        {
            "InvoiceNo": "590005",
            "StockCode": "gift_0001_20",
            "Description": "Dotcomgiftshop Gift Voucher 20",
            "Quantity": 1,
            "InvoiceDate": pd.Timestamp("2011-04-05 15:00:00"),
            "UnitPrice": 17.02,
            "CustomerID": np.nan,
            "Country": "United Kingdom",
        },
    ]
    rows.extend(junk_rows)

    df = pd.DataFrame(rows)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype(str)
    df["Quantity"] = df["Quantity"].astype("int64")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["UnitPrice"] = df["UnitPrice"].astype(float)
    df["CustomerID"] = df["CustomerID"].astype(float)
    df["Country"] = df["Country"].astype(str)
    return df

