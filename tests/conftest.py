"""
Shared fixtures and test configuration for RSG Mini AI Business Intelligence Assistant.
Computes ground truth metrics independently from raw CSV using Pandas.
"""

import os
import pytest
import pandas as pd
from typing import Dict, Any

from app.config import DEFAULT_CSV_PATH
from app.database import clean_and_load_data


@pytest.fixture
def temp_db(tmp_path):
    """Creates an isolated temporary SQLite database for test execution."""
    db_file = str(tmp_path / "test_sales_bi.db")
    result = clean_and_load_data(DEFAULT_CSV_PATH, db_file)
    assert result["status"] == "success"
    return db_file


@pytest.fixture(scope="session")
def ground_truth_metrics() -> Dict[str, Any]:
    """
    Computes all benchmark metrics independently from the raw CSV using Pandas.
    Provides verifiable, dynamic ground-truth values without hardcoded numbers.
    """
    df = pd.read_csv(DEFAULT_CSV_PATH)

    # Impute missing Unit_price with mode of other records for that product
    missing_mask = df["Unit_price"].isna() | (df["Unit_price"].astype(str).str.strip().isin(["", "nan", "NaN"]))
    for idx, row in df[missing_mask].iterrows():
        prod = row["Product"]
        other_prices = df[(df["Product"] == prod) & (~missing_mask)]["Unit_price"].dropna()
        imputed_val = float(other_prices.mode().iloc[0])
        df.loc[idx, "Unit_price"] = imputed_val

    df["Quantity"] = pd.to_numeric(df["Quantity"]).fillna(0).astype(int)
    df["Unit_price"] = pd.to_numeric(df["Unit_price"]).fillna(0.0)
    df["Discount_amount"] = pd.to_numeric(df["Discount_amount"]).fillna(0.0)

    df["parsed_date"] = pd.to_datetime(df["Order_date"], format="%d-%m-%Y")
    df["month_year"] = df["parsed_date"].dt.strftime("%Y-%m")
    df["gross_revenue"] = df["Quantity"] * df["Unit_price"]
    df["net_revenue"] = df["gross_revenue"] - df["Discount_amount"]

    # Completed orders
    df_completed = df[df["Status"] == "Completed"]
    df_returned = df[df["Status"] == "Returned"]

    total_net = float(df_completed["net_revenue"].sum())
    total_gross = float(df_completed["gross_revenue"].sum())
    total_discounts = float(df_completed["Discount_amount"].sum())

    # Product ranking
    prod_grouped = df_completed.groupby("Product")["net_revenue"].sum().sort_values(ascending=False)
    top_product_name = prod_grouped.index[0]
    top_product_net = float(prod_grouped.iloc[0])

    # Region ranking
    region_grouped = df_completed.groupby("Region")["net_revenue"].sum().sort_values(ascending=False)
    top_region_name = region_grouped.index[0]
    top_region_net = float(region_grouped.iloc[0])
    region_totals = {str(k): float(v) for k, v in region_grouped.items()}

    # Monthly trend
    month_grouped = df_completed.groupby("month_year")["net_revenue"].sum().sort_index()
    monthly_trend = {str(k): float(v) for k, v in month_grouped.items()}

    # Jan vs Mar
    jan_rev = monthly_trend["2026-01"]
    mar_rev = monthly_trend["2026-03"]
    jan_mar_delta = round(mar_rev - jan_rev, 2)
    jan_mar_pct = round((jan_mar_delta / jan_rev) * 100, 2)

    # Order performance (Completed and Quantity > 0)
    df_eligible = df_completed[df_completed["Quantity"] > 0]
    aov = float(df_eligible["net_revenue"].mean())
    avg_units = float(df_eligible["Quantity"].mean())

    return {
        "total_records": len(df),
        "completed_count": len(df_completed),
        "returned_count": len(df_returned),
        "returned_amount": float(df_returned["net_revenue"].sum()),
        "total_net_revenue": round(total_net, 2),
        "total_gross_revenue": round(total_gross, 2),
        "total_discounts": round(total_discounts, 2),
        "top_product_name": top_product_name,
        "top_product_net": round(top_product_net, 2),
        "top_region_name": top_region_name,
        "top_region_net": round(top_region_net, 2),
        "region_totals": region_totals,
        "monthly_trend": monthly_trend,
        "jan_mar_delta": jan_mar_delta,
        "jan_mar_pct": jan_mar_pct,
        "aov": round(aov, 2),
        "avg_units": round(avg_units, 2),
        "zero_qty_count": int((df["Quantity"] == 0).sum())
    }
