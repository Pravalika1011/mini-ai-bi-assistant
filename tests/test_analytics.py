"""
Automated unit tests for deterministic analytics functions in app/analytics.py.
Compares outputs against independently computed pandas ground truth metrics with temp_db.
"""

import pytest
import sqlite3

from app.analytics import (
    get_total_net_revenue,
    get_product_highest_net_revenue,
    get_monthly_revenue_trend,
    get_region_highest_net_revenue,
    compare_monthly_revenue,
    get_order_performance_metrics
)
from app.database import get_cleaning_audit_records


def test_total_net_revenue_calculation(temp_db, ground_truth_metrics):
    """Verifies total net revenue, gross revenue, discounts, and order counts."""
    data = get_total_net_revenue(temp_db)

    assert data["metric"] == "total_net_revenue"
    assert data["total_net_revenue"] == ground_truth_metrics["total_net_revenue"]
    assert data["total_gross_revenue"] == ground_truth_metrics["total_gross_revenue"]
    assert data["total_discounts"] == ground_truth_metrics["total_discounts"]
    assert data["completed_orders_count"] == ground_truth_metrics["completed_count"]
    assert data["returned_orders_count"] == ground_truth_metrics["returned_count"]
    assert data["returned_revenue_excluded"] == ground_truth_metrics["returned_amount"]
    assert data["total_records_in_db"] == ground_truth_metrics["total_records"]


def test_product_highest_net_revenue(temp_db, ground_truth_metrics):
    """Verifies top product by net revenue with exact equality against independent pandas baseline."""
    data = get_product_highest_net_revenue(temp_db)

    assert data["metric"] == "highest_net_revenue_product"
    top_product = data["top_product"]
    assert top_product["name"] == ground_truth_metrics["top_product_name"]
    assert top_product["net_revenue"] == ground_truth_metrics["top_product_net"]
    assert len(data["all_products"]) == 7


def test_monthly_revenue_trend(temp_db, ground_truth_metrics):
    """Verifies chronological monthly trend matches independent monthly sums."""
    data = get_monthly_revenue_trend(temp_db)

    assert data["metric"] == "monthly_revenue_trend"
    trend = data["trend"]
    assert len(trend) == 6

    # Verify chronological sorting
    months = [t["month_year"] for t in trend]
    assert months == sorted(months)

    # Check each month matches ground truth
    for t in trend:
        expected_net = ground_truth_metrics["monthly_trend"][t["month_year"]]
        assert t["net_revenue"] == expected_net

    # Sum must equal total completed net revenue
    monthly_sum = sum(t["net_revenue"] for t in trend)
    assert round(monthly_sum, 2) == ground_truth_metrics["total_net_revenue"]


def test_region_highest_net_revenue(temp_db, ground_truth_metrics):
    """Verifies regional rankings and exact sums (South: 411,200.00, North: 384,450.00, etc.)."""
    data = get_region_highest_net_revenue(temp_db)

    assert data["metric"] == "highest_net_revenue_region"
    top_region = data["top_region"]
    assert top_region["name"] == ground_truth_metrics["top_region_name"]
    assert top_region["net_revenue"] == ground_truth_metrics["top_region_net"]

    # Verify each region's exact total
    for r in data["all_regions"]:
        expected = ground_truth_metrics["region_totals"][r["region"]]
        assert r["net_revenue"] == expected


def test_monthly_comparison_valid(temp_db, ground_truth_metrics):
    """Verifies month-to-month comparison delta and percentage change."""
    data = compare_monthly_revenue("January 2026", "March 2026", temp_db)

    assert data["metric"] == "monthly_revenue_comparison"
    assert data["absolute_change"] == ground_truth_metrics["jan_mar_delta"]
    assert data["percentage_change"] == ground_truth_metrics["jan_mar_pct"]
    assert data["growth_direction"] == "increase"


def test_monthly_comparison_no_data_month(temp_db):
    """Verifies graceful handling when comparing against a month with no transactions."""
    data = compare_monthly_revenue("2026-01", "2026-09", temp_db)

    assert data["metric"] == "monthly_revenue_comparison"
    assert data["month_1"]["has_data"] is True
    assert data["month_2"]["has_data"] is False
    assert data["month_2"]["net_revenue"] == 0.0
    assert data["absolute_change"] < 0
    assert data["growth_direction"] == "decrease"


def test_monthly_comparison_invalid_inputs(temp_db):
    """Verifies validation error on unparseable month input."""
    res = compare_monthly_revenue("invalid_month", "2026-03", temp_db)
    assert "error" in res
    assert "Invalid month input" in res["error"]


def test_order_performance_metrics(temp_db, ground_truth_metrics):
    """Verifies AOV and units per order excluding Quantity=0 and Returned orders."""
    data = get_order_performance_metrics(temp_db)

    assert data["metric"] == "order_performance"
    assert data["average_order_value"] == ground_truth_metrics["aov"]
    assert data["average_units_per_order"] == ground_truth_metrics["avg_units"]
    assert data["eligible_order_count"] == ground_truth_metrics["completed_count"] - ground_truth_metrics["zero_qty_count"]


def test_missing_price_imputation_row_txn0051(temp_db):
    """Verifies that missing Unit_price in TXN0051 was imputed as 1200.0 and flagged."""
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM transactions WHERE transaction_id = 'TXN0051'").fetchone()
    conn.close()

    assert row is not None
    assert row["unit_price"] == 1200.0
    assert row["is_imputed"] == 1
    assert row["net_revenue"] == 6000.0  # 5 units * 1200 - 0 discount


def test_returned_order_row_txn0033(temp_db, ground_truth_metrics):
    """Verifies that TXN0033 (Monitor, Returned) is preserved but excluded from completed sales."""
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM transactions WHERE transaction_id = 'TXN0033'").fetchone()
    conn.close()

    assert row is not None
    assert row["status"] == "Returned"
    assert row["net_revenue"] == 19000.0  # 2 units * 9500

    # Ensure this 19000 was excluded from total completed sales
    total_data = get_total_net_revenue(temp_db)
    assert total_data["returned_revenue_excluded"] == 19000.0


def test_zero_quantity_order_row_txn0034(temp_db):
    """Verifies that TXN0034 (Quantity = 0) is retained in DB but excluded from basket metrics."""
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM transactions WHERE transaction_id = 'TXN0034'").fetchone()
    conn.close()

    assert row is not None
    assert row["quantity"] == 0
    assert row["net_revenue"] == 0.0

    perf = get_order_performance_metrics(temp_db)
    # Total completed orders is 50, but eligible is 49
    assert perf["eligible_order_count"] == 49


def test_cleaning_audit_trail_recorded(temp_db):
    """Verifies that audit records capture TXN0051 imputation and policy flags."""
    records = get_cleaning_audit_records(temp_db)
    assert len(records) >= 3

    txn_ids = [r["transaction_id"] for r in records]
    assert "TXN0051" in txn_ids
    assert "TXN0033" in txn_ids
    assert "TXN0034" in txn_ids
