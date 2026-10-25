"""
Deterministic analytics engine for RSG Mini AI Business Intelligence Assistant.
Executes exact SQL aggregations against SQLite to guarantee mathematical fidelity
without numerical hallucination.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from app.config import DEFAULT_DB_PATH
from app.database import get_db_connection, normalize_month_input


def get_total_net_revenue(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Requirement 1: Total net revenue for completed transactions.
    Excludes returned orders in accordance with GAAP revenue-recognition standards.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT 
        COUNT(*) as completed_orders,
        SUM(quantity) as total_units_sold,
        SUM(gross_revenue) as total_gross_revenue,
        SUM(discount_amount) as total_discount_amount,
        SUM(net_revenue) as total_net_revenue
    FROM transactions
    WHERE status = 'Completed'
    """)
    completed_stats = dict(cursor.fetchone())

    cursor.execute("""
    SELECT 
        COUNT(*) as returned_orders,
        SUM(quantity) as returned_units,
        SUM(net_revenue) as returned_amount
    FROM transactions
    WHERE status = 'Returned'
    """)
    returned_stats = dict(cursor.fetchone())

    cursor.execute("SELECT COUNT(*) as total_records FROM transactions")
    total_records = cursor.fetchone()["total_records"]

    conn.close()

    total_net = completed_stats["total_net_revenue"] or 0.0
    total_gross = completed_stats["total_gross_revenue"] or 0.0
    total_disc = completed_stats["total_discount_amount"] or 0.0

    return {
        "metric": "total_net_revenue",
        "total_net_revenue": round(float(total_net), 2),
        "total_gross_revenue": round(float(total_gross), 2),
        "total_discounts": round(float(total_disc), 2),
        "completed_orders_count": int(completed_stats["completed_orders"]),
        "returned_orders_count": int(returned_stats["returned_orders"] or 0),
        "returned_revenue_excluded": round(float(returned_stats["returned_amount"] or 0.0), 2),
        "total_records_in_db": int(total_records),
        "policy_note": (
            "Includes only 'Completed' transactions. Returned transaction TXN0033 (19,000.00) "
            "is excluded from net sales in accordance with accounting revenue-recognition standards."
        )
    }


def get_product_highest_net_revenue(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Requirement 2: Product with the highest net revenue (completed transactions).
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT 
        product,
        category,
        COUNT(*) as order_count,
        SUM(quantity) as total_quantity,
        SUM(gross_revenue) as gross_revenue,
        SUM(discount_amount) as total_discount,
        SUM(net_revenue) as net_revenue
    FROM transactions
    WHERE status = 'Completed'
    GROUP BY product, category
    ORDER BY net_revenue DESC
    """)
    rows = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT SUM(net_revenue) as total FROM transactions WHERE status = 'Completed'")
    total_net_row = cursor.fetchone()
    total_net = total_net_row["total"] if total_net_row and total_net_row["total"] else 1.0

    conn.close()

    if not rows:
        return {"error": "No completed transactions found"}

    top_product = rows[0]
    top_share = round((top_product["net_revenue"] / total_net) * 100, 2)

    return {
        "metric": "highest_net_revenue_product",
        "top_product": {
            "name": top_product["product"],
            "category": top_product["category"],
            "net_revenue": round(float(top_product["net_revenue"]), 2),
            "units_sold": int(top_product["total_quantity"]),
            "order_count": int(top_product["order_count"]),
            "revenue_share_percentage": top_share
        },
        "all_products": [
            {
                "product": r["product"],
                "category": r["category"],
                "net_revenue": round(float(r["net_revenue"]), 2),
                "units_sold": int(r["total_quantity"]),
                "order_count": int(r["order_count"]),
                "revenue_share_percentage": round((r["net_revenue"] / total_net) * 100, 2)
            }
            for r in rows
        ],
        "policy_note": "Evaluated across all 'Completed' orders. Returned units (e.g. 2 Monitor units in TXN0033) are excluded."
    }


def get_monthly_revenue_trend(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Requirement 3: Chronological monthly revenue trend for completed transactions.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT 
        month_year,
        month_name,
        COUNT(*) as order_count,
        SUM(quantity) as total_quantity,
        SUM(gross_revenue) as gross_revenue,
        SUM(discount_amount) as discount_amount,
        SUM(net_revenue) as net_revenue
    FROM transactions
    WHERE status = 'Completed'
    GROUP BY month_year, month_name
    ORDER BY month_year ASC
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    trend_data = [
        {
            "month_year": r["month_year"],
            "month_name": r["month_name"],
            "net_revenue": round(float(r["net_revenue"]), 2),
            "gross_revenue": round(float(r["gross_revenue"]), 2),
            "discount_amount": round(float(r["discount_amount"]), 2),
            "order_count": int(r["order_count"]),
            "units_sold": int(r["total_quantity"])
        }
        for r in rows
    ]

    return {
        "metric": "monthly_revenue_trend",
        "trend": trend_data,
        "total_months": len(trend_data),
        "policy_note": "Trend aggregates completed transactions month-by-month chronologically."
    }


def get_region_highest_net_revenue(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Requirement 4: Region with the highest net revenue (completed transactions).
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT 
        region,
        COUNT(*) as order_count,
        SUM(quantity) as total_quantity,
        SUM(gross_revenue) as gross_revenue,
        SUM(discount_amount) as discount_amount,
        SUM(net_revenue) as net_revenue
    FROM transactions
    WHERE status = 'Completed'
    GROUP BY region
    ORDER BY net_revenue DESC
    """)
    rows = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT SUM(net_revenue) as total FROM transactions WHERE status = 'Completed'")
    total_net_row = cursor.fetchone()
    total_net = total_net_row["total"] if total_net_row and total_net_row["total"] else 1.0

    conn.close()

    if not rows:
        return {"error": "No completed transactions found"}

    top_region = rows[0]
    top_share = round((top_region["net_revenue"] / total_net) * 100, 2)

    return {
        "metric": "highest_net_revenue_region",
        "top_region": {
            "name": top_region["region"],
            "net_revenue": round(float(top_region["net_revenue"]), 2),
            "units_sold": int(top_region["total_quantity"]),
            "order_count": int(top_region["order_count"]),
            "revenue_share_percentage": top_share
        },
        "all_regions": [
            {
                "region": r["region"],
                "net_revenue": round(float(r["net_revenue"]), 2),
                "units_sold": int(r["total_quantity"]),
                "order_count": int(r["order_count"]),
                "revenue_share_percentage": round((r["net_revenue"] / total_net) * 100, 2)
            }
            for r in rows
        ],
        "policy_note": "Calculated on completed transactions. West region return of 19,000 (TXN0033) is excluded from completed net sales."
    }


def compare_monthly_revenue(
    month1: str,
    month2: str,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Requirement 5: Revenue comparison between two specified months,
    including absolute and percentage change.
    """
    norm1 = normalize_month_input(month1)
    norm2 = normalize_month_input(month2)

    if not norm1:
        return {
            "error": f"Invalid month input: '{month1}'. Please provide a valid month (e.g., '2026-01', 'January', or 'Jan 2026')."
        }
    if not norm2:
        return {
            "error": f"Invalid month input: '{month2}'. Please provide a valid month (e.g., '2026-03', 'March', or 'Mar 2026')."
        }

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    def get_month_stats(m_code: str) -> Dict[str, Any]:
        cursor.execute("""
        SELECT 
            month_year,
            month_name,
            COUNT(*) as order_count,
            SUM(quantity) as total_quantity,
            SUM(gross_revenue) as gross_revenue,
            SUM(discount_amount) as discount_amount,
            SUM(net_revenue) as net_revenue
        FROM transactions
        WHERE status = 'Completed' AND month_year = ?
        GROUP BY month_year, month_name
        """, (m_code,))
        row = cursor.fetchone()
        if row:
            return {
                "month_year": row["month_year"],
                "month_name": row["month_name"],
                "net_revenue": round(float(row["net_revenue"]), 2),
                "gross_revenue": round(float(row["gross_revenue"]), 2),
                "discount_amount": round(float(row["discount_amount"]), 2),
                "order_count": int(row["order_count"]),
                "units_sold": int(row["total_quantity"]),
                "has_data": True
            }
        else:
            try:
                display_name = datetime.strptime(m_code, "%Y-%m").strftime("%B %Y")
            except Exception:
                display_name = m_code
            return {
                "month_year": m_code,
                "month_name": display_name,
                "net_revenue": 0.0,
                "gross_revenue": 0.0,
                "discount_amount": 0.0,
                "order_count": 0,
                "units_sold": 0,
                "has_data": False
            }

    data_m1 = get_month_stats(norm1)
    data_m2 = get_month_stats(norm2)
    conn.close()

    rev1 = data_m1["net_revenue"]
    rev2 = data_m2["net_revenue"]
    abs_change = round(rev2 - rev1, 2)

    if rev1 > 0:
        pct_change = round(((rev2 - rev1) / rev1) * 100, 2)
    elif rev1 == 0 and rev2 > 0:
        pct_change = 100.0
    else:
        pct_change = 0.0

    return {
        "metric": "monthly_revenue_comparison",
        "month_1": data_m1,
        "month_2": data_m2,
        "absolute_change": abs_change,
        "percentage_change": pct_change,
        "growth_direction": "increase" if abs_change > 0 else ("decrease" if abs_change < 0 else "unchanged"),
        "policy_note": "Comparison is computed strictly on completed net sales for both target periods."
    }


def get_order_performance_metrics(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Performance KPIs (AOV, units per order).
    Excludes TXN0034 (Quantity = 0) and TXN0033 (Status = 'Returned').
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT 
        COUNT(*) as eligible_order_count,
        SUM(quantity) as total_units,
        SUM(net_revenue) as total_net_revenue,
        AVG(net_revenue) as average_order_value,
        AVG(quantity) as average_units_per_order
    FROM transactions
    WHERE status = 'Completed' AND quantity > 0
    """)
    stats = dict(cursor.fetchone())
    conn.close()

    return {
        "metric": "order_performance",
        "eligible_order_count": int(stats["eligible_order_count"] or 0),
        "total_units": int(stats["total_units"] or 0),
        "average_order_value": round(float(stats["average_order_value"] or 0.0), 2),
        "average_units_per_order": round(float(stats["average_units_per_order"] or 0.0), 2),
        "policy_note": "Excludes TXN0034 (Quantity=0) and TXN0033 (Returned) to prevent deflation of AOV and units/order."
    }
