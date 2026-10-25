"""
Database and data ingestion module for RSG Mini AI Business Intelligence Assistant.
Handles CSV ingestion, data validation, statistical mode imputation, policy tagging,
audit trail persistence, and SQLite table creation.
"""

import os
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from app.config import DEFAULT_CSV_PATH, DEFAULT_DB_PATH


VALID_STATUSES = {"completed", "returned"}


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates a connection to the SQLite database with row access by column name."""
    target_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_month_input(month_input: str) -> Optional[str]:
    """
    Normalizes diverse month inputs into standard 'YYYY-MM' format.
    Handles 'YYYY-MM', 'MM-YYYY', 'Month YYYY', 'Month', '1'..'12'.
    Assumes 2026 for standalone month names as per the business dataset.
    """
    if not month_input or not str(month_input).strip():
        return None

    cleaned = str(month_input).strip()

    # Pattern: YYYY-MM
    try:
        dt = datetime.strptime(cleaned, "%Y-%m")
        return dt.strftime("%Y-%m")
    except ValueError:
        pass

    # Pattern: MM-YYYY or DD-MM-YYYY or YYYY/MM
    for fmt in ("%m-%Y", "%d-%m-%Y", "%Y/%m", "%m/%Y"):
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%Y-%m")
        except ValueError:
            pass

    # Month names with year (e.g., 'January 2026', 'Jan 2026')
    for fmt in ("%B %Y", "%b %Y", "%B, %Y", "%b, %Y"):
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%Y-%m")
        except ValueError:
            pass

    # Month name only (e.g., 'January', 'Jan') -> default to 2026
    for fmt in ("%B", "%b"):
        try:
            dt = datetime.strptime(cleaned, fmt)
            return f"2026-{dt.strftime('%m')}"
        except ValueError:
            pass

    # Numeric month '1' to '12'
    if cleaned.isdigit():
        val = int(cleaned)
        if 1 <= val <= 12:
            return f"2026-{val:02d}"

    return None


def clean_and_load_data(
    csv_path: Optional[str] = None,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ingests raw sales records from CSV, performs comprehensive validation and imputation,
    records an audit trail for data modifications and policy exclusions, and saves tables:
    - 'transactions': Cleaned records with gross/net revenue calculations
    - 'data_cleaning_audit': Log of validations, imputations, and policy tags
    """
    src_csv = csv_path or DEFAULT_CSV_PATH
    tgt_db = db_path or DEFAULT_DB_PATH

    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Sales dataset CSV not found at: {src_csv}")

    # Read raw CSV
    df_raw = pd.read_csv(src_csv)
    audit_records: List[Dict[str, Any]] = []

    validation_summary = {
        "missing_fields": 0,
        "bad_dates": 0,
        "negative_quantities": 0,
        "duplicate_ids": 0,
        "unknown_statuses": 0,
        "imputed_unit_prices": 0
    }

    # 1. Check duplicate Transaction_Ids
    dup_mask = df_raw["Transaction_Id"].duplicated(keep=False)
    if dup_mask.any():
        for _, row in df_raw[dup_mask].iterrows():
            validation_summary["duplicate_ids"] += 1
            audit_records.append({
                "transaction_id": str(row["Transaction_Id"]),
                "field": "Transaction_Id",
                "old_value": str(row["Transaction_Id"]),
                "new_value": "Duplicate Flagged",
                "action": "VALIDATION_WARNING",
                "reason": "Duplicate transaction identifier detected in dataset."
            })

    # 2. Impute missing Unit_price using the statistical mode of that product's other prices
    df = df_raw.copy()
    df["is_imputed"] = False

    missing_price_mask = (
        df["Unit_price"].isna() |
        (df["Unit_price"].astype(str).str.strip().isin(["", "nan", "NaN", "None"]))
    )

    if missing_price_mask.any():
        for idx, row in df[missing_price_mask].iterrows():
            validation_summary["missing_fields"] += 1
            validation_summary["imputed_unit_prices"] += 1
            txn_id = str(row["Transaction_Id"])
            prod = str(row["Product"])

            # Compute mode of other prices for this specific product
            other_prices = df[
                (df["Product"] == prod) & (~missing_price_mask)
            ]["Unit_price"].dropna()

            if not other_prices.empty:
                modes = other_prices.mode()
                imputed_price = float(modes.iloc[0]) if not modes.empty else float(other_prices.median())
            else:
                imputed_price = 0.0

            df.loc[idx, "Unit_price"] = imputed_price
            df.loc[idx, "is_imputed"] = True

            audit_records.append({
                "transaction_id": txn_id,
                "field": "Unit_price",
                "old_value": "NULL / Empty",
                "new_value": str(imputed_price),
                "action": "IMPUTED",
                "reason": f"Imputed with statistical mode ({imputed_price}) of other observed prices for '{prod}'."
            })

    # Convert numeric fields safely
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
    df["Unit_price"] = pd.to_numeric(df["Unit_price"], errors="coerce").fillna(0.0)
    df["Discount_amount"] = pd.to_numeric(df["Discount_amount"], errors="coerce").fillna(0.0)

    # 3. Check for negative quantities
    neg_qty_rows = df[df["Quantity"] < 0]
    if not neg_qty_rows.empty:
        for _, row in neg_qty_rows.iterrows():
            validation_summary["negative_quantities"] += 1
            audit_records.append({
                "transaction_id": str(row["Transaction_Id"]),
                "field": "Quantity",
                "old_value": str(row["Quantity"]),
                "new_value": "Flagged",
                "action": "VALIDATION_WARNING",
                "reason": "Negative quantity detected."
            })

    # 4. Check for unknown statuses
    for idx, row in df.iterrows():
        raw_status = str(row["Status"]).strip()
        if raw_status.lower() not in VALID_STATUSES:
            validation_summary["unknown_statuses"] += 1
            audit_records.append({
                "transaction_id": str(row["Transaction_Id"]),
                "field": "Status",
                "old_value": raw_status,
                "new_value": "Unknown Status",
                "action": "VALIDATION_WARNING",
                "reason": f"Status '{raw_status}' not in recognized statuses {VALID_STATUSES}."
            })

    # 5. Check bad dates
    parsed_dates = pd.to_datetime(df["Order_date"], format="%d-%m-%Y", errors="coerce")
    bad_dates = df[parsed_dates.isna()]
    if not bad_dates.empty:
        for _, row in bad_dates.iterrows():
            validation_summary["bad_dates"] += 1
            audit_records.append({
                "transaction_id": str(row["Transaction_Id"]),
                "field": "Order_date",
                "old_value": str(row["Order_date"]),
                "new_value": "Unparseable Date",
                "action": "VALIDATION_WARNING",
                "reason": "Date does not match expected DD-MM-YYYY format."
            })

    # 6. Check and log Returned orders (e.g. TXN0033)
    returned_rows = df[df["Status"].str.strip().str.lower() == "returned"]
    for _, row in returned_rows.iterrows():
        audit_records.append({
            "transaction_id": str(row["Transaction_Id"]),
            "field": "Status",
            "old_value": str(row["Status"]),
            "new_value": "Returned (Retained in DB)",
            "action": "POLICY_FLAG",
            "reason": "Retained in database for return analytics; excluded from completed-sales net revenue."
        })

    # 7. Check and log Zero-Quantity orders (e.g. TXN0034)
    zero_qty_rows = df[df["Quantity"] == 0]
    for _, row in zero_qty_rows.iterrows():
        audit_records.append({
            "transaction_id": str(row["Transaction_Id"]),
            "field": "Quantity",
            "old_value": "0",
            "new_value": "0 (Retained in DB)",
            "action": "POLICY_FLAG",
            "reason": "Retained in database; excluded from AOV and average units per order performance metrics."
        })

    # Date enrichment
    df["parsed_date"] = parsed_dates
    df["formatted_date"] = df["parsed_date"].dt.strftime("%Y-%m-%d")
    df["month_year"] = df["parsed_date"].dt.strftime("%Y-%m")
    df["month_name"] = df["parsed_date"].dt.strftime("%B %Y")

    # Financial calculations
    df["gross_revenue"] = df["Quantity"] * df["Unit_price"]
    df["net_revenue"] = df["gross_revenue"] - df["Discount_amount"]

    # SQLite Persistence
    # Ensure parent directory exists
    db_dir = os.path.dirname(os.path.abspath(tgt_db))
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = get_db_connection(tgt_db)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS transactions")
    cursor.execute("DROP TABLE IF EXISTS data_cleaning_audit")

    cursor.execute("""
    CREATE TABLE transactions (
        transaction_id TEXT PRIMARY KEY,
        order_date TEXT NOT NULL,
        parsed_date TEXT NOT NULL,
        month_year TEXT NOT NULL,
        month_name TEXT NOT NULL,
        product TEXT NOT NULL,
        category TEXT NOT NULL,
        region TEXT NOT NULL,
        sales_channel TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        discount_amount REAL NOT NULL,
        status TEXT NOT NULL,
        gross_revenue REAL NOT NULL,
        net_revenue REAL NOT NULL,
        is_imputed INTEGER NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE data_cleaning_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL,
        field TEXT NOT NULL,
        old_value TEXT,
        new_value TEXT,
        action TEXT NOT NULL,
        reason TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    for _, r in df.iterrows():
        cursor.execute("""
        INSERT INTO transactions (
            transaction_id, order_date, parsed_date, month_year, month_name,
            product, category, region, sales_channel, quantity, unit_price,
            discount_amount, status, gross_revenue, net_revenue, is_imputed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(r["Transaction_Id"]), str(r["Order_date"]), str(r["formatted_date"]),
            str(r["month_year"]), str(r["month_name"]), str(r["Product"]), str(r["Category"]),
            str(r["Region"]), str(r["Sales_channel"]), int(r["Quantity"]), float(r["Unit_price"]),
            float(r["Discount_amount"]), str(r["Status"]), float(r["gross_revenue"]),
            float(r["net_revenue"]), 1 if r["is_imputed"] else 0
        ))

    for a in audit_records:
        cursor.execute("""
        INSERT INTO data_cleaning_audit (
            transaction_id, field, old_value, new_value, action, reason
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            a["transaction_id"], a["field"], a["old_value"],
            a["new_value"], a["action"], a["reason"]
        ))

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "total_records": len(df),
        "audit_count": len(audit_records),
        "validation_summary": validation_summary,
        "db_path": tgt_db
    }


def get_cleaning_audit_records(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all data cleaning, validation, and imputation audit trail rows."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM data_cleaning_audit ORDER BY id ASC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_all_transactions(db_path: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves transactions for data inspection."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions ORDER BY transaction_id ASC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
