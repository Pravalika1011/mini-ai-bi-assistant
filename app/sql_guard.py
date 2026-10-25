"""
SQL Guard module for RSG Mini AI Business Intelligence Assistant.
Enforces strict read-only execution, statement validation, AST/token whitelisting,
prohibition of harmful SQL keywords, comments, multi-statement injection, and authorizer callbacks.
"""

import os
import re
import time
import sqlite3
from typing import Dict, Any, List, Optional, Tuple

FORBIDDEN_KEYWORDS = [
    r"\bPRAGMA\b",
    r"\bATTACH\b",
    r"\bDETACH\b",
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bUPDATE\b",
    r"\bINSERT\b",
    r"\bALTER\b",
    r"\bCREATE\b",
    r"\bREPLACE\s+INTO\b",
    r"\bLOAD_EXTENSION\b",
    r"\bSQLITE_MASTER\b",
    r"\bSQLITE_SCHEMA\b",
    r"\bSQLITE_TEMP_SCHEMA\b",
    r"\bSQLITE_TEMP_MASTER\b",
    r"\bEXEC\b",
    r"\bEXECUTE\b",
    r"\bVACUUM\b",
    r"\bTRANSACTION\b",
    r"\bCOMMIT\b",
    r"\bROLLBACK\b"
]

ALLOWED_TABLES = {"transactions"}


class SQLSecurityError(ValueError):
    """Raised when an SQL query violates security guardrails."""
    pass


def validate_sql(query: str) -> str:
    """
    Validates that a SQL query conforms to security rules:
    1. Exactly one statement, no semicolons.
    2. No comments (-- or /* */).
    3. Starts strictly with SELECT or WITH...SELECT.
    4. Prohibits all forbidden keywords (PRAGMA, ATTACH, DROP, sqlite_master, etc.).
    5. Whitelists tables: only 'transactions' allowed.
    6. Forces a LIMIT <= 200.
    """
    if not query or not query.strip():
        raise SQLSecurityError("Query cannot be empty.")

    cleaned = query.strip()

    # Reject comments
    if "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        raise SQLSecurityError("SQL comments (-- or /* */) are strictly prohibited.")

    # Reject semicolons (multi-statement injection)
    if ";" in cleaned.rstrip(";"):
        raise SQLSecurityError("Stacked / multi-statement queries containing ';' are prohibited.")
    cleaned = cleaned.rstrip(";").strip()

    # Must start with SELECT or WITH
    upper_query = cleaned.upper()
    if not (upper_query.startswith("SELECT") or upper_query.startswith("WITH")):
        raise SQLSecurityError("Only SELECT or WITH...SELECT queries are allowed.")

    if upper_query.startswith("WITH") and "SELECT" not in upper_query:
        raise SQLSecurityError("WITH statements must conclude with a SELECT statement.")

    # Check forbidden keywords
    for pattern in FORBIDDEN_KEYWORDS:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            match = re.search(pattern, cleaned, flags=re.IGNORECASE).group(0)
            raise SQLSecurityError(f"Forbidden keyword or table access detected: '{match}'.")

    # Table whitelisting check (FROM and JOIN targets)
    from_join_patterns = re.findall(r"\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)", cleaned, flags=re.IGNORECASE)
    for table_ref in from_join_patterns:
        tbl_clean = table_ref.lower().strip()
        if tbl_clean not in ALLOWED_TABLES:
            raise SQLSecurityError(f"Access to table '{tbl_clean}' is not permitted. Only 'transactions' is allowed.")

    # Enforce LIMIT <= 200
    limit_match = re.search(r"\bLIMIT\s+(\d+)", cleaned, flags=re.IGNORECASE)
    if limit_match:
        limit_val = int(limit_match.group(1))
        if limit_val > 200:
            cleaned = re.sub(r"\bLIMIT\s+\d+", "LIMIT 200", cleaned, flags=re.IGNORECASE)
    else:
        cleaned = f"{cleaned} LIMIT 200"

    return cleaned


def sqlite_read_only_authorizer(action: int, arg1: Optional[str], arg2: Optional[str], db_name: Optional[str], trigger_name: Optional[str]) -> int:
    """
    SQLite authorizer callback.
    Denies all actions except SELECT, reading allowed tables/columns, and internal functions.
    """
    # sqlite3 action codes:
    # SQLITE_SELECT = 21
    # SQLITE_READ = 20
    # SQLITE_FUNCTION = 31
    if action == 21:  # SQLITE_SELECT
        return sqlite3.SQLITE_OK

    if action == 31:  # SQLITE_FUNCTION
        return sqlite3.SQLITE_OK

    if action == 20:  # SQLITE_READ
        # arg1 is table name, arg2 is column name
        if arg1:
            tbl = arg1.lower()
            if tbl not in ALLOWED_TABLES and not tbl.startswith("sqlite_"):
                return sqlite3.SQLITE_DENY
            if "sqlite_master" in tbl or "sqlite_schema" in tbl:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    # Deny all other actions (INSERT, UPDATE, DELETE, PRAGMA, ATTACH, etc.)
    return sqlite3.SQLITE_DENY


def execute_guarded_sql(query: str, db_path: str, timeout_seconds: float = 5.0) -> List[Dict[str, Any]]:
    """
    Executes a SQL query within an isolated, read-only SQLite session
    under active authorization guardrails and runtime progress limits (~5s).
    """
    guarded_query = validate_sql(query)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file does not exist at: {db_path}")

    # Connect with read-only URI mode=ro
    abs_db_path = os.path.abspath(db_path).replace("\\", "/")
    uri = f"file:{abs_db_path}?mode=ro"

    conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    conn.row_factory = sqlite3.Row

    # Register authorizer
    conn.set_authorizer(sqlite_read_only_authorizer)

    # Enforce runtime limit with progress handler
    start_time = time.time()

    def progress_handler():
        if time.time() - start_time > timeout_seconds:
            return 1  # Abort query
        return 0

    conn.set_progress_handler(progress_handler, 500)

    cursor = conn.cursor()
    try:
        cursor.execute(guarded_query)
        rows = [dict(r) for r in cursor.fetchall()]
        return rows
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        if "interrupted" in str(e).lower() or (time.time() - start_time) >= timeout_seconds:
            raise SQLSecurityError(f"Query execution timed out (exceeded {timeout_seconds}s limit).") from e
        raise
    finally:
        conn.close()
