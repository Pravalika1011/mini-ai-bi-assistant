"""
Security tests for SQL Guard in app/sql_guard.py.
Verifies blocking of injection attacks, unauthorized keywords, multi-statements,
comment tokens, unauthorized tables, and confirms read-only authorizer enforcement.
"""

import pytest
import sqlite3

from app.sql_guard import (
    validate_sql,
    execute_guarded_sql,
    SQLSecurityError,
    sqlite_read_only_authorizer
)


def test_valid_select_query_appends_limit():
    """Verifies that a valid SELECT query gets a default LIMIT <= 200."""
    q = "SELECT product, SUM(net_revenue) FROM transactions GROUP BY product"
    guarded = validate_sql(q)
    assert "LIMIT 200" in guarded


def test_valid_select_query_clamps_large_limit():
    """Verifies that queries specifying LIMIT > 200 are clamped to 200."""
    q = "SELECT * FROM transactions LIMIT 1000"
    guarded = validate_sql(q)
    assert "LIMIT 200" in guarded
    assert "1000" not in guarded


def test_reject_stacked_statements():
    """Verifies that stacked queries with semicolons are rejected."""
    q = "SELECT * FROM transactions; DROP TABLE transactions;"
    with pytest.raises(SQLSecurityError, match="Stacked / multi-statement"):
        validate_sql(q)


def test_reject_sql_line_comments():
    """Verifies rejection of -- comments."""
    q = "SELECT * FROM transactions -- drop database"
    with pytest.raises(SQLSecurityError, match="SQL comments"):
        validate_sql(q)


def test_reject_sql_block_comments():
    """Verifies rejection of /* */ comments."""
    q = "SELECT /* injected comment */ * FROM transactions"
    with pytest.raises(SQLSecurityError, match="SQL comments"):
        validate_sql(q)


def test_reject_drop_table():
    """Verifies rejection of DROP statements."""
    q = "DROP TABLE transactions; --"
    with pytest.raises(SQLSecurityError):
        validate_sql(q)


def test_reject_update_statement():
    """Verifies rejection of UPDATE statements."""
    q = "UPDATE transactions SET net_revenue = 0"
    with pytest.raises(SQLSecurityError):
        validate_sql(q)


def test_reject_delete_statement():
    """Verifies rejection of DELETE statements."""
    q = "DELETE FROM transactions WHERE quantity = 0"
    with pytest.raises(SQLSecurityError):
        validate_sql(q)


def test_reject_insert_statement():
    """Verifies rejection of INSERT statements."""
    q = "INSERT INTO transactions (transaction_id) VALUES ('HACK')"
    with pytest.raises(SQLSecurityError):
        validate_sql(q)


def test_reject_pragma():
    """Verifies rejection of PRAGMA inspection."""
    q = "SELECT * FROM transactions WHERE 1=1 PRAGMA table_info"
    with pytest.raises(SQLSecurityError):
        validate_sql(q)


def test_reject_attach():
    """Verifies rejection of ATTACH DATABASE."""
    q = "ATTACH DATABASE 'test.db' AS evil"
    with pytest.raises(SQLSecurityError):
        validate_sql(q)


def test_reject_unauthorized_table():
    """Verifies rejection of access to tables other than transactions."""
    q = "SELECT * FROM users"
    with pytest.raises(SQLSecurityError, match="not permitted"):
        validate_sql(q)


def test_reject_sqlite_master():
    """Verifies rejection of sqlite_master table inspection."""
    q = "SELECT name FROM sqlite_master WHERE type='table'"
    with pytest.raises(SQLSecurityError):
        validate_sql(q)


def test_execute_guarded_sql_success(temp_db):
    """Verifies successful execution of valid safe query on temp_db."""
    rows = execute_guarded_sql("SELECT DISTINCT region FROM transactions", temp_db)
    assert len(rows) == 4
    regions = {r["region"] for r in rows}
    assert regions == {"North", "South", "East", "West"}


def test_read_only_authorizer_denies_writes(temp_db):
    """Directly verifies that the authorizer callback denies destructive operations."""
    # Action 9 is SQLITE_DELETE, action 18 is SQLITE_INSERT
    assert sqlite_read_only_authorizer(9, "transactions", None, "main", None) == sqlite3.SQLITE_DENY
    assert sqlite_read_only_authorizer(18, "transactions", None, "main", None) == sqlite3.SQLITE_DENY
    assert sqlite_read_only_authorizer(21, None, None, None, None) == sqlite3.SQLITE_OK  # SQLITE_SELECT


def test_replace_function_allowed_and_replace_into_blocked():
    """Verifies that REPLACE() function is permitted while REPLACE INTO is blocked."""
    valid_replace = "SELECT REPLACE(product, 'Desk', 'Table') FROM transactions"
    guarded = validate_sql(valid_replace)
    assert "REPLACE(product" in guarded

    with pytest.raises(SQLSecurityError):
        validate_sql("REPLACE INTO transactions (transaction_id) VALUES ('TXN9999')")


def test_progress_handler_timeout_triggers(temp_db):
    """Verifies that progress-handler timeout triggers on long-running queries."""
    heavy_query = (
        "SELECT t1.transaction_id "
        "FROM transactions t1 "
        "JOIN transactions t2 "
        "JOIN transactions t3 "
        "JOIN transactions t4 "
        "GROUP BY t1.transaction_id, t2.transaction_id, t3.transaction_id, t4.transaction_id "
        "ORDER BY random()"
    )
    with pytest.raises(SQLSecurityError, match="timed out"):
        execute_guarded_sql(heavy_query, temp_db, timeout_seconds=0.05)
