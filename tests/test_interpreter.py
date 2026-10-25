"""
Unit tests for the natural language query interpreter in app/interpreter.py.
Tests scored intent classification, parameter extraction, and phrasing coverage.
"""

import pytest
from app.interpreter import AIQueryInterpreter, extract_months_from_text


@pytest.fixture
def interpreter():
    return AIQueryInterpreter()


def test_empty_and_whitespace_queries(interpreter):
    """Verifies that empty and whitespace queries return help status."""
    for q in ["", "   ", "\t\n  "]:
        res = interpreter.interpret(q)
        assert res["intent"] == "help"
        assert res["status"] == "help"


def test_core_questions(interpreter):
    """Verifies classification for standard core business inquiries."""
    assert interpreter.interpret("What is our total net revenue?")["intent"] == "total_net_revenue"
    assert interpreter.interpret("Which product generated the highest net revenue?")["intent"] == "product_highest_revenue"
    assert interpreter.interpret("Show me the monthly revenue trend.")["intent"] == "monthly_trend"
    assert interpreter.interpret("Which region had the highest net revenue?")["intent"] == "region_highest_revenue"
    assert interpreter.interpret("Compare revenue between January 2026 and March 2026.")["intent"] == "monthly_comparison"
    assert interpreter.interpret("What are our order performance metrics (AOV)?")["intent"] == "order_performance"
    assert interpreter.interpret("Show data cleaning and imputation audit records.")["intent"] == "data_cleaning_audit"


def test_phrasing_total_revenue_in_march(interpreter):
    """Verifies: 'total revenue in March'."""
    res = interpreter.interpret("total revenue in March")
    assert res["intent"] == "revenue_in_month"
    assert "march" in res["parameters"]["month"].lower()


def test_phrasing_q1_q2(interpreter):
    """Verifies: 'Q1/Q2'."""
    res = interpreter.interpret("Compare Q1/Q2")
    assert res["intent"] == "quarter_revenue"


def test_phrasing_from_feb_to_april(interpreter):
    """Verifies: 'from feb to april'."""
    res = interpreter.interpret("revenue growth from feb to april")
    assert res["intent"] == "monthly_comparison"
    assert "feb" in res["parameters"]["month1"].lower()
    assert "apr" in res["parameters"]["month2"].lower()


def test_phrasing_jan_vs_mar(interpreter):
    """Verifies: 'Jan vs Mar'."""
    res = interpreter.interpret("Jan vs Mar")
    assert res["intent"] == "monthly_comparison"
    assert "jan" in res["parameters"]["month1"].lower()
    assert "mar" in res["parameters"]["month2"].lower()


def test_phrasing_which_category_made_the_most_money(interpreter):
    """Verifies: 'which category made the most money'."""
    res = interpreter.interpret("which category made the most money")
    assert res["intent"] == "category_highest_revenue"


def test_phrasing_online_vs_retail(interpreter):
    """Verifies: 'online vs retail'."""
    res = interpreter.interpret("online vs retail")
    assert res["intent"] == "channel_comparison"


def test_phrasing_best_worst_month(interpreter):
    """Verifies: 'best/worst month'."""
    res = interpreter.interpret("what was our best/worst month?")
    assert res["intent"] == "best_worst_month"


def test_phrasing_top_3_products(interpreter):
    """Verifies: 'top 3 products'."""
    res = interpreter.interpret("top 3 products")
    assert res["intent"] == "top_n_products"
    assert res["parameters"]["n"] == 3


def test_phrasing_product_sold_most_in_east_region(interpreter):
    """Verifies: 'which product sold the most in the East region'."""
    res = interpreter.interpret("which product sold the most in the East region")
    assert res["intent"] == "product_in_region"
    assert res["parameters"]["region"] == "East"


def test_ambiguity_two_months_without_months_specified(interpreter):
    """
    Verifies that 'how did revenue change between two months' without specifying months
    returns status 'needs_clarification' with a helpful prompt (not 'unsupported').
    """
    res = interpreter.interpret("how did revenue change between two months")
    assert res["intent"] == "monthly_comparison"
    assert res["status"] == "needs_clarification"
    assert "specify" in res["explanation"].lower()


def test_only_one_month_given_in_comparison(interpreter):
    """Verifies status 'needs_clarification' when only one month is provided."""
    res = interpreter.interpret("compare revenue for January 2026")
    assert res["intent"] == "monthly_comparison"
    assert res["status"] == "needs_clarification"


def test_unsupported_query(interpreter):
    """Verifies that completely unrelated inquiries receive unsupported status."""
    res = interpreter.interpret("What is the weather in Tokyo?")
    assert res["intent"] == "unsupported"
    assert res["status"] == "unsupported"


def test_narrative_uses_format_money(interpreter):
    """Verifies that narrative synthesis uses currency formatting without raw dollar signs."""
    dummy_payload = {
        "total_net_revenue": 1445200.0,
        "total_gross_revenue": 1449700.0,
        "total_discounts": 4500.0,
        "completed_orders_count": 50,
        "returned_revenue_excluded": 19000.0
    }
    narrative = interpreter.generate_narrative_summary("total_net_revenue", dummy_payload)
    assert "14,45,200.00" in narrative
    assert "₹" in narrative
    assert "$" not in narrative
