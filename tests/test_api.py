"""
Integration and endpoint tests for the FastAPI backend in app/api.py.
Verifies all REST endpoints, health check, NL query handler, and input validation models.
"""

import pytest
from starlette.testclient import TestClient
from fastapi import HTTPException
from pydantic import ValidationError

from app.api import (
    app,
    QueryRequest,
    root,
    health_check,
    process_natural_language_query,
    api_total_net_revenue,
    api_product_highest_net_revenue,
    api_monthly_revenue_trend,
    api_region_highest_net_revenue,
    api_monthly_comparison,
    api_order_performance,
    api_data_cleaning_audit,
    api_get_transactions
)


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    """Verifies that GET / returns 200 and lists service endpoints."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "operational"
    assert "health" in data["endpoints"]


def test_health_endpoint(client):
    """Verifies that GET /health returns 200 and healthy status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True


def test_api_total_net_revenue_endpoint(client):
    """Verifies GET /api/analytics/total-net-revenue."""
    res = client.get("/api/analytics/total-net-revenue")
    assert res.status_code == 200
    data = res.json()
    assert data["metric"] == "total_net_revenue"
    assert data["completed_orders_count"] == 50


def test_api_product_highest_net_revenue_endpoint(client):
    """Verifies GET /api/analytics/product-highest-net-revenue."""
    res = client.get("/api/analytics/product-highest-net-revenue")
    assert res.status_code == 200
    data = res.json()
    assert data["top_product"]["name"] == "Monitor"


def test_api_monthly_revenue_trend_endpoint(client):
    """Verifies GET /api/analytics/monthly-revenue-trend."""
    res = client.get("/api/analytics/monthly-revenue-trend")
    assert res.status_code == 200
    data = res.json()
    assert len(data["trend"]) == 6


def test_api_region_highest_net_revenue_endpoint(client):
    """Verifies GET /api/analytics/region-highest-net-revenue."""
    res = client.get("/api/analytics/region-highest-net-revenue")
    assert res.status_code == 200
    data = res.json()
    assert data["top_region"]["name"] == "South"


def test_api_monthly_comparison_endpoint_valid(client):
    """Verifies GET /api/analytics/monthly-comparison with valid months."""
    res = client.get("/api/analytics/monthly-comparison?month1=2026-01&month2=2026-03")
    assert res.status_code == 200
    data = res.json()
    assert data["metric"] == "monthly_revenue_comparison"
    assert data["growth_direction"] == "increase"


def test_api_monthly_comparison_endpoint_invalid_month(client):
    """Verifies that GET /api/analytics/monthly-comparison returns 400 for bad input."""
    res = client.get("/api/analytics/monthly-comparison?month1=badmonth&month2=2026-03")
    assert res.status_code == 400
    assert "Invalid month input" in res.json()["detail"]


def test_api_order_performance_endpoint(client):
    """Verifies GET /api/analytics/order-performance."""
    res = client.get("/api/analytics/order-performance")
    assert res.status_code == 200
    data = res.json()
    assert data["eligible_order_count"] == 49


def test_api_audit_cleaning_endpoint(client):
    """Verifies GET /api/audit/data-cleaning."""
    res = client.get("/api/audit/data-cleaning")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 3


def test_api_transactions_endpoint(client):
    """Verifies GET /api/transactions with limit parameter."""
    res = client.get("/api/transactions?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert len(data["records"]) == 10


def test_query_request_empty_validation_error():
    """Verifies that QueryRequest model rejects empty or whitespace query."""
    with pytest.raises(ValidationError):
        QueryRequest(query="   ")


def test_api_nl_query_endpoint(client):
    """Verifies POST /api/query returns structured fields and answer."""
    res = client.post("/api/query", json={"query": "What is our total net revenue?"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "answer" in data
    assert "exact_tool" in data
    assert "supporting_rows" in data
    assert "assumptions" in data
    assert "fallback_reason" in data
    assert data["source"] in ["llm_agent", "rule_based"]
