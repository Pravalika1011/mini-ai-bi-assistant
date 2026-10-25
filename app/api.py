"""
FastAPI Backend Application for RSG Mini AI Business Intelligence Assistant.
Exposes deterministic analytics REST endpoints and grounded natural language query processing.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.config import DEFAULT_CSV_PATH, DEFAULT_DB_PATH
from app.database import (
    clean_and_load_data,
    get_cleaning_audit_records,
    get_all_transactions,
    normalize_month_input
)
from app.analytics import (
    get_total_net_revenue,
    get_product_highest_net_revenue,
    get_monthly_revenue_trend,
    get_region_highest_net_revenue,
    compare_monthly_revenue,
    get_order_performance_metrics
)
from app.llm_agent import LLMAgent
from app.interpreter import AIQueryInterpreter


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(DEFAULT_DB_PATH):
        try:
            clean_and_load_data(DEFAULT_CSV_PATH, DEFAULT_DB_PATH)
            print("Successfully initialized and loaded sales database from CSV.")
        except Exception as e:
            print(f"Warning during startup database initialization: {e}")
    yield


app = FastAPI(
    title="RSG Mini AI Business Intelligence Assistant API",
    description="Backend API providing deterministic sales analytics and AI-powered natural language queries.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_agent = LLMAgent()
interpreter = AIQueryInterpreter()


# Request Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language analytics question", min_length=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")
        return v.strip()


class MonthlyComparisonParams(BaseModel):
    month1: str = Field(..., description="First month (e.g. '2026-01', 'January', 'Jan 2026')")
    month2: str = Field(..., description="Second month (e.g. '2026-03', 'March', 'Mar 2026')")





@app.get("/")
def root():
    return {
        "service": "RSG Mini AI Business Intelligence Assistant API",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "ai_query": "POST /api/query",
            "total_net_revenue": "GET /api/analytics/total-net-revenue",
            "top_product": "GET /api/analytics/product-highest-net-revenue",
            "monthly_trend": "GET /api/analytics/monthly-revenue-trend",
            "top_region": "GET /api/analytics/region-highest-net-revenue",
            "monthly_comparison": "GET /api/analytics/monthly-comparison?month1=2026-01&month2=2026-03",
            "order_performance": "GET /api/analytics/order-performance",
            "audit_trail": "GET /api/audit/data-cleaning",
            "transactions": "GET /api/transactions"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint confirming service and database status."""
    db_ok = os.path.exists(DEFAULT_DB_PATH)
    return {
        "status": "healthy" if db_ok else "degraded",
        "database_connected": db_ok,
        "database_path": DEFAULT_DB_PATH
    }


@app.post("/api/init-db")
def initialize_database():
    """Triggers CSV ingestion, validation, cleaning, and database loading."""
    try:
        result = clean_and_load_data(DEFAULT_CSV_PATH, DEFAULT_DB_PATH)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")


# =========================================================================
# AI-Assisted Natural Language Query Endpoint
# =========================================================================

SUGGESTED_QUESTIONS = [
    "What is our total net revenue?",
    "Which product generated the highest net revenue?",
    "Show me the monthly revenue trend.",
    "Which region had the highest net revenue?",
    "Compare revenue between January 2026 and March 2026.",
    "What are our order performance metrics (AOV)?",
    "Show data cleaning and imputation audit records."
]


@app.post("/api/query")
def process_natural_language_query(req: QueryRequest):
    """
    Processes natural language analytics inquiries.
    Uses LLM tool calling when configured or deterministic rule-based matching.
    Returns structured results, exact SQL/tool used, supporting rows, assumptions, and narrative.
    """
    raw_query = req.query
    result = llm_agent.ask(raw_query)

    # Maintain backwards-compatible fields alongside the new structured fields
    return {
        "status": result.get("status", "success"),
        "query": raw_query,
        "answer": result.get("answer", ""),
        "narrative_summary": result.get("answer", ""),
        "exact_tool": result.get("exact_tool"),
        "supporting_rows": result.get("supporting_rows", []),
        "assumptions": result.get("assumptions", []),
        "source": result.get("source", "rule_based"),
        "fallback_reason": result.get("fallback_reason"),
        "data": result.get("data", {}),
        "suggested_chart": result.get("suggested_chart"),
        "glossary_entries": result.get("glossary_entries", []),
        "suggested_questions": SUGGESTED_QUESTIONS if result.get("status") in ["cannot_answer", "needs_clarification", "help"] else None
    }


# =========================================================================
# Deterministic REST Endpoints
# =========================================================================

@app.get("/api/analytics/total-net-revenue")
def api_total_net_revenue():
    """Requirement 1: Total net revenue for completed transactions."""
    return get_total_net_revenue(DEFAULT_DB_PATH)


@app.get("/api/analytics/product-highest-net-revenue")
def api_product_highest_net_revenue():
    """Requirement 2: Product with highest net revenue."""
    return get_product_highest_net_revenue(DEFAULT_DB_PATH)


@app.get("/api/analytics/monthly-revenue-trend")
def api_monthly_revenue_trend():
    """Requirement 3: Monthly revenue trend."""
    return get_monthly_revenue_trend(DEFAULT_DB_PATH)


@app.get("/api/analytics/region-highest-net-revenue")
def api_region_highest_net_revenue():
    """Requirement 4: Region with highest net revenue."""
    return get_region_highest_net_revenue(DEFAULT_DB_PATH)


@app.get("/api/analytics/monthly-comparison")
def api_monthly_comparison(
    month1: str = Query(..., description="First month (e.g., '2026-01', 'January')"),
    month2: str = Query(..., description="Second month (e.g., '2026-03', 'March')")
):
    """Requirement 5: Revenue comparison between two specified months."""
    result = compare_monthly_revenue(month1, month2, DEFAULT_DB_PATH)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/analytics/order-performance")
def api_order_performance():
    """Order performance metrics (AOV, avg units) excluding Quantity=0 and Returned."""
    return get_order_performance_metrics(DEFAULT_DB_PATH)


@app.get("/api/audit/data-cleaning")
def api_data_cleaning_audit():
    """Data cleaning, validation, and imputation audit trail."""
    records = get_cleaning_audit_records(DEFAULT_DB_PATH)
    return {"audit_records": records, "count": len(records)}


@app.get("/api/transactions")
def api_get_transactions(limit: int = Query(100, ge=1, le=500)):
    """Inspection endpoint for database records."""
    records = get_all_transactions(DEFAULT_DB_PATH, limit=limit)
    return {"records": records, "count": len(records)}
