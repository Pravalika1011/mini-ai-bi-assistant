"""
Configuration and shared utilities for RSG Mini AI Business Intelligence Assistant.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Base directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Load environment variables explicitly from PROJECT_ROOT/.env
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

# File paths
DEFAULT_CSV_PATH = os.getenv("CSV_PATH", os.path.join(DATA_DIR, "sales_data.csv"))
DEFAULT_DB_PATH = os.getenv("DB_PATH", os.path.join(PROJECT_ROOT, "sales_bi.db"))

# Currency configuration (Note: Dataset CSV does not define currency; defaulting to INR)
CURRENCY_SYMBOL = os.getenv("CURRENCY_SYMBOL", "₹")

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


def format_indian_digits(n: int) -> str:
    """Formats an integer using Indian numbering system (e.g., 14,45,200)."""
    s = str(abs(n))
    if len(s) <= 3:
        return s
    last_three = s[-3:]
    remaining = s[:-3]
    chunks = []
    while len(remaining) > 2:
        chunks.append(remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        chunks.append(remaining)
    chunks.reverse()
    return ",".join(chunks) + "," + last_three


def format_money(amount: Optional[float], symbol: Optional[str] = None, include_decimals: bool = True) -> str:
    """
    Formats a numeric currency value with Indian digit grouping and the configured currency symbol.
    Example: 1445200.0 -> '₹14,45,200.00' (or '₹14,45,200' if include_decimals=False).
    Negative example: -13650.0 -> '-₹13,650.00'.
    """
    if amount is None:
        amount = 0.0

    curr_sym = symbol if symbol is not None else CURRENCY_SYMBOL
    is_negative = amount < 0
    abs_amt = abs(amount)

    int_part = int(abs_amt)
    cents_part = round((abs_amt - int_part) * 100)
    if cents_part == 100:
        int_part += 1
        cents_part = 0

    formatted_int = format_indian_digits(int_part)

    if include_decimals:
        formatted_val = f"{formatted_int}.{cents_part:02d}"
    else:
        formatted_val = formatted_int

    if is_negative:
        return f"-{curr_sym}{formatted_val}"
    return f"{curr_sym}{formatted_val}"
