"""
LLM Agent and Semantic Retrieval module for RSG Mini AI Business Intelligence Assistant.
Uses Google Gemini API with header authentication ('x-goog-api-key'), bounded agent loops,
BM25 vectorless retrieval over metrics glossary, RapidFuzz entity normalization, SQL guard,
grounding validation, error feedback, and deterministic fallbacks.
"""

import os
import re
import math
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
import httpx
from rapidfuzz import process, fuzz

from app.config import GEMINI_API_KEY, GEMINI_MODEL, format_money, DEFAULT_DB_PATH, PROJECT_ROOT
from app import analytics
from app.sql_guard import execute_guarded_sql, SQLSecurityError
from app.interpreter import AIQueryInterpreter

logger = logging.getLogger(__name__)

GLOSSARY_PATH = os.path.join(PROJECT_ROOT, "docs", "metrics_glossary.md")

# Catalog entities for RapidFuzz normalization
KNOWN_PRODUCTS = [
    "Laptop Stand", "Wireless Mouse", "Office Chair", "Monitor",
    "Keyboard", "Desk", "USB-C Hub"
]
KNOWN_REGIONS = ["North", "South", "East", "West"]
KNOWN_CATEGORIES = ["Electronics", "Furniture", "Accessories"]
KNOWN_CHANNELS = ["Online", "Retail"]


def sanitize_secret(text: str, secret: Optional[str] = None) -> str:
    """Sanitizes text by redacting API keys and sensitive tokens."""
    if not text:
        return ""
    s = str(text)
    if secret and len(secret) > 4 and secret in s:
        s = s.replace(secret, "[REDACTED]")
    # Redact common Gemini / API key patterns
    s = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED]", s)
    s = re.sub(r"key=[0-9A-Za-z-_]+", "key=[REDACTED]", s)
    s = re.sub(r"x-goog-api-key['\":\s]+[0-9A-Za-z-_]+", "x-goog-api-key: [REDACTED]", s)
    # Redact URLs to prevent logging endpoint tokens
    s = re.sub(r"https?://\S+", "[URL_REDACTED]", s)
    return s


def stem_token(w: str) -> str:
    """Lightweight suffix stemmer for BM25 term matching."""
    w_low = w.lower()
    for sfx in ["ing", "ies", "es", "ed", "s"]:
        if w_low.endswith(sfx) and len(w_low) > len(sfx) + 2:
            return w_low[:-len(sfx)]
    return w_low


def tokenize_text(t: str) -> List[str]:
    """Tokenizes and stems a block of text."""
    return [stem_token(w) for w in re.findall(r"[a-zA-Z0-9_]+", t)]


class BM25GlossaryRetriever:
    """
    In-house Okapi BM25 retriever for semantic layer retrieval over metrics glossary.
    Provides deterministic, zero-dependency, vectorless metric retrieval.
    """

    def __init__(self, glossary_file: str = GLOSSARY_PATH):
        self.glossary_file = glossary_file
        self.documents: List[Dict[str, str]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.avgdl: float = 0.0
        self.N: int = 0
        self.load_and_index()

    def load_and_index(self):
        if not os.path.exists(self.glossary_file):
            return

        with open(self.glossary_file, "r", encoding="utf-8") as f:
            content = f.read()

        sections = content.split("### ")[1:]
        self.documents = []
        for sec in sections:
            lines = sec.strip().split("\n")
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            self.documents.append({
                "title": title,
                "body": body,
                "full_text": f"{title} {body}"
            })

        self.tokenized_corpus = [tokenize_text(d["full_text"]) for d in self.documents]
        self.N = len(self.tokenized_corpus)
        if self.N > 0:
            self.avgdl = sum(len(d) for d in self.tokenized_corpus) / self.N

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        if not self.documents or not query.strip():
            return []

        q_tokens = tokenize_text(query)
        if not q_tokens:
            return self.documents[:top_k]

        k1 = 1.5
        b = 0.75
        scores = []

        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            doc_len = len(doc_tokens)
            score = 0.0
            for qt in q_tokens:
                df_count = sum(1 for dt in self.tokenized_corpus if qt in dt)
                if df_count == 0:
                    continue
                idf = math.log((self.N - df_count + 0.5) / (df_count + 0.5) + 1.0)
                tf = doc_tokens.count(qt)
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_len / self.avgdl))
                score += idf * (numerator / denominator)

            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [self.documents[idx] for score, idx in scores[:top_k]]


def normalize_query_entities(query: str) -> Tuple[str, Dict[str, str]]:
    """
    Uses RapidFuzz to detect misspelled products, regions, categories, or channels
    and normalizes them to valid database entities.
    Example: 'wireles mouse' -> 'Wireless Mouse'.
    """
    normalized_query = query
    replacements = {}

    all_entity_groups = [
        ("product", KNOWN_PRODUCTS),
        ("region", KNOWN_REGIONS),
        ("category", KNOWN_CATEGORIES),
        ("sales_channel", KNOWN_CHANNELS)
    ]

    q_words = query.split()
    n_words = len(q_words)

    # Check 2-word combinations
    for i in range(n_words - 1):
        window = f"{q_words[i]} {q_words[i+1]}".strip("?,.!;:")
        for kind, choices in all_entity_groups:
            match = process.extractOne(window, choices, scorer=fuzz.WRatio)
            if match and match[1] >= 78.0:
                standard_name = match[0]
                if standard_name.lower() != window.lower():
                    normalized_query = re.sub(re.escape(window), standard_name, normalized_query, flags=re.IGNORECASE)
                    replacements[window] = standard_name

    # Check 1-word tokens
    for token in q_words:
        clean_tok = token.strip("?,.!;:")
        if len(clean_tok) >= 4:
            for kind, choices in all_entity_groups:
                match = process.extractOne(clean_tok, choices, scorer=fuzz.WRatio)
                if match and match[1] >= 82.0:
                    standard_name = match[0]
                    if standard_name.lower() != clean_tok.lower():
                        normalized_query = re.sub(r"\b" + re.escape(clean_tok) + r"\b", standard_name, normalized_query, flags=re.IGNORECASE)
                        replacements[clean_tok] = standard_name

    return normalized_query, replacements


def build_assumptions_from_execution(tool_name: Optional[str], tool_ref: Optional[str], tool_args: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Constructs dynamic assumptions strictly from what actually ran.
    For deterministic tools: lists their accounting policy notes.
    For run_sql: verifies whether status='Completed' is filtered and reports 'Custom SQL: verify filters' otherwise.
    """
    args = tool_args or {}

    if tool_name == "get_total_net_revenue":
        return [
            "Includes only 'Completed' transactions. Returned orders (TXN0033) excluded in accordance with GAAP revenue-recognition standards."
        ]
    elif tool_name == "get_product_highest_net_revenue":
        return [
            "Evaluated across all 'Completed' orders. Returned units (e.g. 2 Monitor units in TXN0033) are excluded."
        ]
    elif tool_name == "get_monthly_revenue_trend":
        return [
            "Trend aggregates completed transactions month-by-month chronologically. Returned orders excluded."
        ]
    elif tool_name == "get_region_highest_net_revenue":
        return [
            "Calculated on completed transactions. West region return of ₹19,000 (TXN0033) is excluded from completed net sales."
        ]
    elif tool_name == "compare_monthly_revenue":
        return [
            "Comparison is computed strictly on completed net sales for both target periods."
        ]
    elif tool_name == "get_order_performance_metrics":
        return [
            "Excludes TXN0034 (Quantity=0) and TXN0033 (Returned) to prevent deflation of AOV and units/order."
        ]
    elif tool_name == "run_sql":
        query_sql = str(args.get("query", tool_ref or "")).strip()
        # Parse query for status = 'Completed' or status = 'completed'
        if re.search(r"\bstatus\s*=\s*['\"]completed['\"]", query_sql, flags=re.IGNORECASE):
            return ["Filtered for Completed orders (status='Completed')."]
        else:
            return ["Custom SQL: verify filters"]
    else:
        return [
            "Revenue evaluated on Completed orders only.",
            "Returned merchandise excluded from net revenue.",
            "Zero-quantity orders excluded from basket performance."
        ]


def build_grounding_fallback_narrative(tool_name: Optional[str], tool_result: Any, tool_args: Optional[Dict[str, Any]] = None) -> str:
    """
    Constructs a verified fallback narrative strictly conditioned on the tool that actually ran.
    Never relies on rule-based intents or generic success placeholders.
    """
    if tool_name == "get_total_net_revenue" and isinstance(tool_result, dict):
        net = tool_result.get("total_net_revenue", 0.0)
        orders = tool_result.get("completed_orders_count", 0)
        gross = tool_result.get("total_gross_revenue", 0.0)
        disc = tool_result.get("total_discounts", 0.0)
        ret = tool_result.get("returned_revenue_excluded", 0.0)
        return (
            f"Total Net Revenue is {format_money(net)} across {orders} completed transactions. "
            f"Gross revenue was {format_money(gross)} with {format_money(disc)} in discounts applied. "
            f"Returned merchandise of {format_money(ret)} was excluded from net sales."
        )

    elif tool_name == "get_product_highest_net_revenue" and isinstance(tool_result, dict):
        top = tool_result.get("top_product", {})
        return (
            f"The product with the highest net revenue is {top.get('name', 'N/A')} ({top.get('category', 'N/A')}), "
            f"generating {format_money(top.get('net_revenue', 0.0))} ({top.get('revenue_share_percentage', 0.0)}% of completed sales) "
            f"across {top.get('units_sold', 0)} units sold."
        )

    elif tool_name == "get_monthly_revenue_trend" and isinstance(tool_result, dict):
        trend = tool_result.get("trend", [])
        if trend:
            best = max(trend, key=lambda x: x["net_revenue"])
            worst = min(trend, key=lambda x: x["net_revenue"])
            return (
                f"Monthly net revenue spans {len(trend)} months from {trend[0]['month_name']} to {trend[-1]['month_name']}. "
                f"Peak month was {best['month_name']} at {format_money(best['net_revenue'])}, "
                f"and lowest was {worst['month_name']} at {format_money(worst['net_revenue'])}."
            )
        return "No monthly revenue data available."

    elif tool_name == "get_region_highest_net_revenue" and isinstance(tool_result, dict):
        top = tool_result.get("top_region", {})
        return (
            f"The region with the highest net revenue is {top.get('name', 'N/A')}, "
            f"generating {format_money(top.get('net_revenue', 0.0))} ({top.get('revenue_share_percentage', 0.0)}% share) "
            f"across {top.get('units_sold', 0)} units sold."
        )

    elif tool_name == "compare_monthly_revenue" and isinstance(tool_result, dict):
        m1 = tool_result.get("month_1", {})
        m2 = tool_result.get("month_2", {})
        abs_chg = tool_result.get("absolute_change", 0.0)
        pct_chg = tool_result.get("percentage_change", 0.0)
        direction = tool_result.get("growth_direction", "change")
        sign = "+" if abs_chg > 0 else ""
        return (
            f"Comparing {m1.get('month_name', 'Month 1')} ({format_money(m1.get('net_revenue', 0.0))}) to "
            f"{m2.get('month_name', 'Month 2')} ({format_money(m2.get('net_revenue', 0.0))}): "
            f"Net revenue experienced a {direction} of {sign}{format_money(abs_chg)} ({sign}{pct_chg:.2f}%)."
        )

    elif tool_name == "get_order_performance_metrics" and isinstance(tool_result, dict):
        aov = tool_result.get("average_order_value", 0.0)
        units = tool_result.get("average_units_per_order", 0.0)
        eligible = tool_result.get("eligible_order_count", 0)
        return (
            f"Across {eligible} eligible completed orders, the Average Order Value (AOV) is {format_money(aov)} "
            f"with an average of {units:.1f} units per order."
        )

    elif tool_name == "run_sql":
        rows = tool_result if isinstance(tool_result, list) else ([tool_result] if isinstance(tool_result, dict) else [])
        if not rows:
            return "SQL query returned no matching records."
        elif len(rows) == 1:
            row_items = []
            for k, v in rows[0].items():
                if isinstance(v, (int, float)) and ("rev" in k.lower() or "price" in k.lower() or "amt" in k.lower()):
                    row_items.append(f"{k}: {format_money(v)}")
                else:
                    row_items.append(f"{k}: {v}")
            return f"Query returned 1 result row: {', '.join(row_items)}."
        else:
            first_items = [f"{k}: {rows[0][k]}" for k in list(rows[0].keys())[:3]]
            return f"Query returned {len(rows)} rows. Summary of first record: {', '.join(first_items)}."

    return f"Execution of {tool_name or 'analytical tool'} completed with {len(tool_result) if isinstance(tool_result, list) else 1} result record(s)."


SCHEMA_AND_RULES_PROMPT = """
You are the RSG Mini AI Business Intelligence Assistant.
You have access to a SQLite database containing the `transactions` table.

Table Schema:
- transaction_id (TEXT, PK): unique transaction ID (e.g. TXN0001)
- order_date (TEXT): DD-MM-YYYY
- parsed_date (TEXT): YYYY-MM-DD
- month_year (TEXT): YYYY-MM (from 2026-01 to 2026-06)
- month_name (TEXT): e.g. January 2026
- product (TEXT): Laptop Stand, Wireless Mouse, Office Chair, Monitor, Keyboard, Desk, USB-C Hub
- category (TEXT): Electronics, Furniture, Accessories
- region (TEXT): North, South, East, West
- sales_channel (TEXT): Online, Retail
- quantity (INTEGER): items ordered (>= 0)
- unit_price (REAL): unit price
- discount_amount (REAL): total discount applied
- status (TEXT): 'Completed' or 'Returned'
- gross_revenue (REAL): quantity * unit_price
- net_revenue (REAL): gross_revenue - discount_amount
- is_imputed (INTEGER): 1 if price was imputed, 0 otherwise

Business Rules:
1. Recognized revenue is net_revenue on 'Completed' orders ONLY.
2. Returned orders (e.g., status = 'Returned') are strictly excluded from net sales.
3. Quantity-0 rows (e.g. TXN0034) are excluded from Average Order Value (AOV) and units per order.
4. When answering, use the provided tools:
   - For total net revenue: call `get_total_net_revenue`
   - For highest revenue product: call `get_product_highest_net_revenue`
   - For monthly trend: call `get_monthly_revenue_trend`
   - For highest revenue region: call `get_region_highest_net_revenue`
   - For comparing two months: call `compare_monthly_revenue` with month1 and month2
   - For order performance (AOV, avg units): call `get_order_performance_metrics`
   - For all other queries (e.g., category, sales channel, specific product in region, best/worst month, top N, specific filters): call `run_sql` with a safe SELECT query on `transactions`.
5. Your final narrative response MUST be strictly grounded in the tool output. Never invent numbers.
"""

GEMINI_TOOLS_DECLARATION = [
    {
        "function_declarations": [
            {
                "name": "get_total_net_revenue",
                "description": "Calculates total net revenue for completed transactions excluding returned orders.",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "get_product_highest_net_revenue",
                "description": "Finds the product generating the highest net revenue and product rankings.",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "get_monthly_revenue_trend",
                "description": "Provides chronological monthly revenue trend from January to June 2026.",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "get_region_highest_net_revenue",
                "description": "Finds the geographic region generating the highest net revenue.",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "compare_monthly_revenue",
                "description": "Compares revenue between two specified months (e.g., month1='2026-01', month2='2026-03').",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "month1": {"type": "STRING", "description": "First month (e.g. 2026-01 or January)"},
                        "month2": {"type": "STRING", "description": "Second month (e.g. 2026-03 or March)"}
                    },
                    "required": ["month1", "month2"]
                }
            },
            {
                "name": "get_order_performance_metrics",
                "description": "Computes Average Order Value (AOV) and average units per order, excluding zero-quantity and returned orders.",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "run_sql",
                "description": "Executes a read-only SELECT query against the transactions table for customized aggregations.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Safe SELECT SQL query targeting transactions table."}
                    },
                    "required": ["query"]
                }
            }
        ]
    }
]


def extract_numbers_from_text(text: str) -> List[float]:
    """Extracts numeric values from a string, filtering out years like 2026."""
    cleaned = text.replace(",", "")
    tokens = re.findall(r"(?<![a-zA-Z0-9_])-?\d+(?:\.\d+)?", cleaned)
    nums = []
    for t in tokens:
        try:
            val = float(t)
            if val == 2026:
                continue
            nums.append(val)
        except ValueError:
            pass
    return nums


def collect_numbers_from_data(data: Any) -> set:
    """Recursively collects all numeric values present in a tool result payload."""
    nums = set()
    if isinstance(data, (int, float)):
        nums.add(round(float(data), 2))
    elif isinstance(data, dict):
        for v in data.values():
            nums.update(collect_numbers_from_data(v))
    elif isinstance(data, list):
        for item in data:
            nums.update(collect_numbers_from_data(item))
    return nums


def verify_grounding(narrative: str, tool_data: Any) -> bool:
    """
    Verifies that all significant numbers in narrative exist in the tool output payload
    or can be derived from simple percentages/rounding.
    """
    narrative_nums = extract_numbers_from_text(narrative)
    if not narrative_nums:
        return True

    data_nums = collect_numbers_from_data(tool_data)
    if not data_nums:
        return False

    for num in narrative_nums:
        rounded_num = round(num, 2)
        matched = False
        for d in data_nums:
            if abs(rounded_num - d) <= 0.05 or (d > 0 and abs(rounded_num - d) / d < 0.015):
                matched = True
                break
            if d != 0 and (abs(rounded_num - (d * 100)) <= 0.2 or abs(rounded_num - (d / 100)) <= 0.01):
                matched = True
                break
        if not matched:
            if rounded_num in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 50.0, 51.0]:
                continue
            return False
    return True


class LLMAgent:
    """
    Google Gemini powered analytics agent with header authentication ('x-goog-api-key'),
    bounded agent loop (max 3 tool steps), error feedback, SQL guard, and grounded execution.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, db_path: Optional[str] = None):
        self.api_key = api_key if api_key is not None else GEMINI_API_KEY
        self.model = model or GEMINI_MODEL
        self.db_path = db_path or DEFAULT_DB_PATH
        self.interpreter = AIQueryInterpreter()
        self.retriever = BM25GlossaryRetriever()

    def dispatch_tool(self, tool_name: str, args: Dict[str, Any]) -> Tuple[Any, str]:
        """Dispatches tool call to appropriate python function or SQL executor."""
        if tool_name == "get_total_net_revenue":
            return analytics.get_total_net_revenue(self.db_path), "analytics.get_total_net_revenue()"

        elif tool_name == "get_product_highest_net_revenue":
            return analytics.get_product_highest_net_revenue(self.db_path), "analytics.get_product_highest_net_revenue()"

        elif tool_name == "get_monthly_revenue_trend":
            return analytics.get_monthly_revenue_trend(self.db_path), "analytics.get_monthly_revenue_trend()"

        elif tool_name == "get_region_highest_net_revenue":
            return analytics.get_region_highest_net_revenue(self.db_path), "analytics.get_region_highest_net_revenue()"

        elif tool_name == "compare_monthly_revenue":
            m1 = args.get("month1", "")
            m2 = args.get("month2", "")
            return analytics.compare_monthly_revenue(m1, m2, self.db_path), f"analytics.compare_monthly_revenue('{m1}', '{m2}')"

        elif tool_name == "get_order_performance_metrics":
            return analytics.get_order_performance_metrics(self.db_path), "analytics.get_order_performance_metrics()"

        elif tool_name == "run_sql":
            sql_query = args.get("query", "")
            rows = execute_guarded_sql(sql_query, self.db_path)
            return rows, sql_query

        else:
            raise ValueError(f"Unknown tool name: {tool_name}")

    def call_gemini_api(self, contents: List[Dict[str, Any]], system_prompt: str, tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Calls Gemini REST API via httpx using header authentication ('x-goog-api-key').
        Never logs URLs, headers, or response bodies.
        Logs only status code and sanitized message with any key redacted.
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload: Dict[str, Any] = {
            "contents": contents,
            "system_instruction": {"parts": [{"text": system_prompt}]}
        }
        if tools:
            payload["tools"] = tools

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                if resp.status_code == 429:
                    logger.warning("Gemini API rate limit exceeded (HTTP 429)")
                    raise RuntimeError("http_429: rate limit exceeded")
                elif resp.status_code == 400:
                    err_msg = resp.json().get("error", {}).get("message", "invalid argument")[:60]
                    logger.warning(f"Gemini API bad request (HTTP 400): {err_msg}")
                    raise RuntimeError(f"http_400: {err_msg}")
                elif resp.status_code != 200:
                    logger.warning(f"Gemini API returned status code {resp.status_code}")
                    raise RuntimeError(f"http_{resp.status_code}")
                return resp.json()
        except Exception as e:
            sanitized = sanitize_secret(str(e), self.api_key)
            sanitized = re.sub(r"https?://\S+", "[URL_REDACTED]", sanitized)
            logger.warning(f"Gemini API call failed: {sanitized}")
            raise RuntimeError(sanitized) from None

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Processes a natural language question.
        Applies RapidFuzz normalization and BM25 vectorless retrieval.
        Uses a bounded agent loop (max 3 tool steps) with tool error feedback.
        Falls back to rule-based engine ONLY for network/API/no-key failures.
        """
        # 1. Normalize entities via RapidFuzz
        normalized_q, replacements = normalize_query_entities(query)

        # 2. Retrieve relevant glossary entries via BM25
        glossary_matches = self.retriever.retrieve(normalized_q, top_k=3)
        glossary_titles = [m["title"] for m in glossary_matches]

        # 3. Check for ambiguity in monthly comparison
        interp = self.interpreter.interpret(normalized_q)
        if interp.get("status") == "needs_clarification":
            return {
                "answer": interp.get("explanation"),
                "exact_tool": None,
                "supporting_rows": [],
                "assumptions": ["Comparison requested without specifying target months."],
                "status": "needs_clarification",
                "source": "rule_based",
                "fallback_reason": None,
                "data": {},
                "suggested_chart": None,
                "glossary_entries": glossary_titles
            }

        # Fall back to rule-based engine if no API key is provided
        if not self.api_key or not self.api_key.strip():
            fallback_res = self.run_rule_based_fallback(normalized_q, interp, fallback_reason="no_api_key")
            fallback_res["glossary_entries"] = glossary_titles
            return fallback_res

        # Prepare augmented system prompt with retrieved semantic glossary entries
        glossary_context = "\n\nRelevant Semantic Glossary Context:\n" + "\n".join(
            [f"- **{g['title']}**: {g['body']}" for g in glossary_matches]
        )
        augmented_prompt = SCHEMA_AND_RULES_PROMPT + glossary_context

        contents = [{"role": "user", "parts": [{"text": normalized_q}]}]

        # Bounded agent loop (max 3 tool steps)
        max_tool_steps = 3
        tool_steps_taken = 0
        last_tool_name: Optional[str] = None
        last_tool_ref: Optional[str] = None
        last_tool_result: Any = None
        last_tool_args: Dict[str, Any] = {}
        last_error_msg: Optional[str] = None
        final_narrative: Optional[str] = None

        while tool_steps_taken < max_tool_steps:
            try:
                resp = self.call_gemini_api(contents, augmented_prompt, GEMINI_TOOLS_DECLARATION)
            except Exception as api_err:
                # Network / API failure -> fall back to rule-based engine with specific sanitized reason
                err_text = str(api_err).lower()
                if "429" in err_text:
                    reason = "http_429"
                elif "timeout" in err_text:
                    reason = "timeout"
                elif "400" in err_text:
                    reason = "http_400: invalid schema"
                elif "404" in err_text:
                    reason = "http_404: model not found"
                elif "503" in err_text:
                    reason = "http_503: high demand"
                else:
                    reason = "api_error"

                logger.warning(f"Gemini API failure ({reason}): {sanitize_secret(str(api_err), self.api_key)}. Falling back to rule-based engine.")
                fallback_res = self.run_rule_based_fallback(normalized_q, interp, fallback_reason=reason)
                fallback_res["glossary_entries"] = glossary_titles
                return fallback_res

            candidates = resp.get("candidates", [])
            if not candidates:
                last_error_msg = "Model returned empty response."
                break

            candidate_content = candidates[0].get("content", {})
            parts = candidate_content.get("parts", [])
            if not parts:
                last_error_msg = "Model returned empty message parts."
                break

            part = parts[0]

            # Case A: Model issues a tool call
            if "functionCall" in part:
                tool_steps_taken += 1
                fn_call = part["functionCall"]
                fn_name = fn_call.get("name")
                fn_args = fn_call.get("args", {})
                last_tool_name = fn_name
                last_tool_args = fn_args

                # Attempt executing the tool
                try:
                    tool_result, tool_ref = self.dispatch_tool(fn_name, fn_args)
                    last_tool_result = tool_result
                    last_tool_ref = tool_ref
                    last_error_msg = None
                    fn_resp = {"result": tool_result}
                except (SQLSecurityError, sqlite3.Error, ValueError, Exception) as tool_err:
                    sanitized_err = sanitize_secret(str(tool_err), self.api_key)
                    last_error_msg = sanitized_err
                    last_tool_result = None
                    # Feedback error to model so it can correct itself
                    fn_resp = {
                        "error": f"Tool execution failed: {sanitized_err}. Please refine your query or parameters and try again."
                    }

                # Preserve full candidate content (including thoughtSignature and call id)
                contents.append(candidate_content)

                fn_resp_part = {
                    "functionResponse": {
                        "name": fn_name,
                        "response": fn_resp
                    }
                }
                if "id" in fn_call:
                    fn_resp_part["functionResponse"]["id"] = fn_call["id"]

                contents.append({
                    "role": "function",
                    "parts": [fn_resp_part]
                })

                # If last allowed step was used and still in error state, break
                if tool_steps_taken >= max_tool_steps and last_error_msg:
                    break

            # Case B: Model produces final text (no tool call)
            else:
                final_narrative = part.get("text", "").strip()
                break

        # Persistent failure handling: if tool calls failed after all steps, return cannot_answer
        if last_error_msg is not None or (tool_steps_taken > 0 and last_tool_result is None):
            assumptions = build_assumptions_from_execution(last_tool_name, last_tool_ref, last_tool_args)
            return {
                "answer": f"The query could not be completed because the analytical tool encountered persistent errors: {last_error_msg or 'Maximum refinement steps reached without valid result.'}",
                "exact_tool": last_tool_ref,
                "supporting_rows": [],
                "assumptions": assumptions,
                "status": "cannot_answer",
                "source": "llm_agent",
                "fallback_reason": None,
                "data": {"error": last_error_msg},
                "suggested_chart": None,
                "glossary_entries": glossary_titles
            }

        # If a tool ran successfully, evaluate grounding
        if last_tool_result is not None:
            if not final_narrative:
                final_narrative = build_grounding_fallback_narrative(last_tool_name, last_tool_result, last_tool_args)
            elif not verify_grounding(final_narrative, last_tool_result):
                logger.warning("Grounding check failed: hallucinated number detected. Building template from executed tool.")
                final_narrative = build_grounding_fallback_narrative(last_tool_name, last_tool_result, last_tool_args)

            supporting_rows = last_tool_result if isinstance(last_tool_result, list) else ([last_tool_result] if isinstance(last_tool_result, dict) else [])
            assumptions = build_assumptions_from_execution(last_tool_name, last_tool_ref, last_tool_args)

            return {
                "answer": final_narrative,
                "exact_tool": last_tool_ref,
                "supporting_rows": supporting_rows,
                "assumptions": assumptions,
                "status": "success",
                "source": "llm_agent",
                "fallback_reason": None,
                "data": last_tool_result,
                "suggested_chart": interp.get("suggested_chart"),
                "glossary_entries": glossary_titles
            }

        # General conversational answer without tool invocation
        if final_narrative:
            return {
                "answer": final_narrative,
                "exact_tool": None,
                "supporting_rows": [],
                "assumptions": ["General conversational query; no analytical database tools executed."],
                "status": "success",
                "source": "llm_agent",
                "fallback_reason": None,
                "data": {},
                "suggested_chart": interp.get("suggested_chart"),
                "glossary_entries": glossary_titles
            }

        fallback_res = self.run_rule_based_fallback(normalized_q, interp, fallback_reason="empty_response")
        fallback_res["glossary_entries"] = glossary_titles
        return fallback_res

    def run_rule_based_fallback(self, query: str, interp: Optional[Dict[str, Any]] = None, fallback_reason: Optional[str] = None) -> Dict[str, Any]:
        """Deterministic execution fallback using rule-based interpreter."""
        if interp is None:
            interp = self.interpreter.interpret(query)

        intent = interp.get("intent", "unsupported")
        params = interp.get("parameters", {})
        status = interp.get("status", "success")

        data: Any = {}
        exact_tool = None
        tool_name = None

        try:
            if intent == "total_net_revenue":
                tool_name = "get_total_net_revenue"
                data = analytics.get_total_net_revenue(self.db_path)
                exact_tool = "analytics.get_total_net_revenue()"

            elif intent == "product_highest_revenue":
                tool_name = "get_product_highest_net_revenue"
                data = analytics.get_product_highest_net_revenue(self.db_path)
                exact_tool = "analytics.get_product_highest_net_revenue()"

            elif intent == "monthly_trend":
                tool_name = "get_monthly_revenue_trend"
                data = analytics.get_monthly_revenue_trend(self.db_path)
                exact_tool = "analytics.get_monthly_revenue_trend()"

            elif intent == "region_highest_revenue":
                tool_name = "get_region_highest_net_revenue"
                data = analytics.get_region_highest_net_revenue(self.db_path)
                exact_tool = "analytics.get_region_highest_net_revenue()"

            elif intent == "monthly_comparison":
                tool_name = "compare_monthly_revenue"
                m1, m2 = params.get("month1"), params.get("month2")
                if not m1 or not m2:
                    return {
                        "answer": interp.get("explanation", "Comparison requires two target months."),
                        "exact_tool": None,
                        "supporting_rows": [],
                        "assumptions": build_assumptions_from_execution(tool_name, None),
                        "status": "needs_clarification",
                        "source": "rule_based",
                        "fallback_reason": fallback_reason,
                        "data": {},
                        "suggested_chart": None,
                        "glossary_entries": []
                    }
                data = analytics.compare_monthly_revenue(m1, m2, self.db_path)
                exact_tool = f"analytics.compare_monthly_revenue('{m1}', '{m2}')"

            elif intent == "revenue_in_month":
                tool_name = "run_sql"
                target_month = params.get("month", "")
                norm_m = analytics.normalize_month_input(target_month)
                if norm_m:
                    sql = "SELECT month_year, month_name, COUNT(*) as order_count, SUM(net_revenue) as net_revenue FROM transactions WHERE status = 'Completed' AND month_year = ?"
                    conn = analytics.get_db_connection(self.db_path)
                    cur = conn.cursor()
                    cur.execute(sql, (norm_m,))
                    row = cur.fetchone()
                    conn.close()
                    data = dict(row) if row and row["month_year"] else {"month_name": target_month, "net_revenue": 0.0, "order_count": 0}
                    exact_tool = f"SELECT ... WHERE status = 'Completed' AND month_year = '{norm_m}'"
                else:
                    status = "validation_error"
                    data = {"error": f"Invalid month '{target_month}'"}

            elif intent == "quarter_revenue":
                tool_name = "run_sql"
                conn = analytics.get_db_connection(self.db_path)
                cur = conn.cursor()
                cur.execute("""
                SELECT 
                    CASE 
                        WHEN month_year IN ('2026-01', '2026-02', '2026-03') THEN 'Q1 (Jan-Mar 2026)'
                        WHEN month_year IN ('2026-04', '2026-05', '2026-06') THEN 'Q2 (Apr-Jun 2026)'
                        ELSE 'Other'
                    END as quarter,
                    COUNT(*) as order_count,
                    SUM(net_revenue) as net_revenue
                FROM transactions
                WHERE status = 'Completed'
                GROUP BY quarter
                ORDER BY quarter ASC
                """)
                quarters = [dict(r) for r in cur.fetchall()]
                conn.close()
                data = {"quarters": quarters}
                exact_tool = "SELECT ... WHERE status = 'Completed' GROUP BY quarter"

            elif intent == "category_highest_revenue":
                tool_name = "run_sql"
                conn = analytics.get_db_connection(self.db_path)
                cur = conn.cursor()
                cur.execute("""
                SELECT category, COUNT(*) as order_count, SUM(net_revenue) as net_revenue
                FROM transactions
                WHERE status = 'Completed'
                GROUP BY category
                ORDER BY net_revenue DESC
                """)
                rows = [dict(r) for r in cur.fetchall()]
                conn.close()
                top_cat = rows[0] if rows else {}
                data = {"top_category": top_cat, "all_categories": rows}
                exact_tool = "SELECT category, SUM(net_revenue) ... WHERE status = 'Completed' GROUP BY category"

            elif intent == "channel_comparison":
                tool_name = "run_sql"
                conn = analytics.get_db_connection(self.db_path)
                cur = conn.cursor()
                cur.execute("""
                SELECT sales_channel, COUNT(*) as order_count, SUM(net_revenue) as net_revenue
                FROM transactions
                WHERE status = 'Completed'
                GROUP BY sales_channel
                ORDER BY net_revenue DESC
                """)
                channels = [dict(r) for r in cur.fetchall()]
                conn.close()
                data = {"channels": channels}
                exact_tool = "SELECT sales_channel, SUM(net_revenue) ... WHERE status = 'Completed' GROUP BY sales_channel"

            elif intent == "best_worst_month":
                tool_name = "get_monthly_revenue_trend"
                trend_res = analytics.get_monthly_revenue_trend(self.db_path)
                trend = trend_res.get("trend", [])
                if trend:
                    best = max(trend, key=lambda x: x["net_revenue"])
                    worst = min(trend, key=lambda x: x["net_revenue"])
                    data = {"best_month": best, "worst_month": worst, "trend": trend}
                exact_tool = "analytics.get_monthly_revenue_trend()"

            elif intent == "top_n_products":
                tool_name = "run_sql"
                n = params.get("n", 3)
                conn = analytics.get_db_connection(self.db_path)
                cur = conn.cursor()
                cur.execute("""
                SELECT product, category, SUM(net_revenue) as net_revenue, SUM(quantity) as units_sold
                FROM transactions
                WHERE status = 'Completed'
                GROUP BY product, category
                ORDER BY net_revenue DESC
                LIMIT ?
                """, (n,))
                rows = [dict(r) for r in cur.fetchall()]
                conn.close()
                data = {"products": rows, "n": n}
                exact_tool = f"SELECT ... WHERE status = 'Completed' GROUP BY product ORDER BY net_revenue DESC LIMIT {n}"

            elif intent == "product_in_region":
                tool_name = "run_sql"
                reg = params.get("region", "East")
                conn = analytics.get_db_connection(self.db_path)
                cur = conn.cursor()
                cur.execute("""
                SELECT product, category, SUM(net_revenue) as net_revenue, SUM(quantity) as units_sold
                FROM transactions
                WHERE status = 'Completed' AND LOWER(region) = LOWER(?)
                GROUP BY product
                ORDER BY net_revenue DESC
                LIMIT 1
                """, (reg,))
                row = cur.fetchone()
                conn.close()
                top_p = dict(row) if row else {}
                data = {"region": reg, "top_product": top_p}
                exact_tool = f"SELECT ... WHERE status = 'Completed' AND region = '{reg}' GROUP BY product"

            elif intent == "order_performance":
                tool_name = "get_order_performance_metrics"
                data = analytics.get_order_performance_metrics(self.db_path)
                exact_tool = "analytics.get_order_performance_metrics()"

            elif intent == "data_cleaning_audit":
                records = analytics.get_db_connection(self.db_path).cursor().execute("SELECT * FROM data_cleaning_audit ORDER BY id ASC").fetchall()
                data = {"records": [dict(r) for r in records]}
                exact_tool = "SELECT * FROM data_cleaning_audit"

            else:
                return {
                    "answer": "The assistant could not understand or map your question to a supported analytics topic. Please try asking about total revenue, top products, monthly trends, regional sales, or month comparisons.",
                    "exact_tool": None,
                    "supporting_rows": [],
                    "assumptions": build_assumptions_from_execution(None, None),
                    "status": "cannot_answer",
                    "source": "rule_based",
                    "fallback_reason": fallback_reason,
                    "data": {},
                    "suggested_chart": None,
                    "glossary_entries": []
                }

        except Exception as e:
            return {
                "answer": f"Error executing query: {str(e)}",
                "exact_tool": exact_tool,
                "supporting_rows": [],
                "assumptions": build_assumptions_from_execution(tool_name, exact_tool),
                "status": "error",
                "source": "rule_based",
                "fallback_reason": fallback_reason,
                "data": {"error": str(e)},
                "suggested_chart": None,
                "glossary_entries": []
            }

        narrative = self.interpreter.generate_narrative_summary(intent, data if isinstance(data, dict) else {})
        supporting_rows = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
        assumptions = build_assumptions_from_execution(tool_name, exact_tool, {"query": exact_tool})

        return {
            "answer": narrative,
            "exact_tool": exact_tool,
            "supporting_rows": supporting_rows,
            "assumptions": assumptions,
            "status": status,
            "source": "rule_based",
            "fallback_reason": fallback_reason,
            "data": data,
            "suggested_chart": interp.get("suggested_chart"),
            "glossary_entries": []
        }
