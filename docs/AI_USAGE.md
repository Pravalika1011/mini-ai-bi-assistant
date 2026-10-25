# AI Usage & Engineering Record

**Project**: RSG Mini AI Business Intelligence Assistant  
**Tools Employed**: Google Antigravity (Advanced Agentic Pair Programmer) & Google Gemini 3.1 Flash Lite  
**Execution Environment**: Python 3.11, FastAPI, SQLite, Pandas, Streamlit, Plotly, Pytest  

---

## 1. AI Tools Used & Collaboration Model

During the development and refactoring of this BI assistant, **Google Antigravity** was used as an agentic AI pair programmer, alongside the **Google Gemini 3.1 Flash Lite** model (`gemini-3.1-flash-lite`) for natural language intent translation and SQL tool-calling.

### How AI Tools Were Utilized:
1. **Decoupled System Architecture**:
   - Designed a zero-hallucination architecture separating unstructured natural language parsing from deterministic SQL arithmetic.
   - Built a dual-engine interpreter: an offline scored regex/keyword matcher and an LLM agent that dispatches to validated tools and guarded SQL.
2. **Data Cleaning & Governance**:
   - Audited the raw CSV dataset for edge cases: missing `Unit_price` in `TXN0051`, returned order `TXN0033`, and zero-quantity order `TXN0034`.
   - Replaced fragile hardcoded imputation with statistical mode calculation across product groups.
3. **Automated Verification**:
   - Engineered self-verifying test suites where expected values are independently recomputed directly from the raw CSV using pandas, eliminating manual hardcoding.
4. **Security Hardening**:
   - Constructed `app/sql_guard.py` incorporating SQLite connection-level authorizers, table whitelisting, keyword blacklisting, and AST token scanning.

---

## 2. Real Problems Encountered & How They Were Debugged

### Problem 1: Discrepancy in Historical Documentation Numbers
* **Issue**: The original documentation and developer notes listed South region net revenue as **411,400.00**.
* **Root Cause Analysis**: The initial manual calculation in early notes had an arithmetic discrepancy.
* **Resolution**: An automated script recomputed all numbers independently from `data/sales_data.csv` using pandas:
  ```python
  # Completed orders only, excluding returned TXN0033
  df_completed = df[df['Status'] == 'Completed']
  df_completed['net'] = (df_completed['Quantity'] * df_completed['Unit_price']) - df_completed['Discount_amount']
  print(df_completed.groupby('Region')['net'].sum())
  # South: 411,200.00 (NOT 411,400.00)
  # North: 384,450.00
  # East:  352,700.00
  # West:  296,850.00 (TXN0033 19,000 returned order excluded)
  # Total: 1,445,200.00
  ```
  All documentation, test assertions, and KPI displays were updated to the verified figure of **411,200.00**.

### Problem 2: Starlette / TestClient Socket Blocking on Windows
* **Issue**: On Windows sandboxed environments, `starlette.testclient.TestClient` attempts low-level anyio loopback socket creation which can fail with `WinError 10013` (access denied / port blocked).
* **Resolution**: Pytest test cases in `tests/test_api.py` were designed to invoke FastAPI route handlers directly with valid Pydantic models and mock requests, while end-to-end integration was validated using `httpx` and live uvicorn process execution.

### Problem 3: Hardcoded Price Imputation vs. Dynamic Mode
* **Issue**: The original codebase contained hardcoded logic: `if prod == "Laptop Stand": imputed_price = 1200.0`. If a future file had missing prices for other items or changed catalog prices, the logic would break or produce incorrect values.
* **Resolution**: Replaced with generic statistical mode calculation:
  ```python
  other_prices = df[(df["Product"] == prod) & (~missing_price_mask)]["Unit_price"].dropna()
  modes = other_prices.mode()
  imputed_price = float(modes.iloc[0]) if not modes.empty else float(other_prices.median())
  ```
  This preserved the exact result of 1,200.00 for `TXN0051` while ensuring programmatic rigor.

### Problem 4: Numerical Grounding & Hallucination Prevention
* **Issue**: LLMs summarizing tool results can occasionally round numbers unpredictably (e.g., reporting 29.5k instead of 29,493.88) or hallucinate external statistics.
* **Resolution**: Implemented a numerical grounding validator (`verify_grounding`) in `app/llm_agent.py`. It extracts every numeric token from the generated narrative and verifies that each appears within the tool payload (accounting for percentage transformations and rounding tolerances). If an unauthorized number is detected, the agent automatically falls back to a deterministic templated narrative.

### Problem 5: Live Gemini Tool-Calling Silent Fallback & Schema / Thought-Signature Protocol
* **Issue**: When tested live with a valid `GEMINI_API_KEY`, queries like *"revenue by sales channel"* unexpectedly displayed the badge "Deterministic Engine" instead of "AI Verified". The live Gemini call was silently failing and falling back to rule-based execution.
* **Root Cause Analysis**:
  1. **Model ID Deprecation (HTTP 404)**: The environment defaulted to `gemini-2.5-flash`, which Google's API returned `HTTP 404: models/gemini-2.5-flash is no longer available to new users`.
  2. **Multi-Turn Function Calling / Thought Signature (HTTP 400)**: In Gemini's latest API, model candidate turns for tool calls include an internal `thoughtSignature` and `id` in `candidates[0]['content']['parts']`. In our multi-turn tool execution loop, manually reconstructing `parts: [{"functionCall": fn_call}]` stripped `thoughtSignature`. Google's subsequent turn rejected the request with `HTTP 400: INVALID_ARGUMENT: Function call is missing a thought_signature in functionCall parts`.
  3. **Silent Fallback Obscuring Issues**: The top-level `ask()` function caught all exceptions and fell back to `interpreter.py` without propagating why the fallback occurred.
  4. **Subprocess Environment Loading**: When `run.py` launches `uvicorn` as a background subprocess, `.env` could be resolved relative to different working directories unless explicitly targeted.
* **Resolution**:
  1. Updated the default model to `gemini-3.1-flash-lite`, which is active, supported, and responds in ~1.4 seconds.
  2. Preserved the full model candidate response `candidates[0]['content']` verbatim in the multi-turn `contents` array and included the `id` field in `functionResponse`.
  3. Explicitly configured `app/config.py` to load `.env` from `PROJECT_ROOT/.env` with `override=True`.
  4. Added a sanitized `fallback_reason` field to `ask()`, the `/api/query` response, and the frontend "Details" expander (e.g., `no_api_key`, `http_429`, `http_400: invalid schema`, `timeout`), completely eliminating silent fallbacks and making failure modes transparent to operators.
  5. Tested live against the real Gemini API for 5 benchmark questions ("revenue by sales channel", "top 3 products in the East region", "which category made the most money", "what's the weather?", and "DROP TABLE transactions"), confirming `source: "llm_agent"`, valid guarded SQL, and passed grounding checks.

---

## 3. Data Cleaning & Business Policy Decisions

During the preliminary audit of the sales dataset, three anomalies were identified and handled:

1. **Missing `Unit_price` in `TXN0051`**:
   - Row 52 (`TXN0051`, 28-06-2026, Laptop Stand, 5 units) had an empty string `""` for `Unit_price`.
   - All other transactions for `Laptop Stand` (`TXN0001`, `TXN0009`, `TXN0017`, `TXN0025`, `TXN0035`, `TXN0043`) have an invariant price of ₹1,200.00.
   - **Policy**: Imputed ₹1,200.00 via statistical mode. The transaction is flagged with `is_imputed = 1` and logged in the `data_cleaning_audit` table.

2. **Returned Transaction `TXN0033`**:
   - Row 34 (`TXN0033`, 26-04-2026, Monitor, 2 units @ ₹9,500.00 = ₹19,000.00) has `Status = 'Returned'`.
   - **Policy**: In line with GAAP/IFRS standards, returned orders cannot be recognized as realized net sales. It is excluded from the completed net revenue metric (₹1,445,200.00), but preserved in SQLite for returns tracking.

3. **Zero-Quantity Transaction `TXN0034`**:
   - Row 35 (`TXN0034`, 28-04-2026, Keyboard, Quantity = 0, Unit Price = ₹900.00, Status = 'Completed').
   - **Policy**: Retained in the database. When computing Average Order Value (AOV) and average units per order, zero-quantity orders are excluded (₹1,445,200 / 49 = ₹29,493.88 AOV) so that unfulfilled carts do not deflate basket metrics.

---

## 4. What I'd Improve With Another Day

1. **Multi-Turn Contextual Memory**:
   - Enable conversational drill-downs (e.g., "Why did May peak?" followed by "Break that down by channel").
2. **Semantic Vectorless Retrieval (Phase 5)**:
   - Expand the BM25 metrics glossary layer to ingest business KPIs and metadata synonyms dynamically.
3. **Automated Visual Chart Recommendation**:
   - Integrate an LLM reasoning step that selects the most informative chart type based on metric cardinality and distribution.
4. **Automated Data Drift & Anomaly Alerts**:
   - Introduce automated statistical bounds (z-score / IQR) during CSV loading to alert operators of price swings or sudden return spikes.
