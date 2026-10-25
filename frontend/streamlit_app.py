"""
Streamlit Frontend for RSG Mini AI Business Intelligence Assistant.
Clean, executive design with Inter typography, transparent Plotly charts,
card layouts, data unnesting for supporting figures, and robust offline fallback.
"""

import os
import sys
from typing import Dict, Any, Optional, List
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import format_money, CURRENCY_SYMBOL
from app import analytics, database
from app.llm_agent import LLMAgent

# Page Configuration
st.set_page_config(
    page_title="RSG Mini AI Business Intelligence Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inter Typography & Modern CSS Design
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    .stApp {
        background-color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #0F172A;
    }
    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #0F172A;
    }
    .header-banner {
        background-color: #0F172A;
        padding: 12px 24px;
        border-radius: 8px;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .header-title {
        color: #FFFFFF;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .header-sub {
        color: #94A3B8;
        font-size: 0.8125rem;
        font-weight: 500;
    }
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 0.75rem;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .chip-pos {
        display: inline-block;
        background-color: #DCFCE7;
        color: #16A34A;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        margin-top: 0.35rem;
    }
    .chip-neg {
        display: inline-block;
        background-color: #FEE2E2;
        color: #DC2626;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        margin-top: 0.35rem;
    }
    .chip-neutral {
        display: inline-block;
        background-color: #F1F5F9;
        color: #475569;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        margin-top: 0.35rem;
    }
    .answer-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .answer-text {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0F172A;
        line-height: 1.5;
    }
    .badge-ai {
        display: inline-block;
        background-color: #EFF6FF;
        color: #2563EB;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        border: 1px solid #BFDBFE;
    }
    .badge-rules {
        display: inline-block;
        background-color: #F8FAFC;
        color: #475569;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        border: 1px solid #CBD5E1;
    }
    .card-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 0.75rem !important;
    }
</style>
""", unsafe_allow_html=True)


# Transparent Plotly Shared Layout Config
PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#0F172A", size=12),
    margin=dict(l=30, r=30, t=35, b=30),
    xaxis=dict(
        showgrid=True,
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        zeroline=False
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        zeroline=False
    )
)

def get_backend_url() -> str:
    try:
        return st.secrets["BACKEND_URL"].rstrip("/")
    except Exception:
        return os.getenv(
            "BACKEND_URL",
            "http://127.0.0.1:8000"
        ).rstrip("/")

def query_assistant(prompt: str) -> Dict[str, Any]:
    """Queries FastAPI backend with fast fallback to local execution if backend is unreachable."""
    url = f"{get_backend_url()}/api/query"
    try:
        resp = requests.post(url, json={"query": prompt}, timeout=0.8)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    agent = LLMAgent()
    res = agent.ask(prompt)
    return {
        "status": res.get("status", "success"),
        "query": prompt,
        "answer": res.get("answer", ""),
        "exact_tool": res.get("exact_tool"),
        "supporting_rows": res.get("supporting_rows", []),
        "assumptions": res.get("assumptions", []),
        "source": res.get("source", "rule_based"),
        "fallback_reason": res.get("fallback_reason"),
        "data": res.get("data", {}),
        "suggested_chart": res.get("suggested_chart")
    }


def get_kpis_data() -> Dict[str, Any]:
    """Retrieves high-level summary KPIs directly or via backend."""
    try:
        url = f"{get_backend_url()}/api/analytics/total-net-revenue"
        resp = requests.get(url, timeout=0.8)
        if resp.status_code == 200:
            rev_data = resp.json()
        else:
            rev_data = analytics.get_total_net_revenue()
    except Exception:
        rev_data = analytics.get_total_net_revenue()

    try:
        url_perf = f"{get_backend_url()}/api/analytics/order-performance"
        resp_perf = requests.get(url_perf, timeout=0.8)
        if resp_perf.status_code == 200:
            perf_data = resp_perf.json()
        else:
            perf_data = analytics.get_order_performance_metrics()
    except Exception:
        perf_data = analytics.get_order_performance_metrics()

    return {"revenue": rev_data, "performance": perf_data}


def render_supporting_figures(data_payload: Any, supporting_rows: Any):
    """
    Renders supporting figures cleanly without [object Object] serialization.
    Shows tabular dataframes for lists of dicts, or label-value metrics for scalar payloads.
    """
    target_list = None
    if isinstance(supporting_rows, list) and len(supporting_rows) > 0 and isinstance(supporting_rows[0], dict):
        if not any(isinstance(v, (dict, list)) for v in supporting_rows[0].values()):
            target_list = supporting_rows
    elif isinstance(data_payload, dict):
        for key in ["channels", "all_products", "all_regions", "trend", "records", "products", "quarters"]:
            if key in data_payload and isinstance(data_payload[key], list) and len(data_payload[key]) > 0 and isinstance(data_payload[key][0], dict):
                target_list = data_payload[key]
                break

    if target_list:
        df = pd.DataFrame(target_list)
        for col in df.columns:
            if any(term in col.lower() for term in ["revenue", "price", "discount", "amount", "aov"]):
                df[col] = df[col].apply(lambda x: format_money(x) if isinstance(x, (int, float)) else x)
        st.dataframe(df, width="stretch")
    else:
        # Render scalar key-value rows
        items = []
        source_dict = data_payload if isinstance(data_payload, dict) else (
            supporting_rows[0] if isinstance(supporting_rows, list) and len(supporting_rows) > 0 and isinstance(supporting_rows[0], dict) else {}
        )
        for k, v in source_dict.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    label = f"{k.replace('_', ' ').title()} - {sub_k.replace('_', ' ').title()}"
                    val_str = format_money(sub_v) if isinstance(sub_v, (int, float)) and any(term in sub_k.lower() for term in ["revenue", "price", "discount", "amount", "aov"]) else str(sub_v)
                    items.append({"Metric": label, "Value": val_str})
            elif not isinstance(v, list):
                label = k.replace('_', ' ').title()
                val_str = format_money(v) if isinstance(v, (int, float)) and any(term in k.lower() for term in ["revenue", "price", "discount", "amount", "aov"]) else str(v)
                items.append({"Metric": label, "Value": val_str})

        if items:
            df_items = pd.DataFrame(items)
            st.dataframe(df_items, width="stretch", hide_index=True)
        elif supporting_rows:
            st.write(supporting_rows)


# Slim Dark-Navy Header Band
st.markdown("""
<div class="header-banner">
    <div class="header-title">RSG Mini AI Business Intelligence Assistant</div>
    <div class="header-sub">Grounded Executive Analytics & SQL Engine</div>
</div>
""", unsafe_allow_html=True)

# Navigation via Tabs
tab_ask, tab_overview, tab_quality = st.tabs(["Ask", "Overview", "Data & Quality"])

# =========================================================================
# TAB 1: ASK (Natural Language Analytics)
# =========================================================================
with tab_ask:
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "current_input" not in st.session_state:
        st.session_state.current_input = ""

    example_chips = [
        "What is our total net revenue?",
        "Which product generated the highest net revenue?",
        "Show me the monthly revenue trend.",
        "Which region had the highest net revenue?",
        "revenue by sales channel"
    ]

    st.markdown("<div style='font-size: 0.8125rem; font-weight: 600; color: #64748B; margin-bottom: 0.35rem;'>EXAMPLE QUESTIONS</div>", unsafe_allow_html=True)
    chip_cols = st.columns(len(example_chips))
    for i, chip_text in enumerate(example_chips):
        with chip_cols[i]:
            if st.button(chip_text, key=f"chip_{i}", width="stretch"):
                st.session_state.current_input = chip_text

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_q = st.text_input(
            "Ask a business question:",
            value=st.session_state.current_input,
            placeholder="Ask a sales question (e.g., 'What is our total net revenue?')",
            label_visibility="collapsed"
        )
    with col_btn:
        submit = st.button("Run Query", type="primary", width="stretch")

    if submit and user_q.strip():
        with st.spinner("Processing analysis..."):
            response = query_assistant(user_q.strip())
            st.session_state.query_history.insert(0, response)

    if st.session_state.query_history:
        for idx, item in enumerate(st.session_state.query_history):
            q_text = item.get("query", "")
            ans_text = item.get("answer", "")
            source = item.get("source", "rule_based")
            status = item.get("status", "success")
            fallback_reason = item.get("fallback_reason")
            exact_tool = item.get("exact_tool")
            assumptions = item.get("assumptions", [])
            data_payload = item.get("data", {})
            chart_type = item.get("suggested_chart")
            supporting = item.get("supporting_rows", [])

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Answer Card with 4px blue left border
            st.markdown(f"""
            <div class="answer-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.8125rem; font-weight: 600; color: #64748B;">{q_text}</span>
                    <span class="{ 'badge-ai' if source == 'llm_agent' else 'badge-rules' }">
                        { 'AI Verified' if source == 'llm_agent' else 'Deterministic Engine' }
                    </span>
                </div>
                <div class="answer-text">{ans_text}</div>
            </div>
            """, unsafe_allow_html=True)

            # Chart if applicable
            if chart_type == "line" and isinstance(data_payload, dict) and "trend" in data_payload:
                trend_items = data_payload["trend"]
                months = [t["month_name"] for t in trend_items]
                revenues = [t["net_revenue"] for t in trend_items]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=months, y=revenues,
                    mode="lines+markers",
                    line=dict(color="#2563EB", width=2.5),
                    marker=dict(size=6, color="#2563EB"),
                    name="Net Revenue"
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=260, title="Monthly Net Revenue Trend")
                st.plotly_chart(fig, width="stretch")

            elif chart_type == "bar_comparison" and isinstance(data_payload, dict) and "month_1" in data_payload:
                m1 = data_payload.get("month_1", {})
                m2 = data_payload.get("month_2", {})
                m_names = [m1.get("month_name", "Month 1"), m2.get("month_name", "Month 2")]
                m_revs = [m1.get("net_revenue", 0.0), m2.get("net_revenue", 0.0)]
                fig = go.Figure(go.Bar(
                    x=m_names, y=m_revs,
                    marker_color=["#CBD5E1", "#2563EB"],
                    width=0.4
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=260, title="Month-over-Month Revenue Comparison")
                st.plotly_chart(fig, width="stretch")

            elif chart_type == "bar" and isinstance(data_payload, dict):
                if "all_products" in data_payload:
                    items = data_payload["all_products"]
                    names = [it["product"] for it in items]
                    vals = [it["net_revenue"] for it in items]
                    colors = ["#1D4ED8"] + ["#E2E8F0"] * (len(items) - 1)
                    fig = go.Figure(go.Bar(
                        y=names[::-1], x=vals[::-1],
                        orientation="h",
                        marker_color=colors[::-1]
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, height=280, title="Revenue by Product")
                    st.plotly_chart(fig, width="stretch")
                elif "all_regions" in data_payload:
                    items = data_payload["all_regions"]
                    names = [it["region"] for it in items]
                    vals = [it["net_revenue"] for it in items]
                    colors = ["#1D4ED8"] + ["#E2E8F0"] * (len(items) - 1)
                    fig = go.Figure(go.Bar(
                        y=names[::-1], x=vals[::-1],
                        orientation="h",
                        marker_color=colors[::-1]
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, height=230, title="Revenue by Region")
                    st.plotly_chart(fig, width="stretch")

            # Supporting Figures Expander
            with st.expander("Supporting Figures", expanded=False):
                render_supporting_figures(data_payload, supporting)

            # Details Expander (renamed from 'How this was computed')
            with st.expander("Details", expanded=False):
                st.markdown(f"**Execution Tool / SQL:** `{exact_tool or 'Internal SQL Engine'}`")
                st.markdown(f"**Execution Source:** `{source}`")
                if fallback_reason:
                    st.markdown(f"**Fallback Reason:** `{fallback_reason}`")
                if assumptions:
                    st.markdown("**Accounting & Policy Assumptions:**")
                    for a in assumptions:
                        st.markdown(f"- {a}")


# =========================================================================
# TAB 2: OVERVIEW
# =========================================================================
with tab_overview:
    kpis = get_kpis_data()
    rev = kpis.get("revenue", {})
    perf = kpis.get("performance", {})

    total_net = rev.get("total_net_revenue", 0.0)
    orders_cnt = rev.get("completed_orders_count", 0)
    aov = perf.get("average_order_value", 0.0)
    returned_cnt = rev.get("returned_orders_count", 0)
    returned_rev = rev.get("returned_revenue_excluded", 0.0)

    # 4 KPI Cards with small grey uppercase label, 28px bold navy value, green/red delta chip
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Net Revenue</div>
            <div class="kpi-value">{format_money(total_net)}</div>
            <span class="chip-pos">50 Completed Orders</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Completed Orders</div>
            <div class="kpi-value">{orders_cnt:,}</div>
            <span class="chip-pos">100% Fulfillment</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Order Value</div>
            <div class="kpi-value">{format_money(aov)}</div>
            <span class="chip-neutral">{perf.get('average_units_per_order', 0):.1f} Units / Basket</span>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Returns Excluded</div>
            <div class="kpi-value">{format_money(returned_rev)}</div>
            <span class="chip-neg">{returned_cnt} Returned Order</span>
        </div>
        """, unsafe_allow_html=True)

    # Monthly Trend Line (Peak Month Highlighted)
    trend_data = analytics.get_monthly_revenue_trend().get("trend", [])
    if trend_data:
        months = [t["month_name"] for t in trend_data]
        revs = [t["net_revenue"] for t in trend_data]
        peak_idx = revs.index(max(revs))

        marker_colors = ["#93C5FD"] * len(revs)
        marker_colors[peak_idx] = "#2563EB"
        marker_sizes = [8] * len(revs)
        marker_sizes[peak_idx] = 12

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=months, y=revs,
            mode="lines+markers",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=marker_sizes, color=marker_colors),
            name="Net Revenue"
        ))
        fig_trend.add_annotation(
            x=months[peak_idx], y=revs[peak_idx],
            text=f"Peak: {format_money(revs[peak_idx])}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#2563EB",
            font=dict(color="#0F172A", size=11, family="Inter")
        )
        fig_trend.update_layout(**PLOTLY_LAYOUT, height=300, title="Monthly Revenue Trend (Jan - Jun 2026)")
        st.plotly_chart(fig_trend, width="stretch")

    # Horizontal bars: Product & Region
    col_prod, col_reg = st.columns(2)
    with col_prod:
        prod_data = analytics.get_product_highest_net_revenue().get("all_products", [])
        if prod_data:
            p_names = [p["product"] for p in prod_data]
            p_vals = [p["net_revenue"] for p in prod_data]
            p_colors = ["#1D4ED8"] + ["#E2E8F0"] * (len(prod_data) - 1)
            fig_p = go.Figure(go.Bar(
                y=p_names[::-1], x=p_vals[::-1],
                orientation="h",
                marker_color=p_colors[::-1]
            ))
            fig_p.update_layout(**PLOTLY_LAYOUT, height=280, title="Net Revenue by Product")
            st.plotly_chart(fig_p, width="stretch")

    with col_reg:
        reg_data = analytics.get_region_highest_net_revenue().get("all_regions", [])
        if reg_data:
            r_names = [r["region"] for r in reg_data]
            r_vals = [r["net_revenue"] for r in reg_data]
            r_colors = ["#1D4ED8"] + ["#E2E8F0"] * (len(reg_data) - 1)
            fig_r = go.Figure(go.Bar(
                y=r_names[::-1], x=r_vals[::-1],
                orientation="h",
                marker_color=r_colors[::-1]
            ))
            fig_r.update_layout(**PLOTLY_LAYOUT, height=280, title="Net Revenue by Region")
            st.plotly_chart(fig_r, width="stretch")

    # Compare two months card
    st.markdown("<div style='font-size: 0.875rem; font-weight: 600; color: #0F172A; margin-top: 0.5rem;'>Compare Two Months</div>", unsafe_allow_html=True)
    comp_col1, comp_col2 = st.columns(2)
    available_months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    month_labels = ["January 2026", "February 2026", "March 2026", "April 2026", "May 2026", "June 2026"]
    with comp_col1:
        sel_m1 = st.selectbox("First Month", available_months, index=0, format_func=lambda x: month_labels[available_months.index(x)])
    with comp_col2:
        sel_m2 = st.selectbox("Second Month", available_months, index=2, format_func=lambda x: month_labels[available_months.index(x)])

    comp_res = analytics.compare_monthly_revenue(sel_m1, sel_m2)
    abs_chg = comp_res.get("absolute_change", 0.0)
    pct_chg = comp_res.get("percentage_change", 0.0)
    delta_color = "#16A34A" if abs_chg >= 0 else "#DC2626"
    sign = "+" if abs_chg > 0 else ""

    st.markdown(f"""
    <div class="card-box">
        <span style="font-weight: 500; color: #64748B;">Variance:</span>
        <span style="font-size: 1.15rem; font-weight: 700; color: {delta_color}; margin-left: 0.5rem;">
            {sign}{format_money(abs_chg)} ({sign}{pct_chg:.2f}%)
        </span>
        <div style="font-size: 0.8125rem; color: #64748B; margin-top: 0.35rem;">
            {month_labels[available_months.index(sel_m1)]} ({format_money(comp_res['month_1']['net_revenue'])}) vs 
            {month_labels[available_months.index(sel_m2)]} ({format_money(comp_res['month_2']['net_revenue'])})
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# TAB 3: DATA & QUALITY
# =========================================================================
with tab_quality:
    st.markdown("""
    <div class="card-box" style="line-height: 1.6; font-size: 0.875rem; color: #475569;">
        <strong>Data Governance & Cleaning Policies:</strong><br>
        1. <strong>Missing Price Imputation:</strong> TXN0051 (Laptop Stand) had an unpopulated Unit_price, imputed using the statistical mode (₹1,200.00) of other observed catalog records.<br>
        2. <strong>Revenue Recognition:</strong> Returned order TXN0033 (₹19,000.00) is preserved for return audit analysis but excluded from recognized completed net sales.<br>
        3. <strong>Basket Metric Integrity:</strong> Zero-quantity order TXN0034 is retained in SQLite but excluded from Average Order Value and units per order.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 0.875rem; font-weight: 600; color: #0F172A; margin-bottom: 0.5rem;'>Data Cleaning Audit Trail</div>", unsafe_allow_html=True)
    audit_rows = database.get_cleaning_audit_records()
    if audit_rows:
        df_audit = pd.DataFrame(audit_rows)
        st.dataframe(df_audit, width="stretch")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.875rem; font-weight: 600; color: #0F172A; margin-bottom: 0.5rem;'>Inspect Transactions</div>", unsafe_allow_html=True)
    all_txns = database.get_all_transactions(limit=200)
    if all_txns:
        df_txns = pd.DataFrame(all_txns)
        search_kw = st.text_input("Filter transactions by product, region, or channel:", "")
        if search_kw:
            mask = df_txns.astype(str).apply(lambda row: row.str.contains(search_kw, case=False).any(), axis=1)
            filtered_df = df_txns[mask]
        else:
            filtered_df = df_txns
        st.dataframe(filtered_df, width="stretch")
