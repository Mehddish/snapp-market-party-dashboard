# -*- coding: utf-8 -*-
"""
Snapp Market Party Strategy Dashboard v0.1
Visual-first Streamlit dashboard for Snapp Market Party dairy promotion intelligence.

Expected input files can be placed either in the current folder or in these subfolders:
- snapp_market_party_strategy/
- snapp_market_party_history/
- snapp_market_party_features/

Main inputs:
- snapp_product_strategy_matrix.csv/xlsx
- snapp_vendor_strategy_patterns.csv/xlsx
- snapp_brand_vendor_spread.xlsx/csv
- snapp_hero_sku_candidates.xlsx/csv
- snapp_strategy_summary.json
- snapp_market_party_history_cumulative.csv/xlsx
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_TITLE = "Snapp Market Party Strategy Dashboard"
APP_VERSION = "v0.1.2 — Snapp Only"

# IMPORTANT: Snapp-only dashboard.
# It intentionally DOES NOT scan Okala/Atime folders or /mnt/data fallback.
# If Snapp strategy files are missing, it will show a missing-file error instead of silently loading another platform.
CANDIDATE_DIRS = [
    Path.cwd(),
    Path.cwd() / "snapp_market_party_strategy",
    Path.cwd() / "snapp_market_party_features",
    Path.cwd() / "snapp_market_party_history",
]

LOADED_FILES = {}


def find_file(names: Iterable[str]) -> Optional[Path]:
    for d in CANDIDATE_DIRS:
        for name in names:
            p = d / name
            # Hard guard: never accept an Okala/Atime file in this Snapp dashboard.
            low = str(p).lower()
            if ("okala" in low) or ("atime" in low):
                continue
            if p.exists():
                return p
    return None


def read_table(names: Iterable[str], sheet_name: Optional[str] = None) -> pd.DataFrame:
    path = find_file(names)
    key = list(names)[0] if names else "unknown_table"
    if path is None:
        LOADED_FILES[key] = "MISSING"
        return pd.DataFrame()
    LOADED_FILES[key] = str(path)
    try:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            if sheet_name:
                df = pd.read_excel(path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        # Strong platform guard: if a platform column exists and explicitly contains Okala/Atime only, block it.
        if "platform" in df.columns:
            platforms = set(df["platform"].astype(str).str.lower().dropna().unique())
            if platforms and all(("okala" in p or "atime" in p) for p in platforms):
                st.error(f"Blocked non-Snapp input file: {path.name}")
                return pd.DataFrame()
        return df
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def read_json(names: Iterable[str]) -> dict:
    path = find_file(names)
    key = list(names)[0] if names else "unknown_json"
    if path is None:
        LOADED_FILES[key] = "MISSING"
        return {}
    LOADED_FILES[key] = str(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return {}


def num(s, default=0):
    try:
        return pd.to_numeric(s, errors="coerce").fillna(default)
    except Exception:
        return pd.Series([default] * len(s)) if hasattr(s, "__len__") else default


def ensure_bool(series):
    if series is None:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def as_count_label(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return "0"


def topn(df: pd.DataFrame, col: str, n: int = 15, asc: bool = False) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    return df.copy().assign(_sort=pd.to_numeric(df[col], errors="coerce")).sort_values("_sort", ascending=asc).head(n).drop(columns=["_sort"])


def safe_col(df: pd.DataFrame, col: str, default=""):
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def score_band(value):
    try:
        v = float(value)
    except Exception:
        return "unknown"
    if v >= 80:
        return "very strong"
    if v >= 65:
        return "strong"
    if v >= 45:
        return "moderate"
    return "weak"


@st.cache_data(show_spinner=False)
def load_data():
    product = read_table([
        "snapp_product_strategy_matrix.csv",
        "snapp_product_strategy_matrix.xlsx",
    ])
    vendor = read_table([
        "snapp_vendor_strategy_patterns.csv",
        "snapp_vendor_strategy_patterns.xlsx",
    ])
    brand = read_table([
        "snapp_brand_vendor_spread.xlsx",
        "snapp_brand_vendor_spread.csv",
    ])
    hero = read_table([
        "snapp_hero_sku_candidates.xlsx",
        "snapp_hero_sku_candidates.csv",
    ])
    history = read_table([
        "snapp_market_party_history_cumulative.csv",
        "snapp_market_party_history_cumulative.xlsx",
    ])
    summary = read_json(["snapp_strategy_summary.json"])

    # basic cleanup
    for df in [product, vendor, brand, hero, history]:
        if not df.empty:
            for c in df.columns:
                if df[c].dtype == "object":
                    df[c] = df[c].fillna("")

    if not product.empty:
        for c in [
            "presence_rate_percent", "current_discount_percent", "current_unique_vendor_count",
            "market_pull_proxy_score", "snapp_strategy_score", "hero_sku_score",
            "median_dominant_discount_percent", "p75_dominant_discount_percent",
            "dominant_discount_range", "dominant_discount_std", "current_vendor_discount_range",
            "current_vendor_min_discount", "current_vendor_max_discount",
            "recent_3_active_avg_dominant_discount"
        ]:
            if c in product.columns:
                product[c] = pd.to_numeric(product[c], errors="coerce")
        if "current_active_flag" in product.columns:
            product["current_active_flag"] = ensure_bool(product["current_active_flag"])
        if "action_now_flag" in product.columns:
            product["action_now_flag"] = ensure_bool(product["action_now_flag"])
        if "hero_sku_flag" in product.columns:
            product["hero_sku_flag"] = ensure_bool(product["hero_sku_flag"])
        product["score_band"] = product.get("snapp_strategy_score", pd.Series([np.nan] * len(product))).apply(score_band)

    if not vendor.empty:
        for c in ["presence_rate_percent", "current_unique_product_count", "avg_discount_percent",
                  "current_avg_discount_percent", "deep_discount_rate_percent", "avg_snapp_score_proxy",
                  "vendor_strategy_score"]:
            if c in vendor.columns:
                vendor[c] = pd.to_numeric(vendor[c], errors="coerce")

    if not brand.empty:
        for c in ["presence_rate_percent", "current_unique_vendor_count", "current_unique_product_count",
                  "current_avg_discount_percent", "avg_discount_percent", "deep_discount_rate_percent",
                  "brand_vendor_spread_score"]:
            if c in brand.columns:
                brand[c] = pd.to_numeric(brand[c], errors="coerce")

    if not hero.empty:
        for c in ["current_discount_percent", "current_unique_vendor_count", "market_pull_proxy_score", "hero_sku_score", "snapp_strategy_score"]:
            if c in hero.columns:
                hero[c] = pd.to_numeric(hero[c], errors="coerce")
        if "hero_sku_flag" in hero.columns:
            hero["hero_sku_flag"] = ensure_bool(hero["hero_sku_flag"])

    if not history.empty:
        if "run_datetime_parsed" in history.columns:
            history["run_datetime_parsed"] = pd.to_datetime(history["run_datetime_parsed"], errors="coerce")
        elif "run_datetime" in history.columns:
            history["run_datetime_parsed"] = pd.to_datetime(history["run_datetime"], errors="coerce")
        if "discount_percent" in history.columns:
            history["discount_percent"] = pd.to_numeric(history["discount_percent"], errors="coerce")
        if "snapp_score_proxy" in history.columns:
            history["snapp_score_proxy"] = pd.to_numeric(history["snapp_score_proxy"], errors="coerce")
        if "is_core_dairy" in history.columns:
            history["is_core_dairy"] = ensure_bool(history["is_core_dairy"])

    return product, vendor, brand, hero, history, summary


def apply_filters(product, vendor, brand, hero, history):
    st.sidebar.markdown("### فیلترها")
    categories = sorted([x for x in product.get("core_dairy_category", pd.Series(dtype=str)).dropna().unique() if str(x).strip()])
    brands = sorted([x for x in product.get("detected_brand", pd.Series(dtype=str)).dropna().unique() if str(x).strip()])
    signals = sorted([x for x in product.get("snapp_strategy_signal", pd.Series(dtype=str)).dropna().unique() if str(x).strip()])

    selected_cats = st.sidebar.multiselect("دسته لبنی", categories, default=categories)
    selected_brands = st.sidebar.multiselect("برند", brands, default=[])
    selected_signals = st.sidebar.multiselect("سیگنال استراتژیک", signals, default=[])
    action_only = st.sidebar.checkbox("فقط Current Action", value=False)

    def filt(df):
        if df.empty:
            return df
        out = df.copy()
        if selected_cats and "core_dairy_category" in out.columns:
            out = out[out["core_dairy_category"].isin(selected_cats)]
        if selected_brands and "detected_brand" in out.columns:
            out = out[out["detected_brand"].isin(selected_brands)]
        if selected_signals and "snapp_strategy_signal" in out.columns:
            out = out[out["snapp_strategy_signal"].isin(selected_signals)]
        if action_only and "action_now_flag" in out.columns:
            out = out[out["action_now_flag"] == True]
        return out

    product_f = filt(product)
    vendor_f = filt(vendor)
    brand_f = filt(brand)
    hero_f = filt(hero)
    history_f = history.copy()
    if not history_f.empty and selected_cats and "core_dairy_category" in history_f.columns:
        history_f = history_f[history_f["core_dairy_category"].isin(selected_cats)]
    if not history_f.empty and selected_brands and "detected_brand" in history_f.columns:
        history_f = history_f[history_f["detected_brand"].isin(selected_brands)]
    return product_f, vendor_f, brand_f, hero_f, history_f


def metric_card(label, value, help_text=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_empty(title="داده کافی برای نمودار موجود نیست"):
    fig = go.Figure()
    fig.add_annotation(text=title, x=0.5, y=0.5, showarrow=False, font=dict(size=16))
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20), template="plotly_white")
    st.plotly_chart(fig, width='stretch')


def bar_top(df, x, y, color=None, title="", orientation="h", n=12):
    if df.empty or x not in df.columns or y not in df.columns:
        plot_empty()
        return
    d = topn(df, x, n=n).copy()
    if orientation == "h":
        d = d.sort_values(x, ascending=True)
        fig = px.bar(d, x=x, y=y, color=color if color in d.columns else None, orientation="h", title=title, text=x)
    else:
        fig = px.bar(d, x=y, y=x, color=color if color in d.columns else None, title=title, text=x)
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20), template="plotly_white")
    st.plotly_chart(fig, width='stretch')


def scatter(df, x, y, size=None, color=None, hover=None, title="", height=460, trendline=False):
    if df.empty or x not in df.columns or y not in df.columns:
        plot_empty()
        return
    d = df.copy()
    # Make axes numeric-safe where possible
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y]).copy()
    if d.empty:
        plot_empty("داده عددی کافی برای این نمودار موجود نیست")
        return
    # Plotly marker size cannot contain NaN/inf/negative values.
    size_col = None
    if size and size in d.columns:
        safe_size_col = f"__safe_size_{size}"
        vals = pd.to_numeric(d[size], errors="coerce").replace([float("inf"), -float("inf")], pd.NA).fillna(0)
        vals = vals.clip(lower=0)
        if float(vals.sum()) > 0:
            d[safe_size_col] = vals
            size_col = safe_size_col
    hover_cols = [c for c in (hover or []) if c in d.columns and c != size_col]
    fig = px.scatter(
        d,
        x=x,
        y=y,
        size=size_col,
        color=color if color in d.columns else None,
        hover_name="short_label" if "short_label" in d.columns else ("label" if "label" in d.columns else None),
        hover_data=hover_cols,
        title=title,
        size_max=44,
        trendline="ols" if trendline else None,
    )
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=60, b=20), template="plotly_white")
    st.plotly_chart(fig, width='stretch')


def heatmap_pivot(df, index, columns, values, title="", aggfunc="mean", height=430):
    if df.empty or index not in df.columns or columns not in df.columns or values not in df.columns:
        plot_empty()
        return
    piv = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc).fillna(0)
    if piv.empty:
        plot_empty()
        return
    fig = px.imshow(
        piv,
        text_auto=".1f",
        aspect="auto",
        title=title,
        labels=dict(x=columns, y=index, color=values),
    )
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=60, b=20), template="plotly_white")
    st.plotly_chart(fig, width='stretch')


def pie_counts(df, col, title=""):
    if df.empty or col not in df.columns:
        plot_empty()
        return
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = px.pie(counts, names=col, values="count", hole=0.45, title=title)
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20), template="plotly_white")
    st.plotly_chart(fig, width='stretch')


def apply_css():
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 60%); }
        html, body, [class*="css"] { font-family: Tahoma, Arial, sans-serif; }
        .main-title {
            background: linear-gradient(135deg, #202A44, #3B4A6B);
            color: white; padding: 22px 26px; border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.12); margin-bottom: 18px;
        }
        .main-title h1 { margin: 0; font-size: 30px; }
        .main-title p { margin: 8px 0 0 0; opacity: 0.88; }
        .metric-card {
            background: white; border: 1px solid #E8ECF3; border-radius: 16px;
            padding: 16px 18px; box-shadow: 0 8px 24px rgba(32,42,68,0.07); min-height: 108px;
        }
        .metric-label { color: #6B7280; font-size: 13px; margin-bottom: 8px; }
        .metric-value { color: #111827; font-size: 28px; font-weight: 800; margin-bottom: 6px; }
        .metric-help { color: #6B7280; font-size: 12px; line-height: 1.6; }
        .insight-box {
            background: #F4F7FB; border-left: 5px solid #3B82F6; padding: 14px 16px;
            border-radius: 12px; margin: 10px 0 16px 0; color: #1F2937;
        }
        .warning-box {
            background: #FFF7ED; border-left: 5px solid #F97316; padding: 14px 16px;
            border-radius: 12px; margin: 10px 0 16px 0; color: #7C2D12;
        }
        .section-subtitle { font-size: 20px; font-weight: 800; color: #202A44; margin: 8px 0 12px 0; }
        div[data-testid="stMetric"] { background: white; padding: 12px; border-radius: 14px; border: 1px solid #E8ECF3; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_command_center(product, vendor, brand, hero, history, summary):
    st.markdown("<div class='section-subtitle'>Command Center</div>", unsafe_allow_html=True)
    total_runs = summary.get("total_runs") or (history["run_id"].nunique() if not history.empty and "run_id" in history.columns else 0)
    latest_run = summary.get("latest_run_datetime", "-")
    product_rows = len(product)
    current_actions = int(product.get("action_now_flag", pd.Series(dtype=bool)).sum()) if not product.empty and "action_now_flag" in product.columns else summary.get("current_action_rows", 0)
    hero_count = int(hero.get("hero_sku_flag", pd.Series(dtype=bool)).sum()) if not hero.empty and "hero_sku_flag" in hero.columns else summary.get("hero_sku_flag_rows", 0)
    current_active = int(product.get("current_active_flag", pd.Series(dtype=bool)).sum()) if not product.empty and "current_active_flag" in product.columns else summary.get("current_active_products", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("تعداد Run", as_count_label(total_runs), "برای benchmark هنوز کم است")
    with c2: metric_card("آخرین Run", str(latest_run).replace("T", " "), "وضعیت فعلی بازار")
    with c3: metric_card("محصول لبنی", as_count_label(product_rows), "الگوهای محصولی یونیک")
    with c4: metric_card("Current Action", as_count_label(current_actions), "کاندیدهای اقدام فعلی")
    with c5: metric_card("Hero SKU", as_count_label(hero_count), "SKUهای محرک احتمالی")

    if total_runs and total_runs < 8:
        st.markdown(
            "<div class='warning-box'>هشدار بلوغ داده: تعداد run هنوز کم است. سیگنال‌ها برای direction و پایش خوب‌اند، اما برای benchmark نهایی و frequency decision کافی نیستند.</div>",
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns([1.2, 1])
    with col1:
        scatter(
            product,
            x="current_unique_vendor_count",
            y="current_discount_percent",
            size="market_pull_proxy_score",
            color="snapp_strategy_signal",
            hover=["detected_brand", "core_dairy_category", "market_pull_proxy_score", "snapp_strategy_score", "current_seller_examples"],
            title="Product Strategy Map — Vendor Spread × Current Discount",
            height=470,
        )
    with col2:
        pie_counts(product, "snapp_strategy_signal", "Strategy Signal Mix")

    col3, col4 = st.columns([1, 1])
    with col3:
        pie_counts(product, "core_dairy_category", "Category Mix")
    with col4:
        bar_top(product, "snapp_strategy_score", "short_label", color="snapp_strategy_signal", title="Top Product Strategy Score", n=10)


def render_product_strategy(product):
    st.markdown("<div class='section-subtitle'>Product Strategy Matrix</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='insight-box'>این صفحه نشان می‌دهد محصول‌ها بر اساس تخفیف فعلی، پخش فروشنده‌ای، proxy فعالیت بازار و score استراتژیک چه جایگاهی دارند. در اسنپ، تخفیف عمیق تک‌فروشنده‌ای را نباید با حمله سراسری اشتباه گرفت.</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1.25, 1])
    with col1:
        scatter(
            product,
            x="current_unique_vendor_count",
            y="market_pull_proxy_score",
            size="current_discount_percent",
            color="snapp_strategy_signal",
            hover=["detected_brand", "core_dairy_category", "current_discount_percent", "current_vendor_max_discount", "vendor_spread_class"],
            title="Product Decision Map — Vendor Spread × Market Proxy",
        )
    with col2:
        scatter(
            product,
            x="recent_3_active_avg_dominant_discount",
            y="market_pull_proxy_score",
            size="current_unique_vendor_count",
            color="core_dairy_category",
            hover=["short_label", "snapp_strategy_signal", "benchmark_confidence"],
            title="Pricing Action Map — Recent Benchmark × Market Proxy",
        )

    col3, col4 = st.columns([1, 1])
    with col3:
        bar_top(product, "snapp_strategy_score", "short_label", color="snapp_strategy_signal", title="Top Strategy Candidates", n=12)
    with col4:
        heatmap_pivot(product, "detected_brand", "core_dairy_category", "snapp_strategy_score", title="Brand × Category Strategy Heatmap")

    with st.expander("Evidence table — Product Strategy Matrix"):
        cols = [c for c in [
            "short_label", "detected_brand", "core_dairy_category", "product_type", "pack_size",
            "current_active_flag", "current_discount_percent", "current_unique_vendor_count",
            "current_vendor_max_discount", "market_pull_proxy_score", "snapp_strategy_score",
            "snapp_strategy_signal", "strategic_action_fa", "current_seller_examples"
        ] if c in product.columns]
        st.dataframe(product[cols], use_container_width=True, height=420)


def render_brand_vendor_spread(brand):
    st.markdown("<div class='section-subtitle'>Brand × Vendor Spread</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='insight-box'>این صفحه سطح brand-category را نشان می‌دهد: آیا برند در یک category فقط در یک فروشنده تخفیف دارد، یا در چند فروشنده پخش شده و فشار category ساخته است؟</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1.25, 1])
    with col1:
        scatter(
            brand,
            x="current_unique_vendor_count",
            y="current_avg_discount_percent",
            size="brand_vendor_spread_score",
            color="brand_vendor_spread_signal",
            hover=["detected_brand", "core_dairy_category", "current_unique_product_count", "deep_discount_rate_percent", "seller_examples", "product_examples"],
            title="Brand Attack Landscape — Vendor Spread × Avg Discount",
        )
    with col2:
        bar_top(brand, "brand_vendor_spread_score", "label", color="brand_vendor_spread_signal", title="Top Brand Vendor Spread Score", n=10)

    col3, col4 = st.columns([1, 1])
    with col3:
        heatmap_pivot(brand, "detected_brand", "core_dairy_category", "brand_vendor_spread_score", title="Brand × Category Spread Heatmap")
    with col4:
        pie_counts(brand, "brand_vendor_spread_signal", "Brand Spread Signal Mix")

    with st.expander("Evidence table — Brand Vendor Spread"):
        st.dataframe(brand, use_container_width=True, height=420)


def render_vendor_patterns(vendor):
    st.markdown("<div class='section-subtitle'>Vendor Strategy Patterns</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='insight-box'>در اسنپ، فروشنده/وندر اهمیت دارد. این صفحه نشان می‌دهد کدام vendor با کدام برند و category پروموشن قوی‌تری دارد.</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1.25, 1])
    with col1:
        scatter(
            vendor,
            x="current_unique_product_count",
            y="current_avg_discount_percent",
            size="avg_snapp_score_proxy",
            color="vendor_strategy_pattern",
            hover=["seller_vendor", "detected_brand", "core_dairy_category", "vendor_strategy_score", "product_examples"],
            title="Vendor Pattern Map — SKU Count × Avg Discount",
        )
    with col2:
        bar_top(vendor, "vendor_strategy_score", "label", color="vendor_strategy_pattern", title="Top Vendor Strategy Score", n=12)

    col3, col4 = st.columns([1, 1])
    with col3:
        heatmap_pivot(vendor, "seller_vendor", "core_dairy_category", "vendor_strategy_score", title="Vendor × Category Heatmap")
    with col4:
        pie_counts(vendor, "vendor_strategy_pattern", "Vendor Pattern Mix")

    with st.expander("Evidence table — Vendor Strategy Patterns"):
        st.dataframe(vendor, use_container_width=True, height=420)


def render_hero_sku(hero):
    st.markdown("<div class='section-subtitle'>Hero SKU Evidence</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='insight-box'>Hero SKU در این نسخه فقط به معنی SKU محرک احتمالی است، نه فروش قطعی. باید ببینیم hero از جنس تخفیف عمیق تک‌فروشنده‌ای است یا از جنس vendor spread.</div>",
        unsafe_allow_html=True,
    )
    hero_only = hero[hero["hero_sku_flag"] == True] if not hero.empty and "hero_sku_flag" in hero.columns else hero
    col1, col2 = st.columns([1, 1])
    with col1:
        bar_top(hero_only, "hero_sku_score", "short_label", color="snapp_strategy_signal", title="Top Hero SKU Candidates", n=12)
    with col2:
        scatter(
            hero_only,
            x="current_discount_percent",
            y="market_pull_proxy_score",
            size="current_unique_vendor_count",
            color="snapp_strategy_signal",
            hover=["short_label", "detected_brand", "core_dairy_category", "hero_sku_score", "current_seller_examples"],
            title="Hero SKU Map — Discount × Market Proxy",
        )

    col3, col4 = st.columns([1, 1])
    with col3:
        scatter(
            hero_only,
            x="current_unique_vendor_count",
            y="hero_sku_score",
            size="current_discount_percent",
            color="core_dairy_category",
            hover=["short_label", "snapp_strategy_signal"],
            title="Hero Strength — Vendor Spread × Hero Score",
        )
    with col4:
        pie_counts(hero_only, "core_dairy_category", "Hero SKU Category Mix")

    with st.expander("Evidence table — Hero SKU Candidates"):
        st.dataframe(hero_only, use_container_width=True, height=420)


def render_benchmark_activity(product, history):
    st.markdown("<div class='section-subtitle'>Current / Historical Guard & Frequency Notes</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='warning-box'>هدف این صفحه جلوگیری از خطای تفسیری است: هر محصولی که در گذشته تخفیف داشته الزاماً الان کاندید اقدام نیست. برای تصمیم فعلی، current_active_flag و runs_since_last_seen مهم‌اند.</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        scatter(
            product,
            x="median_dominant_discount_percent",
            y="p75_dominant_discount_percent",
            size="current_unique_vendor_count",
            color="benchmark_confidence",
            hover=["short_label", "current_discount_percent", "snapp_strategy_signal"],
            title="Benchmark Zone — Median vs P75",
        )
    with col2:
        scatter(
            product,
            x="dominant_discount_range",
            y="dominant_discount_std",
            size="snapp_strategy_score",
            color="core_dairy_category",
            hover=["short_label", "benchmark_confidence"],
            title="Volatility Map — Range vs Std",
        )

    col3, col4 = st.columns([1, 1])
    with col3:
        pie_counts(product, "freshness_status_class", "Freshness Status Mix")
    with col4:
        pie_counts(product, "market_pull_proxy_class", "Market Proxy Class Mix")

    if not history.empty and "run_id" in history.columns:
        h = history.copy()
        if "is_core_dairy" in h.columns:
            dairy = h[h["is_core_dairy"] == True].copy()
        else:
            dairy = h.copy()
        if not dairy.empty:
            agg = dairy.groupby(["run_id", "run_datetime_parsed"], dropna=False).agg(
                dairy_rows=("product_id_platform", "count") if "product_id_platform" in dairy.columns else ("run_id", "count"),
                unique_products=("normalized_product_key", "nunique") if "normalized_product_key" in dairy.columns else ("run_id", "count"),
                unique_vendors=("chain_or_seller", "nunique") if "chain_or_seller" in dairy.columns else ("run_id", "count"),
                avg_discount=("discount_percent", "mean") if "discount_percent" in dairy.columns else ("run_id", "count"),
            ).reset_index().sort_values("run_datetime_parsed")
            fig = px.line(agg, x="run_datetime_parsed", y=["unique_products", "unique_vendors", "avg_discount"], markers=True, title="Run Timeline — Products, Vendors, Avg Discount")
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20), template="plotly_white")
            st.plotly_chart(fig, width='stretch')

    with st.expander("Evidence table — Current/Historical"):
        cols = [c for c in [
            "short_label", "current_active_flag", "freshness_status_class", "runs_since_last_seen",
            "current_discount_percent", "recent_3_active_avg_dominant_discount", "median_dominant_discount_percent",
            "p75_dominant_discount_percent", "benchmark_confidence", "action_now_flag", "strategic_action_fa"
        ] if c in product.columns]
        st.dataframe(product[cols], use_container_width=True, height=420)


def render_method():
    st.markdown("<div class='section-subtitle'>Method & Interpretation Guide</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='insight-box'>
        <b>منطق اصلی Snapp:</b> برخلاف Atime که chain مثل افق کوروش محور مهم بود، در Snapp محور اصلی <b>Vendor/Seller spread</b> است.
        بنابراین یک تخفیف ۴۴٪ در یک فروشنده را نباید با یک حمله چندفروشنده‌ای اشتباه گرفت.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### قواعد خواندن سیگنال‌ها")
    st.write(
        """
        - **multi_vendor_category_pressure**: محصول در چند vendor فعال است؛ حتی اگر discount خیلی عمیق نباشد، از نظر پخش فروشنده‌ای مهم است.
        - **isolated_deep_discount_action**: تخفیف عمیق دارد، ولی vendor spread محدود است؛ بیشتر tactical/monitor است.
        - **brand_category_multi_vendor_attack**: برند در یک category از چند محصول/فروشنده فشار می‌سازد.
        - **current_monitor**: فعلاً فعال است اما نه تخفیف/پخش کافی برای اقدام فوری دارد.
        - **historical_watchlist**: قبلاً دیده شده اما الان active نیست؛ نباید به عنوان action now استفاده شود.
        """
    )
    st.markdown("### Thresholdهای ذهنی فعلی")
    st.write(
        """
        - Vendor count >= 4: پخش فروشنده‌ای قابل توجه
        - Current discount >= 30%: تخفیف عمیق
        - Market proxy >= 70: proxy فعالیت قوی
        - Strategy score >= 70: کاندید جدی برای بررسی
        - total_runs < 8: هنوز early signal است و benchmark نهایی نیست
        """
    )


def main():
    st.set_page_config(page_title=f"{APP_TITLE} {APP_VERSION}", layout="wide", initial_sidebar_state="expanded")
    apply_css()
    product, vendor, brand, hero, history, summary = load_data()

    st.markdown(
        f"""
        <div class='main-title'>
            <h1>{APP_TITLE} — {APP_VERSION}</h1>
            <p>SNAPP-ONLY visual dashboard for Market Party competitor promotion strategy. Focus: vendor spread, current discount, market proxy, hero SKU, and current-vs-historical guard. It will not load Okala/Atime data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("🔒 Snapp-only data check", expanded=False):
        st.caption("این داشبورد فقط فایل‌های Snapp را از فولدر جاری و فولدرهای snapp_market_party_* می‌خواند. اگر فایل‌ها نبودند، اکالا/اتایم را جایگزین نمی‌کند.")
        for k, v in LOADED_FILES.items():
            st.write(f"**{k}**")
            st.code(v)

    if product.empty and vendor.empty and brand.empty and hero.empty:
        st.error("هیچ فایل Snapp ورودی پیدا نشد. فایل‌های strategy خروجی v0.4 را کنار داشبورد یا در فولدر snapp_market_party_strategy قرار بده.")
        st.code("snapp_product_strategy_matrix.csv\nsnapp_vendor_strategy_patterns.csv\nsnapp_brand_vendor_spread.xlsx\nsnapp_hero_sku_candidates.xlsx\nsnapp_strategy_summary.json")
        st.warning("نکته: این نسخه عمداً فایل‌های Okala/Atime را نمی‌خواند؛ اگر داشبورد اکالا دیدی، یعنی Streamlit قدیمی/تب قدیمی را باز کرده‌ای یا فایل BAT اشتباه اجرا شده است.")
        return

    product_f, vendor_f, brand_f, hero_f, history_f = apply_filters(product, vendor, brand, hero, history)

    tabs = st.tabs([
        "🎯 Command Center",
        "🧭 Product Strategy",
        "🧬 Brand Vendor Spread",
        "🏪 Vendor Patterns",
        "⭐ Hero SKU",
        "⏱️ Current/Historical",
        "📚 Method",
    ])

    with tabs[0]:
        render_command_center(product_f, vendor_f, brand_f, hero_f, history_f, summary)
    with tabs[1]:
        render_product_strategy(product_f)
    with tabs[2]:
        render_brand_vendor_spread(brand_f)
    with tabs[3]:
        render_vendor_patterns(vendor_f)
    with tabs[4]:
        render_hero_sku(hero_f)
    with tabs[5]:
        render_benchmark_activity(product_f, history_f)
    with tabs[6]:
        render_method()


if __name__ == "__main__":
    main()
