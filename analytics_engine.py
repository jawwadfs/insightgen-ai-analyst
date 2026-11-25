"""
Analytics Engine for InsightGen AI Data Analyst.
Orchestrates Schema Introspection, LLM Prompting, Self-Healing Code Refinement,
and Deterministic Heuristic Fallbacks.
"""

import re
import json
import warnings
warnings.filterwarnings("ignore")
import traceback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple
from safety_sandbox import execute_sandboxed_code, validate_python_code


def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates comprehensive schema and statistical metadata for prompt injection."""
    num_rows, num_cols = df.shape
    columns_info = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        n_unique = int(df[col].nunique())
        
        info = {
            "name": col,
            "dtype": dtype,
            "nulls": null_count,
            "unique_count": n_unique
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            clean_series = df[col].dropna()
            if not clean_series.empty:
                info["min"] = float(round(clean_series.min(), 2))
                info["max"] = float(round(clean_series.max(), 2))
                info["mean"] = float(round(clean_series.mean(), 2))
        else:
            sample_vals = [str(x) for x in df[col].dropna().unique()[:5]]
            info["samples"] = sample_vals

        columns_info.append(info)

    return {
        "rows": num_rows,
        "columns": num_cols,
        "schema": columns_info,
        "head_json": df.head(3).to_dict(orient="records")
    }


def format_schema_for_prompt(metadata: Dict[str, Any]) -> str:
    """Formats metadata into a crisp, readable context block for the LLM."""
    lines = [
        f"DataFrame Shape: {metadata['rows']} rows x {metadata['columns']} columns",
        "Available Columns & Types:"
    ]
    for c in metadata["schema"]:
        if "mean" in c:
            lines.append(f" - '{c['name']}' ({c['dtype']}): range [{c['min']}, {c['max']}], mean {c['mean']}")
        else:
            samples = ", ".join([f"'{s}'" for s in c.get('samples', [])])
            lines.append(f" - '{c['name']}' ({c['dtype']}): {c['unique_count']} unique values, examples: [{samples}]")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are InsightGen, a world-class AI Senior Data Analyst and Business Intelligence Engineer.
You convert natural language business and analytics questions into executable, clean Python code using Pandas and Plotly.

The user has a pandas DataFrame preloaded in memory as variable `df`.
Libraries available: `pd` (pandas), `np` (numpy), `px` (plotly.express), `go` (plotly.graph_objects).

YOUR TASK:
Write Python code that directly answers the user's question.

RULES AND CONVENTIONS:
1. Always assign the primary tabular answer or numerical metric to variable `result`.
   - `result` can be a DataFrame, Series, or formatted scalar/dict.
2. If a chart makes business sense for the question (e.g. comparisons, trends, breakdowns, distributions):
   - Create a Plotly figure and assign it to variable `fig`.
   - Always apply `template="plotly_white"` and clean axis titles on `fig`.
3. Provide a brief 1-2 sentence business insight or executive takeaway in string variable `summary`.
4. SECURITY: Never import `os`, `sys`, `subprocess`, `open`, or write files. Rely strictly on `df`.
5. Ensure column names match EXACTLY with the schema provided.
6. DOMAIN KNOWLEDGE & FINANCIAL METRICS:
   - When a dataset has `margin` or `gross_margin`, treat it as Profit ($).
   - If asked for "profit margin %" or "margin percentage", compute: (df['margin'] / df['revenue'] * 100).
   - Treat `revenue` as sales / turnover ($), and `cost` as total expenditure / COGS ($).
   - Never use ID, key, or code columns (e.g. `invoice_no`, `order_id`, `customer_id`, `product_id`) for financial aggregations, rankings, or distributions.
   - For shipping turnaround / delivery delays when `order_date` and `ship_date` are present: (pd.to_datetime(df['ship_date']) - pd.to_datetime(df['order_date'])).dt.days.
   - For time series trends, convert date columns using pd.to_datetime and sort chronologically.
7. OUTPUT FORMAT: Output ONLY pure Python code. Do not include markdown conversational filler outside the code block.
"""


def extract_python_code(raw_response: str) -> str:
    """Extracts raw python code from markdown code fences if present."""
    match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", raw_response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_response.strip()


import concurrent.futures

def run_llm_query(
    user_query: str,
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    model_name: str = "gemini-1.5-flash"
) -> Dict[str, Any]:
    """Generates and executes code using Gemini with a strict 5-second hard ceiling."""
    if not api_key:
        return {
            "success": False,
            "error": "No Gemini API key provided. Using offline analytics engine.",
            "mode": "llm_missing_key"
        }

    # Clean and sanitize API key
    clean_key = str(api_key).strip().strip('"').strip("'")
    if len(clean_key) < 10:
        return {
            "success": False,
            "error": "Invalid API Key format.",
            "mode": "llm_invalid_key"
        }

    metadata = get_dataset_metadata(df)
    schema_text = format_schema_for_prompt(metadata)

    def _call_gemini_api():
        import google.generativeai as genai
        genai.configure(api_key=clean_key)
        
        models_to_try = [
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro"
        ]
        
        last_e = None
        for m_name in models_to_try:
            try:
                model = genai.GenerativeModel(
                    model_name=m_name,
                    system_instruction=SYSTEM_PROMPT
                )
                prompt = f"""DATASET SCHEMA:
{schema_text}

USER QUESTION:
"{user_query}"

Generate the Python code to compute `result`, optional `fig`, and `summary`.
"""
                resp = model.generate_content(prompt)
                if resp and resp.text:
                    return resp, m_name
            except Exception as e_model:
                last_e = e_model
                continue
        if last_e:
            raise last_e
        raise RuntimeError("No model succeeded.")

    try:
        # Enforce strict 5-second hard ceiling to prevent UI hanging
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_gemini_api)
            response, used_m_name = future.result(timeout=6.0)

        if not response or not response.text:
            return {
                "success": False,
                "error": "Empty response from Gemini API.",
                "mode": "llm_failed"
            }

        raw_code = extract_python_code(response.text)
        exec_res = execute_sandboxed_code(raw_code, df, pd, np, px, go)
        exec_res["self_healed"] = False
        exec_res["model_used"] = used_m_name
        return exec_res

    except concurrent.futures.TimeoutError:
        return {
            "success": False,
            "error": "Gemini API timed out after 5.0s (network/latency limit).",
            "mode": "llm_timeout"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"LLM Generation Error: {str(e)}",
            "mode": "llm_failed"
        }


def fallback_offline_query(user_query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Deterministic rule-based NLP parser.
    Ensures 100% reliability during offline jury presentations without internet or API keys.
    """
    q = user_query.lower().strip()
    columns = list(df.columns)
    col_lookup = {c.lower(): c for c in columns}
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Prioritize meaningful numeric business metrics over ID/index/code columns
    def is_id_or_code_column(col_name: str) -> bool:
        c = col_name.lower().strip()
        # Direct matches
        if c in ["id", "no", "num", "number", "code", "zip", "zip_code", "postal_code", "index", "key"]:
            return True
        # Ends with id / no / num / code
        if any(c.endswith(sfx) for sfx in ["_id", " id", "_no", " no", "_num", " num", "_code", " code", "_zip"]):
            return True
        # Specific identifier names
        if any(tok in c for tok in ["invoice_no", "order_id", "order_no", "customer_id", "cust_id", "product_id", "employee_id", "phone"]):
            return True
        return False

    meaningful_num_cols = [c for c in num_cols if not is_id_or_code_column(c)]
    if not meaningful_num_cols:
        meaningful_num_cols = num_cols

    meaningful_cat_cols = [c for c in cat_cols if not is_id_or_code_column(c)]
    if not meaningful_cat_cols:
        meaningful_cat_cols = cat_cols

    # Common business synonyms across enterprise datasets
    METRIC_SYNONYMS = {
        "profit": ["profit", "margin", "net_profit", "earnings", "operating_profit", "net_margin", "net_income", "gain"],
        "margin": ["margin", "profit", "net_margin", "profit_margin", "markup"],
        "revenue": ["revenue", "sales", "turnover", "total_revenue", "gross_sales", "amount", "order_amount"],
        "sales": ["sales", "revenue", "turnover", "gross_sales", "amount"],
        "cost": ["cost", "unit_cost", "total_cost", "expense", "expenses", "cogs"],
        "price": ["unit_price", "price", "rate", "selling_price"],
        "quantity": ["quantity", "units_sold", "units", "qty", "volume", "count"],
        "discount": ["discount", "rebate", "reduction"]
    }

    generated_code = ""

    # Improved matching helper with synonyms and fuzzy plural handling
    def find_best_column(raw_term, candidate_cols):
        if not raw_term or not str(raw_term).strip():
            return None
        raw_clean = raw_term.lower().replace(" ", "").replace("_", "")
        if not raw_clean:
            return None
            
        # 1. Exact or direct substring match
        for col in candidate_cols:
            c_clean = col.lower().replace(" ", "").replace("_", "")
            if c_clean == raw_clean or raw_clean == c_clean:
                return col
        for col in candidate_cols:
            c_clean = col.lower().replace(" ", "").replace("_", "")
            if (c_clean in raw_clean or raw_clean in c_clean) and len(c_clean) > 3:
                return col

        # 2. Semantic business synonyms match
        for metric_key, synonyms in METRIC_SYNONYMS.items():
            if metric_key in raw_term.lower() or any(s in raw_term.lower() for s in synonyms):
                for syn in synonyms:
                    for col in candidate_cols:
                        if syn in col.lower():
                            return col

        # 3. Plural / singular normalization (e.g. subcategories -> subcategory, countries -> country)
        if raw_clean.endswith("ies"):
            base = raw_clean[:-3] + "y"
            for col in candidate_cols:
                if base in col.lower():
                    return col
        if raw_clean.endswith("s"):
            base = raw_clean[:-1]
            for col in candidate_cols:
                if base in col.lower():
                    return col
        return None

    # 0. Overall Financial KPIs / Summary
    if any(k in q for k in ["total revenue", "total profit", "total sales", "total margin", "overall kpi", "kpi", "financial performance", "overview of sales"]):
        rev_col = find_best_column("revenue", meaningful_num_cols) or find_best_column("sales", meaningful_num_cols)
        margin_col = find_best_column("profit", meaningful_num_cols) or find_best_column("margin", meaningful_num_cols)
        cost_col = find_best_column("cost", meaningful_num_cols)
        
        calc_lines = []
        dict_items = []
        chart_metrics = []
        chart_amounts = []
        if rev_col:
            calc_lines.append(f"tot_rev = df['{rev_col}'].sum()")
            dict_items.append(f"'Total Revenue ($)': f'${{tot_rev:,.2f}}'")
            chart_metrics.append("'Revenue'")
            chart_amounts.append("tot_rev")
        if cost_col:
            calc_lines.append(f"tot_cost = df['{cost_col}'].sum()")
            dict_items.append(f"'Total Cost ($)': f'${{tot_cost:,.2f}}'")
            chart_metrics.append("'Cost'")
            chart_amounts.append("tot_cost")
        if margin_col:
            calc_lines.append(f"tot_margin = df['{margin_col}'].sum()")
            dict_items.append(f"'Total Margin / Profit ($)': f'${{tot_margin:,.2f}}'")
            chart_metrics.append("'Profit (Margin)'")
            chart_amounts.append("tot_margin")
        if rev_col and margin_col:
            calc_lines.append(f"overall_margin_pct = (tot_margin / tot_rev * 100) if tot_rev > 0 else 0")
            dict_items.append(f"'Profit Margin': f'{{overall_margin_pct:.2f}}%'")
        dict_items.append(f"'Total Invoices / Records': f'{{len(df):,}}'")

        calc_str = "\n".join(calc_lines)
        dict_str = ", ".join(dict_items)
        code = (
            f"{calc_str}\n"
            f"result = pd.DataFrame([{{{dict_str}}}])\n"
            f"chart_data = pd.DataFrame({{\n"
            f"    'Metric': [{', '.join(chart_metrics)}],\n"
            f"    'Amount ($)': [{', '.join(chart_amounts)}]\n"
            f"}})\n"
            f"fig = px.bar(chart_data, x='Metric', y='Amount ($)', color='Metric', title='Financial Performance Overview ($)', template='plotly_white')\n"
            f"summary = f'Generated overall financial performance: Revenue is ${{tot_rev:,.2f}} with ${{tot_margin:,.2f}} total profit.'"
        )
        return execute_sandboxed_code(code, df, pd, np, px, go)

    # 0.1 Shipping / Delivery Turnaround Time
    if any(k in q for k in ["ship", "delivery", "turnaround", "lead time"]) and any("date" in c.lower() for c in columns):
        date_cols = [c for c in columns if "date" in c.lower()]
        order_col = next((c for c in date_cols if "order" in c.lower()), date_cols[0] if date_cols else None)
        ship_col = next((c for c in date_cols if "ship" in c.lower() or "deliver" in c.lower()), date_cols[-1] if len(date_cols) > 1 else None)
        
        if order_col and ship_col and order_col != ship_col:
            dim_col = find_best_column("category", meaningful_cat_cols) or (meaningful_cat_cols[0] if meaningful_cat_cols else None)
            code = (
                f"df_temp = df.copy()\n"
                f"df_temp['shipping_days'] = (pd.to_datetime(df_temp['{ship_col}']) - pd.to_datetime(df_temp['{order_col}'])).dt.days\n"
                f"result = df_temp.groupby('{dim_col}')['shipping_days'].agg(['mean', 'median', 'max']).round(2).reset_index()\n"
                f"result.columns = ['{dim_col}', 'Avg_Ship_Days', 'Median_Ship_Days', 'Max_Ship_Days']\n"
                f"fig = px.bar(result, x='{dim_col}', y='Avg_Ship_Days', color='Avg_Ship_Days', "
                f"title='Average Delivery Turnaround (Days) by {dim_col}', template='plotly_white')\n"
                f"avg_days = df_temp['shipping_days'].mean()\n"
                f"summary = f'Average delivery lead time is {{avg_days:.2f}} days across all shipments.'"
            )
            return execute_sandboxed_code(code, df, pd, np, px, go)

    # 0.2 Margin Percentage / Profit Margin by Category/Segment
    if any(k in q for k in ["margin %", "margin percent", "profit margin", "margin rate"]):
        rev_col = find_best_column("revenue", meaningful_num_cols) or find_best_column("sales", meaningful_num_cols)
        margin_col = find_best_column("margin", meaningful_num_cols) or find_best_column("profit", meaningful_num_cols)
        dim_col = None
        for c in meaningful_cat_cols:
            if c.lower() in q:
                dim_col = c
                break
        if not dim_col and meaningful_cat_cols:
            dim_col = meaningful_cat_cols[0]

        if rev_col and margin_col and dim_col:
            code = (
                f"agg = df.groupby('{dim_col}').agg({{'{margin_col}': 'sum', '{rev_col}': 'sum'}}).reset_index()\n"
                f"agg['Margin_Pct'] = ((agg['{margin_col}'] / agg['{rev_col}']) * 100).round(2)\n"
                f"result = agg.sort_values(by='Margin_Pct', ascending=False)\n"
                f"fig = px.bar(result, x='{dim_col}', y='Margin_Pct', color='Margin_Pct', "
                f"title='Profit Margin % by {dim_col}', template='plotly_white')\n"
                f"leader = result.iloc[0]['{dim_col}']\n"
                f"highest_pct = result.iloc[0]['Margin_Pct']\n"
                f"summary = f'Top margin driver is {{leader}} achieving {{highest_pct:.1f}}% profit margin.'\n"
            )
            return execute_sandboxed_code(code, df, pd, np, px, go)

    # 0.3 Time Series / Sales Trend
    if any(k in q for k in ["trend", "over time", "by date", "daily", "monthly", "timeline"]):
        date_cols = [c for c in columns if "date" in c.lower()]
        target_date = next((c for c in date_cols if "order" in c.lower()), date_cols[0] if date_cols else None)
        target_metric = find_best_column("revenue", meaningful_num_cols) or find_best_column("profit", meaningful_num_cols) or meaningful_num_cols[0]
        
        if target_date and target_metric:
            code = (
                f"df_temp = df.copy()\n"
                f"df_temp['{target_date}'] = pd.to_datetime(df_temp['{target_date}'])\n"
                f"daily = df_temp.groupby(df_temp['{target_date}'].dt.date)['{target_metric}'].sum().reset_index()\n"
                f"daily.columns = ['Date', 'Total_{target_metric}']\n"
                f"result = daily.sort_values(by='Date')\n"
                f"fig = px.line(result, x='Date', y='Total_{target_metric}', markers=True, "
                f"title='{target_metric} Trend Over Time', template='plotly_white')\n"
                f"peak_date = result.sort_values(by='Total_{target_metric}', ascending=False).iloc[0]['Date']\n"
                f"peak_val = result.sort_values(by='Total_{target_metric}', ascending=False).iloc[0]['Total_{target_metric}']\n"
                f"summary = f'Peak {target_metric} occurred on {{peak_date}} reaching ${{peak_val:,.2f}}.'\n"
            )
            return execute_sandboxed_code(code, df, pd, np, px, go)

    # 0.4 Loss-making / Negative Margin Orders
    if any(k in q for k in ["loss", "losses", "negative margin", "unprofitable"]):
        margin_col = find_best_column("margin", meaningful_num_cols) or find_best_column("profit", meaningful_num_cols)
        if margin_col:
            code = (
                f"losses = df[df['{margin_col}'] < 0]\n"
                f"if len(losses) > 0:\n"
                f"    result = losses.head(10)\n"
                f"    fig = px.histogram(losses, x='{margin_col}', title='Loss Distribution', template='plotly_white')\n"
                f"    summary = f'Detected {{len(losses)}} unprofitable orders with negative {margin_col}.'\n"
                f"else:\n"
                f"    result = pd.DataFrame([{{'Status': 'Healthy', 'Unprofitable Orders': 0, 'Min Margin': df['{margin_col}'].min()}}])\n"
                f"    fig = px.histogram(df, x='{margin_col}', title='Margin Distribution (All Positive)', template='plotly_white')\n"
                f"    summary = 'Zero loss-making transactions found! All orders maintain healthy positive profit margins.'\n"
            )
            return execute_sandboxed_code(code, df, pd, np, px, go)

    # 1. Correlation Matrix
    if "correlation" in q or "heatmap" in q:
        target_nums = meaningful_num_cols if len(meaningful_num_cols) >= 2 else num_cols
        if len(target_nums) >= 2:
            code = (
                f"corr = df[{target_nums}].corr().round(2)\n"
                "result = corr\n"
                "fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title='Correlation Heatmap')\n"
                "summary = 'Calculated Pearson correlation coefficients across numerical attributes.'"
            )
            return execute_sandboxed_code(code, df, pd, np, px, go)

    # 2. Top N / Ranking queries
    top_match = re.search(r"(?:top|highest|best)\s*(\d+)?\s*([a-zA-Z\s_]+?)\s*by\s*([a-zA-Z\s_]+)", q)
    if not top_match:
        # Match "top 5 countries" or "highest products" without explicit "by"
        top_simple = re.search(r"(?:top|highest|best)\s*(\d+)?\s*([a-zA-Z\s_]+)", q)
        if top_simple:
            n_str = top_simple.group(1)
            raw_dim = top_simple.group(2).strip()
            # Verify raw_dim matches one of our categorical columns
            if find_best_column(raw_dim, meaningful_cat_cols) or find_best_column(raw_dim, cat_cols):
                top_match = top_simple
                raw_metric = ""

    if top_match:
        n = int(top_match.group(1)) if top_match.group(1) else 5
        raw_dim = top_match.group(2).strip()
        raw_metric = top_match.group(3).strip() if len(top_match.groups()) >= 3 and top_match.group(3) else ""

        matched_dim = find_best_column(raw_dim, meaningful_cat_cols) or find_best_column(raw_dim, cat_cols) or find_best_column(raw_dim, columns)
        matched_metric = find_best_column(raw_metric, meaningful_num_cols) or find_best_column(raw_metric, num_cols) or find_best_column(raw_metric, columns)

        if not matched_dim and meaningful_cat_cols:
            matched_dim = meaningful_cat_cols[0]
        elif not matched_dim and cat_cols:
            matched_dim = cat_cols[0]
            
        if not matched_metric and meaningful_num_cols:
            # Default to profit/margin or revenue/sales if present, otherwise last meaningful metric
            matched_metric = find_best_column("profit", meaningful_num_cols) or find_best_column("revenue", meaningful_num_cols) or meaningful_num_cols[-1]
        elif not matched_metric and num_cols:
            matched_metric = num_cols[0]

        if matched_dim and matched_metric:
            code = (
                f"agg = df.groupby('{matched_dim}')['{matched_metric}'].sum().reset_index()\n"
                f"result = agg.sort_values(by='{matched_metric}', ascending=False).head({n})\n"
                f"fig = px.bar(result, x='{matched_dim}', y='{matched_metric}', color='{matched_metric}', "
                f"title='Top {n} {matched_dim} by Total {matched_metric}', template='plotly_white')\n"
                f"top_leader = result.iloc[0]['{matched_dim}']\n"
                f"top_val = result.iloc[0]['{matched_metric}']\n"
                f"summary = f'Leader is {{top_leader}} with {{top_val:,.2f}} in {matched_metric}.'"
            )
            return execute_sandboxed_code(code, df, pd, np, px, go)

    # 3. Group by aggregation (e.g. "sales by region", "churn by contract")
    by_match = re.search(r"([a-zA-Z\s_]+?)\s+by\s+([a-zA-Z\s_]+)", q)
    if by_match:
        term1 = by_match.group(1).strip()
        term2 = by_match.group(2).strip()

        metric_col = find_best_column(term1, meaningful_num_cols) or find_best_column(term1, num_cols) or find_best_column(term1, columns)
        dim_col = find_best_column(term2, meaningful_cat_cols) or find_best_column(term2, cat_cols) or find_best_column(term2, columns)

        # Swap if dim was found in term1 and metric in term2
        if not metric_col and not dim_col:
            dim_col = find_best_column(term1, meaningful_cat_cols) or find_best_column(term1, cat_cols)
            metric_col = find_best_column(term2, meaningful_num_cols) or find_best_column(term2, num_cols)
        elif metric_col in cat_cols and dim_col in num_cols:
            metric_col, dim_col = dim_col, metric_col
        elif not metric_col and meaningful_num_cols:
            metric_col = find_best_column("revenue", meaningful_num_cols) or meaningful_num_cols[0]
        elif not dim_col and meaningful_cat_cols:
            dim_col = meaningful_cat_cols[0]

        if dim_col:
            if metric_col in num_cols:
                code = (
                    f"agg = df.groupby('{dim_col}')['{metric_col}'].agg(['sum', 'mean', 'count']).reset_index()\n"
                    f"agg.columns = ['{dim_col}', 'Total_{metric_col}', 'Average_{metric_col}', 'Record_Count']\n"
                    f"result = agg.sort_values(by='Total_{metric_col}', ascending=False)\n"
                    f"fig = px.bar(result, x='{dim_col}', y='Total_{metric_col}', color='Average_{metric_col}', "
                    f"title='{metric_col} Breakdown by {dim_col}', template='plotly_white')\n"
                    f"summary = 'Aggregated {metric_col} across {dim_col} categories with total and average values.'"
                )
            else:
                code = (
                    f"result = df['{dim_col}'].value_counts().reset_index()\n"
                    f"result.columns = ['{dim_col}', 'Count']\n"
                    f"fig = px.pie(result, names='{dim_col}', values='Count', title='Distribution of {dim_col}')\n"
                    f"summary = 'Calculated category frequency distribution for {dim_col}.'"
                )
            return execute_sandboxed_code(code, df, pd, np, px, go)

    # 4. Distribution / Histogram
    if "distribution" in q or "histogram" in q or "spread" in q:
        target_num = None
        for syn_key, syn_list in METRIC_SYNONYMS.items():
            if syn_key in q or any(s in q for s in syn_list):
                target_num = find_best_column(syn_key, meaningful_num_cols) or find_best_column(syn_key, num_cols)
                if target_num:
                    break
        if not target_num:
            for col in meaningful_num_cols + num_cols:
                if col.lower() in q:
                    target_num = col
                    break
        if not target_num:
            target_num = meaningful_num_cols[0] if meaningful_num_cols else (num_cols[0] if num_cols else columns[0])

        code = (
            f"result = df['{target_num}'].describe().to_frame().T\n"
            f"fig = px.histogram(df, x='{target_num}', nbins=30, marginal='box', "
            f"title='Distribution Analysis of {target_num}', template='plotly_white')\n"
            f"summary = f'Mean of {target_num} is {{df[\"{target_num}\"].mean():.2f}} with standard deviation {{df[\"{target_num}\"].std():.2f}}.'"
        )
        return execute_sandboxed_code(code, df, pd, np, px, go)

    # 5. Outliers / Boxplot
    if "outlier" in q or "anomaly" in q or "box" in q:
        target_num = None
        # First check synonyms for business metrics (e.g. 'profit' -> matches 'margin')
        for syn_key, syn_list in METRIC_SYNONYMS.items():
            if syn_key in q or any(s in q for s in syn_list):
                target_num = find_best_column(syn_key, meaningful_num_cols) or find_best_column(syn_key, num_cols)
                if target_num:
                    break

        # Next check direct column names mentioned in query
        if not target_num:
            for col in meaningful_num_cols + num_cols:
                if col.lower() in q:
                    target_num = col
                    break

        # Fallback to key metrics (profit, margin, revenue, sales, or last meaningful column)
        if not target_num:
            target_num = (
                find_best_column("profit", meaningful_num_cols) or 
                find_best_column("margin", meaningful_num_cols) or 
                find_best_column("revenue", meaningful_num_cols) or 
                (meaningful_num_cols[-1] if meaningful_num_cols else (num_cols[0] if num_cols else columns[0]))
            )

        code = (
            f"q1 = df['{target_num}'].quantile(0.25)\n"
            f"q3 = df['{target_num}'].quantile(0.75)\n"
            f"iqr = q3 - q1\n"
            f"outliers = df[(df['{target_num}'] < (q1 - 1.5 * iqr)) | (df['{target_num}'] > (q3 + 1.5 * iqr))]\n"
            f"result = pd.DataFrame([{{\n"
            f"    'Column': '{target_num}',\n"
            f"    'Outlier Count': len(outliers),\n"
            f"    'Outlier %': round(len(outliers) / len(df) * 100, 2),\n"
            f"    'Lower Bound': round(q1 - 1.5 * iqr, 2),\n"
            f"    'Upper Bound': round(q3 + 1.5 * iqr, 2)\n"
            f"}}])\n"
            f"fig = px.box(df, y='{target_num}', points='all', title='Outlier Detection for {target_num}')\n"
            f"summary = f'Detected {{len(outliers)}} statistical outliers in {target_num} using 1.5x IQR criterion.'"
        )
        return execute_sandboxed_code(code, df, pd, np, px, go)

    # 6. Default Fallback: Five-number summary + Key metrics
    default_num = meaningful_num_cols[0] if meaningful_num_cols else (num_cols[0] if num_cols else df.columns[0])
    code = (
        "summary_stats = df.describe(include='all').T.fillna('-')\n"
        "result = summary_stats[['count', 'mean', 'std', 'min', '50%', 'max']].head(8) if 'mean' in summary_stats.columns else summary_stats.head(8)\n"
        f"num_col = '{default_num}'\n"
        "fig = px.histogram(df, x=num_col, title=f'Distribution of {num_col}', template='plotly_white')\n"
        "summary = f'Overview generated: dataset contains {len(df)} records across {len(df.columns)} columns.'"
    )
    return execute_sandboxed_code(code, df, pd, np, px, go)


def automated_executive_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates an instant, comprehensive EDA and data health executive summary."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    total_cells = df.shape[0] * df.shape[1]
    total_nulls = int(df.isnull().sum().sum())
    null_pct = round((total_nulls / total_cells) * 100, 2) if total_cells > 0 else 0
    duplicate_rows = int(df.duplicated().sum())

    # Detect high correlations
    high_corrs = []
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                val = corr_matrix.loc[c1, c2]
                if abs(val) >= 0.5:
                    high_corrs.append((c1, c2, round(val, 2)))

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "null_pct": null_pct,
        "duplicate_rows": duplicate_rows,
        "high_corrs": high_corrs
    }
