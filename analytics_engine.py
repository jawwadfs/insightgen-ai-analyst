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
6. OUTPUT FORMAT: Output ONLY pure Python code. Do not include markdown conversational filler outside the code block.
"""


def extract_python_code(raw_response: str) -> str:
    """Extracts raw python code from markdown code fences if present."""
    match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", raw_response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_response.strip()


def run_llm_query(
    user_query: str,
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    model_name: str = "gemini-1.5-flash"
) -> Dict[str, Any]:
    """Generates and executes code using an LLM (Gemini API) with model failover and self-healing retry."""
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
            "error": "Invalid API Key format (too short).",
            "mode": "llm_invalid_key"
        }

    metadata = get_dataset_metadata(df)
    schema_text = format_schema_for_prompt(metadata)

    try:
        import google.generativeai as genai
        genai.configure(api_key=clean_key)
        
        # Discover available models dynamically for this specific API key
        active_models = []
        try:
            for m in genai.list_models():
                if "generateContent" in getattr(m, "supported_generation_methods", []):
                    # m.name is usually "models/gemini-1.5-flash"
                    active_models.append(m.name)
        except Exception as auth_err:
            err_msg = str(auth_err)
            if "API_KEY_INVALID" in err_msg or "400" in err_msg:
                return {
                    "success": False,
                    "error": "Authentication Failed: Gemini API Key is invalid or expired. Check Streamlit Secrets.",
                    "mode": "llm_auth_error"
                }
            elif "PERMISSION_DENIED" in err_msg or "403" in err_msg:
                return {
                    "success": False,
                    "error": "Access Denied: Gemini API Key does not have permissions for Generative Language API.",
                    "mode": "llm_perm_error"
                }

        # Build prioritized list of models to try
        preferred_order = [
            "models/gemini-1.5-flash", "gemini-1.5-flash",
            "models/gemini-2.0-flash", "gemini-2.0-flash",
            "models/gemini-1.5-flash-8b", "gemini-1.5-flash-8b",
            "models/gemini-1.5-pro", "gemini-1.5-pro"
        ]
        
        models_to_try = []
        if active_models:
            for pref in preferred_order:
                if pref in active_models:
                    models_to_try.append(pref)
            # Add any other active models
            for am in active_models:
                if "gemini" in am.lower() and am not in models_to_try:
                    models_to_try.append(am)
        
        if not models_to_try:
            models_to_try = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]

        model = None
        last_gen_err = None
        
        prompt = f"""DATASET SCHEMA:
{schema_text}

USER QUESTION:
"{user_query}"

Generate the Python code to compute `result`, optional `fig`, and `summary`.
"""

        response = None
        used_model_name = ""
        for m_name in models_to_try:
            try:
                curr_model = genai.GenerativeModel(
                    model_name=m_name,
                    system_instruction=SYSTEM_PROMPT
                )
                response = curr_model.generate_content(prompt)
                if response and response.text:
                    model = curr_model
                    used_model_name = m_name
                    break
            except Exception as e_model:
                last_gen_err = str(e_model)
                continue

        if not response or not response.text:
            return {
                "success": False,
                "error": f"LLM Generation Error: {last_gen_err}",
                "mode": "llm_failed"
            }

        raw_code = extract_python_code(response.text)

        # First execution attempt
        exec_res = execute_sandboxed_code(raw_code, df, pd, np, px, go)
        
        # Self-healing retry if runtime error occurred
        if not exec_res["success"] and "Runtime Execution Error" in str(exec_res.get("error")):
            heal_prompt = f"""The previous code encountered an error:
{exec_res['error']}

Original Code:
```python
{raw_code}
```

DATASET SCHEMA:
{schema_text}

Please fix the error (e.g. check column name casing, data types, or aggregation methods) and return corrected Python code only.
"""
            try:
                heal_response = model.generate_content(heal_prompt)
                healed_code = extract_python_code(heal_response.text)
                healed_res = execute_sandboxed_code(healed_code, df, pd, np, px, go)
                healed_res["self_healed"] = True
                healed_res["original_code"] = raw_code
                healed_res["model_used"] = used_model_name
                return healed_res
            except Exception:
                pass

        exec_res["self_healed"] = False
        exec_res["model_used"] = used_model_name
        return exec_res

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

    generated_code = ""

    # Improved matching helper
    def find_best_column(raw_term, candidate_cols):
        raw_clean = raw_term.lower().replace(" ", "").replace("_", "")
        # exact or contains
        for col in candidate_cols:
            c_clean = col.lower().replace(" ", "").replace("_", "")
            if c_clean == raw_clean or raw_clean == c_clean:
                return col
        for col in candidate_cols:
            c_clean = col.lower().replace(" ", "").replace("_", "")
            if (c_clean in raw_clean or raw_clean in c_clean) and len(c_clean) > 3:
                return col
        # plural/singular normalization (e.g. subcategories -> subcategory)
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

    # 1. Correlation Matrix
    if "correlation" in q or "heatmap" in q:
        if len(num_cols) >= 2:
            code = (
                "corr = df.select_dtypes(include=['number']).corr().round(2)\n"
                "result = corr\n"
                "fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title='Correlation Heatmap')\n"
                "summary = 'Calculated Pearson correlation coefficients across numerical attributes.'"
            )
            return execute_sandboxed_code(code, df, pd, np, px, go)

    # 2. Top N / Ranking queries
    top_match = re.search(r"(?:top|highest|best)\s*(\d+)?\s*([a-zA-Z\s_]+?)\s*by\s*([a-zA-Z\s_]+)", q)
    if top_match:
        n = int(top_match.group(1)) if top_match.group(1) else 5
        raw_dim = top_match.group(2).strip()
        raw_metric = top_match.group(3).strip()

        matched_dim = find_best_column(raw_dim, cat_cols) or find_best_column(raw_dim, columns)
        matched_metric = find_best_column(raw_metric, num_cols) or find_best_column(raw_metric, columns)

        if not matched_dim and cat_cols:
            # exclude ID-like columns if possible
            non_id_cats = [c for c in cat_cols if "id" not in c.lower()]
            matched_dim = non_id_cats[0] if non_id_cats else cat_cols[0]
        if not matched_metric and num_cols:
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

        metric_col = find_best_column(term1, num_cols) or find_best_column(term1, columns)
        dim_col = find_best_column(term2, cat_cols) or find_best_column(term2, columns)

        # Swap if dim was found in term1 and metric in term2
        if not metric_col and not dim_col:
            dim_col = find_best_column(term1, cat_cols)
            metric_col = find_best_column(term2, num_cols)
        elif metric_col in cat_cols and dim_col in num_cols:
            metric_col, dim_col = dim_col, metric_col
        elif not metric_col and num_cols:
            metric_col = num_cols[0]
        elif not dim_col and cat_cols:
            dim_col = [c for c in cat_cols if "id" not in c.lower()][0]

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
        target_num = num_cols[0] if num_cols else columns[0]
        for col in num_cols:
            if col.lower() in q:
                target_num = col
                break
        code = (
            f"result = df['{target_num}'].describe().to_frame().T\n"
            f"fig = px.histogram(df, x='{target_num}', nbins=30, marginal='box', "
            f"title='Distribution Analysis of {target_num}', template='plotly_white')\n"
            f"summary = f'Mean of {target_num} is {{df[\"{target_num}\"].mean():.2f}} with standard deviation {{df[\"{target_num}\"].std():.2f}}.'"
        )
        return execute_sandboxed_code(code, df, pd, np, px, go)

    # 5. Outliers / Boxplot
    if "outlier" in q or "anomaly" in q or "box" in q:
        target_num = num_cols[0] if num_cols else columns[0]
        for col in num_cols:
            if col.lower() in q:
                target_num = col
                break
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
    code = (
        "summary_stats = df.describe(include='all').T.fillna('-')\n"
        "result = summary_stats[['count', 'mean', 'std', 'min', '50%', 'max']].head(8) if 'mean' in summary_stats.columns else summary_stats.head(8)\n"
        f"num_col = '{num_cols[0]}' if len({num_cols}) > 0 else df.columns[0]\n"
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
