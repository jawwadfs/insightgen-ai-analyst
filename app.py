"""
InsightGen: Autonomous AI Data Analyst & Conversational BI Engine
Generative AI Final Jury 2 - Problem Statement 91
Built with Streamlit, Pandas, Plotly, AST Safety Sandbox, and LLM / Heuristic Engine.
"""

import time
import os
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from safety_sandbox import validate_python_code, execute_sandboxed_code
from analytics_engine import (
    run_llm_query,
    fallback_offline_query,
    get_dataset_metadata,
    automated_executive_summary
)

# Page configuration
st.set_page_config(
    page_title="InsightGen — AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/isometric/100/combo-chart.png", width=64)
st.sidebar.title("InsightGen AI")
st.sidebar.caption("PS 91: Natural-Language CSV Analytics Engine")

# Dataset Selection
st.sidebar.markdown("### 1. Ingest Dataset")
dataset_source = st.sidebar.radio(
    "Choose Data Source:",
    ["Sample: Retail Superstore", "Sample: Customer Churn", "Sample: HR Attrition", "Upload Custom CSV"]
)

@st.cache_data
def load_sample_data(name: str):
    if name == "superstore":
        return pd.read_csv("datasets/superstore_sales.csv")
    elif name == "churn":
        return pd.read_csv("datasets/customer_churn.csv")
    elif name == "attrition":
        return pd.read_csv("datasets/employee_attrition.csv")
    return None

df = None
if dataset_source == "Sample: Retail Superstore":
    df = load_sample_data("superstore")
    dataset_name = "Superstore Sales Analytics"
    sample_questions = [
        "What are the top 5 subcategories by profit?",
        "Show total sales by region",
        "What is the distribution of sales?",
        "Check correlation matrix across metrics",
        "Detect outliers in profit"
    ]
elif dataset_source == "Sample: Customer Churn":
    df = load_sample_data("churn")
    dataset_name = "Telecom Customer Churn Analytics"
    sample_questions = [
        "What is the churn breakdown by contract type?",
        "Compare monthly charges across payment methods",
        "Show distribution of tenure months",
        "Correlation between tenure and total charges",
        "Top 5 contracts by monthly charges"
    ]
elif dataset_source == "Sample: HR Attrition":
    df = load_sample_data("attrition")
    dataset_name = "Employee Attrition & HR Analytics"
    sample_questions = [
        "Show attrition rate by department",
        "Compare monthly income across job roles",
        "What is the distribution of employee age?",
        "Check correlation between years at company and salary",
        "Detect outliers in monthly income"
    ]
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            dataset_name = uploaded_file.name
            sample_questions = [
                "Show summary statistics of this dataset",
                "Check correlation matrix",
                "Show distribution of numerical features"
            ]
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
    else:
        st.sidebar.info("Upload a CSV file to begin analysis.")
        sample_questions = []

# Secure API Key resolution from Streamlit Secrets or Environment
api_key = ""
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
    api_key = str(st.secrets["GEMINI_API_KEY"]).strip()
elif os.getenv("GEMINI_API_KEY"):
    api_key = str(os.getenv("GEMINI_API_KEY")).strip()

# Engine Status
st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Execution Engine")
if api_key:
    st.sidebar.success("⚡ **Gemini 1.5 Flash (Live)**\n\n*Authenticated via Streamlit Secrets*")
else:
    st.sidebar.info("⚡ **Smart Offline Heuristic Engine**\n\n*(Zero-API mode active)*")
    with st.sidebar.expander("⚙️ Local Key (Optional)"):
        manual_key = st.text_input("Manual Key:", type="password", help="On Streamlit Cloud, add GEMINI_API_KEY to App Settings -> Secrets.")
        if manual_key:
            api_key = manual_key.strip()

st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Suggested Questions")
selected_sample = None
for sq in sample_questions:
    if st.sidebar.button(f"👉 {sq}", key=f"sq_{sq}"):
        selected_sample = sq


# Main Dashboard Header
st.markdown('<div class="main-header">InsightGen — Autonomous AI Data Analyst</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Active Dataset: <b>{dataset_name if df is not None else "None"}</b> | '
    '<span class="badge-safe">🛡️ AST Sandbox Active</span> | '
    '<span class="badge-safe">⚡ Real-Time Self-Healing</span></div>',
    unsafe_allow_html=True
)

if df is None:
    st.warning("Please select or upload a dataset in the sidebar to begin.")
    st.stop()


# Tabs
tab_chat, tab_eda, tab_security, tab_jury = st.tabs([
    "💬 Conversational Analyst",
    "📊 Automated EDA & Data Health",
    "🛡️ AST Safety Sandbox Demo",
    "🎓 Jury Defense & Rubric"
])


# ==========================================
# TAB 1: Conversational Analyst
# ==========================================
with tab_chat:
    st.markdown("#### Ask Any Analytical or Business Question")
    
    col_input, col_btn = st.columns([5, 1])
    default_text = selected_sample if selected_sample else ""
    user_query = col_input.text_input(
        "Enter your query in plain English:",
        value=default_text,
        placeholder="e.g. What are the top 5 subcategories by profit?"
    )
    submit_btn = col_btn.button("Run Query 🚀", use_container_width=True)

    if submit_btn or selected_sample:
        if not user_query.strip():
            st.warning("Please type a question or select a sample question.")
        else:
            with st.spinner("Analyzing data and generating validated execution pipeline..."):
                start_time = time.time()
                
                # Check execution mode
                if api_key:
                    exec_result = run_llm_query(user_query, df, api_key=api_key)
                    if not exec_result["success"] and "LLM" in str(exec_result.get("error")):
                        st.info("Falling back to local heuristic engine due to API timeout/quota.")
                        exec_result = fallback_offline_query(user_query, df)
                else:
                    exec_result = fallback_offline_query(user_query, df)

                latency = round(time.time() - start_time, 3)

            # Display Results
            if exec_result["success"]:
                st.success(f"Query executed safely in **{latency}s** | Security Verification: **PASSED**")
                
                # 1. Business Insight / Summary
                if exec_result.get("summary"):
                    st.info(f"💡 **Executive Insight:** {exec_result['summary']}")

                # 2. Visualization
                fig = exec_result.get("fig")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

                # 3. Tabular Output
                res_data = exec_result.get("result")
                if res_data is not None:
                    st.markdown("##### 📋 Result Table")
                    if isinstance(res_data, (pd.DataFrame, pd.Series)):
                        st.dataframe(res_data, use_container_width=True)
                    else:
                        st.write(res_data)

                # 4. Under The Hood: Code & Safety Inspection
                with st.expander("🔍 Inspect Generated Python Code & AST Audit"):
                    col_code, col_audit = st.columns([3, 2])
                    with col_code:
                        st.markdown("**Executed Sandboxed Code:**")
                        st.code(exec_result.get("code", "# No code captured"), language="python")
                    with col_audit:
                        st.markdown("**AST Safety Audit:**")
                        st.markdown("- **Syntax Verification:** Valid Python AST")
                        st.markdown("- **Blacklist Check:** `os`, `sys`, `eval`, `open` blocked")
                        st.markdown("- **File Mutations:** Prohibited")
                        st.markdown(f"- **Self-Healing Applied:** `{exec_result.get('self_healed', False)}`")
                        st.markdown(f"- **Execution Latency:** `{latency}s`")
            else:
                st.error(f"Execution Error: {exec_result.get('error')}")
                if exec_result.get("code"):
                    st.code(exec_result["code"], language="python")


# ==========================================
# TAB 2: Automated EDA & Data Health
# ==========================================
with tab_eda:
    st.markdown("#### Comprehensive Exploratory Data Analysis & Quality Audit")
    summary = automated_executive_summary(df)

    # Metric Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Records", f"{summary['rows']:,}")
    kpi2.metric("Total Features", f"{summary['columns']}")
    kpi3.metric("Missing Value Rate", f"{summary['null_pct']}%")
    kpi4.metric("Duplicate Rows", f"{summary['duplicate_rows']}")

    st.markdown("---")
    col_corr, col_dist = st.columns([1, 1])

    with col_corr:
        st.markdown("##### 📌 Correlation Matrix")
        if len(summary["num_cols"]) >= 2:
            corr_df = df[summary["num_cols"]].corr().round(2)
            fig_corr = px.imshow(
                corr_df,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Numerical Correlation Heatmap"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            if summary["high_corrs"]:
                st.caption(f"**Key Associations:** {', '.join([f'{c1} & {c2} (r={r})' for c1, c2, r in summary['high_corrs'][:3]])}")
        else:
            st.info("Not enough numeric columns for correlation.")

    with col_dist:
        st.markdown("##### 📌 Dynamic Feature Distribution")
        if summary["num_cols"]:
            selected_feature = st.selectbox("Select numeric feature to inspect:", summary["num_cols"])
            fig_dist = px.histogram(
                df,
                x=selected_feature,
                marginal="box",
                title=f"Distribution & Outlier Bounds of {selected_feature}",
                template="plotly_white"
            )
            st.plotly_chart(fig_dist, use_container_width=True)


# ==========================================
# TAB 3: AST Safety Sandbox Live Demo
# ==========================================
with tab_security:
    st.markdown("#### 🛡️ AST-Based Security Sandbox Proof of Concept")
    st.markdown("""
    The jury explicitly asks: ***"How do you stop wrong/unsafe queries?"***
    InsightGen solves this using an **Abstract Syntax Tree (AST) static analysis visitor** combined with a **restricted Python scope**.
    """)

    st.markdown("##### Test Malicious Code Injection against the Sandbox:")
    sample_attacks = {
        "1. File System Deletion": "import os\nos.system('rm -rf /')",
        "2. Process Spawning": "import subprocess\nsubprocess.run(['ls', '-la'])",
        "3. Arbitrary Code Eval": "malicious = 'print(123)'\neval(malicious)",
        "4. Dunder Introspection": "df.__class__.__bases__[0].__subclasses__()",
        "5. File Read Attempt": "data = open('/etc/passwd').read()"
    }

    selected_attack = st.selectbox("Select or customize an attack payload:", list(sample_attacks.keys()))
    attack_code = st.text_area("Python Payload to Test:", value=sample_attacks[selected_attack], height=100)

    if st.button("Audit and Attempt Execution 🛡️"):
        is_safe, error_msg = validate_python_code(attack_code)
        if not is_safe:
            st.error(f"🚨 **EXECUTION BLOCKED BY AST FIREWALL!**\n\n**Reason:** `{error_msg}`")
        else:
            st.success("Code is compliant with AST safety rules.")


# ==========================================
# TAB 4: Jury Defense & Rubric
# ==========================================
with tab_jury:
    st.markdown("#### 🎓 Final Year Jury 2 Defense & Rubric Alignment")
    
    st.markdown("### 1. Mandatory Syllabus Questions")
    with st.expander("❓ Question 1: How do you stop wrong/unsafe queries?", expanded=True):
        st.markdown("""
        **Our Technical Defense:**
        1. **Static AST Analysis (Abstract Syntax Tree):** Before `exec()` is ever called, the code is parsed into a Python AST. A custom `CodeSafetyVisitor` walks every node to detect and block forbidden modules (`os`, `sys`, `subprocess`), dangerous builtins (`eval`, `open`, `compile`), and dunder traversal (`__class__`, `__subclasses__`).
        2. **Restricted Execution Scope:** The code is executed in an isolated dictionary scope with a sanitized `__builtins__` whitelist, stripping away file and network I/O.
        3. **Read-Only Data Isolation:** Queries operate on an isolated shallow copy of the DataFrame, preventing in-place mutations or data corruption.
        4. **Self-Healing Loop:** If a query fails with a runtime Pandas error (e.g. casing mismatch), the error message is fed back into the model context to auto-correct syntax.
        """)

    with st.expander("❓ Question 2: Where does natural language translation fail?", expanded=True):
        st.markdown("""
        **Our Technical Defense:**
        1. **Lexical Ambiguity & Polysemy:** When a user asks for *"highest grossing"*, without explicit domain mapping, the model may confuse `Sales` with `Profit` or `MonthlyIncome`.
        2. **Implicit Multi-Step Business Logic:** Queries like *"Who are our most loyal customers?"* require defining loyalty (e.g. Tenure > 24 vs Order Count > 10). Without explicit threshold definitions, assumptions vary.
        3. **Out-of-Vocabulary References:** Typographical errors or abbreviations in category names (e.g. *"Tech"* instead of *"Technology"*).
        4. **Mitigation Strategy:** We inject rich schema metadata (column dtypes, exact unique category samples, statistical ranges) into the system prompt to anchor the translation.
        """)

    st.markdown("### 2. Group of 4 Member Responsibilities")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("**Student 1: Data Architect**\n- Dataset curation & schema profiling\n- Missing value & outlier handling\n- Five-number statistics")
    with c2:
        st.info("**Student 2: LLM Engineer**\n- System prompt engineering\n- Few-shot schema context injection\n- Self-healing retry pipeline")
    with c3:
        st.info("**Student 3: Security & Sandbox**\n- AST Visitor static analysis\n- Prohibited AST node blocking\n- Execution scope isolation")
    with c4:
        st.info("**Student 4: BI & Visuals**\n- Dynamic Plotly chart generation\n- Executive takeaway summaries\n- Streamlit UI & responsiveness")
