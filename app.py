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
from audio_transcriber import transcribe_user_speech

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
    [
        "Sample: Global 100 Sales",
        "Sample: Retail Superstore",
        "Sample: Customer Churn",
        "Sample: HR Attrition",
        "Upload Custom CSV"
    ]
)

@st.cache_data
def load_sample_data(name: str):
    if name == "global100":
        return pd.read_csv("datasets/global_sales_100.csv")
    elif name == "superstore":
        return pd.read_csv("datasets/superstore_sales.csv")
    elif name == "churn":
        return pd.read_csv("datasets/customer_churn.csv")
    elif name == "attrition":
        return pd.read_csv("datasets/employee_attrition.csv")
    return None

df = None
if dataset_source == "Sample: Global 100 Sales":
    df = load_sample_data("global100")
    dataset_name = "Global Sales 100 Records"
    sample_questions = [
        "What are the top 5 countries by Total Profit?",
        "Show Total Revenue by Region",
        "Compare Total Revenue across Sales Channels",
        "What is the distribution of Units Sold?",
        "Check correlation between Total Revenue and Total Profit",
        "Detect outliers in Total Profit"
    ]
elif dataset_source == "Sample: Retail Superstore":
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

# Engine Status & Mode Selection
st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Execution Engine")

mode_choice = st.sidebar.radio(
    "Engine Mode:",
    ["⚡ Smart Auto (Fastest)", "💻 100% Offline Local (Instant)"],
    index=0
)

if api_key:
    st.sidebar.success("⚡ **Gemini 1.5 Flash (Live)**\n\n*Authenticated via Secrets*")
else:
    st.sidebar.info("⚡ **Offline Engine Active**\n\n*(Zero-API mode)*")
    with st.sidebar.expander("⚙️ Local Key (Optional)"):
        manual_key = st.text_input("Manual Key:", type="password", help="On Streamlit Cloud, add GEMINI_API_KEY to App Settings -> Secrets.")
        if manual_key:
            api_key = manual_key.strip()

# Initialize session state for query input
if "query_box_input" not in st.session_state:
    st.session_state["query_box_input"] = sample_questions[0] if sample_questions else ""
if "run_now" not in st.session_state:
    st.session_state["run_now"] = False

st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Suggested Questions")

def select_sample_query(q_text):
    st.session_state["query_box_input"] = q_text
    st.session_state["run_now"] = True

for sq in sample_questions:
    st.sidebar.button(f"👉 {sq}", key=f"sq_{sq}", on_click=select_sample_query, args=(sq,))


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
tab_chat, tab_eda, tab_security = st.tabs([
    "💬 Conversational Analyst",
    "📊 Automated EDA & Data Health",
    "🛡️ AST Safety Sandbox"
])


# ==========================================
# TAB 1: Conversational Analyst
# ==========================================
with tab_chat:
    st.markdown("#### Ask Any Analytical or Business Question")

    # Audio Voice Input via Whisper
    with st.expander("🎙️ Speak Your Question (Whisper Voice Input)", expanded=False):
        st.caption("Click the microphone below, speak your question (e.g. *'What are the top 5 subcategories by profit?'*), and click stop.")
        audio_clip = st.audio_input("Record audio question:")
        if audio_clip is not None:
            audio_bytes = audio_clip.read()
            audio_hash = len(audio_bytes)
            if st.session_state.get("last_audio_hash") != audio_hash:
                st.session_state["last_audio_hash"] = audio_hash
                with st.spinner("Transcribing speech with Whisper / Speech-to-Text..."):
                    ok, speech_text = transcribe_user_speech(audio_bytes, api_key=api_key)
                    if ok and speech_text.strip():
                        st.session_state["query_box_input"] = speech_text
                        st.session_state["run_now"] = True
                        st.rerun()
                    else:
                        st.warning(f"Voice Notice: {speech_text}")
    
    col_input, col_btn = st.columns([5, 1])
    user_query = col_input.text_input(
        "Enter your query in plain English:",
        key="query_box_input",
        placeholder="e.g. What are the top 5 subcategories by profit?"
    )
    submit_btn = col_btn.button("Run Query 🚀", use_container_width=True)

    should_run = submit_btn or st.session_state.get("run_now", False)
    st.session_state["run_now"] = False

    active_query = user_query.strip()
    if should_run:
        if not active_query:
            st.warning("Please type a question or select a sample question.")
        else:
            with st.spinner("Analyzing data and generating execution pipeline..."):
                start_time = time.time()
                
                # Check execution mode
                fallback_reason = None
                if mode_choice == "⚡ Smart Auto (Fastest)" and api_key and len(api_key.strip()) > 10:
                    exec_result = run_llm_query(active_query, df, api_key=api_key)
                    if not exec_result["success"]:
                        fallback_reason = exec_result.get("error")
                        exec_result = fallback_offline_query(active_query, df)
                else:
                    exec_result = fallback_offline_query(active_query, df)

                latency = round(time.time() - start_time, 3)

            # Display Results
            if exec_result["success"]:
                st.success(f"Query executed safely in **{latency}s** | Security Verification: **PASSED**")
                
                # If fallback was used due to API issue, show friendly notification
                if fallback_reason:
                    with st.expander("ℹ️ Connection Notice (Switched to Fast Local Engine)"):
                        st.caption(f"**Notice:** {fallback_reason}")
                        st.caption("Processed seamlessly using the local heuristic analytics engine.")
                
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
    st.markdown("#### 🛡️ AST-Based Security Sandbox")
    st.markdown("""
    InsightGen enforces memory safety and code isolation using an **Abstract Syntax Tree (AST) static analysis visitor** combined with a restricted execution scope.
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
