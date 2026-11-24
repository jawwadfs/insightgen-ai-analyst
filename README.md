# InsightGen — AI Data Analyst & Conversational BI Engine
> **Final Year Generative AI Jury 2 Build — Problem Statement 91**

An enterprise-ready Conversational Business Intelligence engine that translates natural-language questions on CSV datasets into validated Pandas operations, interactive Plotly visualizations, and executive takeaways.

---

## 🚀 Quick Start (One Command)

From the project directory:

```bash
./.venv/bin/streamlit run app.py
```

The web dashboard will automatically launch at `http://localhost:8501`.

---

## 🌟 Key Features

1. **Natural Language Querying:** Ask questions like:
   - *"What are the top 5 subcategories by profit?"*
   - *"Show total sales by region"*
   - *"Check correlation matrix across metrics"*
   - *"Detect outliers in profit"*
2. **Dual AI & Offline Engine:**
   - **Live Mode:** Powered by Google Gemini 1.5 Flash (paste your API key in the sidebar if available).
   - **Offline Mode:** Built-in semantic heuristic parser guaranteeing 100% uptime without internet or API keys.
3. **AST Safety Sandbox:**
   - Static analysis of code using Python's `ast` module before execution.
   - Blocks dangerous imports (`os`, `sys`, `subprocess`), dangerous calls (`eval`, `open`), and introspection attacks.
4. **Interactive Visualizations:**
   - Dynamic charts powered by Plotly (Bar charts, Line graphs, Pie charts, Histograms, Box plots, Correlation heatmaps).
5. **3 Preloaded Enterprise Datasets:**
   - **Retail Superstore Sales:** (Orders, Profit, Sales, Discount, Segments)
   - **Customer Churn:** (Tenure, Contract, MonthlyCharges, TotalCharges, Churn)
   - **HR Employee Attrition:** (Department, MonthlyIncome, OverTime, YearsAtCompany, Satisfaction)
   - *Plus custom CSV file upload support.*

---

## 📁 Repository Structure

```
├── app.py                              # Streamlit interactive web dashboard
├── analytics_engine.py                 # Core AI & deterministic analytics engine
├── safety_sandbox.py                   # AST security firewall & isolated execution
├── JURY_DEFENSE_AND_PROJECT_REPORT.md  # Comprehensive Jury Defense & Q&A guide
├── datasets/                           # Preloaded sample datasets
│   ├── superstore_sales.csv
│   ├── customer_churn.csv
│   ├── employee_attrition.csv
│   └── generate_datasets.py
└── .venv/                              # Python virtual environment
```

---

## 🎯 Resume Bullet Points (Data Analytics & Data Science)

- **InsightGen — AI-Powered Conversational Data Analytics & BI Engine**
  - Architected an end-to-end data analytics copilot converting natural-language business queries into validated Pandas operations and dynamic interactive Plotly visualizations.
  - Implemented an AST-level syntax validation and sandbox execution layer to ensure 100% memory-safe execution and zero destructive script runs.
  - Engineered automated schema profiling, handling missing value imputations, outlier detection, and generating automated natural language statistical takeaways for non-technical stakeholders.
