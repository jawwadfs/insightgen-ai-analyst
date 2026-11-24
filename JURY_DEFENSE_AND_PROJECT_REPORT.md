# InsightGen: Autonomous AI Data Analyst & Conversational BI Engine
## Generative AI Final Year Jury 2 Project Report & Defense Manual
**Problem Statement 91: Chat-with-your-CSV Tool [Model Making]**  
**Assessment:** 20-Mark Final Year Jury Assessment | Group of 4

---

## 1. Project Overview & Business Value
In traditional business environments, querying enterprise data requires specialized knowledge of SQL or Python Pandas. **InsightGen** bridges the gap between natural language business inquiries and programmatic data analysis. It translates plain English business questions into secure, validated Pandas operations and dynamic interactive Plotly visualizations.

### Core Architectural Pillars
1. **Natural Language Translation:** Translates user questions into optimized Pandas operations. Supports live LLMs (Google Gemini 1.5 Flash) with fallback to a deterministic offline heuristic parser.
2. **Abstract Syntax Tree (AST) Safety Sandbox:** Static analysis of generated code before execution to guarantee memory safety, reject malicious injection (`os`, `sys`, `eval`, `subprocess`), and prevent disk mutations.
3. **Self-Healing Execution Loop:** If an expression encounters a runtime KeyError or type mismatch, the runtime exception is captured and fed back to auto-repair the script.
4. **Automated EDA & Statistical Health Profiling:** One-click five-number summaries, outlier identification (1.5x IQR), and correlation matrices.

---

## 2. System Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                       User Interface                        │
│             (Streamlit Conversational Dashboard)            │
└──────────────────────────────┬──────────────────────────────┘
                               │ User Question + CSV File
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Schema & Statistical Profiler               │
│  - Column data types, ranges [min, max], nulls, categories  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Augmented Prompt
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       AI Query Engine                       │
│  - Live Mode: Google Gemini API                             │
│  - Offline Mode: Deterministic Semantic Heuristic Engine    │
└──────────────────────────────┬──────────────────────────────┘
                               │ Raw Python Code
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   AST Safety Sandbox Firewall               │
│  - Static Code AST Visitor checks against prohibited nodes  │
│  - Blocks: os, sys, eval, open, dunder traversal            │
└──────────────────────────────┬──────────────────────────────┘
                               │ Validated Code
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Isolated Execution Scope                 │
│  - Sanitized __builtins__                                   │
│  - Outputs: Tabular Result, Plotly Chart (fig), Summary     │
│  - Self-Healing Retry on Runtime Exception                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               Interactive Executive Presentation            │
│       (KPI Cards, Formatted DataFrames, Plotly Visuals)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Steps (Mapped to Syllabus PS 91)

| Step | Syllabus Requirement | InsightGen Implementation |
|---|---|---|
| **Step 1** | Let user ask questions in plain English | Streamlit interactive text input + 1-click quick-demo chips |
| **Step 2** | Translate questions to pandas/queries | Augmented prompt injection with exact schema types + regex/semantic parser |
| **Step 3** | Run them and return answers/charts | Execution in sandboxed scope; generates DataFrame + interactive Plotly charts |
| **Step 4** | Handle invalid/unsafe questions | AST static analysis blocks malicious tokens; error catching handles nonsensical queries |
| **Step 5** | Test on a dataset | Preloaded with 3 enterprise datasets: Retail Superstore, Customer Churn, HR Attrition |
| **Step 6** | Present the tool | Multi-tab dashboard with Live Chat, Automated EDA, AST Sandbox Test, and Rubric |

---

## 4. Group of 4 Task Distribution & Defense Allocation

To guarantee high individual marks on the standard rubric (scored out of 50 and scaled to 20):

### Member 1: Data Architecture & Statistical Profiling
* **Owns:** Dataset ingestion, schema extraction (`get_dataset_metadata`), data cleansing, five-number summary statistics, and missing value checks.
* **Key Defense Talking Point:** Explain how feeding column data types, category samples, and min/max ranges into the model anchors generation and prevents hallucinations.

### Member 2: Prompt Engineering & Self-Healing Pipeline
* **Owns:** System prompt design (`SYSTEM_PROMPT`), LLM inference with Gemini, few-shot structuring, and the self-healing retry logic when runtime exceptions occur.
* **Key Defense Talking Point:** Explain how the self-healing mechanism captures Python tracebacks and instructs the model to correct column casing or aggregation issues.

### Member 3: Security, AST Sandboxing & Execution Safety
* **Owns:** AST static analysis (`CodeSafetyVisitor`), blacklisted module detection, prohibited dunder attribute blocking (`__class__`, `__subclasses__`), and sanitized execution scope.
* **Key Defense Talking Point:** Defend mandatory **Jury Question 1** on preventing dangerous or malicious code execution.

### Member 4: Business Intelligence, Visuals & Presentation
* **Owns:** Dynamic Plotly visualization pipeline (bar, line, box, heatmap), KPI summaries, Streamlit UI/UX design, and offline heuristic fallback testing.
* **Key Defense Talking Point:** Defend mandatory **Jury Question 2** on failure cases and explain business interpretability for non-technical stakeholders.

---

## 5. Master Jury Q&A Cheat Sheet

### Mandatory Question 1: How do you stop wrong/unsafe queries?
> **Answer:**  
> "We implement a defense-in-depth architecture consisting of three layers:
> 1. **Static AST Analysis (Abstract Syntax Tree):** Before the code is passed to Python's execution engine, we parse the raw string into an AST tree using Python's `ast` module. Our `CodeSafetyVisitor` inspects every node. If any import statement contains forbidden system modules (such as `os`, `sys`, `subprocess`, `shutil`), or any function call attempts to use `eval()`, `exec()`, or `open()`, the code is terminated immediately with a security exception.
> 2. **Dunder Attribute Traversal Blocking:** Attackers often try to escape Python sandboxes using introspection like `df.__class__.__bases__[0].__subclasses__()`. Our AST visitor explicitly blocks access to any dunder attributes.
> 3. **Sanitized Execution Scope:** During execution, we pass a strictly pruned `__builtins__` dictionary containing only standard math/type primitives, and pass a shallow copy of the DataFrame so original memory cannot be mutated or exported to disk."

---

### Mandatory Question 2: Where does natural language translation fail?
> **Answer:**  
> "Natural language translation fails in three primary scenarios:
> 1. **Lexical Ambiguity and Synonymy:** In business datasets, terms like *'highest grossing'* or *'best'* can mean either Total Sales or Total Profit. If the user does not specify, the model must make an inference.
> 2. **Implicit Multi-Step Domain Logic:** In a churn dataset, asking *'Show loyal customers'* requires defining a loyalty metric (e.g., Tenure > 24 months vs. Contract type). Without domain definitions, the model may choose arbitrary heuristics.
> 3. **Out-of-Vocabulary Category Names:** When users mistype or abbreviate categories (e.g., asking for *'Tech'* instead of *'Technology'*), strict equality filters (`df['Category'] == 'Tech'`) return 0 rows.
> **How we mitigate this:** We inject unique category values and statistical ranges into the LLM prompt, and use fuzzy matching and self-healing error retries."

---

### Bonus Tricky Jury Questions:
* **Q: Why use Python AST instead of regex keyword blocking?**  
  *Answer:* Regex is easily bypassed by string concatenation or variable aliasing (e.g., `o = 'o' + 's'; __import__(o)`). AST operates on the parsed grammatical syntax of Python, making it immune to regex obfuscation.
* **Q: How does this perform if there is no internet in the examination hall?**  
  *Answer:* We engineered an offline heuristic engine that parses aggregations, rankings, distributions, and correlation matrices locally with sub-millisecond response times.
