"""
Safety Sandbox Module for InsightGen AI Data Analyst
Provides Abstract Syntax Tree (AST) validation and restricted execution
to prevent malicious or destructive code execution.
"""

import ast
import sys
import copy
from typing import Tuple, Optional, Any, Dict

# Prohibited modules that cannot be imported
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "urllib", "requests",
    "http", "posix", "ctypes", "pathlib", "pickle", "shelve", "tempfile",
    "pty", "commands", "builtin", "builtins", "_thread", "threading", "multiprocessing"
}

# Prohibited built-in functions
BLOCKED_FUNCTIONS = {
    "eval", "exec", "compile", "open", "file", "input", "raw_input",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "hasattr", "__import__", "exit", "quit", "reload", "help", "breakpoint"
}

# Prohibited attribute accesses to prevent Python sandbox escaping
BLOCKED_ATTRIBUTES = {
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__code__", "__globals__", "__builtins__", "__dict__",
    "__import__", "__loader__", "__spec__", "__func__"
}

# Prohibited method calls on DataFrames (destructive operations)
BLOCKED_DF_METHODS = {
    "to_csv", "to_excel", "to_sql", "to_parquet", "to_feather",
    "to_pickle", "to_clipboard", "drop_duplicates_in_place"
}


class CodeSafetyVisitor(ast.NodeVisitor):
    """AST Visitor that flags forbidden constructs before execution."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module in BLOCKED_MODULES:
                self.errors.append(f"Security Violation: Import of module '{alias.name}' is prohibited.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module in BLOCKED_MODULES:
                self.errors.append(f"Security Violation: Import from module '{node.module}' is prohibited.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Direct function calls like open(), eval(), etc.
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_FUNCTIONS:
                self.errors.append(f"Security Violation: Invocation of forbidden function '{node.func.id}()'.")
        
        # Method calls like df.to_csv() or dangerous calls
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in BLOCKED_DF_METHODS:
                self.errors.append(f"Security Policy: File export method '{attr_name}()' is blocked in sandbox.")
            if attr_name in BLOCKED_ATTRIBUTES:
                self.errors.append(f"Security Violation: Access to dunder attribute '{attr_name}' is blocked.")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in BLOCKED_ATTRIBUTES:
            self.errors.append(f"Security Violation: Introspection via attribute '{node.attr}' is prohibited.")
        self.generic_visit(node)


def validate_python_code(code_str: str) -> Tuple[bool, Optional[str]]:
    """
    Parses code into an AST and verifies safety against malicious constructs.
    Returns (is_safe, error_message).
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Syntax Error in generated code: {e}"

    visitor = CodeSafetyVisitor()
    visitor.visit(tree)

    if visitor.errors:
        return False, " | ".join(visitor.errors)

    return True, None


def execute_sandboxed_code(
    code_str: str,
    dataframe,
    pd_module,
    np_module,
    px_module,
    go_module
) -> Dict[str, Any]:
    """
    Executes validated Python code in an isolated dictionary scope.
    Expected output variables:
    - 'result_df' or 'result': filtered/aggregated DataFrame, Series, or scalar
    - 'fig': Plotly figure object (optional)
    - 'summary': text explanation or insight string (optional)
    """
    is_safe, err = validate_python_code(code_str)
    if not is_safe:
        return {
            "success": False,
            "error": err,
            "result": None,
            "fig": None,
            "summary": None,
            "code": code_str
        }

    # Safe built-in functions
    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float, "format": format,
        "int": int, "isinstance": isinstance, "issubclass": issubclass,
        "iter": iter, "len": len, "list": list, "map": map, "max": max,
        "min": min, "next": next, "print": print, "range": range, "reversed": reversed,
        "round": round, "set": set, "slice": slice, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "zip": zip,
        "True": True, "False": False, "None": None
    }

    # Isolated execution environment
    # Use a fresh shallow copy of the dataframe to protect original data
    local_df = dataframe.copy(deep=False)
    
    restricted_globals = {
        "__builtins__": safe_builtins,
        "df": local_df,
        "pd": pd_module,
        "np": np_module,
        "px": px_module,
        "go": go_module
    }
    
    local_scope: Dict[str, Any] = {}

    try:
        exec(code_str, restricted_globals, local_scope)
    except Exception as e:
        return {
            "success": False,
            "error": f"Runtime Execution Error: {str(e)}",
            "result": None,
            "fig": None,
            "summary": None,
            "code": code_str
        }

    # Extract outputs
    result = local_scope.get("result", local_scope.get("result_df", None))
    fig = local_scope.get("fig", None)
    summary = local_scope.get("summary", None)

    return {
        "success": True,
        "error": None,
        "result": result,
        "fig": fig,
        "summary": summary,
        "code": code_str
    }
