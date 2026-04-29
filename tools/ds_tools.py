"""
DS Agent Tools
All tools are scoped to the local workspace/ directory for safety.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain.tools import tool


WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)


def _resolve_workspace_path(filename: str) -> Path:
    candidate = (WORKSPACE / filename).resolve()
    # Prevent "../" path traversal outside workspace
    if WORKSPACE not in candidate.parents and candidate != WORKSPACE:
        raise ValueError("Path escapes workspace directory.")
    return candidate


# ── 1. Run Python ──────────────────────────────────────────────────────────────


@tool
def run_python(code: str) -> str:
    """
    Execute a Python code snippet inside the workspace directory.
    Returns stdout and stderr.
    """
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=WORKSPACE, delete=False)
    try:
        tmp.write(code)
        tmp.close()
        result = subprocess.run(
            [sys.executable, tmp.name],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=WORKSPACE,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        parts: list[str] = []
        if stdout:
            parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            parts.append(f"STDERR:\n{stderr}")
        if not parts:
            parts.append("(no output)")
        return "\n\n".join(parts)
    except subprocess.TimeoutExpired:
        return "ERROR: Code execution timed out after 30 seconds."
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        Path(tmp.name).unlink(missing_ok=True)


# ── 2. Write File ──────────────────────────────────────────────────────────────


@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file in the workspace directory."""
    try:
        target = _resolve_workspace_path(filename)
    except ValueError:
        return "ERROR: Invalid path (must stay within workspace/)."
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Written: workspace/{filename} ({len(content)} chars)"


# ── 3. Read File ───────────────────────────────────────────────────────────────


@tool
def read_file(filename: str) -> str:
    """Read a file from the workspace directory."""
    try:
        target = _resolve_workspace_path(filename)
    except ValueError:
        return "ERROR: Invalid path (must stay within workspace/)."
    if not target.exists():
        return f"ERROR: workspace/{filename} does not exist."
    try:
        return target.read_text()
    except Exception as e:
        return f"ERROR reading file: {e}"


# ── 4. List Directory ──────────────────────────────────────────────────────────


@tool
def list_directory(subpath: str = "") -> str:
    """List files and directories inside workspace (or a subdirectory of it)."""
    try:
        target = _resolve_workspace_path(subpath) if subpath and subpath not in {".", "./"} else WORKSPACE
    except ValueError:
        return "ERROR: Invalid path (must stay within workspace/)."
    if not target.exists():
        return f"ERROR: workspace/{subpath} does not exist."
    lines: list[str] = []
    for p in sorted(target.rglob("*")):
        rel = p.relative_to(WORKSPACE)
        prefix = "  " * (len(rel.parts) - 1)
        icon = "dir" if p.is_dir() else "file"
        lines.append(f"{prefix}{icon} {rel}")
    return "\n".join(lines) if lines else "(workspace is empty)"


# ── 5. Inspect DataFrame ───────────────────────────────────────────────────────


@tool
def inspect_dataframe(filename: str) -> str:
    """
    Load a CSV from the workspace and return a summary:
    shape, column dtypes, null counts, and the first 5 rows.
    """
    try:
        target = _resolve_workspace_path(filename)
    except ValueError:
        return "ERROR: Invalid path (must stay within workspace/)."
    if not target.exists():
        return f"ERROR: workspace/{filename} does not exist."

    code = f"""
import pandas as pd
import json

df = pd.read_csv(r\"\"\"{target}\"\"\")
info = {{
    "shape": list(df.shape),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "null_counts": df.isnull().sum().to_dict(),
    "head": df.head().to_dict(orient="records"),
}}
print(json.dumps(info, default=str))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        return f"ERROR: {result.stderr.strip()}"
    try:
        info = json.loads(result.stdout.strip())
        rows, cols = info["shape"]
        lines = [f"Shape: {rows} rows × {cols} columns", "\nColumn types:"]
        for col, dtype in info["dtypes"].items():
            nulls = info["null_counts"].get(col, 0)
            null_str = f"  WARNING: {nulls} nulls" if nulls else ""
            lines.append(f"  {col}: {dtype}{null_str}")
        lines.append("\nFirst 5 rows:")
        for row in info["head"]:
            lines.append(f"  {row}")
        return "\n".join(lines)
    except Exception as e:
        return f"Raw output:\n{result.stdout}\n\nParse error: {e}"


# ── 6. Install Package ─────────────────────────────────────────────────────────


@tool
def install_package(package_name: str) -> str:
    """
    Install a Python package via pip into the current environment.

    Safety: disabled by default. Enable by setting `ENABLE_PIP_INSTALL=1`.
    Optional allowlist: set `PIP_INSTALL_ALLOWLIST="pandas,numpy,matplotlib"`.
    """
    if os.getenv("ENABLE_PIP_INSTALL", "0") != "1":
        return (
            "ERROR: pip installs are disabled in this demo. "
            "Set ENABLE_PIP_INSTALL=1 to enable."
        )

    allowlist_raw = os.getenv("PIP_INSTALL_ALLOWLIST", "").strip()
    if allowlist_raw:
        allowlist = {p.strip().lower() for p in allowlist_raw.split(",") if p.strip()}
        # Normalize "pkg[extra]==1.2.3" -> "pkg"
        normalized = re.split(r"[<=>\\[]", package_name.strip().lower(), maxsplit=1)[0]
        if normalized not in allowlist:
            return (
                "ERROR: package not in allowlist. "
                f"Requested '{normalized}'. Allowed: {', '.join(sorted(allowlist))}"
            )

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package_name, "-q"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        return f"Successfully installed: {package_name}"
    return f"ERROR installing {package_name}:\n{result.stderr.strip()}"


ALL_TOOLS = [
    run_python,
    write_file,
    read_file,
    list_directory,
    inspect_dataframe,
    install_package,
]

