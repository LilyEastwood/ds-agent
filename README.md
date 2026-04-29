## DS Agent

A data science coding assistant built with LangGraph + Gemini + Streamlit.

The agent can execute Python, read/write files, inspect dataframes, and search Wikipedia — all within a sandboxed `workspace/` directory.

### Run locally (Streamlit)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Required for Gemini via `langchain-google-genai`
export GOOGLE_API_KEY="your_key_here"

streamlit run app.py
```

### Run locally (CLI / REPL)

```bash
source .venv/bin/activate

# One-shot
python cli.py "Load the iris dataset and show a summary"

# REPL
python cli.py
```

### Model configuration

- `MODEL_PROVIDER`: defaults to `google_genai`
- `MODEL_NAME`: defaults to `gemini-2.5-flash`

You can also override these in Streamlit from the sidebar, or via CLI flags:

```bash
python cli.py --model-name gemini-2.5-pro "Explain bias/variance tradeoff"
```

### Architecture

```text
app.py              Streamlit UI
cli.py              Terminal demo (one-shot + REPL)
agent.py            LangGraph agent (model + memory + tools)
tools/ds_tools.py   Custom tool implementations
workspace/          Sandboxed execution directory
```

### Safety: pip install allowlist

`install_package` is **disabled by default**. To enable it:

```bash
export ENABLE_PIP_INSTALL=1
export PIP_INSTALL_ALLOWLIST="pandas,numpy,matplotlib,scikit-learn,seaborn"
```

### Example prompts

- "Inspect `data/raw.csv` and tell me what needs cleaning."
- "Train a baseline model and save evaluation metrics to `workspace/metrics.json`."
- "Generate a plot and save it as `workspace/plot.png`."

### Security note

This is intended for local/demo use. Don’t deploy publicly without stronger sandboxing (e.g. container isolation and stricter execution controls).
