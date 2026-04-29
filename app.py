from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Literal
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent import SYSTEM_PROMPT, build_agent


WORKSPACE_DIR = Path(__file__).resolve().parent / "workspace"
WORKSPACE_DIR.mkdir(exist_ok=True)
CHATS_DIR = WORKSPACE_DIR / "chats"
CHATS_DIR.mkdir(exist_ok=True)


Role = Literal["user", "assistant"]


@dataclass
class ChatTurn:
    role: Role
    content: str


def _turns_to_langchain_messages(turns: list[ChatTurn]) -> list[Any]:
    msgs: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    for t in turns:
        if t.role == "user":
            msgs.append(HumanMessage(content=t.content))
        else:
            msgs.append(AIMessage(content=t.content))
    return msgs


def _extract_last_assistant_text(result: Any) -> str:
    """
    LangGraph agents return a dict with a `messages` key. The final message
    content may be a plain string or a list of typed blocks (e.g. Gemini returns
    [{'type': 'text', 'text': '...'}]). We handle both and fall back to
    stringification if the shape differs.
    """
    if isinstance(result, dict) and "messages" in result and result["messages"]:
        last = result["messages"][-1]
        content = getattr(last, "content", str(last))
        # Handle list-of-dicts format (e.g. from Gemini)
        if isinstance(content, list):
            return " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return content
    return str(result)


@st.cache_resource
def _get_agent(model_name: str, model_provider: str) -> Any:
    return build_agent(model_name=model_name, model_provider=model_provider)


def _list_workspace_files() -> list[Path]:
    return sorted([p for p in WORKSPACE_DIR.rglob("*") if p.is_file()])


def _chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def _list_saved_chats() -> list[Path]:
    return sorted(CHATS_DIR.glob("*.json"), reverse=True)


def _serialize_turns(turns: list[ChatTurn]) -> list[dict[str, str]]:
    return [{"role": t.role, "content": t.content} for t in turns]


def _deserialize_turns(payload: list[dict[str, str]]) -> list[ChatTurn]:
    out: list[ChatTurn] = []
    for item in payload:
        role = item.get("role")
        content = item.get("content", "")
        if role in ("user", "assistant"):
            out.append(ChatTurn(role=role, content=content))
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_chat_data(chat_id: str) -> dict[str, Any]:
    return json.loads(_chat_path(chat_id).read_text(encoding="utf-8"))


def _save_chat(chat_id: str, *, title: str, turns: list[ChatTurn]) -> None:
    existing = _chat_path(chat_id)
    created_at = _now_iso()
    if existing.exists():
        try:
            created_at = _load_chat_data(chat_id).get("created_at", created_at)
        except Exception:
            created_at = created_at

    _chat_path(chat_id).write_text(
        json.dumps(
            {
                "id": chat_id,
                "title": title,
                "created_at": created_at,
                "updated_at": _now_iso(),
                "turns": _serialize_turns(turns),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_chat(chat_id: str) -> tuple[str, list[ChatTurn]]:
    data = _load_chat_data(chat_id)
    title = data.get("title") or chat_id
    return title, _deserialize_turns(data.get("turns", []))


def _chat_label(path: Path) -> str:
    chat_id = path.stem
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        title = (data.get("title") or "").strip()
        if title:
            return f"{title} ({chat_id[:8]})"
    except Exception:
        pass
    return chat_id


st.set_page_config(page_title="DS Agent", page_icon="🧪", layout="wide")
st.title("DS Agent")

with st.sidebar:
    st.subheader("Model")
    if "model_provider" not in st.session_state:
        st.session_state.model_provider = "google_genai"
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gemini-2.5-flash"

    provider = st.selectbox(
        "Provider",
        options=["google_genai"],
        index=["google_genai"].index(st.session_state.model_provider)
        if st.session_state.model_provider in ["google_genai"]
        else 0,
    )
    model_name = st.text_input("Model", value=st.session_state.model_name)

    changed = (provider != st.session_state.model_provider) or (model_name != st.session_state.model_name)
    st.session_state.model_provider = provider
    st.session_state.model_name = model_name

    if changed:
        # Force Streamlit to rebuild the cached agent with new args on rerun.
        st.cache_resource.clear()
        st.rerun()

    st.subheader("Safety")
    if "enable_pip_install" not in st.session_state:
        st.session_state.enable_pip_install = False
    if "pip_install_allowlist" not in st.session_state:
        st.session_state.pip_install_allowlist = "pandas,numpy,matplotlib,scikit-learn,seaborn"

    enable_pip_install = st.toggle(
        "Allow pip installs (off by default)",
        value=st.session_state.enable_pip_install,
        help="Controls the `install_package` tool. Keep off for safer demos.",
    )
    allowlist = st.text_input(
        "pip allowlist (comma-separated)",
        value=st.session_state.pip_install_allowlist,
        help="If set, only these packages can be installed via the tool.",
        disabled=not enable_pip_install,
    )

    safety_changed = (enable_pip_install != st.session_state.enable_pip_install) or (
        allowlist != st.session_state.pip_install_allowlist
    )
    st.session_state.enable_pip_install = enable_pip_install
    st.session_state.pip_install_allowlist = allowlist

    if safety_changed:
        # Ensure newly-built agents see updated env flags.
        st.cache_resource.clear()
        st.rerun()

    st.subheader("Session")
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = uuid.uuid4().hex
    if "chat_title" not in st.session_state:
        st.session_state.chat_title = "Untitled"

    title = st.text_input("Chat title", value=st.session_state.chat_title)
    if title != st.session_state.chat_title:
        st.session_state.chat_title = title

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New chat", use_container_width=True):
            st.session_state.chat_id = uuid.uuid4().hex
            st.session_state.chat_title = "Untitled"
            st.session_state.turns = []
            st.rerun()
    with col_b:
        if st.button("Save chat", use_container_width=True):
            _save_chat(
                st.session_state.chat_id,
                title=st.session_state.chat_title,
                turns=st.session_state.get("turns", []),
            )
            st.toast("Saved.")

        if st.button("Reset conversation", type="primary"):
            st.session_state.pop("turns", None)
            st.session_state.chat_id = uuid.uuid4().hex
            st.session_state.chat_title = "Untitled"
            st.rerun()

    saved = _list_saved_chats()
    if saved:
        st.caption("Saved chats")
        selected_chat = st.selectbox(
            "Load a saved chat",
            options=saved,
            format_func=_chat_label,
            index=None,
            placeholder="Select…",
        )
        if selected_chat is not None and st.button("Load selected chat", use_container_width=True):
            st.session_state.chat_id = selected_chat.stem
            loaded_title, loaded_turns = _load_chat(selected_chat.stem)
            st.session_state.chat_title = loaded_title
            st.session_state.turns = loaded_turns
            st.rerun()

    transcript = json.dumps(
        {
            "id": st.session_state.chat_id,
            "title": st.session_state.chat_title,
            "turns": _serialize_turns(st.session_state.get("turns", [])),
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    st.download_button(
        "Download current chat (JSON)",
        data=transcript,
        file_name=f"chat_{st.session_state.chat_id}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.subheader("Workspace")
    files = _list_workspace_files()
    if not files:
        st.caption("workspace/ is empty")
        selected = None
    else:
        selected = st.selectbox(
            "Open a file created by tools",
            options=files,
            format_func=lambda p: str(p.relative_to(WORKSPACE_DIR)),
        )

    st.caption("Tip: tools save outputs under `workspace/`.")

if "turns" not in st.session_state:
    st.session_state.turns = []

turns: list[ChatTurn] = st.session_state.turns

for t in turns:
    with st.chat_message(t.role):
        st.markdown(t.content)

prompt = st.chat_input("Ask for analysis, code, plots, or help…")
if prompt:
    if st.session_state.get("chat_title", "Untitled") == "Untitled":
        st.session_state.chat_title = (prompt.strip()[:60] or "Untitled")

    turns.append(ChatTurn(role="user", content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    os.environ["ENABLE_PIP_INSTALL"] = "1" if st.session_state.enable_pip_install else "0"
    os.environ["PIP_INSTALL_ALLOWLIST"] = st.session_state.pip_install_allowlist.strip()

    agent = _get_agent(st.session_state.model_name, st.session_state.model_provider)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"configurable": {"thread_id": st.session_state.chat_id}},
            )
            answer = _extract_last_assistant_text(result)
        st.markdown(answer)
    turns.append(ChatTurn(role="assistant", content=answer))

    # Auto-save after each assistant response (local demo-friendly persistence).
    _save_chat(st.session_state.chat_id, title=st.session_state.chat_title, turns=turns)

if selected is not None:
    st.divider()
    rel = selected.relative_to(WORKSPACE_DIR)
    st.subheader(f"workspace/{rel}")
    suffix = selected.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        st.image(str(selected))
    elif suffix in {".csv"}:
        try:
            import pandas as pd

            df = pd.read_csv(selected)
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.code(selected.read_text(errors="replace"))
    else:
        st.code(selected.read_text(errors="replace"))
