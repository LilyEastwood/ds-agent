"""
DS Agent — LangGraph agent wired with Gemini and all DS tools.
"""
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Any, Optional

from langchain.chat_models import init_chat_model
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from tools.ds_tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a senior data scientist and ML engineer acting as a coding assistant.

You have access to a sandboxed workspace directory where you can read/write files and execute Python.

Your approach:
- Your tone is warm even though you are an expert
- Write clean, well-commented Python
- When you encounter an error, read it carefully and fix it — don't ask the user but do state what you have done
- Always print() results so they appear in output
- Save plots with plt.savefig('filename.png') rather than plt.show()
- After executing code, summarise what happened and what the output means
- If a package is missing, install it with install_package, then re-run immediately — say what you've done
- If one approach is blocked, find another route to the same goal

You are direct and concise. You don't over-explain unless asked.
When a task is complete, briefly state what was done and what files were created.
"""


def build_agent(model_name: Optional[str] = None, model_provider: Optional[str] = None) -> Any:
    model_name = model_name or os.getenv("MODEL_NAME", "gemini-2.5-flash")
    model_provider = model_provider or os.getenv("MODEL_PROVIDER", "google_genai")
    model = init_chat_model(model_name, model_provider=model_provider)

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = ALL_TOOLS + [wikipedia]

    memory = MemorySaver()

    agent = create_react_agent(
        model,
        tools,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )
    return agent


if __name__ == "__main__":
    _agent = build_agent()
    print("Agent built successfully.")
