from __future__ import annotations

import argparse
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent import SYSTEM_PROMPT, build_agent


def _extract_last_assistant_text(result: Any) -> str:
    if isinstance(result, dict) and "messages" in result and result["messages"]:
        last = result["messages"][-1]
        return getattr(last, "content", str(last))
    return str(result)


def run_once(*, prompt: str, model_name: str | None, model_provider: str | None) -> int:
    agent = build_agent(model_name=model_name, model_provider=model_provider)
    result = agent.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]}
    )
    print(_extract_last_assistant_text(result))
    return 0


def repl(*, model_name: str | None, model_provider: str | None) -> int:
    agent = build_agent(model_name=model_name, model_provider=model_provider)
    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]

    print("DS Agent REPL. Type 'exit' to quit.\n")
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            return 0

        messages.append(HumanMessage(content=user))
        result = agent.invoke({"messages": messages})
        answer = _extract_last_assistant_text(result)
        print(answer)
        messages.append(AIMessage(content=answer))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the DS Agent from the terminal.")
    parser.add_argument("prompt", nargs="*", help="One-shot prompt. If omitted, starts a REPL.")
    parser.add_argument("--model-name", default=None, help="Override model name for this run.")
    parser.add_argument("--model-provider", default=None, help="Override model provider for this run.")

    args = parser.parse_args(argv)
    if args.prompt:
        return run_once(
            prompt=" ".join(args.prompt),
            model_name=args.model_name,
            model_provider=args.model_provider,
        )
    return repl(model_name=args.model_name, model_provider=args.model_provider)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

