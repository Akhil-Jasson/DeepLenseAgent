"""
DeepLenseSim Agent — Interactive CLI with Human-in-the-Loop

HITL flow:
  1. User gives a (possibly vague) prompt
  2. Agent responds — either with clarifying questions OR runs the simulation
  3. If the agent's text response contains a question mark, we extract and
     present it to the user interactively
  4. User answers are appended to the conversation and the agent runs again
  5. Repeat up to max_rounds times, then simulate with whatever is resolved
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s — %(message)s")

# Ensure parent dir is on path so `deeplense_agent` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def _print_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          DeepLenseSim Agent  •  Strong Gravitational Lensing     ║
║   ML4Sci / DeepLense  •  Pydantic AI  •  Groq + lenstronomy    ║
╚══════════════════════════════════════════════════════════════════╝""")


def _display_result(text: str) -> None:
    print("\n" + "─" * 70)
    print("🤖  Agent response:\n")
    print(text)
    print("─" * 70 + "\n")


def _contains_question(text: str) -> bool:
    """Return True if the agent response is genuinely asking for clarification.
    Ignores lines that are raw tool call artifacts from the LLM."""
    for line in text.splitlines():
        stripped = line.strip()
        # Skip raw tool call artifacts Groq sometimes emits as plain text
        if stripped.startswith("<function") or stripped.startswith("```"):
            continue
        if "?" in stripped:
            return True
    return False


def _extract_questions(text: str) -> list[str]:
    """Pull out lines that look like genuine questions, ignoring tool call artifacts.
    Deduplicates so the same question is never shown twice."""
    seen = set()
    questions = []
    for line in text.splitlines():
        stripped = line.strip()
        if (
            "?" in stripped
            and stripped
            and not stripped.startswith("<function")
            and not stripped.startswith("```")
            and stripped not in seen
        ):
            seen.add(stripped)
            questions.append(stripped)
    return questions


# Known parameter options for numbered menu display
_PARAM_OPTIONS = {
    "substructure": ["no_sub", "cdm", "axion"],
    "telescope":    ["Model_I", "Model_II", "Model_III", "Model_IV"],
}

def _detect_options(question: str) -> list[str]:
    """Return the valid option list for a question, if we recognise the parameter."""
    q = question.lower()
    if any(w in q for w in ["substructure", "no_sub", "cdm", "axion"]):
        return _PARAM_OPTIONS["substructure"]
    if any(w in q for w in ["telescope", "model", "euclid", "hst"]):
        return _PARAM_OPTIONS["telescope"]
    return []


def _ask_user_questions(questions: list[str]) -> str:
    """Present each question as a numbered menu and collect validated answers."""
    print("\n📋  The agent has clarifying questions:\n")
    answers = []
    for q in questions:
        options = _detect_options(q)
        print(f"  ❓ {q}")
        if options:
            for i, opt in enumerate(options, 1):
                print(f"       {i}) {opt}")
            while True:
                raw = input("     Enter number: ").strip()
                if raw.isdigit() and 1 <= int(raw) <= len(options):
                    answer = options[int(raw) - 1]
                    print(f"     ✅ Selected: {answer}")
                    break
                print(f"     ⚠️  Please enter a number between 1 and {len(options)}")
        else:
            answer = input("     Your answer: ").strip()
        answers.append(f"Q: {q}\nA: {answer}")
    return "\n".join(answers)


async def run_agent_loop(
    prompt: str,
    output_dir: Path,
    interactive: bool = True,
    max_rounds: int = 3,
) -> None:
    from deeplense_agent.agent.agent import AgentDeps, build_agent

    agent = build_agent()
    deps = AgentDeps(
        output_dir=output_dir,
        interactive=interactive,
        max_clarification_rounds=max_rounds,
    )

    # Build message history for multi-turn conversation
    message_history = []
    current_prompt = prompt

    for round_num in range(max_rounds + 1):
        print(f"\n🔭  Agent is processing your request (round {round_num + 1})…")

        result = await agent.run(
            current_prompt,
            deps=deps,
            message_history=message_history if message_history else None,
        )

        response_text = str(result.output)

        # Strip any raw function call lines Groq leaks into the response
        clean_lines = [
            line for line in response_text.splitlines()
            if not line.strip().startswith("<function=")
        ]
        response_text = "\n".join(clean_lines).strip()

        # Groq occasionally outputs ONLY a raw tool call — retry
        if response_text.strip().startswith("<function=") or not response_text:
            if round_num < max_rounds:
                current_prompt = "Please do not output raw function call syntax. Give me a plain text response."
                continue
            else:
                print("\n  ⚠️  Agent encountered an internal error. Please try again.\n")
                break

        message_history = result.all_messages()

        # If validation already failed, prefix with ❌ and stop immediately
        if "invalid parameters" in response_text.lower() or "must be greater" in response_text.lower():
            _display_result("❌  " + response_text)
            break

        # Only ask clarifying questions on the very first round before any
        # simulation has run. After round 0 always show the final response.
        if interactive and round_num == 0 and _contains_question(response_text):
            questions = _extract_questions(response_text)

            # Show what the agent said
            print("\n🤖  Agent:\n")
            print(response_text)

            # Collect user answers
            answers = _ask_user_questions(questions)
            current_prompt = f"Here are my answers to your questions:\n{answers}\nPlease now run the simulation."
        else:
            # No questions — final response
            _display_result(response_text)
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepLenseSim Agent — NL-driven strong lensing simulation"
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=2)
    args = parser.parse_args()

    _print_banner()

    if args.prompt:
        user_prompt = args.prompt
    else:
        print("\nEnter your simulation request:\n")
        print("  ── Full parameter control ────────────────────────────────────────")
        print("  'Generate 5 CDM images with HST settings, z_lens=0.4, z_source=1.8, theta_E=1.2'")
        print()
        print("  ── Resolution override ───────────────────────────────────────────")
        print("  'Generate 3 axion images with Model_I settings at 128x128 resolution'")
        print()
        print("  ── Noise-free simulation ─────────────────────────────────────────")
        print("  'Generate 5 no_sub images with Euclid settings, noise-free'")
        print()
        print("  ── HITL: vague prompt triggers clarifying questions ──────────────")
        print("  'Generate some lensing images'")
        print()
        print("  ── Validation: physically invalid parameters are rejected ─────────")
        print("  'Generate 5 CDM images with z_lens=2.0 and z_source=1.0'")
        user_prompt = input("\n  > ").strip()
        if not user_prompt:
            print("No prompt provided. Exiting.")
            sys.exit(0)

    print(f"\n📝  Prompt: {user_prompt!r}")

    asyncio.run(
        run_agent_loop(
            prompt=user_prompt,
            output_dir=args.output_dir,
            interactive=not args.no_interactive,
            max_rounds=args.max_rounds,
        )
    )


if __name__ == "__main__":
    main()