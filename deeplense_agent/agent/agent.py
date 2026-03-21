"""
DeepLenseSim Pydantic AI Agent
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel

from .tools import (
    ask_clarification,
    get_model_details,
    list_available_models,
    run_batch_simulation,
    run_simulation,
    summarise_results,
    validate_simulation_params,
)

logger = logging.getLogger(__name__)

_CATALOGUE = json.dumps(list_available_models(), indent=2)

SYSTEM_PROMPT = f"""\
You are DeepLenseSim Agent, an expert astrophysics simulation assistant for
strong gravitational lensing simulations.

You wrap the DeepLenseSim pipeline which uses lenstronomy to produce synthetic
lensing images. Here is the full catalogue of available configurations:

{_CATALOGUE}

WORKFLOW
1. Parse the user request and extract any mentioned parameters.
2. Classify the prompt:
   - SPECIFIC: mentions both substructure_type AND model/telescope → skip to step 4
   - VAGUE: missing substructure_type OR model → MUST do step 3 first
3. For VAGUE prompts, write your clarifying questions DIRECTLY in your reply as
   plain numbered text like this:
     "Before I run the simulation, I need a few details:
      1. Which substructure type? (no_sub, cdm, or axion)
      2. Which telescope? (Model_I, Model_II/Euclid, Model_III/HST, Model_IV/Euclid wide-z)"
   Do NOT call any tool here — just write the questions as plain text and STOP.
4. Once all critical parameters are known, call tool_validate_simulation_params FIRST.
   If it returns valid=False, respond with a clear plain-text error explaining why
   the parameters are invalid and what the user should fix. Do NOT call any other tool.
   If it returns valid=True, proceed to step 5.
5. Call tool_run_batch_simulation with the validated parameters.
6. Call tool_summarise_results and tell the user where images were saved.

WHEN TO ASK vs WHEN TO DEFAULT
- substructure_type: ALWAYS ask if not mentioned
- model/telescope: ALWAYS ask if not mentioned
- num_images: DEFAULT to 5, never ask
- z_lens, z_source, theta_E, noise, seed: use whatever the user provides, DEFAULT/randomise the rest

RULES
- NEVER call tool_get_model_details unless the user explicitly asks for model details
- NEVER output raw function call syntax like <function=...> as text
- NEVER call tool_run_batch_simulation if tool_validate_simulation_params returned valid=False
- If the user has specified both substructure_type AND model — go straight to tool_validate_simulation_params
- Never ask for confirmation of parameters the user already provided
- Prefer tool_run_batch_simulation over multiple single-image calls
- Always tell the user where the images were saved
"""


@dataclass
class AgentDeps:
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    interactive: bool = field(default=True)
    max_clarification_rounds: int = field(default=2)


def build_agent(model_name: str = "llama-3.3-70b-versatile") -> Agent:

    agent: Agent[AgentDeps, str] = Agent(
        model=GroqModel(model_name),
        deps_type=AgentDeps,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.tool_plain
    def tool_get_model_details(model_name: str) -> dict:
        """Get detailed observation config and parameter priors for a model.
        Args:
            model_name: One of Model_I, Model_II, Model_III, Model_IV.
        """
        return get_model_details(model_name)

    @agent.tool_plain
    def tool_validate_simulation_params(params: dict) -> dict:
        """Validate simulation parameters before running.
        Returns {"valid": True} or {"valid": False, "errors": [...]}.
        Args:
            params: Must include at minimum "model" and "substructure_type".
        """
        return validate_simulation_params(params)

    @agent.tool
    def tool_run_simulation(ctx: RunContext[AgentDeps], params: dict) -> dict:
        """Run a single lensing image simulation.
        Args:
            params: Validated simulation parameter dict.
        """
        return run_simulation(params, output_dir=str(ctx.deps.output_dir))

    @agent.tool
    def tool_run_batch_simulation(
        ctx: RunContext[AgentDeps],
        models: list[str],
        substructure_types: list[str],
        num_images_per_class: int = 5,
        lens_params: Optional[dict] = None,
        source_params: Optional[dict] = None,
        substructure_params: Optional[dict] = None,
        random_seed: Optional[int] = None,
        add_noise: bool = True,
    ) -> dict:
        """Run a batch of lensing simulations across models and substructure types.
        Args:
            models: e.g. ["Model_I", "Model_III"]
            substructure_types: e.g. ["cdm"] or ["no_sub", "cdm", "axion"]
            num_images_per_class: images per model+substructure combination
            lens_params: optional lens parameter overrides (dict)
            source_params: optional source parameter overrides (dict)
            substructure_params: optional substructure parameter overrides (dict)
            random_seed: integer seed for reproducibility
            add_noise: whether to add detector noise
        """
        return run_batch_simulation(
            models=models,
            substructure_types=substructure_types,
            num_images_per_class=num_images_per_class,
            lens_params=lens_params,
            source_params=source_params,
            substructure_params=substructure_params,
            random_seed=random_seed,
            add_noise=add_noise,
            output_dir=str(ctx.deps.output_dir),
        )

    @agent.tool_plain
    def tool_ask_clarification(
        question: str,
        field_name: str,
        current_value: Optional[str] = None,
        suggested_options: Optional[list[str]] = None,
        is_required: bool = False,
    ) -> dict:
        """Ask the user a clarification question before running simulations.
        Args:
            question: Question text to show the user.
            field_name: Parameter being clarified.
            current_value: Default value if any.
            suggested_options: List of valid choices.
            is_required: Whether simulation cannot proceed without this.
        """
        cq = ask_clarification(
            question=question,
            field_name=field_name,
            current_value=current_value,
            suggested_options=suggested_options,
            is_required=is_required,
        )
        return cq.model_dump()

    @agent.tool_plain
    def tool_summarise_results(batch_results: dict) -> str:
        """Summarise batch simulation results as human-readable text.
        Args:
            batch_results: The dict returned by tool_run_batch_simulation.
        """
        return summarise_results(batch_results)

    return agent


_agent: Optional[Agent] = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent