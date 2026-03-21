"""
DeepLenseSim Agent
──────────────────
Pydantic AI–powered agentic workflow for strong gravitational lensing
simulations via the DeepLenseSim / lenstronomy pipeline.

Supports Model_I through Model_IV telescope configurations with
no_sub / cdm / axion dark matter substructure classes.

Quick start
───────────
    from deeplense_agent.agent import get_agent, AgentDeps
    from pathlib import Path
    import asyncio

    agent = get_agent()
    deps  = AgentDeps(output_dir=Path("outputs"), interactive=False)

    result = asyncio.run(
        agent.run(
            "Generate 3 CDM lensing images with Euclid settings (Model_II)",
            deps=deps,
        )
    )
    print(result.data.agent_explanation)
"""

__version__ = "1.0.0"
__author__ = "DeepLenseSim Agent"
