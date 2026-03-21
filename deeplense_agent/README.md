# DeepLenseSim Agent

**Pydantic AI–powered agentic workflow for strong gravitational lensing simulations**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Pydantic AI](https://img.shields.io/badge/pydantic--ai-0.0.14+-green.svg)](https://ai.pydantic.dev)
[![lenstronomy](https://img.shields.io/badge/lenstronomy-1.13+-orange.svg)](https://lenstronomy.readthedocs.io)

---

## Overview

This project wraps the [DeepLenseSim](https://github.com/mwt5345/DeepLenseSim)
simulation pipeline — a lenstronomy-based tool for generating strong gravitational
lensing images used in the ML4Sci DeepLense dark matter research programme — in a
conversational agentic interface.

A user can describe their simulation in plain English:

```
"Generate 10 CDM lensing images with HST settings, z_lens=0.4, z_source=1.8"
"Give me axion dark matter images under Euclid and Model_I configurations"
"I need a balanced training set: 20 images per class (no_sub, cdm, axion) using Model_I"
```

The agent parses the request, asks follow-up questions for any missing parameters
(**human-in-the-loop**), validates all parameters via Pydantic, executes the
lenstronomy simulation pipeline, and returns images with rich structured metadata.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       User (NL prompt)                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Pydantic AI Agent │  ← claude-sonnet-4
                    │  (agent/agent.py)   │
                    └──────────┬──────────┘
                               │ tool calls
           ┌───────────────────┼───────────────────────┐
           │                   │                       │
   ┌───────▼──────┐  ┌────────▼────────┐  ┌──────────▼─────────┐
   │list_available│  │ask_clarification│  │run_batch_simulation │
   │_models       │  │(HITL)           │  │(→ simulator)        │
   └──────────────┘  └────────┬────────┘  └──────────┬─────────┘
                              │                       │
                    ┌─────────▼──────────┐  ┌────────▼────────────┐
                    │  Human answers     │  │ DeepLenseSimulator  │
                    │  (CLI / API)       │  │ (lenstronomy/stub)  │
                    └────────────────────┘  └─────────────────────┘
                                                       │
                                            ┌──────────▼──────────┐
                                            │  SimulationResult   │
                                            │  + ImageMetadata    │
                                            │  (Pydantic models)  │
                                            └─────────────────────┘
```

### Why Pydantic AI over Orchestral AI?

| Dimension | Pydantic AI | Orchestral AI |
|---|---|---|
| Structured I/O | Native `result_type` validation | Manual schema wiring |
| Type safety | End-to-end Pydantic v2 throughout | Looser typing |
| Dependency injection | `RunContext[AgentDeps]` | Ad-hoc globals |
| Tool registration | `@agent.tool` / `@agent.tool_plain` decorator | YAML/config |
| Multi-model support | First class (Anthropic, OpenAI, Gemini, …) | More limited |
| Async | Full async/await | Depends on version |

Pydantic AI was chosen because simulation parameter validation is **safety-critical**:
a wrong redshift ordering or a physically impossible Einstein radius would silently
produce meaningless images. Having every tool input/output run through Pydantic v2
validators provides a natural guardrail that matches exactly how DeepLenseSim defines
its parameter spaces.

---

## Model Configurations

| Config | Telescope | Pixel scale | PSF FWHM | Exposure | Notes |
|--------|-----------|-------------|----------|----------|-------|
| **Model_I** | Synthetic (Gaussian PSF) | 0.08″/px | 0.12″ | 1000 s | Idealized, SNR~25 |
| **Model_II** | Euclid VIS | 0.1″/px | 0.18″ | 1800 s × 4 | ESA survey sim |
| **Model_III** | HST ACS/WFC F814W | 0.05″/px | 0.09″ | 2000 s × 4 | Highest resolution |
| **Model_IV** | Euclid VIS (wide-z) | 0.1″/px | 0.18″ | 1800 s × 4 | Wider z priors |

### Substructure Classes

| Class | Description | Key parameters |
|-------|-------------|----------------|
| `no_sub` | Smooth SIE lens, no dark matter substructure | — |
| `cdm` | Cold Dark Matter: NFW sub-halos | `num_subhalos`, mass range 10^7–10^10 M☉ |
| `axion` | Ultra-light axion DM: vortex substructure | `num_vortices`, `axion_mass` (~10^-22 eV) |

---

## Pydantic Model Hierarchy

```
SimulationRequest                  ← top-level request
├── ModelConfig (enum)             ← Model_I / II / III / IV
├── SubstructureType (enum)        ← no_sub / cdm / axion
├── LensParams                     ← SIE θ_E, redshift, ellipticity, shear
├── SourceParams                   ← Sérsic source, z_source, offset
├── SubstructureParams             ← CDM/axion substructure parameters
└── ObservationConfig (optional)   ← custom telescope override

SimulationResult                   ← output
├── ImageMetadata                  ← rich provenance record (fully JSON-serialisable)
├── image_array (np.ndarray)       ← raw float32 pixel data
├── image_path (Path)              ← saved PNG/NPY location
└── image_b64 (str)               ← base64 PNG thumbnail

AgentResponse                      ← top-level agent output
├── clarification_questions        ← HITL questions (if any)
├── resolved_request               ← final SimulationRequest
├── batch_result                   ← BatchSimulationResult
└── agent_explanation              ← plain-English summary
```

---

## Project Structure

```
deeplense_agent/
├── __init__.py
├── main.py                    ← Interactive CLI with HITL loop
├── requirements.txt
│
├── models/
│   ├── simulation_params.py   ← SimulationRequest, LensParams, etc.
│   └── simulation_results.py  ← SimulationResult, ImageMetadata, AgentResponse
│
├── simulator/
│   └── engine.py              ← DeepLenseSimulator (lenstronomy + stub)
│
├── agent/
│   ├── agent.py               ← Pydantic AI agent + tool registration
│   └── tools.py               ← Tool implementations
│
└── tests/
    └── test_agent.py          ← Pytest test suite
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mwt5345/DeepLenseSim
cd DeepLenseSim   # (then copy deeplense_agent/ here, or use standalone)

# Install dependencies
pip install -r deeplense_agent/requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Usage

### Interactive mode (recommended)

```bash
python -m deeplense_agent.main
```

```
╔══════════════════════════════════════════════════════════════════╗
║          DeepLenseSim Agent  •  Strong Gravitational Lensing     ║
╚══════════════════════════════════════════════════════════════════╝

Enter your simulation request:
  > Generate a balanced training set with 5 images per class using Euclid settings

🔭  Agent is processing your request (round 1)…

📋  The agent has 1 clarifying question:

  ❓ Should I include all three substructure classes (no_sub, cdm, axion)?
     (default: yes — all three)
     [1] yes — all three
     [2] no_sub and cdm only
     [3] cdm and axion only
     Your choice: 1

🔭  Agent is processing your request (round 2)…

✅  Simulation completed successfully!

   Total images:    15
   By model:        {'Model_II': 15}
   By substructure: {'no_sub': 5, 'cdm': 5, 'axion': 5}
   Mean SNR:        23.4 ± 2.1
   Output dir:      outputs/
```

### One-shot mode

```bash
python -m deeplense_agent.main \
    --prompt "Generate 3 CDM images with HST settings, z_lens=0.4, z_source=1.8" \
    --no-interactive \
    --output-dir ./lensing_dataset
```

### Python API

```python
import asyncio
from pathlib import Path
from deeplense_agent.agent import get_agent, AgentDeps

agent = get_agent()
deps  = AgentDeps(output_dir=Path("outputs"), interactive=False)

result = asyncio.run(
    agent.run(
        "Simulate 5 axion dark matter images using Model_I telescope settings",
        deps=deps,
    )
)
response = result.data
print(response.agent_explanation)
print(f"Generated {response.batch_result.total_succeeded} images")
```

### Direct simulator (no LLM)

```python
from deeplense_agent.simulator import DeepLenseSimulator
from deeplense_agent.models import SimulationRequest, ModelConfig, SubstructureType

sim = DeepLenseSimulator(output_dir=Path("outputs"))
req = SimulationRequest(
    model=ModelConfig.MODEL_III,         # HST
    substructure_type=SubstructureType.CDM,
    num_images=1,
    random_seed=42,
)
result = sim.simulate(req)
print(result.metadata.snr_estimate)     # e.g. 24.7
print(result.image_path)                # outputs/lens_3a7f2c1b.png
```

---

## Running Tests

```bash
pytest deeplense_agent/tests/test_agent.py -v
```

Tests do **not** require lenstronomy or an Anthropic API key; the stub simulator
and mocked tool calls cover all logic paths.

---

## Human-in-the-Loop Design

The HITL component works in three tiers:

1. **Automatic** — If the user's prompt fully specifies all critical parameters
   (model, substructure type, num_images), the agent proceeds immediately.

2. **Guided clarification** — For ambiguous prompts, the agent calls
   `ask_clarification()` for each uncertain field, returning structured
   `ClarificationQuestion` objects.  The CLI presents these as numbered menus;
   an API caller can surface them in any UI.

3. **Defaults fallback** — In `--no-interactive` mode, or after `max_rounds`
   clarification rounds, the agent uses physics-motivated defaults derived from
   the original DeepLenseSim parameter priors.

---

## Citation

If you use this work, please also cite the underlying simulation packages:

```bibtex
@software{DeepLenseSim,
  author = {Michael W. Toomey},
  title  = {DeepLenseSim},
  url    = {https://github.com/mwt5345/DeepLenseSim},
}

@article{lenstronomy,
  author  = {Birrer, Simon and Amara, Adam},
  title   = {lenstronomy: Multi-purpose gravitational lens modelling software package},
  journal = {Physics of the Dark Universe},
  year    = {2018},
  doi     = {10.1016/j.dark.2018.11.002},
}
```
