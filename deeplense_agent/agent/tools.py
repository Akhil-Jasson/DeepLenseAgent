"""
Tool functions for the DeepLenseSim agent.

Each function is decorated with @agent.tool (registered in agent.py).
They are documented exhaustively so that the LLM can reason about when
and how to call each tool.

Tool taxonomy
─────────────
  list_available_models        → enumerate supported Model configs & substructure types
  get_model_details            → detailed description of a specific ModelConfig
  validate_simulation_params   → Pydantic-validate a parameter dict, return errors
  run_simulation               → execute ONE simulation and return result + metadata
  run_batch_simulation         → execute MANY simulations across models/substructures
  ask_clarification            → human-in-the-loop: pose a clarifying question
  summarise_results            → produce a human-readable summary of batch results
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError
from pydantic_ai import RunContext

from ..models import (
    BatchSimulationRequest,
    BatchSimulationResult,
    ClarificationQuestion,
    LensParams,
    ModelConfig,
    ObservationConfig,
    SimulationRequest,
    SourceParams,
    SubstructureParams,
    SubstructureType,
)
from ..models.simulation_params import OBSERVATION_CONFIGS
from ..models.simulation_results import SimulationResult
from ..simulator import DeepLenseSimulator

logger = logging.getLogger(__name__)

# Shared simulator instance (output_dir can be overridden via AgentDeps)
_simulator: Optional[DeepLenseSimulator] = None


def get_simulator(output_dir: Optional[Path] = None) -> DeepLenseSimulator:
    global _simulator
    if _simulator is None or (output_dir and _simulator.output_dir != output_dir):
        _simulator = DeepLenseSimulator(output_dir=output_dir or Path("outputs"))
    return _simulator


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations (plain functions — registered in agent.py)
# ─────────────────────────────────────────────────────────────────────────────


def list_available_models() -> dict[str, Any]:
    """
    Return the full catalogue of supported Model configurations and
    substructure classes.

    Returns
    -------
    dict with keys:
      - "model_configs": list of {name, description, pixel_scale, psf_fwhm}
      - "substructure_types": list of {name, description}
    """
    model_info = []
    descriptions = {
        ModelConfig.MODEL_I: (
            "Gaussian PSF (FWHM 0.12\"), Gaussian + Poissonian noise (SNR~25). "
            "The baseline idealized dataset from DeepLenseSim."
        ),
        ModelConfig.MODEL_II: (
            "Euclid VIS instrument characteristics: 0.1\"/px, 1800 s exposure, "
            "4 co-adds.  Realistic space-based survey conditions."
        ),
        ModelConfig.MODEL_III: (
            "Hubble Space Telescope ACS/WFC F814W: 0.05\"/px, 2000 s, excellent "
            "angular resolution (PSF FWHM 0.09\"). Highest resolution configuration."
        ),
        ModelConfig.MODEL_IV: (
            "Euclid VIS variant with a wider redshift prior for the lens (0.2–0.9) "
            "and source (0.5–2.5).  Otherwise identical to Model_II."
        ),
    }
    for mc in ModelConfig:
        obs = OBSERVATION_CONFIGS[mc]
        model_info.append(
            {
                "name": mc.value,
                "description": descriptions[mc],
                "pixel_scale_arcsec": obs.pixel_scale,
                "num_pixels": obs.num_pixels,
                "psf_fwhm_arcsec": obs.psf_fwhm,
                "exposure_time_sec": obs.exposure_time,
                "telescope": _telescope_name(mc),
            }
        )

    sub_info = [
        {
            "name": SubstructureType.NO_SUB.value,
            "description": "No dark matter substructure — smooth SIE lens only.",
        },
        {
            "name": SubstructureType.CDM.value,
            "description": (
                "Cold Dark Matter: NFW sub-halos drawn from a truncated "
                "power-law mass function (10^7 – 10^10 M☉ by default)."
            ),
        },
        {
            "name": SubstructureType.AXION.value,
            "description": (
                "Ultra-light axion dark matter: vortex substructure "
                "parameterised by axion mass (~10^-22 eV by default)."
            ),
        },
    ]
    return {"model_configs": model_info, "substructure_types": sub_info}


def get_model_details(model_name: str) -> dict[str, Any]:
    """
    Return detailed parameter defaults and physical prior ranges for a given
    Model configuration name ("Model_I", "Model_II", "Model_III", "Model_IV").

    Parameters
    ----------
    model_name : str
        One of "Model_I", "Model_II", "Model_III", "Model_IV".
    """
    try:
        mc = ModelConfig(model_name)
    except ValueError:
        return {"error": f"Unknown model '{model_name}'. Valid: {[m.value for m in ModelConfig]}"}

    obs = OBSERVATION_CONFIGS[mc]
    priors = _get_default_priors(mc)
    return {
        "model": mc.value,
        "telescope": _telescope_name(mc),
        "observation_config": obs.model_dump(),
        "default_priors": priors,
        "supported_substructure": [s.value for s in SubstructureType],
    }


def validate_simulation_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a raw parameter dictionary against the SimulationRequest Pydantic model.
    Returns either {"valid": True, "request": <serialised request>} or
    {"valid": False, "errors": [list of error messages]}.

    Parameters
    ----------
    params : dict
        Raw parameter dict, e.g. from NL parsing.  May use nested or flat keys.
    """
    try:
        # Support flat param dicts coming from NL parsing
        req = _coerce_to_request(params)
        return {"valid": True, "request": req.model_dump()}
    except ValidationError as exc:
        errors = [
            f"{' → '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        ]
        return {"valid": False, "errors": errors}
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}


def run_simulation(params: dict[str, Any], output_dir: str = "outputs") -> dict[str, Any]:
    """
    Execute a single strong gravitational lensing simulation.

    Parameters
    ----------
    params : dict
        Simulation parameters.  Will be validated via Pydantic before use.
        Minimum required: "model" (str) and "substructure_type" (str).
    output_dir : str
        Directory where output images will be saved.

    Returns
    -------
    dict with keys:
      - "success": bool
      - "image_id": str
      - "image_path": str
      - "image_b64": str  (base64 PNG thumbnail)
      - "metadata": dict  (full ImageMetadata)
      - "error": str | None
    """
    validation = validate_simulation_params(params)
    if not validation["valid"]:
        return {
            "success": False,
            "error": f"Parameter validation failed: {validation['errors']}",
        }

    req = _coerce_to_request(params)
    sim = get_simulator(Path(output_dir))
    result: SimulationResult = sim.simulate(req)

    return {
        "success": result.success,
        "image_id": result.metadata.image_id if result.metadata else None,
        "image_path": str(result.image_path) if result.image_path else None,
        "image_b64": result.image_b64 or "",
        "metadata": result.metadata.model_dump() if result.metadata else {},
        "error": result.error_message,
    }


def run_batch_simulation(
    models: list[str],
    substructure_types: list[str],
    num_images_per_class: int = 5,
    lens_params: Optional[dict] = None,
    source_params: Optional[dict] = None,
    substructure_params: Optional[dict] = None,
    random_seed: Optional[int] = None,
    add_noise: bool = True,
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """
    Execute a batch of simulations covering multiple models and substructure types.

    Parameters
    ----------
    models : list[str]
        List of model config names, e.g. ["Model_I", "Model_II"].
    substructure_types : list[str]
        List of substructure type names, e.g. ["no_sub", "cdm", "axion"].
    num_images_per_class : int
        Number of images per (model, substructure) combination.
    lens_params : dict | None
        Overrides for LensParams (optional).
    source_params : dict | None
        Overrides for SourceParams (optional).
    substructure_params : dict | None
        Overrides for SubstructureParams (optional).
    random_seed : int | None
        Starting seed (incremented per image for reproducibility).
    add_noise : bool
        Whether to add detector noise.
    output_dir : str
        Directory where output images will be saved.

    Returns
    -------
    dict with keys:
      - "success": bool
      - "total_requested": int
      - "total_succeeded": int
      - "total_failed": int
      - "summary": dict (by-model and by-substructure counts, mean SNR)
      - "output_directory": str
      - "errors": list[str]
    """
    # Coerce strings → enums
    try:
        model_enums = [ModelConfig(m) for m in models]
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    try:
        sub_enums = [SubstructureType(s) for s in substructure_types]
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    try:
        batch_req = BatchSimulationRequest(
            models=model_enums,
            substructure_types=sub_enums,
            num_images_per_class=num_images_per_class,
            lens=LensParams(**(lens_params or {})),
            source=SourceParams(**(source_params or {})),
            substructure=SubstructureParams(**(substructure_params or {})),
            random_seed=random_seed,
            add_noise=add_noise,
        )
        individual_reqs = batch_req.to_individual_requests()
    except Exception as exc:
        msg = str(exc)
        if "Value error," in msg:
            msg = msg.split("Value error,")[-1].strip().split("[type=")[0].strip()
        print(f"\n  ❌ Invalid parameters: {msg}\n")
        return {"success": False, "error": msg}
    sim = get_simulator(Path(output_dir))
    results = sim.simulate_batch(individual_reqs)
    batch_result = BatchSimulationResult.from_results(results, output_directory=output_dir)

    errors = [r.error_message for r in results if r.error_message]
    return {
        "success": batch_result.total_failed == 0,
        "total_requested": batch_result.total_requested,
        "total_succeeded": batch_result.total_succeeded,
        "total_failed": batch_result.total_failed,
        "summary": batch_result.summary,
        "output_directory": output_dir,
        "errors": errors[:5],  # cap error list for brevity
    }


def ask_clarification(
    question: str,
    field_name: str,
    current_value: Optional[Any] = None,
    suggested_options: Optional[list[str]] = None,
    is_required: bool = False,
) -> ClarificationQuestion:
    """
    Construct a structured clarification question to present to the human user.
    The agent should call this when the user's natural language request is
    ambiguous or missing a required parameter.

    Parameters
    ----------
    question : str
        Human-readable question text.
    field_name : str
        Name of the simulation parameter being clarified.
    current_value : any
        The current default or inferred value for the parameter.
    suggested_options : list[str] | None
        Finite set of valid options (shown to user as choices).
    is_required : bool
        Whether this parameter must be resolved before simulation can proceed.

    Returns
    -------
    ClarificationQuestion (Pydantic model)
    """
    return ClarificationQuestion(
        question=question,
        field_name=field_name,
        current_value=current_value,
        suggested_options=suggested_options or [],
        is_required=is_required,
    )


def summarise_results(batch_results: dict[str, Any]) -> str:
    """
    Produce a concise, human-readable summary of a batch simulation run.

    Parameters
    ----------
    batch_results : dict
        The dict returned by run_batch_simulation.

    Returns
    -------
    str — a formatted summary paragraph.
    """
    n_req = batch_results.get("total_requested", "?")
    n_ok = batch_results.get("total_succeeded", "?")
    n_fail = batch_results.get("total_failed", 0)
    summary = batch_results.get("summary", {})
    outdir = batch_results.get("output_directory", "outputs")

    lines = [
        f"✅ Simulation complete: {n_ok}/{n_req} images generated successfully.",
    ]
    if n_fail:
        lines.append(f"⚠️  {n_fail} simulation(s) failed.")

    if "by_model" in summary:
        model_str = ", ".join(f"{k}: {v}" for k, v in summary["by_model"].items())
        lines.append(f"   By model config → {model_str}")
    if "by_substructure" in summary:
        sub_str = ", ".join(f"{k}: {v}" for k, v in summary["by_substructure"].items())
        lines.append(f"   By substructure → {sub_str}")
    if "mean_snr" in summary:
        lines.append(f"   Mean estimated SNR ≈ {summary['mean_snr']:.1f} ± {summary.get('std_snr', 0):.1f}")
    lines.append(f"📁 Images saved to: {outdir}/")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _telescope_name(mc: ModelConfig) -> str:
    return {
        ModelConfig.MODEL_I: "Synthetic (Gaussian PSF)",
        ModelConfig.MODEL_II: "Euclid VIS",
        ModelConfig.MODEL_III: "HST ACS/WFC",
        ModelConfig.MODEL_IV: "Euclid VIS (wide-z variant)",
    }[mc]


def _get_default_priors(mc: ModelConfig) -> dict[str, Any]:
    """Return the DeepLenseSim parameter priors for a given model."""
    base = {
        "z_lens": {"min": 0.2, "max": 0.7, "default": 0.5},
        "z_source": {"min": 0.5, "max": 2.5, "default": 1.5},
        "theta_E_arcsec": {"min": 0.5, "max": 1.5, "default": 1.0},
        "ellipticity_e1": {"min": -0.3, "max": 0.3, "default": 0.0},
        "ellipticity_e2": {"min": -0.3, "max": 0.3, "default": 0.0},
        "external_shear": {"min": 0.0, "max": 0.1, "default": 0.04},
        "source_x_arcsec": {"min": -0.2, "max": 0.2, "default": 0.0},
        "source_y_arcsec": {"min": -0.2, "max": 0.2, "default": 0.0},
    }
    if mc == ModelConfig.MODEL_IV:
        base["z_lens"] = {"min": 0.2, "max": 0.9, "default": 0.5}
        base["z_source"] = {"min": 0.5, "max": 3.0, "default": 1.5}
    return base


def _coerce_to_request(params: dict[str, Any]) -> SimulationRequest:
    """
    Attempt to construct a SimulationRequest from a flat or nested dict,
    handling the most common natural-language abbreviations.
    """
    p = dict(params)

    # Normalise top-level aliases
    if "substructure" in p and isinstance(p["substructure"], str):
        p["substructure_type"] = p.pop("substructure")
    if "telescope" in p:
        p.pop("telescope")  # cosmetic only

    # Extract nested dicts if present; otherwise build from flat keys
    lens_dict = p.pop("lens", {})
    source_dict = p.pop("source", {})
    sub_dict = p.pop("substructure_params", {})
    obs_dict = p.pop("observation", None)

    # Flat overrides (common from NL extraction)
    for k in ("z_lens", "theta_E", "e1", "e2", "gamma_ext", "lens_light_mag"):
        if k in p:
            lens_dict[k] = p.pop(k)
    for k in ("z_source", "source_x", "source_y", "source_mag"):
        if k in p:
            source_dict[k] = p.pop(k)
    for k in ("num_subhalos", "num_vortices", "axion_mass"):
        if k in p:
            sub_dict[k] = p.pop(k)

    return SimulationRequest(
        **p,
        lens=LensParams(**lens_dict) if lens_dict else LensParams(),
        source=SourceParams(**source_dict) if source_dict else SourceParams(),
        substructure=SubstructureParams(**sub_dict) if sub_dict else SubstructureParams(),
        observation=ObservationConfig(**obs_dict) if obs_dict else None,
    )