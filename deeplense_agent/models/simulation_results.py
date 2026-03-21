"""
Pydantic result/output models for DeepLenseSim simulation outputs.
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
from pydantic import BaseModel, Field, field_serializer

from .simulation_params import ModelConfig, SimulationRequest, SubstructureType


class ImageMetadata(BaseModel):
    """
    Rich metadata attached to every generated strong-lensing image.
    Designed to be fully serialisable to JSON for downstream ML pipelines.
    """

    model_config: ClassVar[dict] = {"arbitrary_types_allowed": True}

    image_id: str = Field(description="Unique identifier for this image")
    model: ModelConfig
    substructure_type: SubstructureType
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Physical parameters
    z_lens: float
    z_source: float
    theta_E: float               # Einstein radius (arcsec)
    e1_lens: float
    e2_lens: float
    gamma_ext: float             # External shear magnitude
    source_x: float              # Source offset x (arcsec)
    source_y: float              # Source offset y (arcsec)

    # Image statistics (filled after generation)
    image_shape: tuple[int, int]
    pixel_scale: float           # arcsec / pixel
    snr_estimate: Optional[float] = None
    einstein_ring_radius_px: Optional[float] = None  # in pixels
    peak_flux: Optional[float] = None
    mean_flux: Optional[float] = None

    # Substructure details (if applicable)
    num_subhalos: Optional[int] = None
    num_vortices: Optional[int] = None
    axion_mass_log10_eV: Optional[float] = None

    # Provenance
    random_seed: Optional[int] = None
    simulation_time_sec: Optional[float] = None
    lenstronomy_version: Optional[str] = None




class SimulationResult(BaseModel):
    """
    A single simulated lensing image paired with its metadata and the
    original request that produced it.
    """

    model_config: ClassVar[dict] = {"arbitrary_types_allowed": True}

    request: SimulationRequest
    metadata: ImageMetadata
    image_array: Optional[Any] = Field(
        default=None,
        description="numpy ndarray (H×W float32)",
        exclude=True,            # excluded from JSON serialisation by default
    )
    image_path: Optional[Path] = Field(
        default=None,
        description="Path to saved image file (if persisted to disk)",
    )
    image_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded PNG thumbnail for API responses",
    )
    success: bool = True
    error_message: Optional[str] = None

    @field_serializer("image_path")
    def serialise_path(self, v: Optional[Path]) -> Optional[str]:
        return str(v) if v else None

    def as_dict(self) -> dict:
        """Return a JSON-serialisable dict (excludes raw array)."""
        d = self.model_dump(exclude={"image_array"})
        return d


class BatchSimulationResult(BaseModel):
    """Aggregated result for a batch simulation request."""

    results: list[SimulationResult]
    total_requested: int
    total_succeeded: int
    total_failed: int
    output_directory: Optional[str] = None
    summary: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        results: list[SimulationResult],
        output_directory: Optional[str] = None,
    ) -> "BatchSimulationResult":
        succeeded = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        # Build summary statistics
        summary: dict[str, Any] = {}
        if succeeded:
            by_model: dict[str, int] = {}
            by_sub: dict[str, int] = {}
            for r in succeeded:
                m = r.metadata.model.value
                s = r.metadata.substructure_type.value
                by_model[m] = by_model.get(m, 0) + 1
                by_sub[s] = by_sub.get(s, 0) + 1
            summary["by_model"] = by_model
            summary["by_substructure"] = by_sub
            snrs = [r.metadata.snr_estimate for r in succeeded if r.metadata.snr_estimate]
            if snrs:
                summary["mean_snr"] = float(np.mean(snrs))
                summary["std_snr"] = float(np.std(snrs))

        return cls(
            results=results,
            total_requested=len(results),
            total_succeeded=len(succeeded),
            total_failed=len(failed),
            output_directory=output_directory,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Agent-level models
# ---------------------------------------------------------------------------


class ClarificationQuestion(BaseModel):
    """A structured question the agent asks the human before running a sim."""

    question: str = Field(description="The question text to display to the user")
    field_name: str = Field(description="Simulation parameter being clarified")
    current_value: Optional[Any] = Field(
        default=None,
        description="Current (default or inferred) value for the parameter",
    )
    suggested_options: list[str] = Field(
        default_factory=list,
        description="Suggested answer options (if finite choice set)",
    )
    is_required: bool = Field(
        default=False,
        description="Whether this parameter must be specified before simulation",
    )


class AgentResponse(BaseModel):
    """
    Top-level structured output from the DeepLenseSim agent.
    Returned after a full interaction cycle (optional HITL clarification → simulation).
    """

    user_prompt: str
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    resolved_request: Optional[SimulationRequest] = None
    batch_result: Optional[BatchSimulationResult] = None
    agent_explanation: str = Field(
        description="Human-readable summary of what was simulated and why"
    )
    warnings: list[str] = Field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
