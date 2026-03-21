"""
Pydantic models for DeepLenseSim simulation parameters.

Covers Model_I through Model_IV configurations as defined in the DeepLenseSim
repository (mwt5345/DeepLenseSim), each corresponding to a different telescope
or noise configuration for strong gravitational lensing simulations.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SubstructureType(str, Enum):
    """Dark matter substructure classes supported by DeepLenseSim."""

    NO_SUB = "no_sub"
    CDM = "cdm"       # Cold Dark Matter — NFW sub-halos
    AXION = "axion"   # Ultra-light axion dark matter — vortex substructure


class ModelConfig(str, Enum):
    """
    Telescope / observation configurations mirroring the DeepLenseSim datasets.

    Model_I  – Gaussian PSF, Gaussian + Poissonian noise (SNR ~ 25), 64×64 px
    Model_II – Euclid VIS instrument characteristics, 64×64 px
    Model_III– Hubble Space Telescope (HST) ACS/WFC characteristics, 64×64 px
    Model_IV – Euclid VIS (variant) with wider redshift priors, 64×64 px
    """

    MODEL_I   = "Model_I"
    MODEL_II  = "Model_II"
    MODEL_III = "Model_III"
    MODEL_IV  = "Model_IV"


# ---------------------------------------------------------------------------
# Observation / instrument configuration
# ---------------------------------------------------------------------------


class ObservationConfig(BaseModel):
    """
    Instrument and observing-condition parameters for a single band simulation.
    These values are pre-populated for each ModelConfig but can be overridden.
    """

    pixel_scale: Annotated[float, Field(gt=0.0, description="Arcseconds per pixel")] = 0.101
    num_pixels: Annotated[int, Field(ge=32, le=512, description="Image side length in pixels")] = 64
    exposure_time: Annotated[float, Field(gt=0.0, description="Total exposure time in seconds")] = 1000.0
    sky_brightness: Annotated[float, Field(gt=0.0, description="Sky surface brightness (mag/arcsec²)")] = 22.0
    magnitude_zero_point: Annotated[float, Field(description="Instrument zero-point magnitude")] = 25.0
    read_out_noise: Annotated[float, Field(ge=0.0, description="CCD read-out noise (electrons/pixel)")] = 7.0
    num_exposures: Annotated[int, Field(ge=1, description="Number of co-added exposures")] = 1
    psf_type: Annotated[Literal["gaussian", "pixel"], Field(description="PSF model type")] = "gaussian"
    psf_fwhm: Annotated[float, Field(gt=0.0, description="PSF FWHM in arcseconds")] = 0.1


# Pre-baked observation configs matching each DeepLenseSim model
OBSERVATION_CONFIGS: dict[ModelConfig, ObservationConfig] = {
    ModelConfig.MODEL_I: ObservationConfig(
        pixel_scale=0.08,
        num_pixels=64,
        exposure_time=1000.0,
        sky_brightness=22.0,
        magnitude_zero_point=25.0,
        read_out_noise=7.0,
        num_exposures=1,
        psf_type="gaussian",
        psf_fwhm=0.12,
    ),
    ModelConfig.MODEL_II: ObservationConfig(
        # Euclid VIS instrument
        pixel_scale=0.1,
        num_pixels=64,
        exposure_time=1800.0,
        sky_brightness=22.8,
        magnitude_zero_point=25.58,
        read_out_noise=4.2,
        num_exposures=4,
        psf_type="gaussian",
        psf_fwhm=0.18,
    ),
    ModelConfig.MODEL_III: ObservationConfig(
        # HST ACS/WFC F814W
        pixel_scale=0.05,
        num_pixels=64,
        exposure_time=2000.0,
        sky_brightness=22.3,
        magnitude_zero_point=25.95,
        read_out_noise=4.0,
        num_exposures=4,
        psf_type="gaussian",
        psf_fwhm=0.09,
    ),
    ModelConfig.MODEL_IV: ObservationConfig(
        # Euclid VIS variant (wider redshift prior)
        pixel_scale=0.1,
        num_pixels=64,
        exposure_time=1800.0,
        sky_brightness=22.8,
        magnitude_zero_point=25.58,
        read_out_noise=4.2,
        num_exposures=4,
        psf_type="gaussian",
        psf_fwhm=0.18,
    ),
}


# ---------------------------------------------------------------------------
# Physical parameter models
# ---------------------------------------------------------------------------


class LensParams(BaseModel):
    """
    Parameters for the main lensing galaxy (deflector).
    Uses a SIE (Singular Isothermal Ellipsoid) mass model + Sérsic light.
    """

    z_lens: Annotated[float, Field(gt=0.0, lt=5.0, description="Lens redshift")] = 0.5
    theta_E: Annotated[
        float, Field(gt=0.0, lt=3.0, description="Einstein radius in arcseconds")
    ] = 1.0
    e1: Annotated[
        float, Field(ge=-0.5, le=0.5, description="Ellipticity component e1 (SIE mass)")
    ] = 0.0
    e2: Annotated[
        float, Field(ge=-0.5, le=0.5, description="Ellipticity component e2 (SIE mass)")
    ] = 0.0
    # External shear
    gamma_ext: Annotated[
        float, Field(ge=0.0, le=0.2, description="External shear magnitude")
    ] = 0.05
    psi_ext: Annotated[
        float, Field(ge=0.0, lt=180.0, description="External shear angle (degrees)")
    ] = 0.0
    # Lens light (Sérsic)
    lens_light_mag: Annotated[
        float, Field(gt=0.0, description="Lens galaxy apparent magnitude")
    ] = 25.0
    R_sersic_lens: Annotated[
        float, Field(gt=0.0, description="Sérsic effective radius (arcsec)")
    ] = 1.0
    n_sersic_lens: Annotated[
        float, Field(gt=0.5, le=8.0, description="Sérsic index")
    ] = 4.0

    @field_validator("theta_E")
    @classmethod
    def theta_E_reasonable(cls, v: float) -> float:
        if v < 0.1:
            raise ValueError("Einstein radius < 0.1 arcsec is unphysically small for galaxy-scale lensing.")
        return v


class SourceParams(BaseModel):
    """Parameters for the background source galaxy."""

    z_source: Annotated[float, Field(gt=0.0, lt=10.0, description="Source redshift")] = 1.5
    source_x: Annotated[
        float, Field(ge=-0.5, le=0.5, description="Source x-offset from lens centre (arcsec)")
    ] = 0.1
    source_y: Annotated[
        float, Field(ge=-0.5, le=0.5, description="Source y-offset from lens centre (arcsec)")
    ] = 0.05
    source_mag: Annotated[
        float, Field(gt=0.0, description="Source apparent magnitude")
    ] = 20.0
    R_sersic_source: Annotated[
        float, Field(gt=0.0, description="Source Sérsic effective radius (arcsec)")
    ] = 0.3
    n_sersic_source: Annotated[
        float, Field(gt=0.5, le=6.0, description="Source Sérsic index")
    ] = 1.0
    e1_source: Annotated[
        float, Field(ge=-0.5, le=0.5, description="Source ellipticity e1")
    ] = 0.0
    e2_source: Annotated[
        float, Field(ge=-0.5, le=0.5, description="Source ellipticity e2")
    ] = 0.0

    @model_validator(mode="after")
    def source_behind_lens(self) -> "SourceParams":
        # Will be cross-checked against z_lens in SimulationRequest
        return self


class SubstructureParams(BaseModel):
    """
    Parameters for dark matter substructure (CDM or axion).
    Ignored when substructure_type == SubstructureType.NO_SUB.
    """

    # CDM sub-halos (NFW)
    num_subhalos: Annotated[
        int, Field(ge=1, le=200, description="Number of NFW sub-halos (CDM mode)")
    ] = 20
    subhalo_mass_min: Annotated[
        float, Field(gt=0.0, description="Minimum sub-halo mass (log10 M☉)")
    ] = 7.0
    subhalo_mass_max: Annotated[
        float, Field(gt=0.0, description="Maximum sub-halo mass (log10 M☉)")
    ] = 10.0
    # Axion vortices
    num_vortices: Annotated[
        int, Field(ge=1, le=500, description="Number of axion vortices")
    ] = 100
    axion_mass: Annotated[
        float, Field(gt=0.0, description="Axion mass (log10 eV)")
    ] = -22.0


# ---------------------------------------------------------------------------
# Top-level request models
# ---------------------------------------------------------------------------


class SimulationRequest(BaseModel):
    """
    Complete, validated specification for a single DeepLenseSim run.
    This is the primary structured output produced by the agent after
    parsing and clarifying a natural-language request.
    """

    model: ModelConfig = Field(
        default=ModelConfig.MODEL_I,
        description="Telescope/noise configuration (Model_I–IV)",
    )
    substructure_type: SubstructureType = Field(
        default=SubstructureType.NO_SUB,
        description="Dark matter substructure class to simulate",
    )
    num_images: Annotated[
        int,
        Field(ge=1, le=100, description="Number of images to generate"),
    ] = 1
    lens: LensParams = Field(default_factory=LensParams)
    source: SourceParams = Field(default_factory=SourceParams)
    substructure: SubstructureParams = Field(default_factory=SubstructureParams)
    observation: Optional[ObservationConfig] = Field(
        default=None,
        description="Custom observation config (None → use model defaults)",
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="RNG seed for reproducibility (None = random)",
    )
    add_noise: bool = Field(default=True, description="Whether to add detector noise")

    @model_validator(mode="after")
    def source_behind_lens(self) -> "SimulationRequest":
        if self.source.z_source <= self.lens.z_lens:
            raise ValueError(
                f"Source redshift ({self.source.z_source}) must be greater than "
                f"lens redshift ({self.lens.z_lens})."
            )
        return self

    def effective_observation(self) -> ObservationConfig:
        """Return the observation config to use, applying model defaults."""
        if self.observation is not None:
            return self.observation
        return OBSERVATION_CONFIGS[self.model]


class BatchSimulationRequest(BaseModel):
    """
    Request to generate images across multiple substructure classes
    (and optionally multiple models) in one shot.
    """

    models: list[ModelConfig] = Field(
        default=[ModelConfig.MODEL_I],
        description="List of model configurations to simulate",
        min_length=1,
    )
    substructure_types: list[SubstructureType] = Field(
        default_factory=lambda: list(SubstructureType),
        description="Substructure classes to include",
        min_length=1,
    )
    num_images_per_class: Annotated[
        int, Field(ge=1, le=50, description="Images per (model, substructure) combination")
    ] = 5
    lens: LensParams = Field(default_factory=LensParams)
    source: SourceParams = Field(default_factory=SourceParams)
    substructure: SubstructureParams = Field(default_factory=SubstructureParams)
    random_seed: Optional[int] = None
    add_noise: bool = True

    def to_individual_requests(self) -> list[SimulationRequest]:
        """Expand into individual SimulationRequest objects."""
        requests: list[SimulationRequest] = []
        seed = self.random_seed
        for model in self.models:
            for sub_type in self.substructure_types:
                for _ in range(self.num_images_per_class):
                    requests.append(
                        SimulationRequest(
                            model=model,
                            substructure_type=sub_type,
                            num_images=1,
                            lens=self.lens,
                            source=self.source,
                            substructure=self.substructure,
                            random_seed=seed,
                            add_noise=self.add_noise,
                        )
                    )
                    if seed is not None:
                        seed += 1
        return requests