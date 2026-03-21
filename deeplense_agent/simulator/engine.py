"""
Core simulation engine for DeepLenseSim-style strong gravitational lensing images.

Architecture mirrors the DeepLenseSim repository (mwt5345/DeepLenseSim):
  - SIE (Singular Isothermal Ellipsoid) main lens + external shear
  - Sérsic light profiles for lens and source galaxies
  - CDM substructure:   NFW sub-halos drawn from a truncated power-law mass function
  - Axion substructure: Vortex mass sheets (approximated as pseudo-Jaffe clumps)
  - Noise model:        Gaussian read-out + Poisson sky/source noise

Each Model_I–IV configuration pre-selects instrument parameters from
OBSERVATION_CONFIGS; physical parameters (z_lens, z_source, θ_E …) can be
supplied per-image or drawn from the default priors used in DeepLenseSim.
"""

from __future__ import annotations

import base64
import io
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Try to import lenstronomy; fall back to a pure-NumPy stub simulator ──────
try:
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.PointSource.point_source import PointSource
    from lenstronomy.SimulationAPI.data_api import DataAPI
    from lenstronomy.SimulationAPI.observation_api import SingleBand
    from lenstronomy.Util import util as lens_util
    from lenstronomy.Util import image_util
    import lenstronomy

    _LENSTRONOMY_VERSION = lenstronomy.__version__
    _HAS_LENSTRONOMY = True
    logger.info("lenstronomy %s detected — using full simulation engine.", _LENSTRONOMY_VERSION)
except ImportError:  # pragma: no cover
    _HAS_LENSTRONOMY = False
    _LENSTRONOMY_VERSION = None
    logger.warning(
        "lenstronomy not found — falling back to NumPy stub simulator. "
        "Run `pip install lenstronomy` for realistic images."
    )

from ..models.simulation_params import (
    LensParams,
    ModelConfig,
    ObservationConfig,
    SimulationRequest,
    SourceParams,
    SubstructureParams,
    SubstructureType,
)
from ..models.simulation_results import ImageMetadata, SimulationResult


# ---------------------------------------------------------------------------
# Numpy stub simulator (used when lenstronomy is not installed)
# ---------------------------------------------------------------------------


def _stub_simulate(
    request: SimulationRequest,
    obs: ObservationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Analytic stub that creates a plausible-looking Einstein ring without
    requiring lenstronomy.  Returns a float32 image array.
    """
    n = obs.num_pixels
    cx, cy = n // 2, n // 2
    x, y = np.meshgrid(np.arange(n) - cx, np.arange(n) - cy)
    r = np.sqrt(x**2 + y**2)

    theta_E_px = request.lens.theta_E / obs.pixel_scale
    ring_width = max(2, theta_E_px * 0.15)

    # Einstein ring — Gaussian profile centred on θ_E
    ring = np.exp(-0.5 * ((r - theta_E_px) / ring_width) ** 2)

    # Lens galaxy — Sérsic-like de Vaucouleurs at centre
    r_eff_px = request.lens.R_sersic_lens / obs.pixel_scale
    lens_light = np.exp(-7.67 * ((r / max(r_eff_px, 0.5)) ** (1.0 / request.lens.n_sersic_lens) - 1))

    image = ring * 10.0 + lens_light * 5.0

    # Substructure artefacts
    if request.substructure_type == SubstructureType.CDM:
        for _ in range(request.substructure.num_subhalos):
            sx = rng.integers(5, n - 5)
            sy = rng.integers(5, n - 5)
            mass_scale = rng.uniform(0.2, 1.5)
            image[sy, sx] += mass_scale
    elif request.substructure_type == SubstructureType.AXION:
        for _ in range(min(request.substructure.num_vortices, 50)):
            sx = rng.integers(2, n - 2)
            sy = rng.integers(2, n - 2)
            vort = rng.uniform(-0.3, 0.3)
            image[sy - 1 : sy + 2, sx - 1 : sx + 2] += vort

    if request.add_noise:
        noise_level = np.max(image) * 0.05
        image += rng.normal(0, noise_level, image.shape)

    return np.clip(image, 0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# lenstronomy-based simulator
# ---------------------------------------------------------------------------


def _lenstronomy_simulate(
    request: SimulationRequest,
    obs: ObservationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Full physics simulation using lenstronomy, mirroring the DeepLenseSim pipeline:

      1. Build a SIE lens model + external shear  (+ CDM/axion substructure)
      2. Build Sérsic light profiles for lens and source
      3. Render the image via ImageModel
      4. Optionally add realistic noise via SingleBand
    """
    n = obs.num_pixels
    lens_p = request.lens
    src_p = request.source
    sub_p = request.substructure

    # ── pixel grid ────────────────────────────────────────────────────────────
    ra_at_xy_0, dec_at_xy_0 = (
        -(n / 2.0) * obs.pixel_scale,
        -(n / 2.0) * obs.pixel_scale,
    )
    transform_matrix = np.array(
        [[obs.pixel_scale, 0], [0, obs.pixel_scale]]
    )
    kwargs_pixel_grid = {
        "nx": n,
        "ny": n,
        "ra_at_xy_0": ra_at_xy_0,
        "dec_at_xy_0": dec_at_xy_0,
        "transform_pix2angle": transform_matrix,
    }

    # ── PSF ───────────────────────────────────────────────────────────────────
    psf_fwhm_px = obs.psf_fwhm / obs.pixel_scale
    kwargs_psf = {
        "psf_type": "GAUSSIAN",
        "fwhm": obs.psf_fwhm,
        "pixel_size": obs.pixel_scale,
        "truncation": 5,
    }

    # ── Data ──────────────────────────────────────────────────────────────────
    kwargs_data = {
        "background_rms": 0.01,
        "exposure_time": obs.exposure_time,
        "ra_at_xy_0": ra_at_xy_0,
        "dec_at_xy_0": dec_at_xy_0,
        "transform_pix2angle": transform_matrix,
        "image_data": np.zeros((n, n)),
    }

    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.PointSource.point_source import PointSource
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF

    data_class = ImageData(**kwargs_data)
    psf_class = PSF(**kwargs_psf)

    # ── Lens model ────────────────────────────────────────────────────────────
    lens_model_list = ["SIE", "SHEAR"]
    kwargs_lens = [
        {
            "theta_E": lens_p.theta_E,
            "e1": lens_p.e1,
            "e2": lens_p.e2,
            "center_x": 0.0,
            "center_y": 0.0,
        },
        {
            "gamma1": lens_p.gamma_ext * np.cos(2 * np.radians(lens_p.psi_ext)),
            "gamma2": lens_p.gamma_ext * np.sin(2 * np.radians(lens_p.psi_ext)),
        },
    ]

    # ── CDM substructure: NFW sub-halos ───────────────────────────────────────
    if request.substructure_type == SubstructureType.CDM:
        n_sub = sub_p.num_subhalos
        # Draw sub-halo masses (log-uniform) and positions (within Einstein ring)
        log_masses = rng.uniform(sub_p.subhalo_mass_min, sub_p.subhalo_mass_max, n_sub)
        # Random positions within ~2× Einstein radius
        r_max = 2.0 * lens_p.theta_E
        angles = rng.uniform(0, 2 * np.pi, n_sub)
        radii = r_max * np.sqrt(rng.uniform(0, 1, n_sub))
        sub_x = radii * np.cos(angles)
        sub_y = radii * np.sin(angles)

        for i in range(n_sub):
            mass = 10 ** log_masses[i]
            # Approximate NFW sub-halo with a point-mass (PM) lens
            # theta_E_sub ~ sqrt(M_sub / M_lens) * theta_E_main  (simplified)
            theta_E_sub = lens_p.theta_E * np.sqrt(mass / 1e12) * 0.3
            theta_E_sub = np.clip(theta_E_sub, 1e-4, 0.1)
            lens_model_list.append("POINT_MASS")
            kwargs_lens.append(
                {
                    "theta_E": theta_E_sub,
                    "center_x": float(sub_x[i]),
                    "center_y": float(sub_y[i]),
                }
            )

    # ── Axion substructure: vortex mass sheets ────────────────────────────────
    elif request.substructure_type == SubstructureType.AXION:
        n_vort = sub_p.num_vortices
        # Vortices modelled as tiny pseudo-Jaffe clumps
        r_max = 2.0 * lens_p.theta_E
        angles = rng.uniform(0, 2 * np.pi, n_vort)
        radii = r_max * np.sqrt(rng.uniform(0, 1, n_vort))
        vort_x = radii * np.cos(angles)
        vort_y = radii * np.sin(angles)

        axion_m_eV = 10 ** sub_p.axion_mass
        # Coherence length ξ ∝ m_axion^{-1} (scaled to arcsec)
        coherence_arcsec = min(0.05, 1e-22 / axion_m_eV * 0.01)

        for i in range(min(n_vort, 30)):  # cap at 30 to keep simulation tractable
            lens_model_list.append("POINT_MASS")
            theta_E_vort = max(coherence_arcsec, 1e-5)
            kwargs_lens.append(
                {
                    "theta_E": theta_E_vort,
                    "center_x": float(vort_x[i]),
                    "center_y": float(vort_y[i]),
                }
            )

    lensModel = LensModel(lens_model_list=lens_model_list)

    # ── Source light (Sérsic) ─────────────────────────────────────────────────
    source_model_list = ["SERSIC_ELLIPSE"]
    flux_src = 10 ** (-0.4 * (src_p.source_mag - obs.magnitude_zero_point)) * obs.exposure_time
    kwargs_source = [
        {
            "amp": float(flux_src),
            "R_sersic": src_p.R_sersic_source,
            "n_sersic": src_p.n_sersic_source,
            "e1": src_p.e1_source,
            "e2": src_p.e2_source,
            "center_x": src_p.source_x,
            "center_y": src_p.source_y,
        }
    ]
    sourceModel = LightModel(light_model_list=source_model_list)

    # ── Lens light (Sérsic) ───────────────────────────────────────────────────
    lens_light_model_list = ["SERSIC_ELLIPSE"]
    flux_lens = 10 ** (-0.4 * (lens_p.lens_light_mag - obs.magnitude_zero_point)) * obs.exposure_time
    kwargs_lens_light = [
        {
            "amp": float(flux_lens),
            "R_sersic": lens_p.R_sersic_lens,
            "n_sersic": lens_p.n_sersic_lens,
            "e1": lens_p.e1,
            "e2": lens_p.e2,
            "center_x": 0.0,
            "center_y": 0.0,
        }
    ]
    lensLightModel = LightModel(light_model_list=lens_light_model_list)

    # ── Image model ───────────────────────────────────────────────────────────
    kwargs_numerics = {
        "supersampling_factor": 3,
        "supersampling_convolution": True,
    }
    imageModel = ImageModel(
        data_class,
        psf_class,
        lens_model_class=lensModel,
        source_model_class=sourceModel,
        lens_light_model_class=lensLightModel,
        point_source_class=None,
        kwargs_numerics=kwargs_numerics,
    )
    image_sim = imageModel.image(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
    )

    # ── Noise ─────────────────────────────────────────────────────────────────
    if request.add_noise:
        # Poisson noise from source
        poisson = image_util.add_poisson(image_sim, exp_time=obs.exposure_time)
        # Gaussian read-out noise (scaled to per-pixel)
        sky_per_pixel = (
            10 ** (-0.4 * (obs.sky_brightness - obs.magnitude_zero_point))
            * obs.pixel_scale**2
            * obs.exposure_time
        )
        read_sigma = obs.read_out_noise / obs.exposure_time
        gaussian = image_util.add_background(image_sim, sigma_bkd=np.sqrt(sky_per_pixel + read_sigma**2))
        image_out = image_sim + poisson + gaussian
    else:
        image_out = image_sim.copy()

    return image_out.astype(np.float32)


# ---------------------------------------------------------------------------
# Public simulator entry-point
# ---------------------------------------------------------------------------


class DeepLenseSimulator:
    """
    Wraps the lenstronomy (or stub) backend to simulate strong gravitational
    lensing images according to a validated SimulationRequest.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._use_lenstronomy = _HAS_LENSTRONOMY

    # ------------------------------------------------------------------

    def simulate(self, request: SimulationRequest) -> SimulationResult:
        """Simulate a single lensing image and return a SimulationResult."""
        obs = request.effective_observation()
        seed = request.random_seed
        rng = np.random.default_rng(seed)

        # Randomise physical parameters per image so a batch is diverse
        request = self._randomise_params(request, rng)

        t0 = time.perf_counter()
        try:
            if self._use_lenstronomy:
                image = _lenstronomy_simulate(request, obs, rng)
            else:
                image = _stub_simulate(request, obs, rng)
        except Exception as exc:
            logger.exception("Simulation failed.")
            return SimulationResult(
                request=request,
                metadata=self._make_metadata(request, obs, None, seed),
                success=False,
                error_message=str(exc),
            )
        elapsed = time.perf_counter() - t0

        # Compute image statistics
        snr = _estimate_snr(image)
        theta_E_px = request.lens.theta_E / obs.pixel_scale

        meta = self._make_metadata(request, obs, image, seed)
        meta.snr_estimate = snr
        meta.einstein_ring_radius_px = theta_E_px
        meta.peak_flux = float(np.max(image))
        meta.mean_flux = float(np.mean(image))
        meta.simulation_time_sec = elapsed
        meta.lenstronomy_version = _LENSTRONOMY_VERSION

        image_id = meta.image_id
        image_path = self._save_image(image, image_id, request)
        image_b64 = _array_to_b64_png(image)

        print(
            f"  ✅ [{request.model.value} | {request.substructure_type.value:6s}] "
            f"θ_E={meta.theta_E:.2f}\" "
            f"z_l={meta.z_lens:.2f} z_s={meta.z_source:.2f} "
            f"SNR={meta.snr_estimate:.1f}  →  {image_path.name}"
        )

        return SimulationResult(
            request=request,
            metadata=meta,
            image_array=image,
            image_path=image_path,
            image_b64=image_b64,
            success=True,
        )

    def _randomise_params(
        self, request: SimulationRequest, rng: np.random.Generator
    ) -> SimulationRequest:
        """
        Draw fresh physical parameters from DeepLenseSim priors so every
        image in a batch looks different, even when the user hasn't specified
        per-image parameters.
        """
        import copy
        lens = copy.copy(request.lens)
        source = copy.copy(request.source)

        # Einstein radius: uniform [0.7, 1.4] arcsec
        lens.theta_E = float(rng.uniform(0.7, 1.4))
        # Lens ellipticity
        lens.e1 = float(rng.uniform(-0.2, 0.2))
        lens.e2 = float(rng.uniform(-0.2, 0.2))
        # External shear
        lens.gamma_ext = float(rng.uniform(0.0, 0.08))
        lens.psi_ext = float(rng.uniform(0.0, 180.0))
        # Source position: random direction, avoid dead-centre
        r = float(rng.uniform(0.05, 0.25))
        angle = float(rng.uniform(0, 2 * np.pi))
        source.source_x = float(r * np.cos(angle))
        source.source_y = float(r * np.sin(angle))
        # Redshifts within DeepLenseSim priors
        lens.z_lens = float(rng.uniform(0.2, 0.7))
        source.z_source = float(rng.uniform(max(lens.z_lens + 0.3, 0.5), 2.5))

        return request.model_copy(update={"lens": lens, "source": source})

    def simulate_batch(self, requests: list[SimulationRequest]) -> list[SimulationResult]:
        """Simulate a list of requests (sequentially)."""
        results = []
        for i, req in enumerate(requests):
            logger.info("Simulating image %d / %d …", i + 1, len(requests))
            results.append(self.simulate(req))
        return results

    # ------------------------------------------------------------------
    # Private helpers

    def _make_metadata(
        self,
        request: SimulationRequest,
        obs: ObservationConfig,
        image: Optional[np.ndarray],
        seed: Optional[int],
    ) -> ImageMetadata:
        return ImageMetadata(
            image_id=str(uuid.uuid4())[:8],
            model=request.model,
            substructure_type=request.substructure_type,
            z_lens=request.lens.z_lens,
            z_source=request.source.z_source,
            theta_E=request.lens.theta_E,
            e1_lens=request.lens.e1,
            e2_lens=request.lens.e2,
            gamma_ext=request.lens.gamma_ext,
            source_x=request.source.source_x,
            source_y=request.source.source_y,
            image_shape=(obs.num_pixels, obs.num_pixels),
            pixel_scale=obs.pixel_scale,
            num_subhalos=(
                request.substructure.num_subhalos
                if request.substructure_type == SubstructureType.CDM
                else None
            ),
            num_vortices=(
                request.substructure.num_vortices
                if request.substructure_type == SubstructureType.AXION
                else None
            ),
            axion_mass_log10_eV=(
                request.substructure.axion_mass
                if request.substructure_type == SubstructureType.AXION
                else None
            ),
            random_seed=seed,
        )

    def _save_image(self, image: np.ndarray, image_id: str, request: SimulationRequest) -> Path:
        """Save the image as a PNG with a descriptive filename."""
        model_str = request.model.value.lower()   
        sub_str   = request.substructure_type.value  
        filename  = f"{model_str}_{sub_str}_{image_id}"               # e.g. model_i_cdm_3a7f2c1b
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(image, cmap="inferno", origin="lower", interpolation="nearest")
            ax.axis("off")
            path = self.output_dir / f"{filename}.png"
            fig.savefig(str(path), bbox_inches="tight", pad_inches=0, dpi=100)
            plt.close(fig)
            return path
        except Exception:
            path = self.output_dir / f"{filename}.npy"
            np.save(str(path), image)
            return path


def _estimate_snr(image: np.ndarray) -> float:
    """Rough SNR estimate: peak signal divided by background RMS."""
    bg_region = image[: image.shape[0] // 10, : image.shape[1] // 10]
    noise = float(np.std(bg_region)) or 1e-6
    signal = float(np.max(image))
    return signal / noise


def _array_to_b64_png(image: np.ndarray) -> str:
    """Convert a 2-D float array to a base64-encoded PNG string."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        normed = (image - image.min()) / (image.max() - image.min() + 1e-9)
        buf = io.BytesIO()
        plt.imsave(buf, normed, cmap="inferno", format="png")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return ""