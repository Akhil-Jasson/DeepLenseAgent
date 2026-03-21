"""
Test suite for the DeepLenseSim Agent.

Tests are organised into:
  1. Pydantic model validation (no external deps)
  2. Tool function logic (no LLM calls)
  3. Simulator stub (no lenstronomy required)
  4. Integration smoke test (mocked agent)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── ensure package is importable from repo root ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from deeplense_agent.models import (
    BatchSimulationRequest,
    LensParams,
    ModelConfig,
    ObservationConfig,
    SimulationRequest,
    SourceParams,
    SubstructureParams,
    SubstructureType,
)
from deeplense_agent.models.simulation_params import OBSERVATION_CONFIGS
from deeplense_agent.agent.tools import (
    _coerce_to_request,
    get_model_details,
    list_available_models,
    run_simulation,
    summarise_results,
    validate_simulation_params,
)
from deeplense_agent.simulator.engine import (
    DeepLenseSimulator,
    _estimate_snr,
    _stub_simulate,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Pydantic model validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestPydanticModels:

    def test_default_simulation_request_is_valid(self):
        req = SimulationRequest()
        assert req.model == ModelConfig.MODEL_I
        assert req.substructure_type == SubstructureType.NO_SUB
        assert req.num_images == 1
        assert req.source.z_source > req.lens.z_lens

    def test_source_must_be_behind_lens(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Source redshift"):
            SimulationRequest(
                lens=LensParams(z_lens=2.0),
                source=SourceParams(z_source=1.0),
            )

    def test_theta_E_too_small_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LensParams(theta_E=0.05)

    def test_all_model_configs_have_observation_configs(self):
        for mc in ModelConfig:
            obs = OBSERVATION_CONFIGS[mc]
            assert obs.num_pixels == 64
            assert obs.pixel_scale > 0
            assert obs.psf_fwhm > 0

    def test_model_ii_is_euclid(self):
        obs = OBSERVATION_CONFIGS[ModelConfig.MODEL_II]
        assert abs(obs.pixel_scale - 0.1) < 0.01
        assert obs.num_exposures == 4

    def test_model_iii_is_hst(self):
        obs = OBSERVATION_CONFIGS[ModelConfig.MODEL_III]
        assert obs.pixel_scale <= 0.06  # HST has finer pixels

    def test_batch_request_expands_correctly(self):
        batch = BatchSimulationRequest(
            models=[ModelConfig.MODEL_I, ModelConfig.MODEL_II],
            substructure_types=[SubstructureType.NO_SUB, SubstructureType.CDM],
            num_images_per_class=3,
        )
        reqs = batch.to_individual_requests()
        # 2 models × 2 substructures × 3 images = 12
        assert len(reqs) == 12

    def test_observation_config_override(self):
        custom_obs = ObservationConfig(pixel_scale=0.2, num_pixels=128, psf_fwhm=0.2)
        req = SimulationRequest(observation=custom_obs)
        eff = req.effective_observation()
        assert eff.num_pixels == 128
        assert eff.pixel_scale == 0.2

    def test_substructure_params_defaults(self):
        sp = SubstructureParams()
        assert sp.num_subhalos >= 1
        assert sp.num_vortices >= 1
        assert sp.axion_mass < 0  # log10 eV, should be very negative


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Tool function logic
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolFunctions:

    def test_list_available_models_returns_four_configs(self):
        result = list_available_models()
        assert "model_configs" in result
        assert len(result["model_configs"]) == 4
        names = [m["name"] for m in result["model_configs"]]
        assert "Model_I" in names
        assert "Model_III" in names

    def test_list_available_models_returns_three_substructures(self):
        result = list_available_models()
        subs = [s["name"] for s in result["substructure_types"]]
        assert "no_sub" in subs
        assert "cdm" in subs
        assert "axion" in subs

    def test_get_model_details_model_i(self):
        details = get_model_details("Model_I")
        assert details["telescope"] == "Synthetic (Gaussian PSF)"
        assert "observation_config" in details
        assert "default_priors" in details

    def test_get_model_details_invalid(self):
        details = get_model_details("Model_X")
        assert "error" in details

    def test_validate_simulation_params_valid(self):
        params = {"model": "Model_I", "substructure_type": "cdm", "num_images": 2}
        result = validate_simulation_params(params)
        assert result["valid"] is True

    def test_validate_simulation_params_invalid(self):
        params = {
            "model": "Model_I",
            "substructure_type": "cdm",
            "lens": {"z_lens": 2.0},
            "source": {"z_source": 1.0},  # violates z_source > z_lens
        }
        result = validate_simulation_params(params)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_coerce_to_request_flat_params(self):
        flat = {
            "model": "Model_II",
            "substructure_type": "axion",
            "z_lens": 0.4,
            "z_source": 1.8,
            "theta_E": 1.1,
        }
        req = _coerce_to_request(flat)
        assert req.model == ModelConfig.MODEL_II
        assert req.substructure_type == SubstructureType.AXION
        assert req.lens.z_lens == 0.4
        assert req.lens.theta_E == 1.1

    def test_coerce_to_request_nested_params(self):
        nested = {
            "model": "Model_III",
            "substructure_type": "no_sub",
            "lens": {"z_lens": 0.3, "theta_E": 0.9},
            "source": {"z_source": 2.0},
        }
        req = _coerce_to_request(nested)
        assert req.model == ModelConfig.MODEL_III
        assert req.lens.z_lens == 0.3

    def test_coerce_string_substructure_alias(self):
        p = {"model": "Model_I", "substructure": "cdm"}
        req = _coerce_to_request(p)
        assert req.substructure_type == SubstructureType.CDM

    def test_summarise_results_format(self):
        batch_result_dict = {
            "success": True,
            "total_requested": 6,
            "total_succeeded": 6,
            "total_failed": 0,
            "summary": {
                "by_model": {"Model_I": 3, "Model_II": 3},
                "by_substructure": {"no_sub": 2, "cdm": 2, "axion": 2},
                "mean_snr": 24.3,
                "std_snr": 3.1,
            },
            "output_directory": "outputs",
        }
        summary = summarise_results(batch_result_dict)
        assert "6/6" in summary
        assert "Model_I" in summary
        assert "24.3" in summary


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Stub simulator
# ═══════════════════════════════════════════════════════════════════════════════


class TestStubSimulator:

    def _make_request(self, sub: SubstructureType = SubstructureType.NO_SUB) -> SimulationRequest:
        return SimulationRequest(
            model=ModelConfig.MODEL_I,
            substructure_type=sub,
            num_images=1,
            random_seed=42,
        )

    def _obs(self) -> ObservationConfig:
        return OBSERVATION_CONFIGS[ModelConfig.MODEL_I]

    def test_stub_returns_float32_array(self):
        req = self._make_request()
        rng = np.random.default_rng(42)
        img = _stub_simulate(req, self._obs(), rng)
        assert img.dtype == np.float32
        assert img.shape == (64, 64)

    def test_stub_no_sub_non_negative(self):
        req = self._make_request(SubstructureType.NO_SUB)
        rng = np.random.default_rng(0)
        img = _stub_simulate(req, self._obs(), rng)
        assert np.all(img >= 0)

    def test_stub_cdm_produces_image(self):
        req = self._make_request(SubstructureType.CDM)
        rng = np.random.default_rng(1)
        img = _stub_simulate(req, self._obs(), rng)
        assert img.shape == (64, 64)
        assert img.max() > 0

    def test_stub_axion_produces_image(self):
        req = self._make_request(SubstructureType.AXION)
        rng = np.random.default_rng(2)
        img = _stub_simulate(req, self._obs(), rng)
        assert img.shape == (64, 64)

    def test_snr_estimate_positive(self):
        req = self._make_request()
        rng = np.random.default_rng(42)
        img = _stub_simulate(req, self._obs(), rng)
        snr = _estimate_snr(img)
        assert snr > 0

    def test_reproducibility(self):
        req = self._make_request()
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        img1 = _stub_simulate(req, self._obs(), rng1)
        img2 = _stub_simulate(req, self._obs(), rng2)
        np.testing.assert_array_equal(img1, img2)


class TestDeepLenseSimulator:
    """Integration tests using the DeepLenseSimulator wrapper."""

    def setup_method(self):
        self.sim = DeepLenseSimulator(output_dir=Path("/tmp/test_deeplense_outputs"))
        # Patch lenstronomy detection to force stub mode
        self.sim._use_lenstronomy = False

    def test_simulate_returns_result(self):
        req = SimulationRequest(
            model=ModelConfig.MODEL_I,
            substructure_type=SubstructureType.NO_SUB,
            random_seed=7,
        )
        result = self.sim.simulate(req)
        assert result.success
        assert result.metadata.image_id
        assert result.metadata.snr_estimate is not None
        assert result.metadata.snr_estimate > 0

    def test_simulate_cdm_metadata(self):
        req = SimulationRequest(
            model=ModelConfig.MODEL_II,
            substructure_type=SubstructureType.CDM,
            random_seed=10,
        )
        result = self.sim.simulate(req)
        assert result.success
        assert result.metadata.substructure_type == SubstructureType.CDM
        assert result.metadata.num_subhalos is not None

    def test_simulate_axion_metadata(self):
        req = SimulationRequest(
            model=ModelConfig.MODEL_III,
            substructure_type=SubstructureType.AXION,
            random_seed=11,
        )
        result = self.sim.simulate(req)
        assert result.success
        assert result.metadata.num_vortices is not None
        assert result.metadata.axion_mass_log10_eV is not None

    def test_simulate_model_iii_hst_resolution(self):
        req = SimulationRequest(
            model=ModelConfig.MODEL_III,
            substructure_type=SubstructureType.NO_SUB,
        )
        result = self.sim.simulate(req)
        assert result.metadata.pixel_scale <= 0.06

    def test_batch_simulate_count(self):
        reqs = [
            SimulationRequest(model=ModelConfig.MODEL_I, substructure_type=st, random_seed=i)
            for i, st in enumerate(SubstructureType)
        ]
        results = self.sim.simulate_batch(reqs)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_simulate_image_b64_set(self):
        req = SimulationRequest(
            model=ModelConfig.MODEL_I,
            substructure_type=SubstructureType.NO_SUB,
            random_seed=42,
        )
        result = self.sim.simulate(req)
        # b64 may be empty if matplotlib not available, but should not error
        assert result.image_b64 is not None

    def test_run_simulation_tool_valid_params(self, tmp_path):
        params = {
            "model": "Model_I",
            "substructure_type": "no_sub",
            "num_images": 1,
            "random_seed": 42,
        }
        # Patch the global simulator to use stub mode
        with patch(
            "deeplense_agent.agent.tools._simulator",
            DeepLenseSimulator(output_dir=tmp_path),
        ) as mock_sim:
            mock_sim._use_lenstronomy = False
            result = run_simulation(params, output_dir=str(tmp_path))
        assert result["success"]
        assert result["metadata"]["substructure_type"] == "no_sub"

    def test_run_simulation_tool_invalid_params(self):
        params = {"model": "Model_I", "substructure_type": "cdm", "z_lens": 3.0, "z_source": 1.0}
        result = run_simulation(params)
        assert not result["success"]
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Run tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
