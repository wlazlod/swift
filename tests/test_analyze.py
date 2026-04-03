"""Tests for Phase 2 analysis functions: calibration, power, multi-seed tables/figures.

Tests use synthetic fixture data that mirrors the smoke-test JSON structure.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Ensure scripts/ is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from analyze_common import (
    compute_snr,
    load_calibration_results,
    load_multi_seed_results,
    load_power_results,
    precision_at_k,
    V2_RESULTS_DIR,
)
from analyze_tables import (
    table_calibration,
    table_feature_localization,
    table_multi_seed_stability,
)
from analyze_figures import (
    figure_calibration_qq,
    figure_feature_localization,
    figure_feature_localization_by_magnitude,
    figure_power_curve,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def calibration_data() -> dict[str, Any]:
    """Minimal calibration JSON matching real structure."""
    return {
        "dataset": "taiwan_credit",
        "n_reps": 20,
        "n_permutations": 200,
        "calibration": {
            "0.01": {
                "nominal_alpha": 0.01,
                "empirical_fpr": 0.0,
                "n_rejected": 0,
                "n_reps": 20,
                "ci_95_lower": 0.0,
                "ci_95_upper": 0.161,
                "well_calibrated": True,
            },
            "0.05": {
                "nominal_alpha": 0.05,
                "empirical_fpr": 0.05,
                "n_rejected": 1,
                "n_reps": 20,
                "ci_95_lower": 0.009,
                "ci_95_upper": 0.238,
                "well_calibrated": True,
            },
            "0.1": {
                "nominal_alpha": 0.1,
                "empirical_fpr": 0.10,
                "n_rejected": 2,
                "n_reps": 20,
                "ci_95_lower": 0.028,
                "ci_95_upper": 0.301,
                "well_calibrated": True,
            },
        },
        "all_pvalues": list(np.linspace(0.005, 1.0, 460)),
        "swift_max_under_null": {
            "mean": 0.007,
            "std": 0.003,
            "median": 0.006,
            "q95": 0.011,
        },
        "swift_mean_under_null": {
            "mean": 0.001,
            "std": 0.0003,
            "median": 0.001,
        },
        "rep_details": [
            {
                "rep_idx": i,
                "seed": 42 + i,
                "model_auc": 0.77,
                "swift_max": 0.005 + i * 0.001,
                "swift_mean": 0.001,
                "rejected_at": {"0.01": False, "0.05": False, "0.1": False},
                "n_drifted_at": {"0.01": 0, "0.05": 0, "0.1": 0},
            }
            for i in range(20)
        ],
    }


@pytest.fixture
def multi_seed_data() -> dict[str, dict[str, Any]]:
    """Minimal multi-seed summary JSON matching real structure."""
    def _make_stat(values: list[float]) -> dict[str, Any]:
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "values": values,
            "ci_95_lower": float(np.mean(arr) - 1.96 * np.std(arr)),
            "ci_95_upper": float(np.mean(arr) + 1.96 * np.std(arr)),
        }

    return {
        "taiwan_credit": {
            "dataset": "taiwan_credit",
            "n_seeds": 3,
            "base_seed": 42,
            "scenarios": ["S1", "S2", "S9"],
            "config": {
                "n_permutations": 200,
                "alpha": 0.05,
                "ref_fraction": 0.6,
                "n_features_to_drift": 3,
                "max_samples": 5000,
            },
            "model_auc": _make_stat([0.755, 0.777, 0.791]),
            "scenario_summaries": {
                "S1_mag=2.00": {
                    "scenario": "S1",
                    "magnitude": 2.0,
                    "n_seeds": 3,
                    "swift_max": _make_stat([1.1, 1.2, 1.05]),
                    "swift_mean": _make_stat([0.05, 0.06, 0.04]),
                    "n_drifted": _make_stat([3.0, 3.0, 3.0]),
                    "n_features": _make_stat([23.0, 23.0, 23.0]),
                    "psi_max": _make_stat([15.0, 20.0, 12.0]),
                    "ks_max": _make_stat([0.9, 0.85, 0.88]),
                    "ssi_max": _make_stat([1.0, 1.2, 0.9]),
                    "decker_max": _make_stat([0.5, 0.6, 0.45]),
                    "bbsd_ks": _make_stat([0.12, 0.15, 0.11]),
                },
                "S2_mag=2.00": {
                    "scenario": "S2",
                    "magnitude": 2.0,
                    "n_seeds": 3,
                    "swift_max": _make_stat([0.01, 0.008, 0.012]),
                    "swift_mean": _make_stat([0.001, 0.0008, 0.0012]),
                    "n_drifted": _make_stat([0.0, 0.0, 0.0]),
                    "n_features": _make_stat([23.0, 23.0, 23.0]),
                    "psi_max": _make_stat([15.0, 18.0, 14.0]),
                    "ks_max": _make_stat([0.9, 0.87, 0.85]),
                    "ssi_max": _make_stat([1.0, 1.1, 0.95]),
                    "decker_max": _make_stat([0.7, 0.65, 0.72]),
                    "bbsd_ks": _make_stat([0.13, 0.14, 0.12]),
                },
                "S9_mag=0.00": {
                    "scenario": "S9",
                    "magnitude": 0.0,
                    "n_seeds": 3,
                    "swift_max": _make_stat([0.005, 0.006, 0.004]),
                    "swift_mean": _make_stat([0.001, 0.001, 0.001]),
                    "n_drifted": _make_stat([0.0, 0.0, 0.0]),
                    "n_features": _make_stat([23.0, 23.0, 23.0]),
                    "psi_max": _make_stat([0.05, 0.06, 0.04]),
                    "ks_max": _make_stat([0.07, 0.08, 0.06]),
                    "ssi_max": _make_stat([0.01, 0.012, 0.008]),
                    "decker_max": _make_stat([0.09, 0.1, 0.08]),
                    "bbsd_ks": _make_stat([0.05, 0.06, 0.04]),
                },
            },
        }
    }


@pytest.fixture
def power_data() -> dict[str, Any]:
    """Minimal power analysis JSON matching real structure."""
    return {
        "dataset": "taiwan_credit",
        "config": {
            "sample_sizes": [500, 1000, 5000],
            "magnitudes": [0.0, 0.5, 1.0, 2.0],
            "n_reps": 10,
            "n_permutations": 200,
            "alpha": 0.05,
        },
        "power_curves": {
            "n=500_mag=0.0": {
                "sample_size": 500,
                "magnitude": 0.0,
                "n_reps": 10,
                "swift_detection_rate": 0.0,
                "swift_ci_lower": 0.0,
                "swift_ci_upper": 0.277,
                "swift_max_scores": {"mean": 0.04, "std": 0.015, "median": 0.04},
                "swift_mean_scores": {"mean": 0.01, "std": 0.002},
                "psi_max_scores": {"mean": 0.07, "std": 0.014, "median": 0.07},
                "ks_max_scores": {"mean": 0.08, "std": 0.009, "median": 0.08},
                "decker_max_scores": {"mean": 0.10, "std": 0.013, "median": 0.10},
                "model_auc": {"mean": 0.754, "std": 0.046},
            },
            "n=500_mag=0.5": {
                "sample_size": 500,
                "magnitude": 0.5,
                "n_reps": 10,
                "swift_detection_rate": 0.9,
                "swift_ci_lower": 0.596,
                "swift_ci_upper": 0.982,
                "swift_max_scores": {"mean": 0.49, "std": 0.118, "median": 0.52},
                "swift_mean_scores": {"mean": 0.05, "std": 0.01},
                "psi_max_scores": {"mean": 10.0, "std": 2.93, "median": 11.0},
                "ks_max_scores": {"mean": 0.65, "std": 0.12, "median": 0.68},
                "decker_max_scores": {"mean": 0.53, "std": 0.08, "median": 0.55},
                "model_auc": {"mean": 0.76, "std": 0.03},
            },
            "n=500_mag=1.0": {
                "sample_size": 500,
                "magnitude": 1.0,
                "n_reps": 10,
                "swift_detection_rate": 1.0,
                "swift_ci_lower": 0.722,
                "swift_ci_upper": 1.0,
                "swift_max_scores": {"mean": 0.95, "std": 0.1, "median": 0.97},
                "swift_mean_scores": {"mean": 0.08, "std": 0.01},
                "psi_max_scores": {"mean": 30.0, "std": 5.0, "median": 31.0},
                "ks_max_scores": {"mean": 0.85, "std": 0.05, "median": 0.86},
                "decker_max_scores": {"mean": 0.7, "std": 0.06, "median": 0.71},
                "model_auc": {"mean": 0.75, "std": 0.04},
            },
            "n=500_mag=2.0": {
                "sample_size": 500,
                "magnitude": 2.0,
                "n_reps": 10,
                "swift_detection_rate": 1.0,
                "swift_ci_lower": 0.722,
                "swift_ci_upper": 1.0,
                "swift_max_scores": {"mean": 1.5, "std": 0.15, "median": 1.52},
                "swift_mean_scores": {"mean": 0.12, "std": 0.02},
                "psi_max_scores": {"mean": 80.0, "std": 10.0, "median": 82.0},
                "ks_max_scores": {"mean": 0.95, "std": 0.03, "median": 0.96},
                "decker_max_scores": {"mean": 0.85, "std": 0.05, "median": 0.86},
                "model_auc": {"mean": 0.74, "std": 0.05},
            },
            "n=1000_mag=0.0": {
                "sample_size": 1000,
                "magnitude": 0.0,
                "n_reps": 10,
                "swift_detection_rate": 0.0,
                "swift_ci_lower": 0.0,
                "swift_ci_upper": 0.277,
                "swift_max_scores": {"mean": 0.03, "std": 0.01, "median": 0.03},
                "swift_mean_scores": {"mean": 0.008, "std": 0.001},
                "psi_max_scores": {"mean": 0.05, "std": 0.01, "median": 0.05},
                "ks_max_scores": {"mean": 0.06, "std": 0.008, "median": 0.06},
                "decker_max_scores": {"mean": 0.08, "std": 0.01, "median": 0.08},
                "model_auc": {"mean": 0.76, "std": 0.03},
            },
            "n=1000_mag=0.5": {
                "sample_size": 1000,
                "magnitude": 0.5,
                "n_reps": 10,
                "swift_detection_rate": 1.0,
                "swift_ci_lower": 0.722,
                "swift_ci_upper": 1.0,
                "swift_max_scores": {"mean": 0.55, "std": 0.1, "median": 0.56},
                "swift_mean_scores": {"mean": 0.04, "std": 0.008},
                "psi_max_scores": {"mean": 12.0, "std": 2.0, "median": 12.5},
                "ks_max_scores": {"mean": 0.7, "std": 0.1, "median": 0.71},
                "decker_max_scores": {"mean": 0.6, "std": 0.07, "median": 0.61},
                "model_auc": {"mean": 0.77, "std": 0.02},
            },
            "n=1000_mag=1.0": {
                "sample_size": 1000,
                "magnitude": 1.0,
                "n_reps": 10,
                "swift_detection_rate": 1.0,
                "swift_ci_lower": 0.722,
                "swift_ci_upper": 1.0,
                "swift_max_scores": {"mean": 1.0, "std": 0.08, "median": 1.01},
                "swift_mean_scores": {"mean": 0.09, "std": 0.01},
                "psi_max_scores": {"mean": 35.0, "std": 4.0, "median": 35.5},
                "ks_max_scores": {"mean": 0.88, "std": 0.04, "median": 0.89},
                "decker_max_scores": {"mean": 0.75, "std": 0.05, "median": 0.76},
                "model_auc": {"mean": 0.76, "std": 0.03},
            },
            "n=1000_mag=2.0": {
                "sample_size": 1000,
                "magnitude": 2.0,
                "n_reps": 10,
                "swift_detection_rate": 1.0,
                "swift_ci_lower": 0.722,
                "swift_ci_upper": 1.0,
                "swift_max_scores": {"mean": 1.6, "std": 0.12, "median": 1.61},
                "swift_mean_scores": {"mean": 0.13, "std": 0.02},
                "psi_max_scores": {"mean": 90.0, "std": 12.0, "median": 91.0},
                "ks_max_scores": {"mean": 0.96, "std": 0.02, "median": 0.96},
                "decker_max_scores": {"mean": 0.88, "std": 0.04, "median": 0.88},
                "model_auc": {"mean": 0.75, "std": 0.04},
            },
            "n=5000_mag=0.0": {
                "sample_size": 5000,
                "magnitude": 0.0,
                "n_reps": 10,
                "swift_detection_rate": 0.0,
                "swift_ci_lower": 0.0,
                "swift_ci_upper": 0.277,
                "swift_max_scores": {"mean": 0.015, "std": 0.005, "median": 0.015},
                "swift_mean_scores": {"mean": 0.004, "std": 0.001},
                "psi_max_scores": {"mean": 0.02, "std": 0.005, "median": 0.02},
                "ks_max_scores": {"mean": 0.03, "std": 0.005, "median": 0.03},
                "decker_max_scores": {"mean": 0.04, "std": 0.008, "median": 0.04},
                "model_auc": {"mean": 0.77, "std": 0.02},
            },
            "n=5000_mag=0.5": {
                "sample_size": 5000,
                "magnitude": 0.5,
                "n_reps": 10,
                "swift_detection_rate": 1.0,
                "swift_ci_lower": 0.722,
                "swift_ci_upper": 1.0,
                "swift_max_scores": {"mean": 0.6, "std": 0.08, "median": 0.61},
                "swift_mean_scores": {"mean": 0.035, "std": 0.005},
                "psi_max_scores": {"mean": 14.0, "std": 1.5, "median": 14.2},
                "ks_max_scores": {"mean": 0.75, "std": 0.06, "median": 0.76},
                "decker_max_scores": {"mean": 0.65, "std": 0.05, "median": 0.66},
                "model_auc": {"mean": 0.78, "std": 0.01},
            },
            "n=5000_mag=1.0": {
                "sample_size": 5000,
                "magnitude": 1.0,
                "n_reps": 10,
                "swift_detection_rate": 1.0,
                "swift_ci_lower": 0.722,
                "swift_ci_upper": 1.0,
                "swift_max_scores": {"mean": 1.1, "std": 0.06, "median": 1.11},
                "swift_mean_scores": {"mean": 0.1, "std": 0.008},
                "psi_max_scores": {"mean": 40.0, "std": 3.0, "median": 40.5},
                "ks_max_scores": {"mean": 0.9, "std": 0.03, "median": 0.91},
                "decker_max_scores": {"mean": 0.8, "std": 0.04, "median": 0.81},
                "model_auc": {"mean": 0.77, "std": 0.02},
            },
            "n=5000_mag=2.0": {
                "sample_size": 5000,
                "magnitude": 2.0,
                "n_reps": 10,
                "swift_detection_rate": 1.0,
                "swift_ci_lower": 0.722,
                "swift_ci_upper": 1.0,
                "swift_max_scores": {"mean": 1.7, "std": 0.1, "median": 1.71},
                "swift_mean_scores": {"mean": 0.15, "std": 0.015},
                "psi_max_scores": {"mean": 100.0, "std": 10.0, "median": 101.0},
                "ks_max_scores": {"mean": 0.97, "std": 0.01, "median": 0.97},
                "decker_max_scores": {"mean": 0.9, "std": 0.03, "median": 0.9},
                "model_auc": {"mean": 0.77, "std": 0.02},
            },
        },
        "rep_details": [],
    }


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for test artifacts."""
    d = tmp_path / "output"
    d.mkdir()
    return d


# ── Tests for analyze_common.py loaders ──────────────────────────────────────

class TestLoadCalibration:
    """Tests for load_calibration_results."""

    def test_loads_from_directory(self, tmp_path: Path, calibration_data: dict) -> None:
        """Loader handles directory-style path (calibration_smoketest.json/)."""
        cal_dir = tmp_path / "calibration_results.json"
        cal_dir.mkdir()
        cal_file = cal_dir / "calibration_taiwan_credit.json"
        cal_file.write_text(json.dumps(calibration_data))

        result = load_calibration_results(cal_dir)
        assert "taiwan_credit" in result
        assert result["taiwan_credit"]["n_reps"] == 20
        assert "calibration" in result["taiwan_credit"]

    def test_loads_from_single_file(self, tmp_path: Path, calibration_data: dict) -> None:
        """Loader handles single JSON file."""
        cal_file = tmp_path / "calibration_taiwan_credit.json"
        cal_file.write_text(json.dumps(calibration_data))

        result = load_calibration_results(cal_file)
        assert "taiwan_credit" in result

    def test_returns_empty_on_missing(self, tmp_path: Path) -> None:
        """Returns empty dict for nonexistent path."""
        result = load_calibration_results(tmp_path / "nonexistent")
        assert result == {}


class TestLoadMultiSeed:
    """Tests for load_multi_seed_results."""

    def test_loads_summary(self, tmp_path: Path, multi_seed_data: dict) -> None:
        """Loader reads multi_seed_summary.json from directory."""
        ms_dir = tmp_path / "multi_seed_results.json"
        ms_dir.mkdir()
        summary_file = ms_dir / "multi_seed_summary.json"
        summary_file.write_text(json.dumps(multi_seed_data))

        result = load_multi_seed_results(ms_dir)
        assert "taiwan_credit" in result
        assert result["taiwan_credit"]["n_seeds"] == 3

    def test_returns_empty_on_missing(self, tmp_path: Path) -> None:
        result = load_multi_seed_results(tmp_path / "nonexistent")
        assert result == {}


class TestLoadPower:
    """Tests for load_power_results."""

    def test_loads_single_file(self, tmp_path: Path, power_data: dict) -> None:
        """Loader reads power JSON file."""
        pf = tmp_path / "power_smoketest.json"
        pf.write_text(json.dumps(power_data))

        result = load_power_results(pf)
        assert result["dataset"] == "taiwan_credit"
        assert "power_curves" in result

    def test_returns_none_on_missing(self, tmp_path: Path) -> None:
        result = load_power_results(tmp_path / "nonexistent.json")
        assert result is None


# ── Tests for table_calibration ──────────────────────────────────────────────

class TestTableCalibration:
    """Tests for table_calibration() LaTeX output."""

    def test_generates_valid_latex(
        self, calibration_data: dict, output_dir: Path
    ) -> None:
        """Output is valid LaTeX with begin/end table."""
        tex = table_calibration({"taiwan_credit": calibration_data}, output_dir)
        assert r"\begin{table}" in tex
        assert r"\end{table}" in tex
        assert r"\label{tab:calibration}" in tex

    def test_contains_alpha_levels(
        self, calibration_data: dict, output_dir: Path
    ) -> None:
        """Table includes all three alpha levels."""
        tex = table_calibration({"taiwan_credit": calibration_data}, output_dir)
        assert "0.01" in tex
        assert "0.05" in tex
        assert "0.10" in tex or "0.1" in tex

    def test_contains_fpr_values(
        self, calibration_data: dict, output_dir: Path
    ) -> None:
        """Table shows empirical FPR for each alpha."""
        tex = table_calibration({"taiwan_credit": calibration_data}, output_dir)
        assert "0.000" in tex  # FPR at alpha=0.01 is 0.0

    def test_contains_ci(
        self, calibration_data: dict, output_dir: Path
    ) -> None:
        """Table includes 95% CI bounds."""
        tex = table_calibration({"taiwan_credit": calibration_data}, output_dir)
        # Should contain CI notation
        assert "CI" in tex or "ci" in tex.lower() or "[" in tex

    def test_saves_file(
        self, calibration_data: dict, output_dir: Path
    ) -> None:
        """Table is saved as .tex file."""
        table_calibration({"taiwan_credit": calibration_data}, output_dir)
        assert (output_dir / "table_calibration.tex").exists()

    def test_multi_dataset(
        self, calibration_data: dict, output_dir: Path
    ) -> None:
        """Works with multiple datasets."""
        multi = {
            "taiwan_credit": calibration_data,
            "bank_marketing": {**calibration_data, "dataset": "bank_marketing"},
        }
        tex = table_calibration(multi, output_dir)
        assert "Taiwan" in tex or "taiwan" in tex.lower()


# ── Tests for table_multi_seed_stability ─────────────────────────────────────

class TestTableMultiSeedStability:
    """Tests for table_multi_seed_stability() LaTeX output."""

    def test_generates_valid_latex(
        self, multi_seed_data: dict, output_dir: Path
    ) -> None:
        tex = table_multi_seed_stability(multi_seed_data, output_dir)
        assert r"\begin{table" in tex
        assert r"\end{table" in tex
        assert r"\label{tab:multi_seed}" in tex

    def test_contains_scenarios(
        self, multi_seed_data: dict, output_dir: Path
    ) -> None:
        """Table includes scenario labels."""
        tex = table_multi_seed_stability(multi_seed_data, output_dir)
        assert "S1" in tex
        assert "S2" in tex

    def test_contains_pm_notation(
        self, multi_seed_data: dict, output_dir: Path
    ) -> None:
        """Table uses ± notation for mean ± std."""
        tex = table_multi_seed_stability(multi_seed_data, output_dir)
        assert r"\pm" in tex

    def test_contains_methods(
        self, multi_seed_data: dict, output_dir: Path
    ) -> None:
        """Table includes SWIFT and baseline columns."""
        tex = table_multi_seed_stability(multi_seed_data, output_dir)
        assert "SWIFT" in tex
        assert "PSI" in tex or "Decker" in tex

    def test_saves_file(
        self, multi_seed_data: dict, output_dir: Path
    ) -> None:
        table_multi_seed_stability(multi_seed_data, output_dir)
        assert (output_dir / "table_multi_seed.tex").exists()


# ── Tests for figure_power_curve ─────────────────────────────────────────────

class TestFigurePowerCurve:
    """Tests for figure_power_curve()."""

    def test_generates_pdf(
        self, power_data: dict, output_dir: Path
    ) -> None:
        """Saves a PDF file."""
        figure_power_curve(power_data, output_dir)
        pdfs = list(output_dir.glob("power_curve*.pdf"))
        assert len(pdfs) >= 1

    def test_generates_png(
        self, power_data: dict, output_dir: Path
    ) -> None:
        """Also saves PNG."""
        figure_power_curve(power_data, output_dir)
        pngs = list(output_dir.glob("power_curve*.png"))
        assert len(pngs) >= 1


# ── Tests for figure_calibration_qq ──────────────────────────────────────────

class TestFigureCalibrationQQ:
    """Tests for figure_calibration_qq()."""

    def test_generates_pdf(
        self, calibration_data: dict, output_dir: Path
    ) -> None:
        figure_calibration_qq({"taiwan_credit": calibration_data}, output_dir)
        pdfs = list(output_dir.glob("calibration_qq*.pdf"))
        assert len(pdfs) >= 1

    def test_generates_png(
        self, calibration_data: dict, output_dir: Path
    ) -> None:
        figure_calibration_qq({"taiwan_credit": calibration_data}, output_dir)
        pngs = list(output_dir.glob("calibration_qq*.png"))
        assert len(pngs) >= 1


# ── Feature Localization Fixtures ────────────────────────────────────────────

@pytest.fixture
def controlled_data_with_features() -> dict[str, dict[str, Any]]:
    """Controlled experiment results with per-feature scores for localization.

    Mimics taiwan_credit_controlled_smoketest.json structure with S1 at
    multiple magnitudes and per-feature swift_scores + baseline_scores.
    """
    features = [
        "PAY_0", "LIMIT_BAL", "PAY_2", "SEX", "EDUCATION",
        "MARRIAGE", "AGE", "PAY_3", "PAY_4", "PAY_5",
    ]
    drifted = ["PAY_0", "LIMIT_BAL", "PAY_2"]

    def _make_scenario(
        scenario: str,
        magnitude: float,
        swift_drifted_val: float,
        swift_nondrifted_val: float,
        decker_drifted_val: float,
        decker_nondrifted_val: float,
        psi_drifted_val: float,
        psi_nondrifted_val: float,
    ) -> dict[str, Any]:
        swift_scores = {}
        decker_scores = {}
        psi_scores = {}
        ks_scores = {}
        ssi_scores = {}
        mmd_scores = {}
        raw_w1_scores = {}
        psi_mb_scores = {}

        for f in features:
            is_drifted = f in drifted
            swift_scores[f] = swift_drifted_val if is_drifted else swift_nondrifted_val
            decker_scores[f] = decker_drifted_val if is_drifted else decker_nondrifted_val
            psi_scores[f] = psi_drifted_val if is_drifted else psi_nondrifted_val
            ks_scores[f] = (0.9 if is_drifted else 0.05) * magnitude
            ssi_scores[f] = (1.5 if is_drifted else 0.01) * magnitude
            mmd_scores[f] = (1.0 if is_drifted else 0.0) * magnitude
            raw_w1_scores[f] = (5000.0 if is_drifted else 50.0) * magnitude
            psi_mb_scores[f] = (15.0 if is_drifted else 0.01) * magnitude

        swift_max = max(swift_scores.values())
        swift_mean = float(np.mean(list(swift_scores.values())))

        return {
            "scenario": scenario,
            "magnitude": magnitude,
            "swift_scores": swift_scores,
            "swift_pvalues": {f: 0.0 for f in features},
            "swift_drifted": [f for f in features if f in drifted] if magnitude > 0 else [],
            "swift_max": swift_max,
            "swift_mean": swift_mean,
            "baseline_scores": {
                "PSI": psi_scores,
                "KS": ks_scores,
                "Raw_W1": raw_w1_scores,
                "MMD": mmd_scores,
                "SSI": ssi_scores,
                "PSI_model_buckets": psi_mb_scores,
                "BBSD_KS": {"_model_output": 0.5},
                "BBSD_PSI": {"_model_output": 3.0},
                "Decker_KS": decker_scores,
            },
            "drifted_features": drifted if scenario != "S9" else [],
            "description": f"{scenario} at magnitude {magnitude}",
        }

    scenario_results = [
        # S1 at multiple magnitudes
        _make_scenario("S1", 0.5, 0.15, 0.001, 0.40, 0.15, 5.0, 0.01),
        _make_scenario("S1", 1.0, 0.45, 0.001, 0.55, 0.18, 10.0, 0.01),
        _make_scenario("S1", 2.0, 0.90, 0.001, 0.70, 0.20, 16.0, 0.01),
        _make_scenario("S1", 3.0, 1.46, 0.001, 0.81, 0.22, 20.0, 0.01),
        # S9 (null) for completeness
        _make_scenario("S9", 0.0, 0.002, 0.001, 0.05, 0.04, 0.01, 0.01),
    ]

    return {
        "Taiwan Credit": {
            "n_ref": 18000,
            "n_mon": 12000,
            "n_features": 10,
            "model_auc": 0.777,
            "total_time_seconds": 45.0,
            "fit_time_seconds": 12.0,
            "scenario_results": scenario_results,
        },
    }


# ── Tests for compute_snr ────────────────────────────────────────────────────

class TestComputeSNR:
    """Tests for compute_snr() helper."""

    def test_perfect_separation(self) -> None:
        """Drifted scores high, non-drifted zero -> inf."""
        scores = {"A": 1.0, "B": 0.5, "C": 0.0, "D": 0.0}
        drifted = {"A", "B"}
        result = compute_snr(scores, drifted)
        assert result == float("inf")

    def test_clear_signal(self) -> None:
        """Drifted much higher than non-drifted -> large SNR."""
        scores = {"A": 1.0, "B": 0.5, "C": 0.01, "D": 0.02}
        drifted = {"A", "B"}
        result = compute_snr(scores, drifted)
        assert result == pytest.approx(0.75 / 0.015, rel=1e-6)

    def test_no_separation(self) -> None:
        """All scores equal -> SNR = 1.0."""
        scores = {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5}
        drifted = {"A", "B"}
        result = compute_snr(scores, drifted)
        assert result == pytest.approx(1.0)

    def test_empty_drifted(self) -> None:
        """No drifted features -> SNR = 0.0."""
        scores = {"A": 1.0, "B": 0.5}
        result = compute_snr(scores, set())
        assert result == 0.0

    def test_all_drifted(self) -> None:
        """All features drifted -> inf (no non-drifted denominator)."""
        scores = {"A": 1.0, "B": 0.5}
        drifted = {"A", "B"}
        result = compute_snr(scores, drifted)
        assert result == float("inf")


# ── Tests for precision_at_k ─────────────────────────────────────────────────

class TestPrecisionAtK:
    """Tests for precision_at_k() helper."""

    def test_perfect_precision(self) -> None:
        """Top-k are all truly drifted -> 1.0."""
        scores = {"A": 1.0, "B": 0.8, "C": 0.01, "D": 0.02}
        drifted = {"A", "B"}
        assert precision_at_k(scores, drifted, k=2) == pytest.approx(1.0)

    def test_zero_precision(self) -> None:
        """Top-k are all non-drifted -> 0.0."""
        scores = {"A": 0.01, "B": 0.02, "C": 1.0, "D": 0.8}
        drifted = {"A", "B"}
        assert precision_at_k(scores, drifted, k=2) == pytest.approx(0.0)

    def test_partial_precision(self) -> None:
        """2 out of 3 top-k are drifted -> 2/3."""
        scores = {"A": 1.0, "B": 0.9, "C": 0.95, "D": 0.01}
        drifted = {"A", "B"}
        # Top-3: A(1.0), C(0.95), B(0.9) -> A and B are drifted = 2/3
        assert precision_at_k(scores, drifted, k=3) == pytest.approx(2 / 3)

    def test_k_equals_one(self) -> None:
        """k=1, highest score is drifted -> 1.0."""
        scores = {"A": 1.0, "B": 0.5, "C": 0.9}
        drifted = {"A"}
        assert precision_at_k(scores, drifted, k=1) == pytest.approx(1.0)

    def test_k_larger_than_features(self) -> None:
        """k > number of features -> uses all features."""
        scores = {"A": 1.0, "B": 0.5}
        drifted = {"A"}
        result = precision_at_k(scores, drifted, k=10)
        assert result == pytest.approx(0.5)  # 1 out of 2

    def test_empty_scores(self) -> None:
        """Empty scores dict -> 0.0."""
        result = precision_at_k({}, {"A"}, k=3)
        assert result == 0.0


# ── Tests for figure_feature_localization ────────────────────────────────────

class TestFigureFeatureLocalization:
    """Tests for figure_feature_localization() bar chart."""

    def test_generates_pdf(
        self,
        controlled_data_with_features: dict,
        output_dir: Path,
    ) -> None:
        figure_feature_localization(
            controlled_data_with_features, output_dir,
        )
        pdfs = list(output_dir.glob("feature_localization*.pdf"))
        assert len(pdfs) >= 1

    def test_generates_png(
        self,
        controlled_data_with_features: dict,
        output_dir: Path,
    ) -> None:
        figure_feature_localization(
            controlled_data_with_features, output_dir,
        )
        pngs = list(output_dir.glob("feature_localization*.png"))
        assert len(pngs) >= 1


# ── Tests for figure_feature_localization_by_magnitude ───────────────────────

class TestFigureFeatureLocalizationByMagnitude:
    """Tests for figure_feature_localization_by_magnitude() line plot."""

    def test_generates_pdf(
        self,
        controlled_data_with_features: dict,
        output_dir: Path,
    ) -> None:
        figure_feature_localization_by_magnitude(
            controlled_data_with_features, output_dir,
        )
        pdfs = list(output_dir.glob("feature_localization_by_mag*.pdf"))
        assert len(pdfs) >= 1

    def test_generates_png(
        self,
        controlled_data_with_features: dict,
        output_dir: Path,
    ) -> None:
        figure_feature_localization_by_magnitude(
            controlled_data_with_features, output_dir,
        )
        pngs = list(output_dir.glob("feature_localization_by_mag*.png"))
        assert len(pngs) >= 1


# ── Tests for table_feature_localization ─────────────────────────────────────

class TestTableFeatureLocalization:
    """Tests for table_feature_localization() LaTeX output."""

    def test_generates_valid_latex(
        self,
        controlled_data_with_features: dict,
        output_dir: Path,
    ) -> None:
        tex = table_feature_localization(
            controlled_data_with_features, output_dir,
        )
        assert r"\begin{table" in tex
        assert r"\end{table" in tex
        assert r"\label{tab:feature_localization}" in tex

    def test_contains_methods(
        self,
        controlled_data_with_features: dict,
        output_dir: Path,
    ) -> None:
        tex = table_feature_localization(
            controlled_data_with_features, output_dir,
        )
        assert "SWIFT" in tex
        assert "Decker" in tex
        assert "PSI" in tex

    def test_contains_snr(
        self,
        controlled_data_with_features: dict,
        output_dir: Path,
    ) -> None:
        """Table includes SNR values."""
        tex = table_feature_localization(
            controlled_data_with_features, output_dir,
        )
        # Should contain numeric SNR values (could be inf for perfect sep)
        assert "SNR" in tex or "snr" in tex.lower() or "$\\infty$" in tex

    def test_contains_precision(
        self,
        controlled_data_with_features: dict,
        output_dir: Path,
    ) -> None:
        """Table includes Precision@k values."""
        tex = table_feature_localization(
            controlled_data_with_features, output_dir,
        )
        assert "P@" in tex or "Precision" in tex or "1.000" in tex

    def test_saves_file(
        self,
        controlled_data_with_features: dict,
        output_dir: Path,
    ) -> None:
        table_feature_localization(
            controlled_data_with_features, output_dir,
        )
        assert (output_dir / "table_feature_localization.tex").exists()
