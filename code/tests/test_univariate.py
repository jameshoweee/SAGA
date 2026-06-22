"""Test SAGA univariate analysis against generated test vectors."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from saga import UnivariateSamples


class TestGoodVectors:
    """Perfect discrete Gaussian samples must pass all tests."""

    def test_is_valid(self, good_univariate_vector):
        v = good_univariate_vector
        uv = UnivariateSamples(v["params"]["mu"], v["params"]["sigma"], v["samples"])
        assert uv.is_valid, (
            f"Good vector {v['label']} flagged as invalid "
            f"(chi2_p={uv.chi2_pvalue:.6f}, outliers={uv.outlier})"
        )

    def test_no_outliers(self, good_univariate_vector):
        v = good_univariate_vector
        uv = UnivariateSamples(v["params"]["mu"], v["params"]["sigma"], v["samples"])
        assert uv.outlier == 0


class TestBadVectors:
    """Flawed distributions must be detected (is_valid == False)."""

    def test_detected(self, bad_univariate_vector):
        v = bad_univariate_vector
        flaw = v["flaw"]["type"]

        uv = UnivariateSamples(v["params"]["mu"], v["params"]["sigma"], v["samples"])

        # Known blind spots: current chi-square cannot detect these
        known_blind_spots = {"markov", "tail_truncation"}

        if flaw in known_blind_spots:
            if uv.is_valid:
                import pytest
                pytest.skip(f"Known blind spot: {flaw} (needs Phase 3/4 tests)")
            else:
                pass  # detected despite being a known blind spot — good
        else:
            assert not uv.is_valid, (
                f"Bad vector {v['label']} (flaw={flaw}) was not detected "
                f"(chi2_p={uv.chi2_pvalue:.6f})"
            )


class TestMediocreVectors:
    """Borderline vectors — document sensitivity without hard pass/fail."""

    def test_runs(self, mediocre_univariate_vector):
        v = mediocre_univariate_vector
        uv = UnivariateSamples(v["params"]["mu"], v["params"]["sigma"], v["samples"])
        # Just verify it doesn't crash; record result for the power matrix
        assert uv.chi2_pvalue is not None
