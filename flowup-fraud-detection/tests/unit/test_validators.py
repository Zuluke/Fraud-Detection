"""Unit tests for input validators: ranges, completeness, hashing, sanitization."""

import pytest

from src.utils.validators import (
    REQUIRED_FEATURES,
    hash_input,
    sanitize_input,
    validate_completeness,
    validate_feature_ranges,
)


class TestValidateFeatureRanges:
    """Tests for validate_feature_ranges function."""

    def test_valid_features_no_errors(self, valid_transaction: dict[str, float]) -> None:
        """Valid transaction should produce no validation errors."""
        errors = validate_feature_ranges(valid_transaction)
        assert errors == []

    def test_out_of_range_v1(self) -> None:
        """V1 outside [-20, 20] should produce a range error."""
        features = {
            "Time": 0.0,
            **{f"V{i}": 0.0 for i in range(1, 29)},
            "Amount": 10.0,
        }
        features["V1"] = 25.0  # Out of [-20, 20]
        errors = validate_feature_ranges(features)
        assert any("V1" in e for e in errors)

    def test_negative_time(self) -> None:
        """Negative Time should produce a range error."""
        features = {
            "Time": -10.0,
            **{f"V{i}": 0.0 for i in range(1, 29)},
            "Amount": 10.0,
        }
        errors = validate_feature_ranges(features)
        assert any("Time" in e for e in errors)

    def test_negative_amount(self) -> None:
        """Negative Amount should produce a range error."""
        features = {
            "Time": 0.0,
            **{f"V{i}": 0.0 for i in range(1, 29)},
            "Amount": -5.0,
        }
        errors = validate_feature_ranges(features)
        assert any("Amount" in e for e in errors)

    def test_missing_feature(self) -> None:
        """Missing feature should produce an error."""
        features = {"Time": 0.0, "Amount": 10.0}
        errors = validate_feature_ranges(features)
        assert len(errors) > 0

    def test_non_numeric_value(self) -> None:
        """Non-numeric value should produce a type error."""
        features = {
            "Time": 0.0,
            **{f"V{i}": 0.0 for i in range(1, 29)},
            "Amount": "not_a_number",
        }
        errors = validate_feature_ranges(features)
        assert any("Amount" in e for e in errors)


class TestValidateCompleteness:
    """Tests for validate_completeness function."""

    def test_complete_features(self, valid_transaction: dict[str, float]) -> None:
        """Complete feature set should return empty list."""
        missing = validate_completeness(valid_transaction)
        assert missing == []

    def test_missing_features(self) -> None:
        """Missing features should be identified."""
        features = {"Time": 0.0, "V1": 1.0}
        missing = validate_completeness(features)
        assert len(missing) == len(REQUIRED_FEATURES) - 2
        assert "Amount" in missing
        assert "V2" in missing


class TestHashInput:
    """Tests for hash_input function."""

    def test_deterministic(self, valid_transaction: dict[str, float]) -> None:
        """Same input should always produce the same hash."""
        h1 = hash_input(valid_transaction)
        h2 = hash_input(valid_transaction)
        assert h1 == h2

    def test_different_inputs_different_hashes(self) -> None:
        """Different inputs should produce different hashes."""
        features1 = {"Time": 0.0, "V1": 1.0, "Amount": 10.0}
        features2 = {"Time": 0.0, "V1": 2.0, "Amount": 10.0}
        assert hash_input(features1) != hash_input(features2)

    def test_hash_length(self, valid_transaction: dict[str, float]) -> None:
        """Hash should be 16 characters long (truncated SHA-256)."""
        h = hash_input(valid_transaction)
        assert len(h) == 16

    def test_order_independent(self) -> None:
        """Hash should be independent of dictionary key insertion order."""
        f1 = {"Time": 0.0, "V1": 1.0, "Amount": 10.0}
        f2 = {"Amount": 10.0, "Time": 0.0, "V1": 1.0}
        assert hash_input(f1) == hash_input(f2)


class TestSanitizeInput:
    """Tests for sanitize_input function."""

    def test_valid_input(self, valid_transaction: dict[str, float]) -> None:
        """Valid input should be sanitized without errors."""
        sanitized = sanitize_input(valid_transaction)
        assert len(sanitized) == 30
        assert all(isinstance(v, float) for v in sanitized.values())

    def test_string_numbers_coerced(self) -> None:
        """Numeric strings should be coerced to float."""
        features = {
            "Time": "100.0",
            **{f"V{i}": "0.5" for i in range(1, 29)},
            "Amount": "50",
        }
        sanitized = sanitize_input(features)
        assert sanitized["Time"] == 100.0
        assert sanitized["Amount"] == 50.0

    def test_missing_field_raises(self) -> None:
        """Missing required field should raise ValueError."""
        features = {"Time": 0.0}
        with pytest.raises(ValueError, match="Missing required feature"):
            sanitize_input(features)

    def test_non_numeric_raises(self) -> None:
        """Non-numeric value should raise ValueError."""
        features = {
            "Time": 0.0,
            **{f"V{i}": 0.0 for i in range(1, 29)},
            "Amount": "abc",
        }
        with pytest.raises(ValueError, match="must be numeric"):
            sanitize_input(features)
