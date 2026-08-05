"""Integration tests for the FastAPI endpoints: /health and /predict."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Integration tests for GET /health."""

    def test_health_returns_200(self, test_client: TestClient) -> None:
        """GET /health should return 200 OK."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, test_client: TestClient) -> None:
        """GET /health should return status, version, and timestamp."""
        response = test_client.get("/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"

    def test_health_version_format(self, test_client: TestClient) -> None:
        """Health version should be a valid semver string."""
        response = test_client.get("/health")
        data = response.json()
        # Version should be a non-empty string
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0


class TestPredictEndpoint:
    """Integration tests for POST /predict."""

    def test_predict_valid_returns_200(
        self, test_client: TestClient, valid_transaction: dict[str, float]
    ) -> None:
        """POST /predict with valid data should return 200."""
        response = test_client.post("/predict", json=valid_transaction)
        assert response.status_code == 200

    def test_predict_response_structure(
        self, test_client: TestClient, valid_transaction: dict[str, float]
    ) -> None:
        """POST /predict should return all required response fields."""
        response = test_client.post("/predict", json=valid_transaction)
        data = response.json()

        required_fields = [
            "transaction_id",
            "probability",
            "decision",
            "threshold_used",
            "latency_ms",
            "timestamp",
            "model_version",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_predict_probability_range(
        self, test_client: TestClient, valid_transaction: dict[str, float]
    ) -> None:
        """Probability should be between 0.0 and 1.0."""
        response = test_client.post("/predict", json=valid_transaction)
        data = response.json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_decision_values(
        self, test_client: TestClient, valid_transaction: dict[str, float]
    ) -> None:
        """Decision should be either 'approved' or 'suspected_fraud'."""
        response = test_client.post("/predict", json=valid_transaction)
        data = response.json()
        assert data["decision"] in ("approved", "suspected_fraud")

    def test_predict_transaction_id_is_uuid(
        self, test_client: TestClient, valid_transaction: dict[str, float]
    ) -> None:
        """Transaction ID should be a valid UUID string."""
        import uuid

        response = test_client.post("/predict", json=valid_transaction)
        data = response.json()
        # Should not raise
        uuid.UUID(data["transaction_id"])

    def test_predict_latency_positive(
        self, test_client: TestClient, valid_transaction: dict[str, float]
    ) -> None:
        """Latency should be a positive number."""
        response = test_client.post("/predict", json=valid_transaction)
        data = response.json()
        assert data["latency_ms"] >= 0

    def test_predict_missing_field_returns_422(
        self,
        test_client: TestClient,
        invalid_transaction_missing_field: dict[str, float],
    ) -> None:
        """POST /predict with missing field should return 422."""
        response = test_client.post("/predict", json=invalid_transaction_missing_field)
        assert response.status_code == 422

    def test_predict_out_of_range_returns_422(
        self,
        test_client: TestClient,
        invalid_transaction_out_of_range: dict[str, float],
    ) -> None:
        """POST /predict with out-of-range V1 should return 422."""
        response = test_client.post("/predict", json=invalid_transaction_out_of_range)
        assert response.status_code == 422

    def test_predict_negative_amount_returns_422(
        self,
        test_client: TestClient,
        invalid_transaction_negative_amount: dict[str, float],
    ) -> None:
        """POST /predict with negative Amount should return 422."""
        response = test_client.post(
            "/predict", json=invalid_transaction_negative_amount
        )
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self, test_client: TestClient) -> None:
        """POST /predict with empty body should return 422."""
        response = test_client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_422_has_error_detail(self, test_client: TestClient) -> None:
        """422 response should contain a 'detail' key with error info."""
        response = test_client.post("/predict", json={})
        data = response.json()
        assert "detail" in data
        assert len(data["detail"]) > 0

    def test_predict_string_values_returns_422(self, test_client: TestClient) -> None:
        """POST /predict with string values should return 422."""
        bad_txn = {
            "Time": "not_a_number",
            **{f"V{i}": 0.0 for i in range(1, 29)},
            "Amount": 10.0,
        }
        response = test_client.post("/predict", json=bad_txn)
        assert response.status_code == 422


class TestCorrelationId:
    """Tests for correlation ID middleware."""

    def test_response_has_correlation_id(self, test_client: TestClient) -> None:
        """Responses should include X-Correlation-ID header."""
        response = test_client.get("/health")
        assert "X-Correlation-ID" in response.headers

    def test_custom_correlation_id_echoed(self, test_client: TestClient) -> None:
        """Client-provided correlation ID should be echoed back."""
        custom_id = "test-correlation-12345"
        response = test_client.get(
            "/health", headers={"X-Correlation-ID": custom_id}
        )
        assert response.headers["X-Correlation-ID"] == custom_id
