# =============================================================================
# FlowUp Fraud Detection - Makefile
# =============================================================================

.PHONY: install test lint train docker-build docker-run clean help

PYTHON := python
PIP := pip
PYTEST := pytest
RUFF := ruff

# Default target
help: ## Show this help message
	@echo "FlowUp Fraud Detection - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	$(PIP) install -r requirements.txt

test: ## Run tests with coverage report
	$(PYTEST) -v --cov=src --cov-report=term-missing --cov-fail-under=80

test-unit: ## Run only unit tests
	$(PYTEST) tests/unit/ -v

test-integration: ## Run only integration tests
	$(PYTEST) tests/integration/ -v

lint: ## Run linter (ruff)
	$(RUFF) check src/ tests/

lint-fix: ## Run linter with auto-fix
	$(RUFF) check --fix src/ tests/

format: ## Format code with ruff
	$(RUFF) format src/ tests/

train: ## Train the ML model
	$(PYTHON) -m src.models.trainer

docker-build: ## Build Docker image
	docker build -t flowup-fraud-detection:latest .

docker-run: ## Run with Docker Compose
	docker-compose up -d

docker-stop: ## Stop Docker Compose services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f app

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov .coverage .coverage.* 2>/dev/null || true
	rm -rf logs/*.json logs/*.log 2>/dev/null || true

health: ## Check API health
	curl -s http://localhost:8000/health | python -m json.tool

predict-sample: ## Send a sample prediction request
	curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"Time":406.0,"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,"V6":-0.47,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":-0.34,"V25":-0.05,"V26":-0.23,"V27":0.04,"V28":0.01,"Amount":149.62}' \
		| python -m json.tool
