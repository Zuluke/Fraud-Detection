# FlowUp Fraud Detection API

Real-time credit card fraud detection microservice built with **FastAPI**, **XGBoost**, and **scikit-learn**. Designed for production-grade performance with sub-100ms inference latency.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

---

## 🏗️ Architecture

```
Client → FastAPI (Pydantic Validation) → Preprocessing Pipeline → XGBoost Model → Decision
                                              ↓                        ↓
                                        ColumnTransformer         predict_proba()
                                        (Time→cyclic,             threshold → 
                                         V1-28→IQR+Scale,        approved/suspected_fraud
                                         Amount→Scale)
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Framework | FastAPI + Uvicorn | Async HTTP, auto-docs, Pydantic validation |
| ML Model | XGBoost (primary) | Gradient boosted trees for tabular fraud patterns |
| Pipeline | scikit-learn Pipeline | Leak-free preprocessing with ColumnTransformer |
| Imbalance | SMOTE + class_weight | Synthetic minority oversampling + model-level weighting |
| Serialization | joblib | Efficient numpy/sklearn object persistence |
| Logging | python-json-logger | Structured JSON logs with correlation IDs |
| Container | Docker multi-stage | Slim production image with non-root user |

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone and enter project
cd flowup-fraud-detection

# 2. Train the model (generates models/fraud_model.joblib)
pip install -r requirements.txt
python -m src.models.trainer

# 3. Start services
docker-compose up -d

# 4. Verify
curl http://localhost:8000/health
```

### Option 2: Local Development

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model
python -m src.models.trainer

# 4. Run server
uvicorn src.main:app --reload --port 8000
```

---

## 📡 API Reference

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00.123456+00:00"
}
```

### `POST /predict`

Evaluate a transaction for fraud risk.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": -0.47, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": -0.34,
    "V25": -0.05, "V26": -0.23, "V27": 0.04, "V28": 0.01,
    "Amount": 149.62
  }'
```

**Response (200 OK):**
```json
{
  "transaction_id": "a7b3c4d5-e6f7-8901-2345-abcdef012345",
  "probability": 0.0342,
  "decision": "approved",
  "threshold_used": 0.5,
  "latency_ms": 2.15,
  "timestamp": "2025-01-15T10:30:01.456789+00:00",
  "model_version": "1.0.0"
}
```

**Error Response (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "Amount"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

### Input Validation Rules

| Field | Type | Constraint |
|-------|------|-----------|
| `Time` | float | `>= 0` |
| `V1`–`V28` | float | `>= -20`, `<= 20` |
| `Amount` | float | `>= 0` |

---

## 🧪 Testing

```bash
# Run all tests with coverage
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# Lint
make lint
```

**Coverage target:** > 80% (enforced in `pyproject.toml`)

---

## 🐳 Docker

```bash
# Build image
make docker-build

# Run with Docker Compose
make docker-run

# View logs
make docker-logs

# Stop services
make docker-stop
```

**Image details:**
- Base: `python:3.11-slim` (multi-stage build)
- Size: ~350MB
- User: non-root (`appuser`)
- Healthcheck: `GET /health` every 30s

---

## ⚙️ Configuration

All settings via environment variables (`.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/fraud_model.joblib` | Path to trained model artifact |
| `LOG_LEVEL` | `INFO` | Logging level |
| `THRESHOLD` | `0.5` | Fraud decision threshold (0.0–1.0) |
| `REDIS_URL` | `redis://redis:6379/0` | Redis URL for optional caching |
| `APP_PORT` | `8000` | Server port |
| `MODEL_VERSION` | `1.0.0` | Version identifier |

---

## 📊 Model Details

### Training Pipeline

1. **Data Loading**: Kaggle Credit Card Fraud dataset (or synthetic fallback)
2. **Feature Engineering**: Time → cyclic sin/cos encoding
3. **Preprocessing**: IQR outlier capping → StandardScaler (via ColumnTransformer)
4. **Resampling**: SMOTE on preprocessed training data only
5. **Training**: XGBoost vs RandomForest comparison (selected by PR-AUC)
6. **Threshold Optimization**: Youden's J statistic on Precision-Recall curve

### Metrics (on test set)

| Metric | Description |
|--------|-------------|
| **PR-AUC** | Primary metric (imbalanced-aware) |
| **F1-Score** | Harmonic mean of precision and recall |
| **Recall** | Fraction of actual fraud caught |
| **Confusion Matrix** | TP, FP, TN, FN breakdown |

---

## 🔬 Technical Justifications

### Why XGBoost over RandomForest / LightGBM?

| Factor | XGBoost | RandomForest | LightGBM |
|--------|---------|-------------|----------|
| **Tabular data performance** | ✅ State-of-the-art | Good but slower convergence | ✅ Comparable to XGBoost |
| **Imbalanced data** | ✅ `scale_pos_weight` | `class_weight='balanced'` | ✅ `is_unbalance` |
| **Inference speed** | ✅ <1ms per sample | Slower (many trees) | ✅ Very fast |
| **Regularization** | ✅ L1/L2 built-in | Limited (tree depth only) | ✅ L1/L2 built-in |
| **Interpretability** | ✅ Feature importance | ✅ Feature importance | ✅ Feature importance |
| **Production maturity** | ✅ Battle-tested | ✅ Battle-tested | Good, fewer deployments |

**Decision**: XGBoost is selected as the primary model because it consistently achieves higher PR-AUC on tabular fraud data, offers sub-millisecond inference, and has native support for imbalanced classes via `scale_pos_weight`. RandomForest is trained as a comparison baseline.

### Why SMOTE over ADASYN / Tomek Links?

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **SMOTE** | Synthetic minority interpolation | Proven on fraud data, stable | Can create noise in overlapping regions |
| ADASYN | Density-adaptive synthesis | Focuses on hard examples | Unstable on very low minority counts |
| Tomek Links | Removes ambiguous majority samples | Clean boundaries | Loses training data, insufficient alone |
| Random Oversampling | Duplicate minority samples | Simple | Overfitting risk |

**Decision**: SMOTE is used because it generates *new* synthetic fraud samples (not duplicates), is well-tested on credit card fraud benchmarks (Dal Pozzolo et al., 2015), and combined with `class_weight='balanced'` provides dual-layer imbalance handling. Applied *after* preprocessing and *only on training data* to prevent data leakage.

### Why FastAPI over Flask?

| Factor | FastAPI | Flask |
|--------|---------|-------|
| **Performance** | ✅ 2-5x faster (async + Starlette) | Synchronous WSGI |
| **Validation** | ✅ Pydantic built-in, auto-422 errors | Manual validation |
| **Documentation** | ✅ Auto OpenAPI/Swagger | Requires extensions |
| **Type safety** | ✅ Native type hints | No type enforcement |
| **Async support** | ✅ Native async/await | Requires Flask 2.0+ workarounds |

**Decision**: FastAPI provides automatic request validation via Pydantic (critical for 30-field input), built-in OpenAPI docs at `/docs`, and significantly better performance under concurrent load—essential for a real-time fraud detection service processing thousands of transactions per minute.

---

## 📈 Production Monitoring

### Data Drift Detection (PSI)

**Population Stability Index (PSI)** measures distributional shift between training and production data:

```
PSI = Σ (actual% - expected%) × ln(actual% / expected%)
```

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant change | Continue monitoring |
| 0.1–0.2 | Moderate shift | Investigate features |
| ≥ 0.2 | **Significant drift** | **Retrain model** |

**Implementation**: Compute PSI weekly on each feature using `src/utils/metrics.compute_psi()`. Alert via Prometheus/Grafana when any feature's PSI exceeds 0.2.

### Performance Degradation

Monitor model quality using delayed ground truth:

1. **Batch evaluation**: When fraud labels arrive (typically 30-90 days delayed), compute F1, Recall, PR-AUC against production predictions
2. **Alert thresholds**:
   - F1 < 0.7 → WARNING: Model degradation
   - F1 < 0.5 → CRITICAL: Immediate retraining required
   - Recall < 0.8 → WARNING: Missing too many fraud cases

### Evidently AI Integration

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
report.run(reference_data=train_df, current_data=prod_df)
report.save_html("monitoring/drift_report.html")
```

### Prometheus Metrics

Expose via `/metrics` endpoint:
- `fraud_prediction_latency_seconds` (histogram)
- `fraud_prediction_total` (counter, labels: decision)
- `fraud_model_drift_psi` (gauge, per feature)
- `fraud_model_f1_score` (gauge)

---

## 📁 Project Structure

```
flowup-fraud-detection/
├── src/
│   ├── main.py                    # FastAPI app factory + entrypoint
│   ├── api/
│   │   ├── routes.py              # /health and /predict endpoints
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── dependencies.py        # DI + correlation ID middleware
│   ├── core/
│   │   ├── config.py              # Environment-based settings
│   │   └── logging_config.py      # Structured JSON logging
│   ├── models/
│   │   ├── trainer.py             # Training pipeline (XGBoost + RF)
│   │   ├── predictor.py           # Real-time prediction service
│   │   └── pipeline_builder.py    # sklearn Pipeline construction
│   ├── data/
│   │   ├── dataset_loader.py      # Kaggle download + synthetic fallback
│   │   ├── preprocessor.py        # IQR capper + scaler
│   │   └── feature_engineering.py # Cyclic time encoding
│   └── utils/
│       ├── validators.py          # Input validation helpers
│       └── metrics.py             # PR-AUC, Youden's J, PSI
├── tests/
│   ├── unit/                      # Component-level tests
│   └── integration/               # API + pipeline tests
├── models/                        # Trained .joblib artifacts
├── logs/                          # Structured JSON log files
├── notebooks/                     # EDA and training notebooks
├── Dockerfile                     # Multi-stage production build
├── docker-compose.yml             # App + Redis services
├── Makefile                       # Dev workflow commands
└── requirements.txt               # Pinned dependencies
```

---

## 📜 License

MIT License — FlowUp Engineering
