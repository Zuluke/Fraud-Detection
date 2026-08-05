"""Dataset loading with Kaggle download and synthetic fallback.

Attempts to download the Credit Card Fraud Detection dataset from Kaggle.
If unavailable, generates synthetic data with realistic distributions:
- 284,807 samples
- 30 features (V1–V28 from PCA, Time, Amount)
- 0.17% fraud rate (highly imbalanced)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"
]
TOTAL_SAMPLES = 284_807
FRAUD_RATIO = 0.0017


def load_from_kaggle() -> pd.DataFrame | None:
    """Attempt to download the credit card fraud dataset from Kaggle.

    Requires kagglehub to be installed and Kaggle credentials configured.

    Returns:
        pd.DataFrame | None: The loaded dataset, or None if download fails.
    """
    try:
        import kagglehub  # noqa: F811

        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        csv_path = Path(path) / "creditcard.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.info(
                "Loaded Kaggle dataset",
                extra={"shape": str(df.shape), "source": "kaggle"},
            )
            return df
    except Exception as exc:
        logger.warning(
            "Kaggle download failed, will use synthetic data",
            extra={"error": str(exc)},
        )
    return None


def generate_synthetic_data(
    n_samples: int = TOTAL_SAMPLES,
    fraud_ratio: float = FRAUD_RATIO,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic credit card fraud data with realistic distributions.

    Produces data mimicking the Kaggle Credit Card Fraud dataset:
    - V1–V28: PCA-transformed features (normal distribution, std ~ 1–3)
    - Time: seconds elapsed since first transaction (0–172,800 ≈ 2 days)
    - Amount: transaction amount (log-normal, median ~22, range 0–25,000)
    - Class: binary target (0 = legitimate, 1 = fraud)

    Fraud transactions have shifted feature distributions to create
    learnable patterns for the classifier.

    Args:
        n_samples: Total number of samples to generate.
        fraud_ratio: Fraction of samples that are fraudulent.
        random_state: Seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic dataset with columns matching the Kaggle schema.
    """
    rng = np.random.RandomState(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    logger.info(
        "Generating synthetic data",
        extra={
            "n_samples": n_samples,
            "n_fraud": n_fraud,
            "n_legit": n_legit,
            "fraud_ratio": fraud_ratio,
        },
    )

    # --- Legitimate transactions ---
    legit_features = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    legit_time = rng.uniform(0, 172_800, size=n_legit)
    legit_amount = rng.lognormal(mean=3.0, sigma=1.5, size=n_legit)
    legit_amount = np.clip(legit_amount, 0, 25_000)

    # --- Fraud transactions (shifted distributions for separability) ---
    fraud_features = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, 28))
    # Shift key features to create learnable fraud patterns
    # V1, V3, V4 strongly shifted (typical in real PCA-transformed fraud data)
    fraud_features[:, 0] -= 3.0   # V1: strong negative shift
    fraud_features[:, 2] -= 2.5   # V3: negative shift
    fraud_features[:, 3] += 2.0   # V4: positive shift
    fraud_features[:, 9] -= 2.0   # V10: negative shift
    fraud_features[:, 11] += 1.5  # V12: positive shift
    fraud_features[:, 13] -= 2.0  # V14: strong negative shift
    fraud_features[:, 16] -= 1.5  # V17: negative shift

    fraud_time = rng.uniform(0, 172_800, size=n_fraud)
    fraud_amount = rng.lognormal(mean=4.5, sigma=1.8, size=n_fraud)
    fraud_amount = np.clip(fraud_amount, 0, 25_000)

    # --- Combine ---
    features = np.vstack([legit_features, fraud_features])
    time_col = np.concatenate([legit_time, fraud_time])
    amount_col = np.concatenate([legit_amount, fraud_amount])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    df = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time_col)
    df["Amount"] = amount_col
    df["Class"] = labels.astype(int)

    # Shuffle rows
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(
        "Synthetic data generated",
        extra={
            "shape": str(df.shape),
            "fraud_count": int(labels.sum()),
            "fraud_pct": f"{labels.mean() * 100:.4f}%",
        },
    )

    return df


def load_dataset(cache_dir: Path | None = None) -> pd.DataFrame:
    """Load the credit card fraud dataset, with automatic fallback.

    Strategy:
    1. Try loading from a cached CSV in cache_dir.
    2. Try downloading from Kaggle.
    3. Fall back to synthetic data generation.

    Args:
        cache_dir: Optional directory to look for / save cached CSV.

    Returns:
        pd.DataFrame: Dataset with columns [Time, V1..V28, Amount, Class].
    """
    # 1. Check local cache
    if cache_dir is not None:
        cached_file = cache_dir / "creditcard.csv"
        if cached_file.exists():
            logger.info("Loading from cache", extra={"path": str(cached_file)})
            return pd.read_csv(cached_file)

    # 2. Try Kaggle
    df = load_from_kaggle()
    if df is not None:
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_dir / "creditcard.csv", index=False)
        return df

    # 3. Synthetic fallback
    logger.info("Using synthetic data fallback")
    df = generate_synthetic_data()

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_dir / "creditcard.csv", index=False)

    return df
