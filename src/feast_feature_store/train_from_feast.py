import json
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

FEAST_REPO = "feast_repo"
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

PG_USER = os.getenv("POSTGRES_USER", "feast")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "feast")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5050")
PG_DB = os.getenv("POSTGRES_DB", "feast_demo")
ENGINE_URL = f"postgresql+psycopg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

FEATURE_SERVICE_NAME = "sms_service"


def fetch_entity_df():
    """Build the entity dataframe with entity key + event_timestamp + label."""
    engine = create_engine(ENGINE_URL, future=True)
    with engine.connect() as conn:
        q = text(
            """
            SELECT message_id,
                   event_timestamp,
                   label
            FROM sms_messages
            ORDER BY event_timestamp
        """
        )
        df = pd.read_sql(q, conn)
    # Found that Feast requires pandas datetime64[ns, UTC]
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    return df


def main():
    entity_df = fetch_entity_df()
    print("Entity DF sample:\n", entity_df.head(3).to_string(index=False))

    fs = FeatureStore(repo_path=FEAST_REPO)
    feature_service = fs.get_feature_service(FEATURE_SERVICE_NAME)

    training_df = fs.get_historical_features(
        entity_df=entity_df, features=feature_service
    ).to_df()

    print("Training DF columns:", training_df.columns.tolist())
    print("Training DF sample:\n", training_df.head(3).to_string(index=False))

    label_col = "label"
    feature_cols = [
        c for c in training_df.columns if c.startswith("f_")
    ]  # I named all features with f_ prefix
    X = training_df[feature_cols].copy()
    y = training_df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    """Simple pipeline: scale then Logistic Regression"""
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    clf.fit(X_train_scaled, y_train)

    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("\n=== Evaluation ===")
    print(classification_report(y_test, y_pred, digits=3))
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        pass

    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    joblib.dump(clf, ARTIFACTS_DIR / "model.joblib")
    with open(ARTIFACTS_DIR / "feature_service.txt", "w") as f:
        f.write(FEATURE_SERVICE_NAME + "\n")
    with open(ARTIFACTS_DIR / "training_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("\nSaved artifacts:")
    print(" - artifacts/scaler.joblib")
    print(" - artifacts/model.joblib")
    print(" - artifacts/feature_service.txt")
    print(" - artifacts/training_columns.json")


if __name__ == "__main__":
    main()
