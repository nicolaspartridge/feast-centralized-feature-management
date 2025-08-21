import os, json, random
import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from feast import FeatureStore

FEAST_REPO = "feast_repo"
FEATURE_SERVICE_NAME = "sms_service"

PG_USER = os.getenv("POSTGRES_USER", "feast")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "feast")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5050")
PG_DB = os.getenv("POSTGRES_DB", "feast_demo")
ENGINE_URL = f"postgresql+psycopg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"


def sample_recent_message_ids(k=5):
    eng = create_engine(ENGINE_URL, future=True)
    with eng.connect() as conn:
        df = pd.read_sql(
            text(
                """
            SELECT message_id
            FROM sms_messages
            ORDER BY event_timestamp DESC
            LIMIT 100
            """
            ),
            conn,
        )
    ids = df["message_id"].tolist()
    random.seed(0)
    return random.sample(ids, k=min(k, len(ids)))


def main():
    scaler = joblib.load("artifacts/scaler.joblib")
    model = joblib.load("artifacts/model.joblib")
    with open("artifacts/training_columns.json") as f:
        feature_cols = json.load(f)

    message_ids = sample_recent_message_ids(k=5)
    print("Testing message_ids:", message_ids)

    fs = FeatureStore(repo_path=FEAST_REPO)
    feature_service = fs.get_feature_service(FEATURE_SERVICE_NAME)

    entity_rows = [{"message_id": mid} for mid in message_ids]
    feats = fs.get_online_features(
        features=feature_service, entity_rows=entity_rows
    ).to_dict()

    data = {col: feats[col] for col in feature_cols}
    X_online = pd.DataFrame(data, index=message_ids)

    # ---- FIX: fill missing values ----
    X_online = X_online.fillna(0.0)

    X_scaled = scaler.transform(X_online)
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = pd.DataFrame(
        {"message_id": message_ids, "score": proba, "prediction": pred}
    ).set_index("message_id")

    print("\nOnline predictions (head):")
    print(out.head().to_string())


if __name__ == "__main__":
    main()
