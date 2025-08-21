# Centralized Feature Management with Feast

This project shows how to use **Feast** to manage features consistently between **model training (offline)** and **real-time inference (online)**. The goal was to build something small but production-like, so I set up a real feature store pipeline instead of just using Pandas in memory.

---

## Why this project?

A common pain in ML is _train/serve skew_, engineers compute features one way during training, then re-implement them at serving time, leading to mismatches. A feature store solves this by defining features once and letting both training and inference pull from the same definitions.

Here I demonstrate that workflow end-to-end:

- **Postgres** as the offline store (historical features, training sets).
- **Redis** as the online store (low-latency inference).
- **Feast** as the glue managing definitions, registry, and retrieval.

---

## Data

I used the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) dataset (5.5k messages labeled ham/spam).  
Each row was enriched with:

- `message_id` (entity key for Feast)
- `event_timestamp` (needed for time travel and materialization)

Features engineered:

- Text length, word count, average word length
- Digit and exclamation counts
- Uppercase ratio

They’re deliberately simple: easy to compute in Pandas, but realistic enough to showcase feature definitions.

---

## Workflow

1. **Ingest & compute features**

   - Raw messages + labels stored in Postgres
   - Feature table (`sms_features`) created with engineered stats

2. **Define Feast objects**

   - `Entity`: `message_id`
   - `FeatureView`: `sms_base_features` (all engineered features)
   - `FeatureService`: `sms_service` (single contract for train + serve)

3. **Offline training**

   - Used `get_historical_features()` to build a labeled training set
   - Trained Logistic Regression with scikit-learn
   - Saved scaler + model artifacts

4. **Online inference**
   - Materialized features into Redis
   - Retrieved with `get_online_features()`
   - Applied the same scaler + model for predictions

---

## Key insight

Even in a toy dataset, the workflow looks like production:

- **Consistency**: both training and serving use `sms_service`, no duplicated feature code.
- **Reproducibility**: historical retrieval is “as-of” event time, so the model never trains on future data.
- **Speed**: Redis makes features instantly available at inference time.

It’s overkill for SMS spam, but that’s the point — showing the mechanics you’d use for fraud detection, recommendations, or any ML system that needs reliable features in prod.

---

## Run it yourself

```bash
# spin up infra
docker compose up -d

# fetch dataset
poetry run python src/feast_feature_store/data_download.py

# ingest + compute features
poetry run python src/feast_feature_store/ingest_and_features.py

# register Feast objects
poetry run feast -c feast_repo apply

# train
poetry run python src/feast_feature_store/train_from_feast.py

# materialize to Redis
poetry run python src/feast_feature_store/materialize_online.py

# online inference
poetry run python src/feast_feature_store/online_infer.py
```
