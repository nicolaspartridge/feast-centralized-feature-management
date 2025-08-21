from feast import Entity, FeatureView, Field, FeatureService
from feast.types import Int32, Float32
from datetime import timedelta
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)

message = Entity(
    name="message",
    join_keys=["message_id"],
    description="Unique SMS message id",
)

sms_features_source = PostgreSQLSource(
    name="sms_features_source",
    table="sms_features",  # public.sms_features
    timestamp_field="event_timestamp",
)

sms_base_features = FeatureView(
    name="sms_base_features",
    entities=[message],
    ttl=timedelta(days=365),
    schema=[
        Field(name="f_char_len", dtype=Int32),
        Field(name="f_word_count", dtype=Int32),
        Field(name="f_avg_word_len", dtype=Float32),
        Field(name="f_digit_count", dtype=Int32),
        Field(name="f_excl_count", dtype=Int32),
        Field(name="f_upper_ratio", dtype=Float32),
    ],
    online=True,
    source=sms_features_source,
    tags={"owner": "ml-demo", "domain": "nlp"},
)

sms_service = FeatureService(
    name="sms_service",
    features=[sms_base_features],
    tags={"purpose": "training-serving-parity"},
)
