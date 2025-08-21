import os
import re
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text

PG_USER = os.getenv("POSTGRES_USER", "feast")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "feast")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5050")
PG_DB = os.getenv("POSTGRES_DB", "feast_demo")

CSV_PATH = "data/raw/sms_spam.csv"
RAW_TABLE = "sms_messages"
FEAT_TABLE = "sms_features"

ENGINE_URL = f"postgresql+psycopg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

word_pat = re.compile(r"\b\w+\b", flags=re.UNICODE)
digit_pat = re.compile(r"\d")
excl_pat = re.compile(r"!")


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    s = df["text"].fillna("")

    """
    Here we compute:
    1. Character length
    2. Word tokens
    3. Average word length
    4. Counts of digits and exclamations
    5. Uppercase ratio ( share of A–Z among letters )
    """
    char_len = s.str.len().fillna(0)

    word_counts = s.apply(lambda x: len(word_pat.findall(x)))

    avg_word_len = (char_len / word_counts.replace(0, pd.NA)).fillna(0)

    digit_count = s.apply(lambda x: len(digit_pat.findall(x)))
    excl_count = s.apply(lambda x: len(excl_pat.findall(x)))

    letters_only = s.str.replace(r"[^A-Za-z]", "", regex=True)
    upper_only = s.str.replace(r"[^A-Z]", "", regex=True)
    upper_ratio = (
        upper_only.str.len() / letters_only.str.len().replace(0, pd.NA)
    ).fillna(0)

    feats = pd.DataFrame(
        {
            "message_id": df["message_id"],
            "event_timestamp": pd.to_datetime(df["event_timestamp"], utc=True),
            "f_char_len": char_len.astype("int32"),
            "f_word_count": word_counts.astype("int32"),
            "f_avg_word_len": avg_word_len.astype("float32"),
            "f_digit_count": digit_count.astype("int16"),
            "f_excl_count": excl_count.astype("int16"),
            "f_upper_ratio": upper_ratio.astype("float32"),
        }
    )
    return feats


def main():
    df = pd.read_csv(CSV_PATH)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

    engine = create_engine(ENGINE_URL, future=True)

    with engine.begin() as conn:
        df.to_sql(RAW_TABLE, conn, if_exists="replace", index=False)

        conn.execute(text(f"ALTER TABLE {RAW_TABLE} ADD PRIMARY KEY (message_id);"))
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_{RAW_TABLE}_evt ON {RAW_TABLE}(event_timestamp);"
            )
        )
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_{RAW_TABLE}_label ON {RAW_TABLE}(label);"
            )
        )

        feats = compute_features(df)
        feats.to_sql(FEAT_TABLE, conn, if_exists="replace", index=False)

        conn.execute(text(f"ALTER TABLE {FEAT_TABLE} ADD PRIMARY KEY (message_id);"))
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_{FEAT_TABLE}_evt ON {FEAT_TABLE}(event_timestamp);"
            )
        )

    print(f"Ingested raw -> {RAW_TABLE} ({len(df)} rows)")
    print(f"Wrote features -> {FEAT_TABLE} ({len(feats)} rows)")
    print("Sample features:\n", feats.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
