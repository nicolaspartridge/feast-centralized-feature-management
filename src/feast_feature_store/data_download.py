import io, os, zipfile, uuid, random
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests

RAW_DIR = "data/raw"
OUT_CSV = os.path.join(RAW_DIR, "sms_spam.csv")
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

os.makedirs(RAW_DIR, exist_ok=True)

print("Downloading dataset...")
r = requests.get(URL, timeout=60)
r.raise_for_status()
data = r.content

print("Unzipping and parsing...")
zf = zipfile.ZipFile(io.BytesIO(data))
with zf.open("SMSSpamCollection") as f:
    """
    File format: <label>\t<text>
    """
    rows = [
        line.decode("utf-8", errors="ignore").rstrip("\n").split("\t", 1) for line in f
    ]

df = pd.DataFrame(rows, columns=["label_str", "text"])
df["label"] = (df["label_str"] == "spam").astype(int)

"""
Add a reproducible synthetic message_id and event_timestamp
"""
random.seed(42)
now = datetime.now(timezone.utc)
start = now - timedelta(days=28)
timestamps = []
ids = []
for i in range(len(df)):
    t = start + timedelta(seconds=int((i / len(df)) * 28 * 24 * 3600))
    jitter = timedelta(seconds=random.randint(0, 3600))
    timestamps.append(t + jitter)
    ids.append(str(uuid.uuid4()))
df["message_id"] = ids
df["event_timestamp"] = timestamps

df = df[["message_id", "event_timestamp", "label", "label_str", "text"]]
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV} with {len(df)} rows.")
print(df.head(3).to_string(index=False))
