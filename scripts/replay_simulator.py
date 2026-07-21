"""
Phase 7: replays real transactions from fraudTest.csv against the live API
on a timer, to demo "real-time" scoring behavior.

Streams rows with csv.DictReader instead of loading the full 143MB test set
into a DataFrame - the whole point of a replay simulator is to behave like
a feed of individual events, not a batch job, and streaming keeps memory
flat regardless of --limit.

Since fraudTest.csv has ground-truth is_fraud labels, each line prints the
model's prediction next to the actual label - useful for watching the
system work correctly over a live-looking stream, which a bare prediction
number wouldn't show.
"""

import argparse
import csv
import sys
import time
from datetime import datetime

import requests

DEFAULT_SOURCE = "data/raw/fraudTest.csv"
DEFAULT_API_URL = "http://localhost:8000/predict"
REQUEST_FIELDS = [
    "cc_num",
    "amt",
    "category",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "trans_date_trans_time",
]


def to_payload(row: dict) -> dict:
    payload = {k: row[k] for k in REQUEST_FIELDS}
    payload["cc_num"] = int(payload["cc_num"])
    payload["amt"] = float(payload["amt"])
    payload["lat"] = float(payload["lat"])
    payload["long"] = float(payload["long"])
    payload["merch_lat"] = float(payload["merch_lat"])
    payload["merch_long"] = float(payload["merch_long"])
    return payload


def replay(source: str, api_url: str, interval: float, limit: int | None):
    correct, total = 0, 0

    with open(source, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit is not None and total >= limit:
                break

            payload = to_payload(row)
            actual_fraud = row["is_fraud"] == "1"

            try:
                resp = requests.post(api_url, json=payload, timeout=5)
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as e:
                print(f"[ERROR] request failed: {e}", file=sys.stderr)
                time.sleep(interval)
                continue

            total += 1
            predicted_fraud = result["is_fraud"]
            match = "✓" if predicted_fraud == actual_fraud else "✗"
            correct += predicted_fraud == actual_fraud

            ts = datetime.fromisoformat(payload["trans_date_trans_time"])
            print(
                f"[{ts}] cc_num=...{str(payload['cc_num'])[-4:]} "
                f"amt=${payload['amt']:.2f} category={payload['category']:<15} "
                f"-> predicted={'FRAUD' if predicted_fraud else 'legit':<5} "
                f"(score={result['fraud_score']:.3f}) actual={'FRAUD' if actual_fraud else 'legit':<5} {match}"
            )

            time.sleep(interval)

    if total:
        print(f"\nReplayed {total} transactions, {correct}/{total} matched actual label ({100 * correct / total:.1f}%).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="CSV to replay")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Fraud API /predict URL")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between requests")
    parser.add_argument("--limit", type=int, default=50, help="Number of transactions to replay")
    args = parser.parse_args()

    replay(args.source, args.api_url, args.interval, args.limit)
