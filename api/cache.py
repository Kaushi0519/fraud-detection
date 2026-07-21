"""
Redis cache for per-user features (avg_amt, count, last_trans_time).

Strategy:
- What's cached: {"avg_amt": float, "count": int, "last_trans_time": iso
  string} - not the full transaction history, just the derived numbers
  /predict needs. count is carried along so the next update can do a
  proper count-weighted running average instead of re-deriving it.
- TTL: 24h. This is a safety net, not the primary invalidation mechanism -
  see below - so it just bounds how long a stale entry could theoretically
  live if the write-through step below were ever skipped, and keeps Redis
  from accumulating entries for users who never come back.
- Invalidation: write-through, not TTL-driven. Every time /predict scores
  a transaction, the cache is overwritten immediately with the transaction
  just seen (see update_user_features), so readers always get the latest
  state on the next request rather than waiting for an expiry.
- Read path (in model_service.py): Redis -> Postgres aggregate -> frozen
  training snapshot, in that order, so a cache miss degrades gracefully
  instead of failing.
"""

import json
import os

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
USER_FEATURES_TTL_SECONDS = 24 * 60 * 60

_client = redis.from_url(REDIS_URL, decode_responses=True)


def _key(cc_num: int) -> str:
    return f"user_features:{cc_num}"


def get_user_features(cc_num: int) -> dict | None:
    raw = _client.get(_key(cc_num))
    return json.loads(raw) if raw else None


def set_user_features(cc_num: int, avg_amt: float, count: int, last_trans_time: str):
    _client.set(
        _key(cc_num),
        json.dumps({"avg_amt": avg_amt, "count": count, "last_trans_time": last_trans_time}),
        ex=USER_FEATURES_TTL_SECONDS,
    )
