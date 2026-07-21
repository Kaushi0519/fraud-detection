"""
Shared test fixtures. API tests run against SQLite in-memory (not a real
Postgres) and a monkeypatched in-process dict standing in for Redis - fast,
no external services required, suitable for CI. This is deliberately NOT a
substitute for the manual integration testing already done against real
Postgres/Redis/Docker in Phases 6-8 (see DECISIONS.md); it verifies API
request/response contracts and business logic in isolation.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api import cache, db, main


@pytest.fixture
def client(monkeypatch):
    fake_redis_store: dict = {}

    def fake_get_user_features(cc_num):
        return fake_redis_store.get(str(cc_num))

    def fake_set_user_features(cc_num, avg_amt, count, last_trans_time):
        fake_redis_store[str(cc_num)] = {
            "avg_amt": avg_amt,
            "count": count,
            "last_trans_time": last_trans_time,
        }

    monkeypatch.setattr(cache, "get_user_features", fake_get_user_features)
    monkeypatch.setattr(cache, "set_user_features", fake_set_user_features)

    # StaticPool keeps a single connection alive for the engine's lifetime -
    # without it, each checkout of a SQLite ":memory:" engine gets its own
    # empty database, so the table created below wouldn't be visible to the
    # session a request actually uses.
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    db.Base.metadata.create_all(test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine)

    def override_get_db():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr(db, "init_db", lambda: None)
    main.app.dependency_overrides[main.get_db] = override_get_db

    with TestClient(main.app) as test_client:
        yield test_client

    main.app.dependency_overrides.clear()
