"""
PostgreSQL persistence for scored transactions.

Stores every transaction the API scores, building real transaction history
over time. This is deliberately NOT backfilled with the full 1.3M-row
training dataset - it starts empty and accumulates as the API (and the
Phase 7 replay simulator) score live traffic, which is what a production
system's transaction log actually looks like.
"""

import os
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://fraud:fraud@localhost:5432/fraud_detection"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cc_num: Mapped[int] = mapped_column(BigInteger, index=True)
    amt: Mapped[float] = mapped_column(Float)
    category: Mapped[str] = mapped_column(String)
    trans_date_trans_time: Mapped[datetime] = mapped_column(DateTime)
    fraud_score: Mapped[float] = mapped_column(Float)
    is_fraud: Mapped[bool] = mapped_column(Boolean)
    scored_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


def init_db():
    Base.metadata.create_all(engine)


def save_transaction(
    session: Session,
    cc_num: int,
    amt: float,
    category: str,
    trans_date_trans_time: datetime,
    fraud_score: float,
    is_fraud: bool,
):
    txn = Transaction(
        cc_num=cc_num,
        amt=amt,
        category=category,
        trans_date_trans_time=trans_date_trans_time,
        fraud_score=fraud_score,
        is_fraud=is_fraud,
    )
    session.add(txn)
    session.commit()


def get_live_aggregate(session: Session, cc_num: int) -> dict | None:
    """Aggregate of transactions scored by THIS system for this user (i.e.
    since Postgres started accumulating - it holds no training history).
    Returns sum_amt + count, not avg_amt, so the caller can count-weight
    blend this with the frozen training snapshot rather than one silently
    overriding the other. Returns None if we've never scored a transaction
    for this user (a real cold start, not just a cache miss)."""
    row = session.execute(
        select(
            func.sum(Transaction.amt),
            func.count(Transaction.id),
            func.max(Transaction.trans_date_trans_time),
        ).where(Transaction.cc_num == cc_num)
    ).one()
    sum_amt, count, last_trans_time = row
    if count == 0:
        return None
    return {
        "sum_amt": float(sum_amt),
        "count": count,
        "last_trans_time": last_trans_time.isoformat(),
    }
