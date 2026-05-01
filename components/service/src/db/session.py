from typing import Generator

from sqlmodel import Session, SQLModel, create_engine, text

_engine = None


def init_db(db_url: str) -> None:
    global _engine
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    _engine = create_engine(db_url, connect_args=connect_args)
    SQLModel.metadata.create_all(_engine)
    _migrate(_engine, db_url)


def _migrate(engine, db_url: str) -> None:
    """Apply lightweight additive migrations for columns added after initial release."""
    with engine.connect() as conn:
        try:
            if db_url.startswith("sqlite"):
                conn.execute(text("ALTER TABLE job ADD COLUMN algorithm TEXT"))
            else:
                conn.execute(text("ALTER TABLE job ADD COLUMN IF NOT EXISTS algorithm TEXT"))
            conn.commit()
        except Exception:
            pass  # column already exists (SQLite doesn't support IF NOT EXISTS)


def get_engine():
    if _engine is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _engine


def get_session() -> Generator[Session, None, None]:
    with Session(get_engine()) as session:
        yield session
