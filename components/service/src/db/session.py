from typing import Generator

from sqlmodel import Session, SQLModel, create_engine

_engine = None


def init_db(db_url: str) -> None:
    global _engine
    _engine = create_engine(db_url, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(_engine)


def get_engine():
    if _engine is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _engine


def get_session() -> Generator[Session, None, None]:
    with Session(get_engine()) as session:
        yield session
