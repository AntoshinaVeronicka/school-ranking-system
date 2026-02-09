from __future__ import annotations

from collections.abc import Callable, Generator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.sql import func as sqlfunc

try:
    from .config import DATABASE_URL, DEFAULT_ADMIN_LOGIN, DEFAULT_ADMIN_PASSWORD
except ImportError:
    from config import DATABASE_URL, DEFAULT_ADMIN_LOGIN, DEFAULT_ADMIN_PASSWORD


class Base(DeclarativeBase):
    pass


engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    login = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=sqlfunc.now(), nullable=False)


class ImportJob(Base):
    __tablename__ = "import_job"
    id = Column(Integer, primary_key=True)
    job_type = Column(String(32), nullable=False)  # ege | admissions | events
    filename = Column(String(255), nullable=False)
    status = Column(String(32), nullable=False, default="uploaded")  # uploaded | validated | loaded | error
    details = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=sqlfunc.now(), nullable=False)


def init_db(password_hasher: Callable[[str], str]) -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.login == DEFAULT_ADMIN_LOGIN).one_or_none()
        if admin is None:
            db.add(
                User(
                    login=DEFAULT_ADMIN_LOGIN,
                    password_hash=password_hasher(DEFAULT_ADMIN_PASSWORD),
                    is_admin=True,
                )
            )
            db.commit()
    finally:
        db.close()
