from __future__ import annotations

from collections.abc import Callable, Generator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
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


class School(Base):
    __tablename__ = "school"
    id = Column(Integer, primary_key=True)
    region = Column(String(128), nullable=False)
    municipality = Column(String(128), nullable=False)
    name = Column(String(256), nullable=False)

    __table_args__ = (
        UniqueConstraint("region", "municipality", "name", name="uq_school"),
    )


class EgeStat(Base):
    __tablename__ = "ege_stat"
    id = Column(Integer, primary_key=True)
    school_id = Column(Integer, ForeignKey("school.id", ondelete="CASCADE"), nullable=False, index=True)
    year = Column(Integer, nullable=False)
    subject = Column(String(64), nullable=False)
    avg_score = Column(Float, nullable=False)
    graduates = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("school_id", "year", "subject", name="uq_ege_school_year_subject"),
    )


class StudyProgram(Base):
    __tablename__ = "study_program"
    id = Column(Integer, primary_key=True)
    institute = Column(String(128), nullable=False)
    name = Column(String(256), nullable=False)


class ProgramRequirement(Base):
    __tablename__ = "program_requirement"
    id = Column(Integer, primary_key=True)
    program_id = Column(Integer, ForeignKey("study_program.id", ondelete="CASCADE"), nullable=False, index=True)
    subject = Column(String(64), nullable=False)
    role = Column(String(16), nullable=False)  # required | optional
    min_score = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint("program_id", "subject", name="uq_program_subject"),
    )


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

        # Seed one demo program for rating section.
        program = db.query(StudyProgram).filter(StudyProgram.name == "Прикладная информатика").one_or_none()
        if program is None:
            program = StudyProgram(institute="ИМКТ", name="Прикладная информатика")
            db.add(program)
            db.flush()
            db.add_all(
                [
                    ProgramRequirement(program_id=program.id, subject="Русский язык", role="required", min_score=40),
                    ProgramRequirement(program_id=program.id, subject="Математика", role="required", min_score=40),
                    ProgramRequirement(program_id=program.id, subject="Информатика", role="required", min_score=40),
                ]
            )
            db.commit()
    finally:
        db.close()
