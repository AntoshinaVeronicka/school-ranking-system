from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from fastapi import FastAPI, Request, Depends, Form, UploadFile, File
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from passlib.context import CryptContext
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey,
    UniqueConstraint, select, func
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.sql import func as sqlfunc
from starlette.middleware.sessions import SessionMiddleware



# Настройки (можно вынести в .env, но для КП достаточно так)
APP_NAME = "DVFU – Профориентации и анализ школ"
SECRET_KEY = "CHANGE_ME"
DATABASE_URL = "sqlite:///./app.db"
UPLOAD_DIR = "./uploads"

DEFAULT_ADMIN_LOGIN = "admin"
DEFAULT_ADMIN_PASSWORD = "admin"

REQUIRED_EGE_COLUMNS = {
    "Муниципальное образование",
    "Образовательная организация (школа)",
    "Год",
    "Предмет",
    "Средний балл",
    "Количество выпускников",
}



# База данных

class Base(DeclarativeBase):
    pass


engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def get_db() -> Iterable[Session]:
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



# Безопасность / сессии
pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd.verify(password, password_hash)


def get_current_user(request: Request, db: Session) -> User | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.get(User, int(user_id))


def require_user(request: Request, db: Session) -> User:
    user = get_current_user(request, db)
    if user is None:
        raise PermissionError("not authenticated")
    return user


def require_admin(request: Request, db: Session) -> User:
    user = require_user(request, db)
    if not user.is_admin:
        raise PermissionError("admin required")
    return user



# Расчёты (демо)
def subjects_for_program(db: Session, program_id: int) -> list[str]:
    rows = db.execute(select(ProgramRequirement.subject).where(ProgramRequirement.program_id == program_id)).all()
    return [r[0] for r in rows]


def calculate_school_metrics(db: Session, school_ids: list[int], year: int, program_id: int | None) -> list[dict[str, Any]]:
    if not school_ids:
        return []

    # В реальной модели выпускники хранятся отдельно; здесь эвристика: max(graduates) по предметам
    subq = (
        select(
            EgeStat.school_id.label("school_id"),
            func.max(EgeStat.graduates).label("graduates"),
            func.avg(EgeStat.avg_score).label("avg_score_all"),
        )
        .where(EgeStat.school_id.in_(school_ids), EgeStat.year == year)
        .group_by(EgeStat.school_id)
        .subquery()
    )

    rows = db.execute(
        select(School, subq.c.graduates, subq.c.avg_score_all).join(subq, School.id == subq.c.school_id)
    ).all()

    req_subjects = subjects_for_program(db, program_id) if program_id else []

    result: list[dict[str, Any]] = []
    for school, grads, avg_score_all in rows:
        match_share = None
        if req_subjects:
            have = db.execute(
                select(func.count(func.distinct(EgeStat.subject)))
                .where(EgeStat.school_id == school.id, EgeStat.year == year, EgeStat.subject.in_(req_subjects))
            ).scalar_one()
            match_share = float(have) / float(len(req_subjects)) if req_subjects else None

        result.append({
            "school_id": school.id,
            "region": school.region,
            "municipality": school.municipality,
            "name": school.name,
            "graduates": int(grads or 0),
            "avg_score_all": float(avg_score_all) if avg_score_all is not None else None,
            "match_share": match_share,
        })

    return result


def rank_schools(metrics: list[dict[str, Any]], w_graduates: float, w_avg_score: float, w_match_share: float) -> list[dict[str, Any]]:
    if not metrics:
        return []

    max_grads = max(m["graduates"] for m in metrics) or 1
    max_score = max((m["avg_score_all"] or 0.0) for m in metrics) or 1.0

    ranked = []
    for m in metrics:
        score = (
            w_graduates * (m["graduates"] / max_grads)
            + w_avg_score * ((m["avg_score_all"] or 0.0) / max_score)
            + w_match_share * (m["match_share"] or 0.0)
        )
        ranked.append({**m, "rating_score": float(score)})

    ranked.sort(key=lambda x: x["rating_score"], reverse=True)
    return ranked


# Импорт ЕГЭ (Excel) – реализовано
def validate_ege_xlsx(path: str) -> tuple[bool, list[str]]:
    df = pd.read_excel(path)
    cols = set(str(c).strip() for c in df.columns)
    missing = sorted(REQUIRED_EGE_COLUMNS - cols)
    return (len(missing) == 0, missing)


def load_ege_xlsx(db: Session, path: str, region: str) -> dict[str, int]:
    df = pd.read_excel(path)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    df = df.dropna(subset=["Образовательная организация (школа)", "Муниципальное образование", "Год", "Предмет"])

    inserted_schools = 0
    inserted_stats = 0
    updated_stats = 0

    for _, row in df.iterrows():
        municipality = str(row["Муниципальное образование"]).strip()
        name = str(row["Образовательная организация (школа)"]).strip()
        year = int(row["Год"])
        subject = str(row["Предмет"]).strip()
        avg_score = float(row["Средний балл"])
        graduates = int(row["Количество выпускников"])

        school = db.query(School).filter_by(region=region, municipality=municipality, name=name).one_or_none()
        if school is None:
            school = School(region=region, municipality=municipality, name=name)
            db.add(school)
            db.flush()
            inserted_schools += 1

        stat = db.query(EgeStat).filter_by(school_id=school.id, year=year, subject=subject).one_or_none()
        if stat is None:
            stat = EgeStat(school_id=school.id, year=year, subject=subject, avg_score=avg_score, graduates=graduates)
            db.add(stat)
            inserted_stats += 1
        else:
            stat.avg_score = avg_score
            stat.graduates = graduates
            updated_stats += 1

    db.commit()
    return {"inserted_schools": inserted_schools, "inserted_stats": inserted_stats, "updated_stats": updated_stats}


# Шаблоны (Jinja2)
templates_env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html", "xml"]),
)


def render(template_name: str, context: dict[str, Any]) -> HTMLResponse:
    tpl = templates_env.get_template(template_name)
    html = tpl.render(**context)
    return HTMLResponse(html)


# Приложение
app = FastAPI(title=APP_NAME, default_response_class=HTMLResponse)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

app.mount("/static", StaticFiles(directory="static"), name="static")


def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # admin
        admin = db.query(User).filter(User.login == DEFAULT_ADMIN_LOGIN).one_or_none()
        if admin is None:
            db.add(User(
                login=DEFAULT_ADMIN_LOGIN,
                password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
                is_admin=True
            ))
            db.commit()

        # seed one program (для демонстрации рейтинга)
        program = db.query(StudyProgram).filter(StudyProgram.name == "Прикладная информатика").one_or_none()
        if program is None:
            program = StudyProgram(institute="ИМКТ", name="Прикладная информатика")
            db.add(program)
            db.flush()
            db.add_all([
                ProgramRequirement(program_id=program.id, subject="Русский язык", role="required", min_score=40),
                ProgramRequirement(program_id=program.id, subject="Математика", role="required", min_score=40),
                ProgramRequirement(program_id=program.id, subject="Информатика", role="required", min_score=40),
            ])
            db.commit()
    finally:
        db.close()


init_db()


# 0 Вход / 0.1 Восстановление / 1 Меню / 1.1 Выход
@app.get("/")
def main_menu(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return render("main.html", {"request": request, "user": user})


@app.get("/login")
def login_page(request: Request):
    return render("login.html", {"request": request, "user": None, "error": None})


@app.post("/login")
def login_action(
    request: Request,
    login: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.login == login).one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        return render("login.html", {"request": request, "user": None, "error": "Неверный логин или пароль."})

    request.session["user_id"] = user.id
    return RedirectResponse("/", status_code=303)


@app.get("/recovery")
def recovery_page(request: Request):
    return render("recovery.html", {"request": request, "user": None, "info": None})


@app.post("/recovery")
def recovery_action(request: Request, login: str = Form(...)):
    info = f"Запрос на восстановление для пользователя «{login}» зарегистрирован (демо-режим)."
    return render("recovery.html", {"request": request, "user": None, "info": info})


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# 2 Данные и загрузки
@app.get("/data")
def data_mgmt(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("data_mgmt.html", {"request": request, "user": user})


@app.get("/data/ege")
def import_ege_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("import_ege.html", {"request": request, "user": user, "error": None, "ok": None})


@app.post("/data/ege")
def import_ege_action(
    request: Request,
    region: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    filepath = str(Path(UPLOAD_DIR) / safe_name)

    content = file.file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    job = ImportJob(job_type="ege", filename=safe_name, status="uploaded", details=json.dumps({"region": region}, ensure_ascii=False))
    db.add(job)
    db.commit()
    db.refresh(job)

    ok, missing = validate_ege_xlsx(filepath)
    if not ok:
        job.status = "error"
        job.details = f"Missing columns: {missing}"
        db.commit()
        return render("import_ege.html", {"request": request, "user": user, "error": f"В файле отсутствуют столбцы: {', '.join(missing)}", "ok": None})

    details = load_ege_xlsx(db, filepath, region=region)
    job.status = "loaded"
    job.details = json.dumps(details, ensure_ascii=False)
    db.commit()

    return render("import_ege.html", {"request": request, "user": user, "error": None, "ok": f"Импорт выполнен. Результат: {details}"})


@app.get("/data/admissions")
def import_admissions_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "2.2 Импорт приёма", "back_url": "/data"})


@app.get("/data/events")
def import_events_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "2.3 Импорт профориентации", "back_url": "/data"})


@app.get("/data/validate")
def validation_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    jobs = db.query(ImportJob).order_by(ImportJob.created_at.desc()).limit(20).all()
    return render("validation.html", {"request": request, "user": user, "jobs": jobs})


@app.get("/data/history")
def history_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    jobs = db.query(ImportJob).order_by(ImportJob.created_at.desc()).all()
    return render("history.html", {"request": request, "user": user, "jobs": jobs})



# 3 Поиск и карточка школы
@app.get("/search")
def search_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    programs = db.execute(select(StudyProgram)).scalars().all()
    return render("search.html", {"request": request, "user": user, "programs": programs, "results": None, "q": "", "year": 2025, "program_id": None})


@app.post("/search")
def search_action(
    request: Request,
    q: str = Form(""),
    year: int = Form(2025),
    program_id: int | None = Form(None),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
    programs = db.execute(select(StudyProgram)).scalars().all()

    stmt = select(School).where(School.name.ilike(f"%{q}%")) if q else select(School)
    schools = db.execute(stmt.limit(100)).scalars().all()

    school_ids = [s.id for s in schools]
    metrics = calculate_school_metrics(db, school_ids, year, program_id)

    return render("search.html", {"request": request, "user": user, "programs": programs, "results": metrics, "q": q, "year": year, "program_id": program_id})


@app.get("/search/calc/{school_id}")
def calc_page(request: Request, school_id: int, year: int = 2025, program_id: int | None = None, db: Session = Depends(get_db)):
    user = require_user(request, db)
    metrics = calculate_school_metrics(db, [school_id], year, program_id)
    item = metrics[0] if metrics else None
    return render("calc.html", {"request": request, "user": user, "item": item, "year": year, "program_id": program_id})


@app.get("/search/school/{school_id}")
def school_card(request: Request, school_id: int, db: Session = Depends(get_db)):
    user = require_user(request, db)
    school = db.get(School, school_id)
    return render("school_card.html", {"request": request, "user": user, "school": school})



# 4 Подбор и рейтинг
@app.get("/rating/profile")
def rating_profile(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    programs = db.execute(select(StudyProgram)).scalars().all()
    return render("rating_profile.html", {"request": request, "user": user, "programs": programs})


@app.post("/rating/filters")
def rating_filters(request: Request, program_id: int = Form(...), db: Session = Depends(get_db)):
    user = require_user(request, db)
    program = db.get(StudyProgram, program_id)
    return render("rating_filters.html", {
        "request": request, "user": user, "program": program,
        "year": 2025, "min_graduates": 10, "w_graduates": 0.4, "w_avg_score": 0.4, "w_match_share": 0.2
    })


@app.post("/rating/calc")
def rating_calc(
    request: Request,
    program_id: int = Form(...),
    year: int = Form(2025),
    min_graduates: int = Form(10),
    w_graduates: float = Form(0.4),
    w_avg_score: float = Form(0.4),
    w_match_share: float = Form(0.2),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
    program = db.get(StudyProgram, program_id)

    school_ids = [r[0] for r in db.execute(select(School.id)).all()]
    metrics = calculate_school_metrics(db, school_ids, year, program_id)

    filtered = [m for m in metrics if m["graduates"] >= min_graduates]
    ranked = rank_schools(filtered, w_graduates, w_avg_score, w_match_share)

    return render("rating_list.html", {"request": request, "user": user, "program": program, "ranked": ranked, "year": year})



# 5 Настройки анализа (заглушки)
@app.get("/settings")
def settings_home(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("settings_home.html", {"request": request, "user": user})


@app.get("/settings/filters")
def settings_filters(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "5.1 Конструктор фильтров", "back_url": "/settings"})


@app.get("/settings/weights")
def settings_weights(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "5.2 Метрики и веса", "back_url": "/settings"})


@app.get("/settings/profiles")
def settings_profiles(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "5.3 Профили расчёта", "back_url": "/settings"})



# 6 Отчётность (частично)
@app.get("/reports/generate")
def report_generate(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "6. Выгрузка / отчёт", "back_url": "/"})


@app.get("/reports/archive")
def report_archive(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    jobs = db.query(ImportJob).order_by(ImportJob.created_at.desc()).limit(50).all()
    return render("report_archive.html", {"request": request, "user": user, "jobs": jobs})


@app.get("/reports/calc-history")
def calc_history(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "6.2 История расчётов", "back_url": "/"})



# 7 Администрирование (только admin)
@app.get("/admin")
def admin_home(request: Request, db: Session = Depends(get_db)):
    user = require_admin(request, db)
    return render("admin_home.html", {"request": request, "user": user})


@app.get("/admin/roles")
def admin_roles(request: Request, db: Session = Depends(get_db)):
    user = require_admin(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "7.1 Роли и права", "back_url": "/admin"})


@app.get("/admin/directories")
def admin_directories(request: Request, db: Session = Depends(get_db)):
    user = require_admin(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "7.2 Справочники", "back_url": "/admin"})


@app.get("/admin/methodologies")
def admin_methodologies(request: Request, db: Session = Depends(get_db)):
    user = require_admin(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "7.3 Методики", "back_url": "/admin"})
