from __future__ import annotations

import json
import subprocess
import sys
import traceback
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



# РќР°СЃС‚СЂРѕР№РєРё (РјРѕР¶РЅРѕ РІС‹РЅРµСЃС‚Рё РІ .env, РЅРѕ РґР»СЏ РљРџ РґРѕСЃС‚Р°С‚РѕС‡РЅРѕ С‚Р°Рє)
APP_NAME = "DVFU вЂ“ РџСЂРѕС„РѕСЂРёРµРЅС‚Р°С†РёРё Рё Р°РЅР°Р»РёР· С€РєРѕР»"
SECRET_KEY = "CHANGE_ME"
DATABASE_URL = "sqlite:///./app.db"
UPLOAD_DIR = "./uploads"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOAD_EGE_SCRIPT = PROJECT_ROOT / "load" / "load_ege_to_db.py"
LOAD_SCHOOLS_SCRIPT = PROJECT_ROOT / "load" / "load_schools_to_db.py"
LOAD_PROGRAMS_SCRIPT = PROJECT_ROOT / "load" / "load_programs_requirements.py"

DEFAULT_EGE_SUBJECT_SCORES = [
    {"subject_id": 1, "name": "Русский язык", "min_passing_score": 40},
    {"subject_id": 2, "name": "Математика", "min_passing_score": 40},
    {"subject_id": 3, "name": "Физика", "min_passing_score": 41},
    {"subject_id": 4, "name": "Обществознание", "min_passing_score": 45},
    {"subject_id": 5, "name": "История", "min_passing_score": 40},
    {"subject_id": 6, "name": "ИКТ", "min_passing_score": 46},
    {"subject_id": 7, "name": "Иностранный язык", "min_passing_score": 40},
    {"subject_id": 8, "name": "Литература", "min_passing_score": 40},
    {"subject_id": 9, "name": "Биология", "min_passing_score": 40},
    {"subject_id": 10, "name": "География", "min_passing_score": 40},
    {"subject_id": 11, "name": "Химия", "min_passing_score": 40},
]

DEFAULT_ADMIN_LOGIN = "admin"
DEFAULT_ADMIN_PASSWORD = "admin"

REQUIRED_EGE_COLUMNS = {
    "РњСѓРЅРёС†РёРїР°Р»СЊРЅРѕРµ РѕР±СЂР°Р·РѕРІР°РЅРёРµ",
    "РћР±СЂР°Р·РѕРІР°С‚РµР»СЊРЅР°СЏ РѕСЂРіР°РЅРёР·Р°С†РёСЏ (С€РєРѕР»Р°)",
    "Р“РѕРґ",
    "РџСЂРµРґРјРµС‚",
    "РЎСЂРµРґРЅРёР№ Р±Р°Р»Р»",
    "РљРѕР»РёС‡РµСЃС‚РІРѕ РІС‹РїСѓСЃРєРЅРёРєРѕРІ",
}



# Р‘Р°Р·Р° РґР°РЅРЅС‹С…

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



# Р‘РµР·РѕРїР°СЃРЅРѕСЃС‚СЊ / СЃРµСЃСЃРёРё
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



# Р Р°СЃС‡С‘С‚С‹ (РґРµРјРѕ)
def subjects_for_program(db: Session, program_id: int) -> list[str]:
    rows = db.execute(select(ProgramRequirement.subject).where(ProgramRequirement.program_id == program_id)).all()
    return [r[0] for r in rows]


def calculate_school_metrics(db: Session, school_ids: list[int], year: int, program_id: int | None) -> list[dict[str, Any]]:
    if not school_ids:
        return []

    # Р’ СЂРµР°Р»СЊРЅРѕР№ РјРѕРґРµР»Рё РІС‹РїСѓСЃРєРЅРёРєРё С…СЂР°РЅСЏС‚СЃСЏ РѕС‚РґРµР»СЊРЅРѕ; Р·РґРµСЃСЊ СЌРІСЂРёСЃС‚РёРєР°: max(graduates) РїРѕ РїСЂРµРґРјРµС‚Р°Рј
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


# РРјРїРѕСЂС‚ Р•Р“Р­ (Excel) вЂ“ СЂРµР°Р»РёР·РѕРІР°РЅРѕ
def validate_ege_xlsx(path: str) -> tuple[bool, list[str]]:
    df = pd.read_excel(path)
    cols = set(str(c).strip() for c in df.columns)
    missing = sorted(REQUIRED_EGE_COLUMNS - cols)
    return (len(missing) == 0, missing)


def load_ege_xlsx(db: Session, path: str, region: str) -> dict[str, int]:
    df = pd.read_excel(path)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    df = df.dropna(subset=["РћР±СЂР°Р·РѕРІР°С‚РµР»СЊРЅР°СЏ РѕСЂРіР°РЅРёР·Р°С†РёСЏ (С€РєРѕР»Р°)", "РњСѓРЅРёС†РёРїР°Р»СЊРЅРѕРµ РѕР±СЂР°Р·РѕРІР°РЅРёРµ", "Р“РѕРґ", "РџСЂРµРґРјРµС‚"])

    inserted_schools = 0
    inserted_stats = 0
    updated_stats = 0

    for _, row in df.iterrows():
        municipality = str(row["РњСѓРЅРёС†РёРїР°Р»СЊРЅРѕРµ РѕР±СЂР°Р·РѕРІР°РЅРёРµ"]).strip()
        name = str(row["РћР±СЂР°Р·РѕРІР°С‚РµР»СЊРЅР°СЏ РѕСЂРіР°РЅРёР·Р°С†РёСЏ (С€РєРѕР»Р°)"]).strip()
        year = int(row["Р“РѕРґ"])
        subject = str(row["РџСЂРµРґРјРµС‚"]).strip()
        avg_score = float(row["РЎСЂРµРґРЅРёР№ Р±Р°Р»Р»"])
        graduates = int(row["РљРѕР»РёС‡РµСЃС‚РІРѕ РІС‹РїСѓСЃРєРЅРёРєРѕРІ"])

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


# РЁР°Р±Р»РѕРЅС‹ (Jinja2)
def default_ege_form() -> dict[str, Any]:
    return {"region": "", "kind": "actual", "sheet": "", "year": "", "dry_run": False}


def parse_year_value(raw: str) -> int | None:
    value = raw.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("РџРѕР»Рµ 'Р“РѕРґ' РґРѕР»Р¶РЅРѕ Р±С‹С‚СЊ С‡РёСЃР»РѕРј.") from exc


def resolve_sheet_name(path: Path, requested: str) -> tuple[str, list[str]]:
    with pd.ExcelFile(path, engine="openpyxl") as xls:
        sheets = list(xls.sheet_names)
    if not sheets:
        raise ValueError("Р’ С„Р°Р№Р»Рµ РѕС‚СЃСѓС‚СЃС‚РІСѓСЋС‚ Р»РёСЃС‚С‹ Excel.")

    value = requested.strip()
    if not value:
        return sheets[0], sheets

    if value.isdigit():
        idx = int(value)
        if 1 <= idx <= len(sheets):
            return sheets[idx - 1], sheets
        if idx == 0:
            return sheets[0], sheets

    for sheet in sheets:
        if sheet.casefold() == value.casefold():
            return sheet, sheets

    raise ValueError(f"Р›РёСЃС‚ '{value}' РЅРµ РЅР°Р№РґРµРЅ. Р”РѕСЃС‚СѓРїРЅС‹Рµ Р»РёСЃС‚С‹: {', '.join(sheets)}")


def infer_year_from_sheet_name(sheet_name: str) -> int | None:
    if str(LOAD_EGE_SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(LOAD_EGE_SCRIPT.parent))
    from load_ege_to_db import infer_year_from_sheet

    return infer_year_from_sheet(sheet_name)


def run_ege_loader_script(
    *,
    file_path: Path,
    kind: str,
    sheet: str,
    year: int,
    region: str | None,
    dry_run: bool,
) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(LOAD_EGE_SCRIPT),
        "--file",
        str(file_path.resolve()),
        "--kind",
        kind,
        "--sheet",
        sheet,
        "--year",
        str(year),
    ]
    if region:
        cmd.extend(["--region", region])
    if dry_run:
        cmd.append("--dry-run")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "Р—Р°РіСЂСѓР·РєР° РїСЂРµСЂРІР°РЅР° РїРѕ С‚Р°Р№Рј-Р°СѓС‚Сѓ (РїСЂРµРІС‹С€РµРЅРѕ 300 СЃРµРєСѓРЅРґ)."
    except Exception:
        return False, traceback.format_exc()

    output_parts = [part.strip() for part in [proc.stdout, proc.stderr] if part and part.strip()]
    output_text = "\n\n".join(output_parts) if output_parts else "(РЎРєСЂРёРїС‚ Р·Р°РІРµСЂС€РёР»СЃСЏ Р±РµР· РІС‹РІРѕРґР°.)"
    return proc.returncode == 0, output_text


def run_script_with_output(
    script_path: Path,
    args: list[str],
    timeout_seconds: int = 300,
) -> tuple[bool, str]:
    cmd = [sys.executable, str(script_path), *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"Выполнение прервано по тайм-ауту ({timeout_seconds} секунд)."
    except Exception:
        return False, traceback.format_exc()

    output_parts = [part.strip() for part in [proc.stdout, proc.stderr] if part and part.strip()]
    output_text = "\n\n".join(output_parts) if output_parts else "(Скрипт завершился без вывода.)"
    return proc.returncode == 0, output_text


def default_directories_form() -> dict[str, Any]:
    return {
        "loader_type": "schools",
        "sheet": "",
        "dry_run": False,
        "create_missing_subjects": False,
    }


def default_subject_scores() -> list[dict[str, Any]]:
    return [dict(item) for item in DEFAULT_EGE_SUBJECT_SCORES]


def get_load_db_config() -> dict[str, Any]:
    if str(LOAD_EGE_SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(LOAD_EGE_SCRIPT.parent))
    from load_common import get_db_config

    return get_db_config(search_from=LOAD_EGE_SCRIPT.parent)


def get_subject_scores_from_db() -> list[dict[str, Any]]:
    defaults = default_subject_scores()
    default_by_id = {item["subject_id"]: item for item in defaults}
    default_ids = [item["subject_id"] for item in defaults]
    try:
        import psycopg2

        db_cfg = get_load_db_config()
        with psycopg2.connect(**db_cfg) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT subject_id, name, min_passing_score
                    FROM edu.ege_subject
                    WHERE subject_id = ANY(%s)
                    ORDER BY subject_id
                    """,
                    (default_ids,),
                )
                rows = cur.fetchall()
        if not rows:
            return defaults

        db_by_id: dict[int, dict[str, Any]] = {}
        for subject_id, name, min_score in rows:
            db_by_id[int(subject_id)] = {
                "name": str(name),
                "min_passing_score": min_score,
            }

        result: list[dict[str, Any]] = []
        for subject_id in default_ids:
            default_item = default_by_id[subject_id]
            db_item = db_by_id.get(subject_id)
            if db_item is None:
                result.append(dict(default_item))
                continue

            score_value = db_item["min_passing_score"]
            if score_value is None:
                score_value = default_item["min_passing_score"]

            result.append(
                {
                    "subject_id": subject_id,
                    "name": db_item["name"] or str(default_item["name"]),
                    "min_passing_score": int(score_value) if score_value is not None else 0,
                }
            )
        return result
    except Exception:
        return defaults


def apply_subject_scores_from_form(
    base_rows: list[dict[str, Any]],
    form_values: dict[str, str],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in base_rows]
    for row in rows:
        key = f"score_{row['subject_id']}"
        if key in form_values:
            raw = form_values[key].strip()
            if raw:
                row["min_passing_score"] = raw
    return rows


templates_env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html", "xml"]),
)


def render(template_name: str, context: dict[str, Any]) -> HTMLResponse:
    tpl = templates_env.get_template(template_name)
    html = tpl.render(**context)
    return HTMLResponse(html)


# РџСЂРёР»РѕР¶РµРЅРёРµ
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

        # seed one program (РґР»СЏ РґРµРјРѕРЅСЃС‚СЂР°С†РёРё СЂРµР№С‚РёРЅРіР°)
        program = db.query(StudyProgram).filter(StudyProgram.name == "РџСЂРёРєР»Р°РґРЅР°СЏ РёРЅС„РѕСЂРјР°С‚РёРєР°").one_or_none()
        if program is None:
            program = StudyProgram(institute="РРњРљРў", name="РџСЂРёРєР»Р°РґРЅР°СЏ РёРЅС„РѕСЂРјР°С‚РёРєР°")
            db.add(program)
            db.flush()
            db.add_all([
                ProgramRequirement(program_id=program.id, subject="Р СѓСЃСЃРєРёР№ СЏР·С‹Рє", role="required", min_score=40),
                ProgramRequirement(program_id=program.id, subject="РњР°С‚РµРјР°С‚РёРєР°", role="required", min_score=40),
                ProgramRequirement(program_id=program.id, subject="РРЅС„РѕСЂРјР°С‚РёРєР°", role="required", min_score=40),
            ])
            db.commit()
    finally:
        db.close()


init_db()


# 0 Р’С…РѕРґ / 0.1 Р’РѕСЃСЃС‚Р°РЅРѕРІР»РµРЅРёРµ / 1 РњРµРЅСЋ / 1.1 Р’С‹С…РѕРґ
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
        return render("login.html", {"request": request, "user": None, "error": "РќРµРІРµСЂРЅС‹Р№ Р»РѕРіРёРЅ РёР»Рё РїР°СЂРѕР»СЊ."})

    request.session["user_id"] = user.id
    return RedirectResponse("/", status_code=303)


@app.get("/recovery")
def recovery_page(request: Request):
    return render("recovery.html", {"request": request, "user": None, "info": None})


@app.post("/recovery")
def recovery_action(request: Request, login: str = Form(...)):
    info = f"Р—Р°РїСЂРѕСЃ РЅР° РІРѕСЃСЃС‚Р°РЅРѕРІР»РµРЅРёРµ РґР»СЏ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ В«{login}В» Р·Р°СЂРµРіРёСЃС‚СЂРёСЂРѕРІР°РЅ (РґРµРјРѕ-СЂРµР¶РёРј)."
    return render("recovery.html", {"request": request, "user": None, "info": info})


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# 2 Р”Р°РЅРЅС‹Рµ Рё Р·Р°РіСЂСѓР·РєРё
@app.get("/data")
def data_mgmt(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("data_mgmt.html", {"request": request, "user": user})


@app.get("/data/ege")
def import_ege_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render(
        "import_ege.html",
        {
            "request": request,
            "user": user,
            "error": None,
            "ok": None,
            "dialog_output": None,
            "form": default_ege_form(),
        },
    )


@app.post("/data/ege")
def import_ege_action(
    request: Request,
    region: str = Form(""),
    kind: str = Form("actual"),
    sheet: str = Form(""),
    year: str = Form(""),
    dry_run: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    form_state = {
        "region": region,
        "kind": kind,
        "sheet": sheet,
        "year": year,
        "dry_run": dry_run is not None,
    }

    kind_value = kind.strip().lower()
    if kind_value not in {"plan", "actual"}:
        return render(
            "import_ege.html",
            {
                "request": request,
                "user": user,
                "error": "Тип данных должен быть 'plan' или 'actual'.",
                "ok": None,
                "dialog_output": None,
                "form": form_state,
            },
        )

    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    filepath = Path(UPLOAD_DIR) / safe_name

    content = file.file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    job = ImportJob(job_type="ege", filename=safe_name, status="uploaded", details=json.dumps({"region": region}, ensure_ascii=False))
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        sheet_name, available_sheets = resolve_sheet_name(filepath, sheet)
        year_value = parse_year_value(year)
        if year_value is None:
            year_value = infer_year_from_sheet_name(sheet_name)
        if year_value is None:
            raise ValueError("Не удалось определить год автоматически. Укажите поле 'Год' вручную.")

        region_value = region.strip() or filepath.stem.replace("_", " ").strip()
        is_dry_run = dry_run is not None

        dialog_header = [
            "Параметры запуска:",
            f"- Файл: {safe_name}",
            f"- Доступные листы: {', '.join(available_sheets)}",
            f"- Лист: {sheet_name}",
            f"- Тип данных: {kind_value}",
            f"- Год: {year_value}",
            f"- Регион: {region_value}",
            f"- Режим: {'dry-run' if is_dry_run else 'запись в БД'}",
            "",
            "Вывод скрипта:",
        ]

        success, script_output = run_ege_loader_script(
            file_path=filepath,
            kind=kind_value,
            sheet=sheet_name,
            year=year_value,
            region=region_value,
            dry_run=is_dry_run,
        )
        dialog_output = "\n".join(dialog_header + [script_output])

        job.status = "loaded" if success else "error"
        job.details = json.dumps(
            {
                "kind": kind_value,
                "sheet": sheet_name,
                "year": year_value,
                "region": region_value,
                "dry_run": is_dry_run,
                "output": script_output[-12000:],
            },
            ensure_ascii=False,
        )
        db.commit()

        form_state["sheet"] = sheet_name
        form_state["year"] = str(year_value)

        return render(
            "import_ege.html",
            {
                "request": request,
                "user": user,
                "error": None if success else "Загрузка ЕГЭ завершилась с ошибкой. Смотрите лог ниже.",
                "ok": "Загрузка ЕГЭ выполнена успешно." if success else None,
                "dialog_output": dialog_output,
                "form": form_state,
            },
        )
    except Exception as exc:
        dialog_output = traceback.format_exc()
        job.status = "error"
        job.details = json.dumps({"error": str(exc), "traceback": dialog_output[-12000:]}, ensure_ascii=False)
        db.commit()
        return render(
            "import_ege.html",
            {
                "request": request,
                "user": user,
                "error": str(exc),
                "ok": None,
                "dialog_output": dialog_output,
                "form": form_state,
            },
        )


@app.get("/data/admissions")
def import_admissions_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {
        "request": request,
        "user": user,
        "title": "РРјРїРѕСЂС‚ РїСЂРёС‘РјР°",
        "back_url": "/data",
        "validate_url": "/data/validate",
    })


@app.get("/data/events")
def import_events_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {
        "request": request,
        "user": user,
        "title": "РРјРїРѕСЂС‚ РїСЂРѕС„РѕСЂРёРµРЅС‚Р°С†РёРё",
        "back_url": "/data",
        "validate_url": "/data/validate",
    })


@app.get("/data/directories")
def directories_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render(
        "directories_import.html",
        {
            "request": request,
            "user": user,
            "loader_form": default_directories_form(),
            "subject_scores": get_subject_scores_from_db(),
            "loader_error": None,
            "loader_ok": None,
            "loader_output": None,
            "scores_error": None,
            "scores_ok": None,
            "scores_output": None,
        },
    )


@app.post("/data/directories/load")
def directories_load_action(
    request: Request,
    loader_type: str = Form("schools"),
    sheet: str = Form(""),
    dry_run: str | None = Form(None),
    create_missing_subjects: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
    directories_upload_dir = Path(UPLOAD_DIR) / "directories"
    directories_upload_dir.mkdir(parents=True, exist_ok=True)

    form_state = {
        "loader_type": loader_type,
        "sheet": sheet,
        "dry_run": dry_run is not None,
        "create_missing_subjects": create_missing_subjects is not None,
    }

    if loader_type not in {"schools", "programs"}:
        return render(
            "directories_import.html",
            {
                "request": request,
                "user": user,
                "loader_form": form_state,
                "subject_scores": get_subject_scores_from_db(),
                "loader_error": "Неизвестный тип загрузчика.",
                "loader_ok": None,
                "loader_output": None,
                "scores_error": None,
                "scores_ok": None,
                "scores_output": None,
            },
        )

    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    filepath = directories_upload_dir / safe_name
    with open(filepath, "wb") as f:
        f.write(file.file.read())

    job = ImportJob(
        job_type="directories",
        filename=safe_name,
        status="uploaded",
        details=json.dumps({"loader_type": loader_type}, ensure_ascii=False),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        sheet_name, available_sheets = resolve_sheet_name(filepath, sheet)
        args: list[str] = [str(filepath.resolve()), "--sheet", sheet_name]

        if loader_type == "schools":
            script_path = LOAD_SCHOOLS_SCRIPT
            title = "Загрузка: Регион / Муниципалитет / Школа / Профиль"
            if dry_run is not None:
                args.append("--dry-run")
        else:
            script_path = LOAD_PROGRAMS_SCRIPT
            title = "Загрузка: Направления и требования по ВИ"
            if dry_run is not None:
                args.append("--dry-run")
            if create_missing_subjects is not None:
                args.append("--create-missing-subjects")

        success, script_output = run_script_with_output(script_path, args)
        loader_output = "\n".join(
            [
                f"{title}",
                f"- Файл: {safe_name}",
                f"- Доступные листы: {', '.join(available_sheets)}",
                f"- Лист: {sheet_name}",
                f"- Dry-run: {'да' if dry_run is not None else 'нет'}",
                f"- create-missing-subjects: {'да' if create_missing_subjects is not None else 'нет'}",
                "",
                "Вывод скрипта:",
                script_output,
            ]
        )

        job.status = "loaded" if success else "error"
        job.details = json.dumps(
            {
                "loader_type": loader_type,
                "sheet": sheet_name,
                "dry_run": dry_run is not None,
                "create_missing_subjects": create_missing_subjects is not None,
                "output": script_output[-12000:],
            },
            ensure_ascii=False,
        )
        db.commit()

        form_state["sheet"] = sheet_name

        return render(
            "directories_import.html",
            {
                "request": request,
                "user": user,
                "loader_form": form_state,
                "subject_scores": get_subject_scores_from_db(),
                "loader_error": None if success else "Загрузка справочников завершилась с ошибкой.",
                "loader_ok": "Загрузка справочников выполнена успешно." if success else None,
                "loader_output": loader_output,
                "scores_error": None,
                "scores_ok": None,
                "scores_output": None,
            },
        )
    except Exception as exc:
        loader_output = traceback.format_exc()
        job.status = "error"
        job.details = json.dumps({"error": str(exc), "traceback": loader_output[-12000:]}, ensure_ascii=False)
        db.commit()
        return render(
            "directories_import.html",
            {
                "request": request,
                "user": user,
                "loader_form": form_state,
                "subject_scores": get_subject_scores_from_db(),
                "loader_error": str(exc),
                "loader_ok": None,
                "loader_output": loader_output,
                "scores_error": None,
                "scores_ok": None,
                "scores_output": None,
            },
        )


@app.post("/data/directories/min-scores")
async def directories_min_scores_action(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    submitted = await request.form()
    submitted_map = {str(k): str(v) for k, v in submitted.items()}

    rows = apply_subject_scores_from_form(default_subject_scores(), submitted_map)
    updates: list[tuple[int, str, int]] = []
    errors: list[str] = []

    for row in rows:
        subject_id = int(row["subject_id"])
        name = str(row["name"])
        raw = str(row["min_passing_score"]).strip()
        if not raw:
            errors.append(f"Для предмета '{name}' не заполнен минимальный балл.")
            continue
        try:
            score = int(raw)
        except ValueError:
            errors.append(f"Для предмета '{name}' значение '{raw}' не является числом.")
            continue
        if score < 0 or score > 100:
            errors.append(f"Для предмета '{name}' минимальный балл должен быть в диапазоне 0..100.")
            continue
        updates.append((subject_id, name, score))
        row["min_passing_score"] = score

    if errors:
        return render(
            "directories_import.html",
            {
                "request": request,
                "user": user,
                "loader_form": default_directories_form(),
                "subject_scores": rows,
                "loader_error": None,
                "loader_ok": None,
                "loader_output": None,
                "scores_error": "\n".join(errors),
                "scores_ok": None,
                "scores_output": None,
            },
        )

    job = ImportJob(
        job_type="directories",
        filename="ege_subject.min_scores",
        status="uploaded",
        details=json.dumps({"rows": len(updates)}, ensure_ascii=False),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        import psycopg2

        db_cfg = get_load_db_config()
        updated = 0
        updated_by_name = 0
        inserted = 0

        with psycopg2.connect(**db_cfg) as conn:
            with conn.cursor() as cur:
                for subject_id, name, score in updates:
                    cur.execute(
                        """
                        UPDATE edu.ege_subject
                        SET name = %s,
                            min_passing_score = %s
                        WHERE subject_id = %s
                        """,
                        (name, score, subject_id),
                    )
                    if cur.rowcount:
                        updated += cur.rowcount
                        continue

                    cur.execute(
                        """
                        SELECT subject_id
                        FROM edu.ege_subject
                        WHERE name = %s
                        """,
                        (name,),
                    )
                    existing_by_name = cur.fetchone()
                    if existing_by_name:
                        cur.execute(
                            """
                            UPDATE edu.ege_subject
                            SET min_passing_score = %s
                            WHERE subject_id = %s
                            """,
                            (score, int(existing_by_name[0])),
                        )
                        updated_by_name += cur.rowcount
                        continue

                    cur.execute(
                        """
                        INSERT INTO edu.ege_subject (subject_id, name, min_passing_score)
                        VALUES (%s, %s, %s)
                        """,
                        (subject_id, name, score),
                    )
                    inserted += 1
            conn.commit()

        score_output = "\n".join(
            [
                "Обновление минимальных баллов ЕГЭ:",
                f"- Обновлено по subject_id: {updated}",
                f"- Обновлено по name: {updated_by_name}",
                f"- Вставлено новых строк: {inserted}",
            ]
        )

        job.status = "loaded"
        job.details = json.dumps(
            {
                "updated": updated,
                "updated_by_name": updated_by_name,
                "inserted": inserted,
            },
            ensure_ascii=False,
        )
        db.commit()

        return render(
            "directories_import.html",
            {
                "request": request,
                "user": user,
                "loader_form": default_directories_form(),
                "subject_scores": get_subject_scores_from_db(),
                "loader_error": None,
                "loader_ok": None,
                "loader_output": None,
                "scores_error": None,
                "scores_ok": "Минимальные баллы обновлены.",
                "scores_output": score_output,
            },
        )
    except Exception as exc:
        score_output = traceback.format_exc()
        job.status = "error"
        job.details = json.dumps({"error": str(exc), "traceback": score_output[-12000:]}, ensure_ascii=False)
        db.commit()
        return render(
            "directories_import.html",
            {
                "request": request,
                "user": user,
                "loader_form": default_directories_form(),
                "subject_scores": rows,
                "loader_error": None,
                "loader_ok": None,
                "loader_output": None,
                "scores_error": str(exc),
                "scores_ok": None,
                "scores_output": score_output,
            },
        )


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



# 3 РџРѕРёСЃРє Рё РєР°СЂС‚РѕС‡РєР° С€РєРѕР»С‹
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



# 4 РџРѕРґР±РѕСЂ Рё СЂРµР№С‚РёРЅРі
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



# 5 РќР°СЃС‚СЂРѕР№РєРё Р°РЅР°Р»РёР·Р° (Р·Р°РіР»СѓС€РєРё)
@app.get("/settings")
def settings_home(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("settings_home.html", {"request": request, "user": user})


@app.get("/settings/filters")
def settings_filters(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "РљРѕРЅСЃС‚СЂСѓРєС‚РѕСЂ С„РёР»СЊС‚СЂРѕРІ", "back_url": "/settings"})


@app.get("/settings/weights")
def settings_weights(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "РњРµС‚СЂРёРєРё Рё РІРµСЃР°", "back_url": "/settings"})


@app.get("/settings/profiles")
def settings_profiles(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "РџСЂРѕС„РёР»Рё СЂР°СЃС‡С‘С‚Р°", "back_url": "/settings"})



# 6 РћС‚С‡С‘С‚РЅРѕСЃС‚СЊ
@app.get("/reports")
def reports_home(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("reports_home.html", {"request": request, "user": user})


@app.get("/reports/generate")
def report_generate(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("report_generate.html", {"request": request, "user": user})


@app.get("/reports/archive")
def report_archive(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    jobs = db.query(ImportJob).order_by(ImportJob.created_at.desc()).limit(50).all()
    return render("report_archive.html", {"request": request, "user": user, "jobs": jobs})


@app.get("/reports/calc-history")
def calc_history(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("calc_history.html", {"request": request, "user": user})



# 7 РђРґРјРёРЅРёСЃС‚СЂРёСЂРѕРІР°РЅРёРµ (С‚РѕР»СЊРєРѕ admin)
@app.get("/admin")
def admin_home(request: Request, db: Session = Depends(get_db)):
    user = require_admin(request, db)
    return render("admin_home.html", {"request": request, "user": user})


@app.get("/admin/roles")
def admin_roles(request: Request, db: Session = Depends(get_db)):
    user = require_admin(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "Р РѕР»Рё Рё РїСЂР°РІР°", "back_url": "/admin"})


@app.get("/admin/directories")
def admin_directories(request: Request, db: Session = Depends(get_db)):
    require_admin(request, db)
    return RedirectResponse("/data/directories", status_code=303)


@app.get("/admin/methodologies")
def admin_methodologies(request: Request, db: Session = Depends(get_db)):
    user = require_admin(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "РњРµС‚РѕРґРёРєРё", "back_url": "/admin"})
