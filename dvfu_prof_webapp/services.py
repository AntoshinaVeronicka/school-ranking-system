from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Request, UploadFile
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from passlib.context import CryptContext
from sqlalchemy import func, select
from sqlalchemy.orm import Session

try:
    from .config import (
        DEFAULT_EGE_SUBJECT_SCORES,
        LOAD_EGE_SCRIPT,
        PROJECT_ROOT,
        TEMPLATES_DIR,
        UPLOAD_DIR,
    )
    from .db import EgeStat, ImportJob, ProgramRequirement, School, User
except ImportError:
    from config import (
        DEFAULT_EGE_SUBJECT_SCORES,
        LOAD_EGE_SCRIPT,
        PROJECT_ROOT,
        TEMPLATES_DIR,
        UPLOAD_DIR,
    )
    from db import EgeStat, ImportJob, ProgramRequirement, School, User


# Security / sessions
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


# Calculations (demo)
def subjects_for_program(db: Session, program_id: int) -> list[str]:
    rows = db.execute(select(ProgramRequirement.subject).where(ProgramRequirement.program_id == program_id)).all()
    return [r[0] for r in rows]


def calculate_school_metrics(db: Session, school_ids: list[int], year: int, program_id: int | None) -> list[dict[str, Any]]:
    if not school_ids:
        return []

    # In real model graduates may be stored separately; here we use max(graduates) per subject as heuristic.
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

        result.append(
            {
                "school_id": school.id,
                "region": school.region,
                "municipality": school.municipality,
                "name": school.name,
                "graduates": int(grads or 0),
                "avg_score_all": float(avg_score_all) if avg_score_all is not None else None,
                "match_share": match_share,
            }
        )

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


# Forms/helpers
def default_ege_form() -> dict[str, Any]:
    return {"region": "", "kind": "actual", "sheet": "", "year": "", "dry_run": False}


def default_upload_form() -> dict[str, Any]:
    return {"region": "", "sheet": "", "year": "", "dry_run": False}


def parse_year_value(raw: str) -> int | None:
    value = raw.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("Поле 'Год' должно быть числом.") from exc


def resolve_sheet_name(path: Path, requested: str) -> tuple[str, list[str]]:
    with pd.ExcelFile(path, engine="openpyxl") as xls:
        sheets = list(xls.sheet_names)
    if not sheets:
        raise ValueError("В файле отсутствуют листы Excel.")

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

    raise ValueError(f"Лист '{value}' не найден. Доступные листы: {', '.join(sheets)}")


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
        return False, "Загрузка прервана по тайм-ауту (превышено 300 секунд)."
    except Exception:
        return False, traceback.format_exc()

    output_parts = [part.strip() for part in [proc.stdout, proc.stderr] if part and part.strip()]
    output_text = "\n\n".join(output_parts) if output_parts else "(Скрипт завершился без вывода.)"
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


def render_generic_import_page(
    *,
    request: Request,
    user: User,
    title: str,
    submit_url: str,
    hint: str,
    form: dict[str, Any],
    error: str | None,
    ok: str | None,
    dialog_output: str | None,
) -> HTMLResponse:
    return render(
        "import_generic.html",
        {
            "request": request,
            "user": user,
            "title": title,
            "submit_url": submit_url,
            "hint": hint,
            "form": form,
            "error": error,
            "ok": ok,
            "dialog_output": dialog_output,
        },
    )


def process_generic_import_upload(
    *,
    request: Request,
    user: User,
    db: Session,
    title: str,
    submit_url: str,
    hint: str,
    job_type: str,
    upload_subdir: str,
    region: str,
    sheet: str,
    year: str,
    dry_run: str | None,
    file: UploadFile,
) -> HTMLResponse:
    form_state = {
        "region": region,
        "sheet": sheet,
        "year": year,
        "dry_run": dry_run is not None,
    }

    upload_dir = UPLOAD_DIR / upload_subdir
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    filepath = upload_dir / safe_name
    with open(filepath, "wb") as f:
        f.write(file.file.read())

    job = ImportJob(
        job_type=job_type,
        filename=safe_name,
        status="uploaded",
        details=json.dumps({"region": region}, ensure_ascii=False),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        sheet_name, available_sheets = resolve_sheet_name(filepath, sheet)
        year_value = parse_year_value(year)
        if year_value is None:
            year_value = infer_year_from_sheet_name(sheet_name)

        region_value = region.strip() or filepath.stem.replace("_", " ").strip()
        is_dry_run = dry_run is not None

        dialog_lines = [
            "Параметры запуска:",
            f"- Файл: {safe_name}",
            f"- Доступные листы: {', '.join(available_sheets)}",
            f"- Лист: {sheet_name}",
            f"- Год: {year_value if year_value is not None else 'не указан'}",
            f"- Регион: {region_value if region_value else 'не указан'}",
            f"- Режим: {'dry-run' if is_dry_run else 'подготовка файла'}",
            "",
            "Вывод обработчика:",
            "Форма загрузки подключена. Импорт в БД для этого раздела пока в разработке.",
        ]
        dialog_output = "\n".join(dialog_lines)

        job.status = "validated"
        job.details = json.dumps(
            {
                "sheet": sheet_name,
                "year": year_value,
                "region": region_value,
                "dry_run": is_dry_run,
                "note": "upload_form_only",
            },
            ensure_ascii=False,
        )
        db.commit()

        form_state["sheet"] = sheet_name
        form_state["year"] = str(year_value) if year_value is not None else ""

        return render_generic_import_page(
            request=request,
            user=user,
            title=title,
            submit_url=submit_url,
            hint=hint,
            form=form_state,
            error=None,
            ok="Файл принят. Черновая форма импорта отработала успешно.",
            dialog_output=dialog_output,
        )
    except Exception as exc:
        dialog_output = traceback.format_exc()
        job.status = "error"
        job.details = json.dumps({"error": str(exc), "traceback": dialog_output[-12000:]}, ensure_ascii=False)
        db.commit()
        return render_generic_import_page(
            request=request,
            user=user,
            title=title,
            submit_url=submit_url,
            hint=hint,
            form=form_state,
            error=str(exc),
            ok=None,
            dialog_output=dialog_output,
        )


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
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


def render(template_name: str, context: dict[str, Any]) -> HTMLResponse:
    tpl = templates_env.get_template(template_name)
    html = tpl.render(**context)
    return HTMLResponse(html)
