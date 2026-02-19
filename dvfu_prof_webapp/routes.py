from __future__ import annotations

import io
import json
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from urllib.parse import urlencode

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

try:
    from .analytics_repo import (
        create_reports_for_run,
        fetch_calc_history as fetch_calc_history_rows,
        fetch_rating_runs,
        fetch_report_archive as fetch_report_archive_rows,
        fetch_run_export_data,
        fetch_run_schools,
        get_report_payload,
        save_rating_run,
        save_search_request,
    )
    from .config import (
        DEFAULT_RATING_LIMIT,
        DEFAULT_RATING_W_AVG_SCORE,
        DEFAULT_RATING_W_GRADUATES,
        DEFAULT_RATING_W_MATCH_SHARE,
        DEFAULT_RATING_W_THRESHOLD_SHARE,
        DEFAULT_SEARCH_PER_PAGE,
        LOAD_PROGRAMS_SCRIPT,
        LOAD_SCHOOLS_SCRIPT,
        MAX_RATING_LIMIT,
        MAX_SEARCH_PER_PAGE,
        MIN_RATING_LIMIT,
        MIN_SEARCH_PER_PAGE,
        UPLOAD_DIR,
    )
    from .db import ImportJob, User, get_db
    from .filter_options_repo import clear_filter_options_cache
    from .rating_repo import (
        RatingFilters,
        RatingWeights,
        calculate_school_rating,
        fetch_program_requirements,
        fetch_programs,
        fetch_rating_filter_options,
    )
    from .search_repo import (
        SchoolSearchFilters,
        fetch_filter_options,
        fetch_municipalities,
        fetch_school_card,
        search_schools,
    )
    from .services import (
        apply_subject_scores_from_form,
        default_directories_form,
        default_ege_form,
        default_subject_scores,
        default_upload_form,
        get_current_user,
        get_load_db_config,
        get_subject_scores_from_db,
        infer_year_from_sheet_name,
        parse_year_value,
        process_generic_import_upload,
        render,
        render_generic_import_page,
        require_admin,
        require_user,
        resolve_sheet_name,
        run_ege_loader_script,
        run_script_with_output,
        validate_csrf_token,
        verify_password,
    )
    from .subject_analytics_service import build_subject_analytics
except ImportError:
    from analytics_repo import (
        create_reports_for_run,
        fetch_calc_history as fetch_calc_history_rows,
        fetch_rating_runs,
        fetch_report_archive as fetch_report_archive_rows,
        fetch_run_export_data,
        fetch_run_schools,
        get_report_payload,
        save_rating_run,
        save_search_request,
    )
    from config import (
        DEFAULT_RATING_LIMIT,
        DEFAULT_RATING_W_AVG_SCORE,
        DEFAULT_RATING_W_GRADUATES,
        DEFAULT_RATING_W_MATCH_SHARE,
        DEFAULT_RATING_W_THRESHOLD_SHARE,
        DEFAULT_SEARCH_PER_PAGE,
        LOAD_PROGRAMS_SCRIPT,
        LOAD_SCHOOLS_SCRIPT,
        MAX_RATING_LIMIT,
        MAX_SEARCH_PER_PAGE,
        MIN_RATING_LIMIT,
        MIN_SEARCH_PER_PAGE,
        UPLOAD_DIR,
    )
    from db import ImportJob, User, get_db
    from filter_options_repo import clear_filter_options_cache
    from rating_repo import (
        RatingFilters,
        RatingWeights,
        calculate_school_rating,
        fetch_program_requirements,
        fetch_programs,
        fetch_rating_filter_options,
    )
    from search_repo import (
        SchoolSearchFilters,
        fetch_filter_options,
        fetch_municipalities,
        fetch_school_card,
        search_schools,
    )
    from services import (
        apply_subject_scores_from_form,
        default_directories_form,
        default_ege_form,
        default_subject_scores,
        default_upload_form,
        get_current_user,
        get_load_db_config,
        get_subject_scores_from_db,
        infer_year_from_sheet_name,
        parse_year_value,
        process_generic_import_upload,
        render,
        render_generic_import_page,
        require_admin,
        require_user,
        resolve_sheet_name,
        run_ege_loader_script,
        run_script_with_output,
        validate_csrf_token,
        verify_password,
    )
    from subject_analytics_service import build_subject_analytics

router = APIRouter()
logger = logging.getLogger(__name__)

SEARCH_QUERY_KEYS = {
    "q",
    "region_id",
    "municipality_id",
    "profile_ids",
    "year",
    "kind",
    "subject_ids",
    "subject_id",
    "page",
    "per_page",
}

RATING_QUERY_KEYS = {
    "q",
    "region_id",
    "municipality_id",
    "profile_ids",
    "year",
    "kind",
    "subject_ids",
    "institute_ids",
    "institute_id",
    "program_ids",
    "min_graduates",
    "min_avg_score",
    "enforce_subject_threshold",
    "w_graduates",
    "w_avg_score",
    "w_match_share",
    "w_threshold_share",
    "limit",
}

# Раздел 0. Вход, восстановление доступа и выход.
@router.get("/")
def main_menu(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return render("main.html", {"request": request, "user": user})


@router.get("/login")
def login_page(request: Request):
    return render("login.html", {"request": request, "user": None, "error": None})


@router.post("/login")
def login_action(
    request: Request,
    login: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(""),
    db: Session = Depends(get_db),
):
    validate_csrf_token(request, csrf_token)
    user = db.query(User).filter(User.login == login).one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        return render("login.html", {"request": request, "user": None, "error": "Неверный логин или пароль."})

    request.session["user_id"] = user.id
    return RedirectResponse("/", status_code=303)


@router.get("/recovery")
def recovery_page(request: Request):
    return render("recovery.html", {"request": request, "user": None, "info": None})


@router.post("/recovery")
def recovery_action(request: Request, login: str = Form(...), csrf_token: str = Form("")):
    validate_csrf_token(request, csrf_token)
    info = f"Запрос на восстановление для пользователя «{login}» зарегистрирован (демо-режим)."
    return render("recovery.html", {"request": request, "user": None, "info": info})


@router.post("/logout")
def logout(request: Request, csrf_token: str = Form("")):
    validate_csrf_token(request, csrf_token)
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# Раздел 2. Загрузка данных.
@router.get("/data")
def data_mgmt(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("data_mgmt.html", {"request": request, "user": user})


@router.get("/data/ege")
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


@router.post("/data/ege")
def import_ege_action(
    request: Request,
    region: str = Form(""),
    kind: str = Form("actual"),
    sheet: str = Form(""),
    year: str = Form(""),
    dry_run: str | None = Form(None),
    csrf_token: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
    validate_csrf_token(request, csrf_token)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    filepath = UPLOAD_DIR / safe_name

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
        if success and not is_dry_run:
            clear_filter_options_cache()

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
    except (ValueError, RuntimeError, OSError, PermissionError) as exc:
        logger.warning("Ошибка импорта ЕГЭ: file=%s, year=%s, kind=%s, err=%s", safe_name, year, kind_value, exc)
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
    except Exception as exc:
        logger.exception("Непредвиденная ошибка импорта ЕГЭ: file=%s", safe_name)
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


@router.get("/data/admissions")
def import_admissions_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render_generic_import_page(
        request=request,
        user=user,
        title="Импорт приёма",
        submit_url="/data/admissions",
        hint="Черновая форма загрузки файла по разделу приёма. Подключение бизнес-логики можно добавить на следующем шаге.",
        form=default_upload_form(),
        error=None,
        ok=None,
        dialog_output=None,
    )


@router.post("/data/admissions")
def import_admissions_action(
    request: Request,
    region: str = Form(""),
    sheet: str = Form(""),
    year: str = Form(""),
    dry_run: str | None = Form(None),
    csrf_token: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
    validate_csrf_token(request, csrf_token)
    return process_generic_import_upload(
        request=request,
        user=user,
        db=db,
        title="Импорт приёма",
        submit_url="/data/admissions",
        hint="Черновая форма загрузки файла по разделу приёма. Подключение бизнес-логики можно добавить на следующем шаге.",
        job_type="admissions",
        upload_subdir="admissions",
        region=region,
        sheet=sheet,
        year=year,
        dry_run=dry_run,
        file=file,
    )


@router.get("/data/events")
def import_events_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render_generic_import_page(
        request=request,
        user=user,
        title="Импорт профориентации",
        submit_url="/data/events",
        hint="Черновая форма загрузки файла по разделу профориентации. Подключение бизнес-логики можно добавить на следующем шаге.",
        form=default_upload_form(),
        error=None,
        ok=None,
        dialog_output=None,
    )


@router.post("/data/events")
def import_events_action(
    request: Request,
    region: str = Form(""),
    sheet: str = Form(""),
    year: str = Form(""),
    dry_run: str | None = Form(None),
    csrf_token: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
    validate_csrf_token(request, csrf_token)
    return process_generic_import_upload(
        request=request,
        user=user,
        db=db,
        title="Импорт профориентации",
        submit_url="/data/events",
        hint="Черновая форма загрузки файла по разделу профориентации. Подключение бизнес-логики можно добавить на следующем шаге.",
        job_type="events",
        upload_subdir="events",
        region=region,
        sheet=sheet,
        year=year,
        dry_run=dry_run,
        file=file,
    )


@router.get("/data/directories")
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


@router.post("/data/directories/load")
def directories_load_action(
    request: Request,
    loader_type: str = Form("schools"),
    sheet: str = Form(""),
    dry_run: str | None = Form(None),
    create_missing_subjects: str | None = Form(None),
    csrf_token: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
    validate_csrf_token(request, csrf_token)
    directories_upload_dir = UPLOAD_DIR / "directories"
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
        if success and dry_run is None:
            clear_filter_options_cache()

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
    except (ValueError, RuntimeError, OSError, PermissionError) as exc:
        logger.warning("Ошибка загрузки справочников: loader=%s, file=%s, err=%s", loader_type, safe_name, exc)
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
    except Exception as exc:
        logger.exception("Непредвиденная ошибка загрузки справочников: loader=%s, file=%s", loader_type, safe_name)
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


@router.post("/data/directories/min-scores")
async def directories_min_scores_action(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    submitted = await request.form()
    validate_csrf_token(request, str(submitted.get("csrf_token", "")))
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
        clear_filter_options_cache()

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
    except (ValueError, OSError, RuntimeError, PermissionError) as exc:
        logger.warning("Ошибка обновления минимальных баллов ЕГЭ: rows=%s, err=%s", len(updates), exc)
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
    except Exception as exc:
        logger.exception("Непредвиденная ошибка обновления минимальных баллов ЕГЭ: rows=%s", len(updates))
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


# Раздел 3. Поиск и карточка школы.
def _build_search_filters(
    *,
    q: str,
    region_id: int | None,
    municipality_id: int | None,
    profile_ids: list[int] | None,
    year: int | None,
    kind: str,
    subject_ids: list[int] | None,
) -> SchoolSearchFilters:
    return SchoolSearchFilters(
        q=q.strip(),
        region_id=region_id,
        municipality_id=municipality_id,
        profile_ids=tuple(sorted(set(profile_ids or []))),
        year=year,
        kind=kind.strip() or None,
        subject_ids=tuple(sorted(set(subject_ids or []))),
    )


def _build_search_query_without_page(
    *,
    q: str,
    region_id: int | None,
    municipality_id: int | None,
    profile_ids: list[int],
    year: int | None,
    kind: str,
    subject_ids: list[int],
    per_page: int,
) -> str:
    params: list[tuple[str, str]] = []
    if q:
        params.append(("q", q))
    if region_id is not None:
        params.append(("region_id", str(region_id)))
    if municipality_id is not None:
        params.append(("municipality_id", str(municipality_id)))
    for pid in profile_ids:
        params.append(("profile_ids", str(pid)))
    if year is not None:
        params.append(("year", str(year)))
    if kind:
        params.append(("kind", kind))
    for sid in subject_ids:
        params.append(("subject_ids", str(sid)))
    params.append(("per_page", str(per_page)))
    return urlencode(params, doseq=True)


def _parse_optional_int(value: str | None) -> int | None:
    raw = (value or "").strip()
    if not raw:
        return None
    return int(raw) if raw.isdigit() else None


def _parse_int_list(values: list[str] | None) -> list[int]:
    result: list[int] = []
    for value in values or []:
        raw = str(value).strip()
        if raw.isdigit():
            result.append(int(raw))
    return result


def _parse_bounded_int(value: str | None, *, default: int, min_value: int, max_value: int | None = None) -> int:
    parsed = _parse_optional_int(value)
    if parsed is None:
        parsed = default
    if parsed < min_value:
        parsed = min_value
    if max_value is not None and parsed > max_value:
        parsed = max_value
    return parsed


def _parse_optional_float(value: str | None) -> float | None:
    raw = (value or "").strip().replace(",", ".")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _excel_safe_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.isoformat()
        return value
    if isinstance(value, pd.Timestamp):
        if value.tz is not None:
            return value.isoformat()
        py_dt = value.to_pydatetime()
        return py_dt
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _prepare_df_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    prepared = df.copy()
    for col in prepared.columns:
        prepared[col] = prepared[col].map(_excel_safe_value)
    return prepared


def _excel_response(*, filename: str, sheets: list[tuple[str, pd.DataFrame]]) -> StreamingResponse:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet_name, df in sheets:
            safe_sheet = (sheet_name or "sheet")[:31]
            _prepare_df_for_excel(df).to_excel(writer, index=False, sheet_name=safe_sheet)
    out.seek(0)
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


def _has_known_query_params(request: Request, keys: set[str]) -> bool:
    params = request.query_params
    return any(key in params for key in keys)


@dataclass(frozen=True)
class SearchRequestParams:
    q: str
    region_id: int | None
    municipality_id: int | None
    profile_ids: list[int]
    year: int | None
    kind: str
    subject_ids: list[int]
    page: int
    per_page: int


def _parse_search_request_params(
    *,
    q: str,
    region_id: str | None,
    municipality_id: str | None,
    profile_ids: list[str] | None,
    year: str | None,
    kind: str,
    subject_ids: list[str] | None,
    legacy_subject_id: str | None,
    page: str | None,
    per_page: str | None,
) -> SearchRequestParams:
    region_id_value = _parse_optional_int(region_id)
    municipality_id_value = _parse_optional_int(municipality_id)
    year_value = _parse_optional_int(year)

    subject_ids_value = _parse_int_list(subject_ids)
    legacy_subject_id_value = _parse_optional_int(legacy_subject_id)
    if legacy_subject_id_value is not None:
        subject_ids_value.append(legacy_subject_id_value)
    subject_ids_value = sorted(set(subject_ids_value))

    profile_ids_value = _parse_int_list(profile_ids)
    safe_page = _parse_bounded_int(page, default=1, min_value=1)
    safe_per_page = _parse_bounded_int(
        per_page,
        default=DEFAULT_SEARCH_PER_PAGE,
        min_value=MIN_SEARCH_PER_PAGE,
        max_value=MAX_SEARCH_PER_PAGE,
    )

    return SearchRequestParams(
        q=(q or "").strip(),
        region_id=region_id_value,
        municipality_id=municipality_id_value,
        profile_ids=profile_ids_value,
        year=year_value,
        kind=(kind or "").strip(),
        subject_ids=subject_ids_value,
        page=safe_page,
        per_page=safe_per_page,
    )


def _build_search_filters_from_params(params: SearchRequestParams) -> SchoolSearchFilters:
    return _build_search_filters(
        q=params.q,
        region_id=params.region_id,
        municipality_id=params.municipality_id,
        profile_ids=params.profile_ids,
        year=params.year,
        kind=params.kind,
        subject_ids=params.subject_ids,
    )


def _build_search_query_without_page_from_params(params: SearchRequestParams) -> str:
    return _build_search_query_without_page(
        q=params.q,
        region_id=params.region_id,
        municipality_id=params.municipality_id,
        profile_ids=params.profile_ids,
        year=params.year,
        kind=params.kind,
        subject_ids=params.subject_ids,
        per_page=params.per_page,
    )


@router.get("/search")
def search_page(
    request: Request,
    q: str = "",
    region_id: str | None = Query(default=None),
    municipality_id: str | None = Query(default=None),
    profile_ids: list[str] = Query(default=[]),
    year: str | None = Query(default=None),
    kind: str = "",
    subject_ids: list[str] = Query(default=[]),
    subject_id: str | None = Query(default=None),
    page: str | None = Query(default="1"),
    per_page: str | None = Query(default=str(DEFAULT_SEARCH_PER_PAGE)),
    save_status: str | None = Query(default=None),
    saved_request_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
    params = _parse_search_request_params(
        q=q,
        region_id=region_id,
        municipality_id=municipality_id,
        profile_ids=profile_ids,
        year=year,
        kind=kind,
        subject_ids=subject_ids,
        legacy_subject_id=subject_id,
        page=page,
        per_page=per_page,
    )
    region_id_value = params.region_id
    municipality_id_value = params.municipality_id
    year_value = params.year
    subject_ids_value = params.subject_ids
    profile_ids_value = params.profile_ids
    safe_page = params.page
    safe_per_page = params.per_page
    has_query = _has_known_query_params(request, SEARCH_QUERY_KEYS)
    save_status_value = (save_status or "").strip().lower()
    saved_request_id_value = _parse_optional_int(saved_request_id)
    save_message: str | None = None
    save_message_kind = "ok"
    if save_status_value == "ok":
        if saved_request_id_value is not None:
            save_message = f"Запрос сохранен в системе (request_id={saved_request_id_value})."
        else:
            save_message = "Запрос сохранен в системе."
    elif save_status_value == "empty":
        save_message_kind = "alert"
        save_message = "Нечего сохранять: по выбранным фильтрам нет данных."
    elif save_status_value == "error":
        save_message_kind = "alert"
        save_message = "Не удалось сохранить запрос. Проверьте лог приложения."

    filters = _build_search_filters_from_params(params)
    options = fetch_filter_options(region_id=region_id_value)
    results, total = search_schools(filters, page=safe_page, per_page=safe_per_page, apply_pagination=True)

    total_pages = max(1, (total + safe_per_page - 1) // safe_per_page)
    if safe_page > total_pages:
        safe_page = total_pages
        results, _ = search_schools(filters, page=safe_page, per_page=safe_per_page, apply_pagination=True)

    query_without_page = _build_search_query_without_page_from_params(params)
    current_query = query_without_page
    if current_query:
        current_query = f"{current_query}&page={safe_page}"
    else:
        current_query = f"page={safe_page}"

    return render(
        "search.html",
        {
            "request": request,
            "user": user,
            "results": results,
            "total": total,
            "page": safe_page,
            "per_page": safe_per_page,
            "total_pages": total_pages,
            "query_without_page": query_without_page,
            "current_query": current_query,
            "has_query": has_query,
            "save_message": save_message,
            "save_message_kind": save_message_kind,
            "saved_request_id": saved_request_id_value,
            "filters": {
                "q": params.q,
                "region_id": region_id_value,
                "municipality_id": municipality_id_value,
                "profile_ids": profile_ids_value,
                "year": year_value,
                "kind": params.kind,
                "subject_ids": subject_ids_value,
            },
            "regions": options["regions"],
            "municipalities": options["municipalities"],
            "profiles": options["profiles"],
            "years": options["years"],
            "kinds": options["kinds"],
            "subjects": options["subjects"],
        },
    )


@router.post("/search")
async def search_action(request: Request, db: Session = Depends(get_db)):
    require_user(request, db)
    form = await request.form()
    validate_csrf_token(request, str(form.get("csrf_token", "")))
    params = _parse_search_request_params(
        q=str(form.get("q", "")),
        region_id=str(form.get("region_id", "")),
        municipality_id=str(form.get("municipality_id", "")),
        profile_ids=form.getlist("profile_ids"),
        year=str(form.get("year", "")),
        kind=str(form.get("kind", "")),
        subject_ids=form.getlist("subject_ids"),
        legacy_subject_id=str(form.get("subject_id", "")),
        page=None,
        per_page=str(form.get("per_page", str(DEFAULT_SEARCH_PER_PAGE))),
    )

    query_without_page = _build_search_query_without_page_from_params(params)
    location = "/search"
    if query_without_page:
        location = f"/search?{query_without_page}&page=1"
    return RedirectResponse(location, status_code=303)


@router.post("/search/save")
async def search_save_action(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    form = await request.form()
    validate_csrf_token(request, str(form.get("csrf_token", "")))

    params = _parse_search_request_params(
        q=str(form.get("q", "")),
        region_id=str(form.get("region_id", "")),
        municipality_id=str(form.get("municipality_id", "")),
        profile_ids=form.getlist("profile_ids"),
        year=str(form.get("year", "")),
        kind=str(form.get("kind", "")),
        subject_ids=form.getlist("subject_ids"),
        legacy_subject_id=str(form.get("subject_id", "")),
        page=None,
        per_page=str(form.get("per_page", str(DEFAULT_SEARCH_PER_PAGE))),
    )

    q = params.q
    region_id = params.region_id
    municipality_id = params.municipality_id
    year = params.year
    kind = params.kind
    subject_ids = params.subject_ids
    profile_ids = params.profile_ids
    per_page = params.per_page

    filters = _build_search_filters_from_params(params)

    save_status = "error"
    saved_request_id: int | None = None
    try:
        rows, total = search_schools(filters, page=1, per_page=per_page, apply_pagination=False)
        saved_request_id = save_search_request(
            created_by=user.login,
            filters={
                "q": q,
                "region_id": region_id,
                "municipality_id": municipality_id,
                "profile_ids": profile_ids,
                "year": year,
                "kind": kind,
                "subject_ids": subject_ids,
            },
            rows=rows,
            total_rows=total,
            page=1,
            per_page=per_page,
        )
        save_status = "ok" if saved_request_id is not None else "empty"
    except (ValueError, RuntimeError, TimeoutError) as exc:
        logger.warning("Не удалось сохранить поисковый запрос: %s", exc)
        save_status = "error"

    query_without_page = _build_search_query_without_page_from_params(params)
    status_params: list[tuple[str, str]] = [("page", "1"), ("save_status", save_status)]
    if saved_request_id is not None:
        status_params.append(("saved_request_id", str(saved_request_id)))
    status_query = urlencode(status_params, doseq=True)

    location = "/search"
    if query_without_page:
        location = f"/search?{query_without_page}&{status_query}"
    else:
        location = f"/search?{status_query}"
    return RedirectResponse(location, status_code=303)


@router.get("/search/municipalities")
def municipalities_api(region_id: int, request: Request, db: Session = Depends(get_db)):
    require_user(request, db)
    rows = fetch_municipalities(region_id)
    return JSONResponse({"items": rows})


@router.get("/search/export")
def search_export(
    request: Request,
    q: str = "",
    region_id: str | None = Query(default=None),
    municipality_id: str | None = Query(default=None),
    profile_ids: list[str] = Query(default=[]),
    year: str | None = Query(default=None),
    kind: str = "",
    subject_ids: list[str] = Query(default=[]),
    subject_id: str | None = Query(default=None),
    per_page: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    require_user(request, db)
    params = _parse_search_request_params(
        q=q,
        region_id=region_id,
        municipality_id=municipality_id,
        profile_ids=profile_ids,
        year=year,
        kind=kind,
        subject_ids=subject_ids,
        legacy_subject_id=subject_id,
        page=None,
        per_page=per_page,
    )
    filters = _build_search_filters_from_params(params)
    rows, _ = search_schools(filters, page=1, per_page=DEFAULT_SEARCH_PER_PAGE, apply_pagination=False)
    export_rows = [
        {
            "school_id": r["school_id"],
            "region": r["region_name"],
            "municipality": r["municipality_name"],
            "school": r["full_name"],
            "short_name": r["short_name"],
            "profiles": r["profile_names"],
            "is_active": r["is_active"],
            "ege_last_year": r["last_year"],
            "ege_years": r["ege_years"],
            "ege_avg_graduates": float(r["avg_graduates"]) if r["avg_graduates"] is not None else None,
            "ege_avg_score": float(r["avg_score"]) if r["avg_score"] is not None else None,
        }
        for r in rows
    ]
    df = pd.DataFrame(export_rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "school_id",
                "region",
                "municipality",
                "school",
                "short_name",
                "profiles",
                "is_active",
                "ege_last_year",
                "ege_years",
                "ege_avg_graduates",
                "ege_avg_score",
            ]
        )

    return _excel_response(
        filename="school_search_export.xlsx",
        sheets=[("search", df)],
    )


@router.get("/search/school/{school_id}/export")
def school_card_export(request: Request, school_id: int, db: Session = Depends(get_db)):
    require_user(request, db)
    data = fetch_school_card(school_id)
    if data is None:
        return RedirectResponse("/search", status_code=303)

    school = data["school"]
    school_df = pd.DataFrame(
        [
            {
                "school_id": school.get("school_id"),
                "school_name": school.get("full_name"),
                "region_id": school.get("region_id"),
                "region_name": school.get("region_name"),
                "municipality_id": school.get("municipality_id"),
                "municipality_name": school.get("municipality_name"),
                "is_active": school.get("is_active"),
            }
        ]
    )

    profiles_df = pd.DataFrame([{"profile_name": p} for p in (school.get("profiles") or [])])
    if profiles_df.empty:
        profiles_df = pd.DataFrame(columns=["profile_name"])

    external_df = pd.DataFrame(data["external_keys"])
    if external_df.empty:
        external_df = pd.DataFrame(columns=["source_name", "external_key", "normalized_name"])

    admission_df = pd.DataFrame(data["admission_stats"])
    if admission_df.empty:
        admission_df = pd.DataFrame(columns=["year", "applicants_cnt", "enrolled_cnt", "enrolled_avg_score"])

    prof_events_df = pd.DataFrame(data["prof_events"])
    if prof_events_df.empty:
        prof_events_df = pd.DataFrame(columns=["year", "events_cnt", "coverage_cnt"])

    ege_subject_rows: list[dict[str, object]] = []
    ege_period_rows: list[dict[str, object]] = []

    for period in data["ege_timeline"]:
        subjects = period.get("subjects") or []
        participants_total = sum(int(s.get("participants_cnt") or 0) for s in subjects)
        not_passed_total = sum(int(s.get("not_passed_cnt") or 0) for s in subjects)
        high_80_99_total = sum(int(s.get("high_80_99_cnt") or 0) for s in subjects)
        score_100_total = sum(int(s.get("score_100_cnt") or 0) for s in subjects)
        chosen_total = sum(int(s.get("chosen_cnt") or 0) for s in subjects)
        avg_values = [float(s["avg_score"]) for s in subjects if s.get("avg_score") is not None]
        avg_score_mean = round(sum(avg_values) / len(avg_values), 2) if avg_values else None

        ege_period_rows.append(
            {
                "year": period.get("year"),
                "kind": period.get("kind"),
                "graduates_total": period.get("graduates_total"),
                "subjects_cnt": len(subjects),
                "participants_total": participants_total,
                "not_passed_total": not_passed_total,
                "high_80_99_total": high_80_99_total,
                "score_100_total": score_100_total,
                "chosen_total": chosen_total,
                "avg_score_mean": avg_score_mean,
            }
        )

        if not subjects:
            ege_subject_rows.append(
                {
                    "year": period.get("year"),
                    "kind": period.get("kind"),
                    "graduates_total": period.get("graduates_total"),
                    "subject_name": None,
                    "participants_cnt": None,
                    "not_passed_cnt": None,
                    "high_80_99_cnt": None,
                    "score_100_cnt": None,
                    "avg_score": None,
                    "chosen_cnt": None,
                }
            )
            continue

        for subject in subjects:
            ege_subject_rows.append(
                {
                    "year": period.get("year"),
                    "kind": period.get("kind"),
                    "graduates_total": period.get("graduates_total"),
                    "subject_name": subject.get("subject_name"),
                    "participants_cnt": subject.get("participants_cnt"),
                    "not_passed_cnt": subject.get("not_passed_cnt"),
                    "high_80_99_cnt": subject.get("high_80_99_cnt"),
                    "score_100_cnt": subject.get("score_100_cnt"),
                    "avg_score": subject.get("avg_score"),
                    "chosen_cnt": subject.get("chosen_cnt"),
                }
            )

    ege_periods_df = pd.DataFrame(ege_period_rows)
    if ege_periods_df.empty:
        ege_periods_df = pd.DataFrame(
            columns=[
                "year",
                "kind",
                "graduates_total",
                "subjects_cnt",
                "participants_total",
                "not_passed_total",
                "high_80_99_total",
                "score_100_total",
                "chosen_total",
                "avg_score_mean",
            ]
        )

    ege_subjects_df = pd.DataFrame(ege_subject_rows)
    if ege_subjects_df.empty:
        ege_subjects_df = pd.DataFrame(
            columns=[
                "year",
                "kind",
                "graduates_total",
                "subject_name",
                "participants_cnt",
                "not_passed_cnt",
                "high_80_99_cnt",
                "score_100_cnt",
                "avg_score",
                "chosen_cnt",
            ]
        )

    return _excel_response(
        filename=f"school_card_{school_id}.xlsx",
        sheets=[
            ("school", school_df),
            ("profiles", profiles_df),
            ("external_keys", external_df),
            ("ege_periods", ege_periods_df),
            ("ege_subjects", ege_subjects_df),
            ("admission", admission_df),
            ("prof_events", prof_events_df),
        ],
    )


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_subject_analytics(ege_timeline: list[dict[str, object]]) -> dict[str, object]:
    return build_subject_analytics(ege_timeline)


@router.get("/search/school/{school_id}")
def school_card(request: Request, school_id: int, db: Session = Depends(get_db)):
    user = require_user(request, db)
    src = (request.query_params.get("src") or "").strip().lower()
    back_base = "/rating/profile" if src == "rating" else "/search"
    back_params = [(k, v) for (k, v) in request.query_params.multi_items() if k != "src"]
    back_url = back_base
    if back_params:
        back_url = f"{back_base}?{urlencode(back_params, doseq=True)}"

    data = fetch_school_card(school_id)
    if data is None:
        return render(
            "school_card.html",
            {
                "request": request,
                "user": user,
                "school": None,
                "external_keys": [],
                "admission_stats": [],
                "prof_events": [],
                "ege_timeline": [],
                "subject_heatmap_period": "",
                "subject_heatmap_rows": [],
                "subject_histogram_rows": [],
                "subject_histogram_latest_year": None,
                "subject_histogram_prev_year": None,
                "back_url": back_url,
            },
        )

    subject_analytics = _build_subject_analytics(data["ege_timeline"])

    return render(
        "school_card.html",
        {
            "request": request,
            "user": user,
            "school": data["school"],
            "external_keys": data["external_keys"],
            "admission_stats": data["admission_stats"],
            "prof_events": data["prof_events"],
            "ege_timeline": data["ege_timeline"],
            "subject_heatmap_period": subject_analytics["heatmap_period_label"],
            "subject_heatmap_rows": subject_analytics["heatmap_rows"],
            "subject_histogram_rows": subject_analytics["histogram_rows"],
            "subject_histogram_latest_year": subject_analytics["histogram_latest_year"],
            "subject_histogram_prev_year": subject_analytics["histogram_prev_year"],
            "back_url": back_url,
        },
    )


# Раздел 4. Подбор и рейтинг.
@dataclass(frozen=True)
class RatingRequestParams:
    q: str
    region_id: int | None
    municipality_id: int | None
    profile_ids: list[int]
    year: int | None
    kind: str
    subject_ids: list[int]
    institute_ids: list[int]
    program_ids: list[int]
    min_graduates: int | None
    min_avg_score: float | None
    enforce_subject_threshold: bool
    w_graduates: float
    w_avg_score: float
    w_match_share: float
    w_threshold_share: float
    limit: int
    show_admission_subject_avg: bool


@dataclass(frozen=True)
class RatingQueryContext:
    filters: RatingFilters
    weights: RatingWeights
    current_query: str
    show_admission_subject_avg: bool
    show_potential_applicants: bool


def _parse_bool_flag(value: str | None) -> bool:
    return (value or "").strip().lower() not in {"", "0", "false", "off", "no"}


def _has_rating_scope(subject_ids: list[int], institute_ids: list[int], program_ids: list[int]) -> bool:
    return bool(subject_ids) or bool(institute_ids) or bool(program_ids)


def _parse_rating_request_params(
    *,
    q: str,
    region_id: str | None,
    municipality_id: str | None,
    profile_ids: list[str] | None,
    year: str | None,
    kind: str,
    subject_ids: list[str] | None,
    legacy_subject_id: str | None,
    institute_ids: list[str] | None,
    legacy_institute_id: str | None,
    program_ids: list[str] | None,
    min_graduates: str | None,
    min_avg_score: str | None,
    enforce_subject_threshold: str | None,
    w_graduates: str | None,
    w_avg_score: str | None,
    w_match_share: str | None,
    w_threshold_share: str | None,
    limit: str | None,
) -> RatingRequestParams:
    region_id_value = _parse_optional_int(region_id)
    municipality_id_value = _parse_optional_int(municipality_id)
    year_value = _parse_optional_int(year)
    profile_ids_value = _parse_int_list(profile_ids)

    subject_ids_value = _parse_int_list(subject_ids)
    legacy_subject_id_value = _parse_optional_int(legacy_subject_id)
    if legacy_subject_id_value is not None:
        subject_ids_value.append(legacy_subject_id_value)
    subject_ids_value = sorted(set(subject_ids_value))

    institute_ids_value = _parse_int_list(institute_ids)
    legacy_institute_id_value = _parse_optional_int(legacy_institute_id)
    if legacy_institute_id_value is not None:
        institute_ids_value.append(legacy_institute_id_value)
    institute_ids_value = sorted(set(institute_ids_value))
    program_ids_value = _parse_int_list(program_ids)

    min_graduates_value = _parse_optional_int(min_graduates)
    if min_graduates_value is not None and min_graduates_value < 0:
        min_graduates_value = 0

    min_avg_score_value = _parse_optional_float(min_avg_score)
    if min_avg_score_value is not None:
        min_avg_score_value = max(0.0, min(100.0, min_avg_score_value))

    w_graduates_value = _parse_optional_float(w_graduates)
    w_avg_score_value = _parse_optional_float(w_avg_score)
    w_match_share_value = _parse_optional_float(w_match_share)
    w_threshold_share_value = _parse_optional_float(w_threshold_share)
    if w_graduates_value is None:
        w_graduates_value = DEFAULT_RATING_W_GRADUATES
    if w_avg_score_value is None:
        w_avg_score_value = DEFAULT_RATING_W_AVG_SCORE
    if w_match_share_value is None:
        w_match_share_value = DEFAULT_RATING_W_MATCH_SHARE
    if w_threshold_share_value is None:
        w_threshold_share_value = DEFAULT_RATING_W_THRESHOLD_SHARE

    limit_value = _parse_bounded_int(
        limit,
        default=DEFAULT_RATING_LIMIT,
        min_value=MIN_RATING_LIMIT,
        max_value=MAX_RATING_LIMIT,
    )

    show_admission_subject_avg = _has_rating_scope(subject_ids_value, institute_ids_value, program_ids_value)

    return RatingRequestParams(
        q=(q or "").strip(),
        region_id=region_id_value,
        municipality_id=municipality_id_value,
        profile_ids=profile_ids_value,
        year=year_value,
        kind=(kind or "").strip(),
        subject_ids=subject_ids_value,
        institute_ids=institute_ids_value,
        program_ids=program_ids_value,
        min_graduates=min_graduates_value,
        min_avg_score=min_avg_score_value,
        enforce_subject_threshold=_parse_bool_flag(enforce_subject_threshold),
        w_graduates=w_graduates_value,
        w_avg_score=w_avg_score_value,
        w_match_share=w_match_share_value,
        w_threshold_share=w_threshold_share_value,
        limit=limit_value,
        show_admission_subject_avg=show_admission_subject_avg,
    )


def _build_rating_filters_from_params(params: RatingRequestParams) -> RatingFilters:
    return RatingFilters(
        q=params.q,
        region_id=params.region_id,
        municipality_id=params.municipality_id,
        profile_ids=tuple(sorted(set(params.profile_ids))),
        year=params.year,
        kind=params.kind or None,
        subject_ids=tuple(sorted(set(params.subject_ids))),
        institute_ids=tuple(sorted(set(params.institute_ids))),
        program_ids=tuple(sorted(set(params.program_ids))),
        min_graduates=params.min_graduates,
        min_avg_score=params.min_avg_score,
        enforce_subject_threshold=params.enforce_subject_threshold,
        limit=params.limit,
    )


def _build_rating_weights_from_params(params: RatingRequestParams) -> RatingWeights:
    return RatingWeights(
        graduates=max(0.0, params.w_graduates),
        avg_score=max(0.0, params.w_avg_score),
        match_share=max(0.0, params.w_match_share),
        threshold_share=max(0.0, params.w_threshold_share),
    )


def _build_rating_query_from_params(params: RatingRequestParams) -> str:
    return _build_rating_query(
        q=params.q,
        region_id=params.region_id,
        municipality_id=params.municipality_id,
        profile_ids=params.profile_ids,
        year=params.year,
        kind=params.kind,
        subject_ids=params.subject_ids,
        institute_ids=params.institute_ids,
        program_ids=params.program_ids,
        min_graduates=params.min_graduates,
        min_avg_score=params.min_avg_score,
        enforce_subject_threshold=params.enforce_subject_threshold,
        w_graduates=params.w_graduates,
        w_avg_score=params.w_avg_score,
        w_match_share=params.w_match_share,
        w_threshold_share=params.w_threshold_share,
        limit=params.limit,
    )


def _build_rating_context(params: RatingRequestParams) -> RatingQueryContext:
    has_scope = _has_rating_scope(params.subject_ids, params.institute_ids, params.program_ids)
    return RatingQueryContext(
        filters=_build_rating_filters_from_params(params),
        weights=_build_rating_weights_from_params(params),
        current_query=_build_rating_query_from_params(params),
        show_admission_subject_avg=has_scope,
        show_potential_applicants=has_scope,
    )


def _build_rating_query(
    *,
    q: str,
    region_id: int | None,
    municipality_id: int | None,
    profile_ids: list[int],
    year: int | None,
    kind: str,
    subject_ids: list[int],
    institute_ids: list[int],
    program_ids: list[int],
    min_graduates: int | None,
    min_avg_score: float | None,
    enforce_subject_threshold: bool,
    w_graduates: float,
    w_avg_score: float,
    w_match_share: float,
    w_threshold_share: float,
    limit: int,
) -> str:
    params: list[tuple[str, str]] = []
    if q:
        params.append(("q", q))
    if region_id is not None:
        params.append(("region_id", str(region_id)))
    if municipality_id is not None:
        params.append(("municipality_id", str(municipality_id)))
    for pid in sorted(set(profile_ids)):
        params.append(("profile_ids", str(pid)))
    if year is not None:
        params.append(("year", str(year)))
    if kind:
        params.append(("kind", kind))
    for sid in sorted(set(subject_ids)):
        params.append(("subject_ids", str(sid)))
    for iid in sorted(set(institute_ids)):
        params.append(("institute_ids", str(iid)))
    for program_id in sorted(set(program_ids)):
        params.append(("program_ids", str(program_id)))
    if min_graduates is not None:
        params.append(("min_graduates", str(min_graduates)))
    if min_avg_score is not None:
        params.append(("min_avg_score", str(min_avg_score)))
    if enforce_subject_threshold:
        params.append(("enforce_subject_threshold", "1"))
    params.append(("w_graduates", str(w_graduates)))
    params.append(("w_avg_score", str(w_avg_score)))
    params.append(("w_match_share", str(w_match_share)))
    params.append(("w_threshold_share", str(w_threshold_share)))
    params.append(("limit", str(limit)))
    return urlencode(params, doseq=True)


@router.get("/rating/programs")
def rating_programs_api(
    request: Request,
    institute_ids: list[int] = Query(default=[]),
    institute_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
):
    require_user(request, db)
    selected_ids = list(institute_ids)
    if institute_id is not None:
        selected_ids.append(institute_id)
    selected_ids = sorted(set(selected_ids))
    items = fetch_programs(tuple(selected_ids))
    return JSONResponse({"items": items})


@router.get("/rating/profile")
def rating_profile(
    request: Request,
    q: str = "",
    region_id: str | None = Query(default=None),
    municipality_id: str | None = Query(default=None),
    profile_ids: list[str] = Query(default=[]),
    year: str | None = Query(default=None),
    kind: str = "",
    subject_ids: list[str] = Query(default=[]),
    institute_ids: list[str] = Query(default=[]),
    institute_id: str | None = Query(default=None),
    program_ids: list[str] = Query(default=[]),
    min_graduates: str | None = Query(default="0"),
    min_avg_score: str | None = Query(default=""),
    enforce_subject_threshold: str | None = Query(default=None),
    w_graduates: str | None = Query(default=str(DEFAULT_RATING_W_GRADUATES)),
    w_avg_score: str | None = Query(default=str(DEFAULT_RATING_W_AVG_SCORE)),
    w_match_share: str | None = Query(default=str(DEFAULT_RATING_W_MATCH_SHARE)),
    w_threshold_share: str | None = Query(default=str(DEFAULT_RATING_W_THRESHOLD_SHARE)),
    limit: str | None = Query(default=str(DEFAULT_RATING_LIMIT)),
    save_status: str | None = Query(default=None),
    saved_request_id: str | None = Query(default=None),
    saved_run_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)

    params = _parse_rating_request_params(
        q=q,
        region_id=region_id,
        municipality_id=municipality_id,
        profile_ids=profile_ids,
        year=year,
        kind=kind,
        subject_ids=subject_ids,
        legacy_subject_id=None,
        institute_ids=institute_ids,
        legacy_institute_id=institute_id,
        program_ids=program_ids,
        min_graduates=min_graduates,
        min_avg_score=min_avg_score,
        enforce_subject_threshold=enforce_subject_threshold,
        w_graduates=w_graduates,
        w_avg_score=w_avg_score,
        w_match_share=w_match_share,
        w_threshold_share=w_threshold_share,
        limit=limit,
    )

    rating_ctx = _build_rating_context(params)
    region_id_value = params.region_id
    municipality_id_value = params.municipality_id
    year_value = params.year
    profile_ids_value = params.profile_ids
    subject_ids_value = params.subject_ids
    institute_ids_value = params.institute_ids
    program_ids_value = params.program_ids
    min_graduates_value = params.min_graduates
    min_avg_score_value = params.min_avg_score
    enforce_threshold_value = params.enforce_subject_threshold
    w_graduates_value = params.w_graduates
    w_avg_score_value = params.w_avg_score
    w_match_share_value = params.w_match_share
    w_threshold_share_value = params.w_threshold_share
    limit_value = params.limit
    show_admission_subject_avg = rating_ctx.show_admission_subject_avg
    show_potential_applicants = rating_ctx.show_potential_applicants
    has_query = _has_known_query_params(request, RATING_QUERY_KEYS)
    save_status_value = (save_status or "").strip().lower()
    saved_request_id_value = _parse_optional_int(saved_request_id)
    saved_run_id_value = _parse_optional_int(saved_run_id)
    save_message: str | None = None
    save_message_kind = "ok"
    if save_status_value == "ok":
        if saved_run_id_value is not None:
            save_message = f"Рейтинг сохранен в системе (run_id={saved_run_id_value})."
        elif saved_request_id_value is not None:
            save_message = f"Рейтинг сохранен в системе (request_id={saved_request_id_value})."
        else:
            save_message = "Рейтинг сохранен в системе."
    elif save_status_value == "empty":
        save_message_kind = "alert"
        save_message = "Нечего сохранять: по выбранным фильтрам нет данных."
    elif save_status_value == "error":
        save_message_kind = "alert"
        save_message = "Не удалось сохранить рейтинг. Проверьте лог приложения."

    options = fetch_rating_filter_options(region_id=region_id_value, institute_ids=tuple(institute_ids_value))
    program_requirements = fetch_program_requirements(
        institute_ids=tuple(institute_ids_value),
        program_ids=tuple(sorted(set(program_ids_value))),
    )

    ranked: list[dict[str, object]] = []
    if has_query:
        ranked = calculate_school_rating(
            rating_ctx.filters,
            rating_ctx.weights,
        )
    rating_chart_stacked: list[dict[str, object]] = []
    if ranked:
        for row in ranked[:10]:
            part_graduates = float(_to_float_or_none(row.get("score_part_graduates")) or 0.0)
            part_avg_score = float(_to_float_or_none(row.get("score_part_avg_score")) or 0.0)
            part_potential = float(_to_float_or_none(row.get("score_part_potential")) or 0.0)
            part_threshold = float(_to_float_or_none(row.get("score_part_threshold")) or 0.0)
            score_total = part_graduates + part_avg_score + part_potential + part_threshold
            rating_chart_stacked.append(
                {
                    "rank_pos": int(_to_int_or_none(row.get("rank_pos")) or 0),
                    "name": str(row.get("full_name") or ""),
                    "score_total": round(score_total, 6),
                    "score_part_graduates": round(part_graduates, 6),
                    "score_part_avg_score": round(part_avg_score, 6),
                    "score_part_potential": round(part_potential, 6),
                    "score_part_threshold": round(part_threshold, 6),
                }
            )

    current_query = rating_ctx.current_query

    return render(
        "rating_profile.html",
        {
            "request": request,
            "user": user,
            "main_class": "container-rating",
            "has_query": has_query,
            "show_admission_subject_avg": show_admission_subject_avg,
            "show_potential_applicants": show_potential_applicants,
            "rating_chart_stacked": rating_chart_stacked,
            "ranked": ranked,
            "total": len(ranked),
            "program_requirements": program_requirements,
            "current_query": current_query,
            "save_message": save_message,
            "save_message_kind": save_message_kind,
            "saved_request_id": saved_request_id_value,
            "saved_run_id": saved_run_id_value,
            "filters": {
                "q": params.q,
                "region_id": region_id_value,
                "municipality_id": municipality_id_value,
                "profile_ids": profile_ids_value,
                "year": year_value,
                "kind": params.kind,
                "subject_ids": subject_ids_value,
                "institute_ids": institute_ids_value,
                "program_ids": program_ids_value,
                "min_graduates": min_graduates_value if min_graduates_value is not None else 0,
                "min_avg_score": min_avg_score_value if min_avg_score_value is not None else "",
                "enforce_subject_threshold": enforce_threshold_value,
                "w_graduates": w_graduates_value,
                "w_avg_score": w_avg_score_value,
                "w_match_share": w_match_share_value,
                "w_threshold_share": w_threshold_share_value,
                "limit": limit_value,
            },
            "regions": options["regions"],
            "municipalities": options["municipalities"],
            "profiles": options["profiles"],
            "years": options["years"],
            "kinds": options["kinds"],
            "subjects": options["subjects"],
            "institutes": options["institutes"],
            "programs": options["programs"],
        },
    )


@router.post("/rating/profile/save")
async def rating_profile_save_action(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    form = await request.form()
    validate_csrf_token(request, str(form.get("csrf_token", "")))

    params = _parse_rating_request_params(
        q=str(form.get("q", "")),
        region_id=str(form.get("region_id", "")),
        municipality_id=str(form.get("municipality_id", "")),
        profile_ids=form.getlist("profile_ids"),
        year=str(form.get("year", "")),
        kind=str(form.get("kind", "")),
        subject_ids=form.getlist("subject_ids"),
        legacy_subject_id=str(form.get("subject_id", "")),
        institute_ids=form.getlist("institute_ids"),
        legacy_institute_id=str(form.get("institute_id", "")),
        program_ids=form.getlist("program_ids"),
        min_graduates=str(form.get("min_graduates", "0")),
        min_avg_score=str(form.get("min_avg_score", "")),
        enforce_subject_threshold=str(form.get("enforce_subject_threshold", "")),
        w_graduates=str(form.get("w_graduates", str(DEFAULT_RATING_W_GRADUATES))),
        w_avg_score=str(form.get("w_avg_score", str(DEFAULT_RATING_W_AVG_SCORE))),
        w_match_share=str(form.get("w_match_share", str(DEFAULT_RATING_W_MATCH_SHARE))),
        w_threshold_share=str(form.get("w_threshold_share", str(DEFAULT_RATING_W_THRESHOLD_SHARE))),
        limit=str(form.get("limit", str(DEFAULT_RATING_LIMIT))),
    )
    rating_ctx = _build_rating_context(params)
    current_query = rating_ctx.current_query

    q = params.q
    kind = params.kind
    region_id_value = params.region_id
    municipality_id_value = params.municipality_id
    year_value = params.year
    profile_ids_value = params.profile_ids
    subject_ids_value = params.subject_ids
    institute_ids_value = params.institute_ids
    program_ids_value = params.program_ids
    min_graduates_value = params.min_graduates
    min_avg_score_value = params.min_avg_score
    enforce_threshold_value = params.enforce_subject_threshold
    w_graduates_value = params.w_graduates
    w_avg_score_value = params.w_avg_score
    w_match_share_value = params.w_match_share
    w_threshold_share_value = params.w_threshold_share
    limit_value = params.limit

    save_status = "error"
    saved_request_id_value: int | None = None
    saved_run_id_value: int | None = None
    try:
        ranked = calculate_school_rating(
            rating_ctx.filters,
            rating_ctx.weights,
        )
        saved = save_rating_run(
            created_by=user.login,
            filters={
                "q": q,
                "region_id": region_id_value,
                "municipality_id": municipality_id_value,
                "profile_ids": profile_ids_value,
                "year": year_value,
                "kind": kind,
                "subject_ids": subject_ids_value,
                "institute_ids": institute_ids_value,
                "program_ids": program_ids_value,
                "min_graduates": min_graduates_value,
                "min_avg_score": min_avg_score_value,
                "enforce_subject_threshold": enforce_threshold_value,
                "limit": limit_value,
            },
            weights={
                "w_graduates": w_graduates_value,
                "w_avg_score": w_avg_score_value,
                "w_match_share": w_match_share_value,
                "w_threshold_share": w_threshold_share_value,
            },
            ranked_rows=ranked,
        )
        if saved:
            saved_request_id_value = int(saved.get("request_id", 0) or 0) or None
            saved_run_id_value = int(saved.get("run_id", 0) or 0) or None
            save_status = "ok" if saved_run_id_value is not None else "empty"
        else:
            save_status = "empty"
    except (ValueError, RuntimeError, TimeoutError) as exc:
        logger.warning("Не удалось сохранить рейтинг: %s", exc)
        save_status = "error"

    status_params: list[tuple[str, str]] = [("save_status", save_status)]
    if saved_request_id_value is not None:
        status_params.append(("saved_request_id", str(saved_request_id_value)))
    if saved_run_id_value is not None:
        status_params.append(("saved_run_id", str(saved_run_id_value)))
    status_query = urlencode(status_params, doseq=True)

    if current_query:
        location = f"/rating/profile?{current_query}&{status_query}"
    else:
        location = f"/rating/profile?{status_query}"
    return RedirectResponse(location, status_code=303)


@router.get("/rating/export")
def rating_export(
    request: Request,
    q: str = "",
    region_id: str | None = Query(default=None),
    municipality_id: str | None = Query(default=None),
    profile_ids: list[str] = Query(default=[]),
    year: str | None = Query(default=None),
    kind: str = "",
    subject_ids: list[str] = Query(default=[]),
    institute_ids: list[str] = Query(default=[]),
    institute_id: str | None = Query(default=None),
    program_ids: list[str] = Query(default=[]),
    min_graduates: str | None = Query(default="0"),
    min_avg_score: str | None = Query(default=""),
    enforce_subject_threshold: str | None = Query(default=None),
    w_graduates: str | None = Query(default=str(DEFAULT_RATING_W_GRADUATES)),
    w_avg_score: str | None = Query(default=str(DEFAULT_RATING_W_AVG_SCORE)),
    w_match_share: str | None = Query(default=str(DEFAULT_RATING_W_MATCH_SHARE)),
    w_threshold_share: str | None = Query(default=str(DEFAULT_RATING_W_THRESHOLD_SHARE)),
    limit: str | None = Query(default=str(DEFAULT_RATING_LIMIT)),
    db: Session = Depends(get_db),
):
    require_user(request, db)

    params = _parse_rating_request_params(
        q=q,
        region_id=region_id,
        municipality_id=municipality_id,
        profile_ids=profile_ids,
        year=year,
        kind=kind,
        subject_ids=subject_ids,
        legacy_subject_id=None,
        institute_ids=institute_ids,
        legacy_institute_id=institute_id,
        program_ids=program_ids,
        min_graduates=min_graduates,
        min_avg_score=min_avg_score,
        enforce_subject_threshold=enforce_subject_threshold,
        w_graduates=w_graduates,
        w_avg_score=w_avg_score,
        w_match_share=w_match_share,
        w_threshold_share=w_threshold_share,
        limit=limit,
    )

    rating_ctx = _build_rating_context(params)
    region_id_value = params.region_id
    municipality_id_value = params.municipality_id
    year_value = params.year
    profile_ids_value = params.profile_ids
    subject_ids_value = params.subject_ids
    institute_ids_value = params.institute_ids
    program_ids_value = params.program_ids
    min_graduates_value = params.min_graduates
    min_avg_score_value = params.min_avg_score
    enforce_threshold_value = params.enforce_subject_threshold
    w_graduates_value = params.w_graduates
    w_avg_score_value = params.w_avg_score
    w_match_share_value = params.w_match_share
    w_threshold_share_value = params.w_threshold_share
    limit_value = params.limit
    show_potential_applicants = rating_ctx.show_potential_applicants
    has_query = bool(request.query_params)

    ranked: list[dict[str, object]] = []
    if has_query:
        ranked = calculate_school_rating(
            rating_ctx.filters,
            rating_ctx.weights,
        )

    export_rows: list[dict[str, object]] = []
    for r in ranked:
        row: dict[str, object] = {
            "rank_pos": r.get("rank_pos"),
            "school_id": r.get("school_id"),
            "region_name": r.get("region_name"),
            "municipality_name": r.get("municipality_name"),
            "full_name": r.get("full_name"),
            "profile_names": r.get("profile_names"),
            "avg_graduates": r.get("avg_graduates"),
            "avg_score_all": r.get("avg_score_all"),
            "ege_years": r.get("ege_years"),
            "last_year": r.get("last_year"),
            "programs_total": r.get("programs_total"),
            "programs_matched": r.get("programs_matched"),
            "threshold_share_pct": round(float(r.get("threshold_share", 0)) * 100.0, 2),
            "rating_score": r.get("rating_score"),
            "matched_programs": r.get("matched_programs"),
        }
        if show_potential_applicants:
            row["potential_applicants_avg"] = r.get("potential_applicants_avg")
        export_rows.append(row)
    df = pd.DataFrame(export_rows)
    if df.empty:
        rating_columns = [
            "rank_pos",
            "school_id",
            "region_name",
            "municipality_name",
            "full_name",
            "profile_names",
            "avg_graduates",
            "avg_score_all",
            "ege_years",
            "last_year",
            "programs_total",
            "programs_matched",
        ]
        if show_potential_applicants:
            rating_columns.append("potential_applicants_avg")
        rating_columns.extend(
            [
                "threshold_share_pct",
                "rating_score",
                "matched_programs",
            ]
        )
        df = pd.DataFrame(
            columns=rating_columns
        )

    filters_df = pd.DataFrame(
        [
            {"key": "q", "value": params.q},
            {"key": "region_id", "value": region_id_value},
            {"key": "municipality_id", "value": municipality_id_value},
            {"key": "profile_ids", "value": ",".join(str(v) for v in sorted(set(profile_ids_value)))},
            {"key": "year", "value": year_value},
            {"key": "kind", "value": params.kind},
            {"key": "subject_ids", "value": ",".join(str(v) for v in sorted(set(subject_ids_value)))},
            {"key": "institute_ids", "value": ",".join(str(v) for v in sorted(set(institute_ids_value)))},
            {"key": "program_ids", "value": ",".join(str(v) for v in sorted(set(program_ids_value)))},
            {"key": "min_graduates", "value": min_graduates_value},
            {"key": "min_avg_score", "value": min_avg_score_value},
            {"key": "enforce_subject_threshold", "value": enforce_threshold_value},
            {"key": "w_graduates", "value": w_graduates_value},
            {"key": "w_avg_score", "value": w_avg_score_value},
            {"key": "w_match_share", "value": w_match_share_value},
            {"key": "w_threshold_share", "value": w_threshold_share_value},
            {"key": "limit", "value": limit_value},
        ]
    )

    return _excel_response(
        filename="school_rating_export.xlsx",
        sheets=[("rating", df), ("filters", filters_df)],
    )


# Раздел 5. Настройки анализа (заглушки).
@router.get("/settings")
def settings_home(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("settings_home.html", {"request": request, "user": user})


@router.get("/settings/filters")
def settings_filters(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "Конструктор фильтров", "back_url": "/settings"})


@router.get("/settings/weights")
def settings_weights(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "Метрики и веса", "back_url": "/settings"})


@router.get("/settings/profiles")
def settings_profiles(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "Профили расчёта", "back_url": "/settings"})


# Раздел 6. Отчётность.
@router.get("/reports")
def reports_home(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("reports_home.html", {"request": request, "user": user})


def _render_report_generate_page(
    *,
    request: Request,
    user: User,
    selected_run_id: int | None,
    selected_school_id: int | None = None,
    report_type: str = "standard",
    report_format: str = "json",
    error: str | None = None,
    ok: str | None = None,
):
    page_error = error
    runs: list[dict[str, object]] = []
    try:
        runs = fetch_rating_runs(limit=200)
    except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
        logger.warning("Не удалось загрузить список расчетов: %s", exc)
        page_error = page_error or f"Не удалось загрузить список расчетов: {exc}"
    except Exception as exc:
        logger.exception("Непредвиденная ошибка при загрузке списка расчетов")
        page_error = page_error or f"Не удалось загрузить список расчетов: {exc}"

    run_ids = {int(item["run_id"]) for item in runs}
    if selected_run_id is not None and selected_run_id not in run_ids:
        selected_run_id = None
    if selected_run_id is None and runs:
        selected_run_id = int(runs[0]["run_id"])

    selected_run = (
        next((item for item in runs if int(item["run_id"]) == selected_run_id), None)
        if selected_run_id is not None
        else None
    )

    run_schools: list[dict[str, object]] = []
    if selected_run_id is not None:
        try:
            run_schools = fetch_run_schools(selected_run_id, limit=3000)
        except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
            logger.warning("Не удалось загрузить школы для run_id=%s: %s", selected_run_id, exc)
            page_error = page_error or f"Не удалось загрузить школы для расчета #{selected_run_id}: {exc}"
        except Exception as exc:
            logger.exception("Непредвиденная ошибка при загрузке школ для run_id=%s", selected_run_id)
            page_error = page_error or f"Не удалось загрузить школы для расчета #{selected_run_id}: {exc}"

    return render(
        "report_generate.html",
        {
            "request": request,
            "user": user,
            "runs": runs,
            "selected_run": selected_run,
            "selected_run_id": selected_run_id,
            "run_schools": run_schools,
            "selected_school_id": selected_school_id,
            "report_type": report_type,
            "report_format": report_format,
            "error": page_error,
            "ok": ok,
        },
    )


@router.get("/reports/generate")
def report_generate(
    request: Request,
    run_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    selected_run_id = _parse_optional_int(run_id)
    return _render_report_generate_page(
        request=request,
        user=user,
        selected_run_id=selected_run_id,
    )


@router.post("/reports/generate")
def report_generate_action(
    request: Request,
    run_id: str = Form(""),
    school_id: str = Form(""),
    report_type: str = Form("standard"),
    report_format: str = Form("json"),
    csrf_token: str = Form(""),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    validate_csrf_token(request, csrf_token)
    run_id_value = _parse_optional_int(run_id)
    school_id_value = _parse_optional_int(school_id)

    if run_id_value is None:
        return _render_report_generate_page(
            request=request,
            user=user,
            selected_run_id=None,
            selected_school_id=school_id_value,
            report_type=report_type,
            report_format=report_format,
            error="Выберите расчет (run_id).",
        )

    try:
        created_count = create_reports_for_run(
            run_id=run_id_value,
            created_by=user.login,
            report_type=report_type,
            report_format=report_format,
            school_id=school_id_value,
        )
    except (ValueError, RuntimeError, TimeoutError) as exc:
        logger.warning("Не удалось сформировать отчет: %s", exc)
        return _render_report_generate_page(
            request=request,
            user=user,
            selected_run_id=run_id_value,
            selected_school_id=school_id_value,
            report_type=report_type,
            report_format=report_format,
            error="Не удалось сформировать отчет. Проверьте лог приложения.",
        )

    if created_count <= 0:
        return _render_report_generate_page(
            request=request,
            user=user,
            selected_run_id=run_id_value,
            selected_school_id=school_id_value,
            report_type=report_type,
            report_format=report_format,
            error="Для выбранного run_id нет школ для отчета.",
        )

    return RedirectResponse(
        f"/reports/archive?generated={created_count}&run_id={run_id_value}",
        status_code=303,
    )


@router.get("/reports/run/{run_id}/export")
def report_run_export(request: Request, run_id: int, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    payload = fetch_run_export_data(run_id)
    if payload is None:
        return RedirectResponse("/reports/generate", status_code=303)

    meta = payload["meta"]
    ranking_rows = payload["ranking_rows"]
    filter_rows = payload["filter_rows"]
    weight_rows = payload["weight_rows"]

    ranking_df = pd.DataFrame(ranking_rows)
    if ranking_df.empty:
        ranking_df = pd.DataFrame(
            columns=[
                "rank_pos",
                "school_id",
                "full_name",
                "region_name",
                "municipality_name",
                "total_score",
                "students_cnt",
                "ege_avg_all",
            ]
        )

    filters_df = pd.DataFrame(filter_rows)
    if filters_df.empty:
        filters_df = pd.DataFrame(
            columns=[
                "filter_type",
                "region_id",
                "municipality_id",
                "institute_id",
                "profile_id",
                "program_id",
                "subject_id",
                "school_id",
                "min_score",
            ]
        )

    weights_df = pd.DataFrame(weight_rows)
    if weights_df.empty:
        weights_df = pd.DataFrame(columns=["metric_code", "weight", "is_primary"])

    meta_items = []
    for key, value in meta.items():
        if isinstance(value, (dict, list)):
            value_repr = json.dumps(value, ensure_ascii=False)
        else:
            value_repr = value
        meta_items.append({"key": key, "value": value_repr})
    meta_df = pd.DataFrame(meta_items)
    return _excel_response(
        filename=f"rating_run_{run_id}.xlsx",
        sheets=[
            ("rating_run", ranking_df),
            ("filters", filters_df),
            ("weights", weights_df),
            ("meta", meta_df),
        ],
    )


@router.get("/reports/report/{report_id}/export")
def report_payload_export(request: Request, report_id: int, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    payload = get_report_payload(report_id)
    if payload is None:
        return RedirectResponse("/reports/archive", status_code=303)

    report_format = str(payload.get("report_format") or "json").strip().lower()
    report_payload = payload.get("report_payload") or {}
    if report_format == "xlsx":
        summary_df = pd.DataFrame(
            [
                {"key": "report_id", "value": payload.get("report_id")},
                {"key": "generated_at", "value": payload.get("generated_at")},
                {"key": "report_type", "value": payload.get("report_type")},
                {"key": "report_format", "value": payload.get("report_format")},
                {"key": "run_id", "value": payload.get("run_id")},
                {"key": "school_id", "value": payload.get("school_id")},
                {"key": "full_name", "value": payload.get("full_name")},
            ]
        )

        payload_df = pd.json_normalize(report_payload, sep=".")
        if payload_df.empty:
            payload_df = pd.DataFrame([{"payload_json": json.dumps(report_payload, ensure_ascii=False, default=str)}])
        return _excel_response(
            filename=f"school_report_{report_id}.xlsx",
            sheets=[("summary", summary_df), ("payload", payload_df)],
        )

    content = json.dumps(report_payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    out = io.BytesIO(content)
    out.seek(0)
    headers = {"Content-Disposition": f"attachment; filename=school_report_{report_id}.json"}
    return StreamingResponse(
        out,
        media_type="application/json; charset=utf-8",
        headers=headers,
    )


@router.get("/reports/archive")
def report_archive(
    request: Request,
    generated: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    generated_value = _parse_optional_int(generated)
    run_id_value = _parse_optional_int(run_id)
    reports = fetch_report_archive_rows(limit=300)
    ok_message: str | None = None
    if generated_value is not None and generated_value > 0 and run_id_value is not None:
        ok_message = f"Сформировано отчетов: {generated_value} (run_id={run_id_value})."
    return render(
        "report_archive.html",
        {
            "request": request,
            "user": user,
            "reports": reports,
            "ok_message": ok_message,
        },
    )


@router.get("/reports/calc-history")
def calc_history(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    rows = fetch_calc_history_rows(limit=400)
    return render(
        "calc_history.html",
        {
            "request": request,
            "user": user,
            "rows": rows,
        },
    )


# Раздел 7. Администрирование (только администратор).
@router.get("/admin")
def admin_home(request: Request, db: Session = Depends(get_db)):
    user = require_admin(request, db)
    return render("admin_home.html", {"request": request, "user": user})


@router.get("/admin/roles")
def admin_roles(request: Request, db: Session = Depends(get_db)):
    user = require_admin(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "Роли и права", "back_url": "/admin"})


@router.get("/admin/directories")
def admin_directories(request: Request, db: Session = Depends(get_db)):
    require_admin(request, db)
    return RedirectResponse("/data/directories", status_code=303)


@router.get("/admin/methodologies")
def admin_methodologies(request: Request, db: Session = Depends(get_db)):
    user = require_admin(request, db)
    return render("stub_page.html", {"request": request, "user": user, "title": "Методики", "back_url": "/admin"})


