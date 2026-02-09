from __future__ import annotations

import io
import json
import traceback
from urllib.parse import urlencode

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

try:
    from .config import LOAD_PROGRAMS_SCRIPT, LOAD_SCHOOLS_SCRIPT, UPLOAD_DIR
    from .db import ImportJob, User, get_db
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
        verify_password,
    )
except ImportError:
    from config import LOAD_PROGRAMS_SCRIPT, LOAD_SCHOOLS_SCRIPT, UPLOAD_DIR
    from db import ImportJob, User, get_db
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
        verify_password,
    )

router = APIRouter()


# 0 Вход / 0.1 Восстановление / 1 Меню / 1.1 Выход
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
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.login == login).one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        return render("login.html", {"request": request, "user": None, "error": "Неверный логин или пароль."})

    request.session["user_id"] = user.id
    return RedirectResponse("/", status_code=303)


@router.get("/recovery")
def recovery_page(request: Request):
    return render("recovery.html", {"request": request, "user": None, "info": None})


@router.post("/recovery")
def recovery_action(request: Request, login: str = Form(...)):
    info = f"Запрос на восстановление для пользователя «{login}» зарегистрирован (демо-режим)."
    return render("recovery.html", {"request": request, "user": None, "info": info})


@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# 2 Данные и загрузки
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
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
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
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
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
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
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
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
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


@router.post("/data/directories/min-scores")
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


@router.get("/data/validate")
def validation_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    jobs = db.query(ImportJob).order_by(ImportJob.created_at.desc()).limit(20).all()
    return render("validation.html", {"request": request, "user": user, "jobs": jobs})


@router.get("/data/history")
def history_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    jobs = db.query(ImportJob).order_by(ImportJob.created_at.desc()).all()
    return render("history.html", {"request": request, "user": user, "jobs": jobs})


# 3 Поиск и карточка школы
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
    per_page: str | None = Query(default="20"),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)
    region_id_value = _parse_optional_int(region_id)
    municipality_id_value = _parse_optional_int(municipality_id)
    year_value = _parse_optional_int(year)
    subject_ids_value = _parse_int_list(subject_ids)
    legacy_subject_id = _parse_optional_int(subject_id)
    if legacy_subject_id is not None:
        subject_ids_value.append(legacy_subject_id)
    subject_ids_value = sorted(set(subject_ids_value))
    profile_ids_value = _parse_int_list(profile_ids)
    safe_page = _parse_bounded_int(page, default=1, min_value=1)
    safe_per_page = _parse_bounded_int(per_page, default=20, min_value=5, max_value=200)

    filters = _build_search_filters(
        q=q,
        region_id=region_id_value,
        municipality_id=municipality_id_value,
        profile_ids=profile_ids_value,
        year=year_value,
        kind=kind,
        subject_ids=subject_ids_value,
    )
    options = fetch_filter_options(region_id=region_id_value)
    results, total = search_schools(filters, page=safe_page, per_page=safe_per_page, apply_pagination=True)

    total_pages = max(1, (total + safe_per_page - 1) // safe_per_page)
    if safe_page > total_pages:
        safe_page = total_pages
        results, _ = search_schools(filters, page=safe_page, per_page=safe_per_page, apply_pagination=True)

    query_without_page = _build_search_query_without_page(
        q=q.strip(),
        region_id=region_id_value,
        municipality_id=municipality_id_value,
        profile_ids=profile_ids_value,
        year=year_value,
        kind=kind.strip(),
        subject_ids=subject_ids_value,
        per_page=safe_per_page,
    )
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
            "filters": {
                "q": q.strip(),
                "region_id": region_id_value,
                "municipality_id": municipality_id_value,
                "profile_ids": profile_ids_value,
                "year": year_value,
                "kind": kind.strip(),
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

    q = str(form.get("q", "")).strip()
    region_raw = str(form.get("region_id", "")).strip()
    municipality_raw = str(form.get("municipality_id", "")).strip()
    year_raw = str(form.get("year", "")).strip()
    kind = str(form.get("kind", "")).strip()
    subject_raw_values = form.getlist("subject_ids")
    subject_raw_legacy = str(form.get("subject_id", "")).strip()
    per_page_raw = str(form.get("per_page", "20")).strip()
    profile_raw_values = form.getlist("profile_ids")

    profile_ids = [int(v) for v in profile_raw_values if str(v).strip().isdigit()]
    subject_ids = [int(v) for v in subject_raw_values if str(v).strip().isdigit()]
    if subject_raw_legacy.isdigit():
        subject_ids.append(int(subject_raw_legacy))
    subject_ids = sorted(set(subject_ids))
    region_id = int(region_raw) if region_raw.isdigit() else None
    municipality_id = int(municipality_raw) if municipality_raw.isdigit() else None
    year = int(year_raw) if year_raw.isdigit() else None
    per_page = int(per_page_raw) if per_page_raw.isdigit() else 20
    per_page = max(5, min(per_page, 200))

    query_without_page = _build_search_query_without_page(
        q=q,
        region_id=region_id,
        municipality_id=municipality_id,
        profile_ids=profile_ids,
        year=year,
        kind=kind,
        subject_ids=subject_ids,
        per_page=per_page,
    )
    location = "/search"
    if query_without_page:
        location = f"/search?{query_without_page}&page=1"
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
    region_id_value = _parse_optional_int(region_id)
    municipality_id_value = _parse_optional_int(municipality_id)
    year_value = _parse_optional_int(year)
    subject_ids_value = _parse_int_list(subject_ids)
    legacy_subject_id = _parse_optional_int(subject_id)
    if legacy_subject_id is not None:
        subject_ids_value.append(legacy_subject_id)
    subject_ids_value = sorted(set(subject_ids_value))
    profile_ids_value = _parse_int_list(profile_ids)
    _parse_bounded_int(per_page, default=20, min_value=5, max_value=200)

    filters = _build_search_filters(
        q=q,
        region_id=region_id_value,
        municipality_id=municipality_id_value,
        profile_ids=profile_ids_value,
        year=year_value,
        kind=kind,
        subject_ids=subject_ids_value,
    )
    rows, _ = search_schools(filters, page=1, per_page=20, apply_pagination=False)
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

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="search")
    out.seek(0)

    headers = {"Content-Disposition": "attachment; filename=school_search_export.xlsx"}
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@router.get("/search/calc/{school_id}")
def calc_page(school_id: int):
    return RedirectResponse(f"/search/school/{school_id}", status_code=303)


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

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        school_df.to_excel(writer, index=False, sheet_name="school")
        profiles_df.to_excel(writer, index=False, sheet_name="profiles")
        external_df.to_excel(writer, index=False, sheet_name="external_keys")
        ege_periods_df.to_excel(writer, index=False, sheet_name="ege_periods")
        ege_subjects_df.to_excel(writer, index=False, sheet_name="ege_subjects")
        admission_df.to_excel(writer, index=False, sheet_name="admission")
        prof_events_df.to_excel(writer, index=False, sheet_name="prof_events")
    out.seek(0)

    headers = {"Content-Disposition": f"attachment; filename=school_card_{school_id}.xlsx"}
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


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
                "back_url": back_url,
            },
        )

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
            "back_url": back_url,
        },
    )


# 4 Подбор и рейтинг
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
    w_graduates: str | None = Query(default="0.25"),
    w_avg_score: str | None = Query(default="0.45"),
    w_match_share: str | None = Query(default="0.20"),
    w_threshold_share: str | None = Query(default="0.10"),
    limit: str | None = Query(default="300"),
    db: Session = Depends(get_db),
):
    user = require_user(request, db)

    region_id_value = _parse_optional_int(region_id)
    municipality_id_value = _parse_optional_int(municipality_id)
    year_value = _parse_optional_int(year)
    profile_ids_value = _parse_int_list(profile_ids)
    subject_ids_value = _parse_int_list(subject_ids)
    institute_ids_value = _parse_int_list(institute_ids)
    legacy_institute_id = _parse_optional_int(institute_id)
    if legacy_institute_id is not None:
        institute_ids_value.append(legacy_institute_id)
    institute_ids_value = sorted(set(institute_ids_value))
    program_ids_value = _parse_int_list(program_ids)

    min_graduates_value = _parse_optional_int(min_graduates)
    if min_graduates_value is not None and min_graduates_value < 0:
        min_graduates_value = 0

    min_avg_score_value = _parse_optional_float(min_avg_score)
    if min_avg_score_value is not None:
        min_avg_score_value = max(0.0, min(100.0, min_avg_score_value))

    enforce_threshold_value = (enforce_subject_threshold or "").strip().lower() not in {
        "",
        "0",
        "false",
        "off",
        "no",
    }

    w_graduates_value = _parse_optional_float(w_graduates)
    w_avg_score_value = _parse_optional_float(w_avg_score)
    w_match_share_value = _parse_optional_float(w_match_share)
    w_threshold_share_value = _parse_optional_float(w_threshold_share)
    if w_graduates_value is None:
        w_graduates_value = 0.25
    if w_avg_score_value is None:
        w_avg_score_value = 0.45
    if w_match_share_value is None:
        w_match_share_value = 0.20
    if w_threshold_share_value is None:
        w_threshold_share_value = 0.10

    limit_value = _parse_bounded_int(limit, default=300, min_value=10, max_value=2000)
    has_query = bool(request.query_params)

    options = fetch_rating_filter_options(region_id=region_id_value, institute_ids=tuple(institute_ids_value))
    program_requirements = fetch_program_requirements(
        institute_ids=tuple(institute_ids_value),
        program_ids=tuple(sorted(set(program_ids_value))),
    )

    ranked: list[dict[str, object]] = []
    if has_query:
        filters = RatingFilters(
            q=q.strip(),
            region_id=region_id_value,
            municipality_id=municipality_id_value,
            profile_ids=tuple(sorted(set(profile_ids_value))),
            year=year_value,
            kind=kind.strip() or None,
            subject_ids=tuple(sorted(set(subject_ids_value))),
            institute_ids=tuple(sorted(set(institute_ids_value))),
            program_ids=tuple(sorted(set(program_ids_value))),
            min_graduates=min_graduates_value,
            min_avg_score=min_avg_score_value,
            enforce_subject_threshold=enforce_threshold_value,
            limit=limit_value,
        )
        weights = RatingWeights(
            graduates=max(0.0, w_graduates_value),
            avg_score=max(0.0, w_avg_score_value),
            match_share=max(0.0, w_match_share_value),
            threshold_share=max(0.0, w_threshold_share_value),
        )
        ranked = calculate_school_rating(filters, weights)

    current_query = _build_rating_query(
        q=q.strip(),
        region_id=region_id_value,
        municipality_id=municipality_id_value,
        profile_ids=profile_ids_value,
        year=year_value,
        kind=kind.strip(),
        subject_ids=subject_ids_value,
        institute_ids=institute_ids_value,
        program_ids=program_ids_value,
        min_graduates=min_graduates_value,
        min_avg_score=min_avg_score_value,
        enforce_subject_threshold=enforce_threshold_value,
        w_graduates=w_graduates_value,
        w_avg_score=w_avg_score_value,
        w_match_share=w_match_share_value,
        w_threshold_share=w_threshold_share_value,
        limit=limit_value,
    )

    return render(
        "rating_profile.html",
        {
            "request": request,
            "user": user,
            "has_query": has_query,
            "ranked": ranked,
            "total": len(ranked),
            "program_requirements": program_requirements,
            "current_query": current_query,
            "filters": {
                "q": q.strip(),
                "region_id": region_id_value,
                "municipality_id": municipality_id_value,
                "profile_ids": profile_ids_value,
                "year": year_value,
                "kind": kind.strip(),
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
    w_graduates: str | None = Query(default="0.25"),
    w_avg_score: str | None = Query(default="0.45"),
    w_match_share: str | None = Query(default="0.20"),
    w_threshold_share: str | None = Query(default="0.10"),
    limit: str | None = Query(default="300"),
    db: Session = Depends(get_db),
):
    require_user(request, db)

    region_id_value = _parse_optional_int(region_id)
    municipality_id_value = _parse_optional_int(municipality_id)
    year_value = _parse_optional_int(year)
    profile_ids_value = _parse_int_list(profile_ids)
    subject_ids_value = _parse_int_list(subject_ids)
    institute_ids_value = _parse_int_list(institute_ids)
    legacy_institute_id = _parse_optional_int(institute_id)
    if legacy_institute_id is not None:
        institute_ids_value.append(legacy_institute_id)
    institute_ids_value = sorted(set(institute_ids_value))
    program_ids_value = _parse_int_list(program_ids)

    min_graduates_value = _parse_optional_int(min_graduates)
    if min_graduates_value is not None and min_graduates_value < 0:
        min_graduates_value = 0

    min_avg_score_value = _parse_optional_float(min_avg_score)
    if min_avg_score_value is not None:
        min_avg_score_value = max(0.0, min(100.0, min_avg_score_value))

    enforce_threshold_value = (enforce_subject_threshold or "").strip().lower() not in {
        "",
        "0",
        "false",
        "off",
        "no",
    }

    w_graduates_value = _parse_optional_float(w_graduates)
    w_avg_score_value = _parse_optional_float(w_avg_score)
    w_match_share_value = _parse_optional_float(w_match_share)
    w_threshold_share_value = _parse_optional_float(w_threshold_share)
    if w_graduates_value is None:
        w_graduates_value = 0.25
    if w_avg_score_value is None:
        w_avg_score_value = 0.45
    if w_match_share_value is None:
        w_match_share_value = 0.20
    if w_threshold_share_value is None:
        w_threshold_share_value = 0.10

    limit_value = _parse_bounded_int(limit, default=300, min_value=10, max_value=2000)
    has_query = bool(request.query_params)

    ranked: list[dict[str, object]] = []
    if has_query:
        filters = RatingFilters(
            q=q.strip(),
            region_id=region_id_value,
            municipality_id=municipality_id_value,
            profile_ids=tuple(sorted(set(profile_ids_value))),
            year=year_value,
            kind=kind.strip() or None,
            subject_ids=tuple(sorted(set(subject_ids_value))),
            institute_ids=tuple(sorted(set(institute_ids_value))),
            program_ids=tuple(sorted(set(program_ids_value))),
            min_graduates=min_graduates_value,
            min_avg_score=min_avg_score_value,
            enforce_subject_threshold=enforce_threshold_value,
            limit=limit_value,
        )
        weights = RatingWeights(
            graduates=max(0.0, w_graduates_value),
            avg_score=max(0.0, w_avg_score_value),
            match_share=max(0.0, w_match_share_value),
            threshold_share=max(0.0, w_threshold_share_value),
        )
        ranked = calculate_school_rating(filters, weights)

    export_rows = [
        {
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
            "match_share_pct": round(float(r.get("match_share", 0)) * 100.0, 2),
            "threshold_share_pct": round(float(r.get("threshold_share", 0)) * 100.0, 2),
            "rating_score": r.get("rating_score"),
            "matched_programs": r.get("matched_programs"),
        }
        for r in ranked
    ]
    df = pd.DataFrame(export_rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
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
                "match_share_pct",
                "threshold_share_pct",
                "rating_score",
                "matched_programs",
            ]
        )

    filters_df = pd.DataFrame(
        [
            {"key": "q", "value": q.strip()},
            {"key": "region_id", "value": region_id_value},
            {"key": "municipality_id", "value": municipality_id_value},
            {"key": "profile_ids", "value": ",".join(str(v) for v in sorted(set(profile_ids_value)))},
            {"key": "year", "value": year_value},
            {"key": "kind", "value": kind.strip()},
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

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="rating")
        filters_df.to_excel(writer, index=False, sheet_name="filters")
    out.seek(0)

    headers = {"Content-Disposition": "attachment; filename=school_rating_export.xlsx"}
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@router.post("/rating/filters")
def rating_filters(request: Request, program_id: int = Form(...), db: Session = Depends(get_db)):
    require_user(request, db)
    location = f"/rating/profile?{urlencode([('program_ids', str(program_id))], doseq=True)}"
    return RedirectResponse(location, status_code=303)


@router.post("/rating/calc")
def rating_calc(
    request: Request,
    program_id: int = Form(...),
    year: int = Form(2025),
    min_graduates: int = Form(10),
    w_graduates: float = Form(0.25),
    w_avg_score: float = Form(0.45),
    w_match_share: float = Form(0.20),
    w_threshold_share: float = Form(0.10),
    db: Session = Depends(get_db),
):
    require_user(request, db)
    params = [
        ("program_ids", str(program_id)),
        ("year", str(year)),
        ("min_graduates", str(min_graduates)),
        ("w_graduates", str(w_graduates)),
        ("w_avg_score", str(w_avg_score)),
        ("w_match_share", str(w_match_share)),
        ("w_threshold_share", str(w_threshold_share)),
        ("enforce_subject_threshold", "1"),
    ]
    location = f"/rating/profile?{urlencode(params, doseq=True)}"
    return RedirectResponse(location, status_code=303)


# 5 Настройки анализа (заглушки)
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


# 6 Отчётность
@router.get("/reports")
def reports_home(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("reports_home.html", {"request": request, "user": user})


@router.get("/reports/generate")
def report_generate(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("report_generate.html", {"request": request, "user": user})


@router.get("/reports/archive")
def report_archive(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    jobs = db.query(ImportJob).order_by(ImportJob.created_at.desc()).limit(50).all()
    return render("report_archive.html", {"request": request, "user": user, "jobs": jobs})


@router.get("/reports/calc-history")
def calc_history(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    return render("calc_history.html", {"request": request, "user": user})


# 7 Администрирование (только admin)
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
