from __future__ import annotations

import json
import traceback

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from .config import LOAD_PROGRAMS_SCRIPT, LOAD_SCHOOLS_SCRIPT, UPLOAD_DIR
    from .db import ImportJob, School, StudyProgram, User, get_db
    from .services import (
        apply_subject_scores_from_form,
        calculate_school_metrics,
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
        rank_schools,
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
    from db import ImportJob, School, StudyProgram, User, get_db
    from services import (
        apply_subject_scores_from_form,
        calculate_school_metrics,
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
        rank_schools,
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
@router.get("/search")
def search_page(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    programs = db.execute(select(StudyProgram)).scalars().all()
    return render("search.html", {"request": request, "user": user, "programs": programs, "results": None, "q": "", "year": 2025, "program_id": None})


@router.post("/search")
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


@router.get("/search/calc/{school_id}")
def calc_page(request: Request, school_id: int, year: int = 2025, program_id: int | None = None, db: Session = Depends(get_db)):
    user = require_user(request, db)
    metrics = calculate_school_metrics(db, [school_id], year, program_id)
    item = metrics[0] if metrics else None
    return render("calc.html", {"request": request, "user": user, "item": item, "year": year, "program_id": program_id})


@router.get("/search/school/{school_id}")
def school_card(request: Request, school_id: int, db: Session = Depends(get_db)):
    user = require_user(request, db)
    school = db.get(School, school_id)
    return render("school_card.html", {"request": request, "user": user, "school": school})


# 4 Подбор и рейтинг
@router.get("/rating/profile")
def rating_profile(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    programs = db.execute(select(StudyProgram)).scalars().all()
    return render("rating_profile.html", {"request": request, "user": user, "programs": programs})


@router.post("/rating/filters")
def rating_filters(request: Request, program_id: int = Form(...), db: Session = Depends(get_db)):
    user = require_user(request, db)
    program = db.get(StudyProgram, program_id)
    return render(
        "rating_filters.html",
        {
            "request": request,
            "user": user,
            "program": program,
            "year": 2025,
            "min_graduates": 10,
            "w_graduates": 0.4,
            "w_avg_score": 0.4,
            "w_match_share": 0.2,
        },
    )


@router.post("/rating/calc")
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
