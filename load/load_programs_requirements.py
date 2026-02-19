# -*- coding: utf-8 -*-
"""
???????? ??????????? ?????????? ? ?????????? ??? ?? Excel ? PostgreSQL.

????????? ??????????? ??????????/?????????, ????????? ? ?????
?????????? ?? ????????? ??? ?????? ?????????.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from load_common import (
    fetch_map,
    get_db_config,
    normalize_text,
    normalize_upper_key,
    norm_spaces,
    pick_file_dialog_desktop,
    resolve_excel_sheet_name,
    resolve_user_path,
    table_is_empty,
)


INSTITUTE_SEED: List[Tuple[int, str]] = [
    (1, "ВИ"),
    (2, "ИМКТ"),
    (3, "ИМО"),
    (4, "ИТПМ"),
    (5, "ИФКС"),
    (6, "ПИ"),
    (7, "ПИШ"),
    (8, "ШИГН"),
    (9, "ШМиНЖ"),
    (10, "ШП"),
    (11, "ШЭМ"),
    (12, "ЮШ"),
]

SUBJECT_SEED: List[Tuple[int, str, Optional[float]]] = [
    (1, "Русский язык", 40),
    (2, "Математика", 40),
    (3, "Физика", 41),
    (4, "Обществознание", 45),
    (5, "История", 40),
    (6, "ИКТ", 46),
    (7, "Иностранный язык", 40),
    (8, "Литература", 40),
    (9, "Биология", 40),
    (10, "География", 40),
    (11, "Химия", 40),
]

INSTITUTE_ALIASES: Dict[str, str] = {
    "ИНТПМ": "ИТПМ",
}

SUBJECT_ALIASES: Dict[str, str] = {
    "ИНФОРМАТИКА": "ИКТ",
    "ИНФОРМАТИКА И ИКТ": "ИКТ",
    "ИНФОРМАТИКА ИКТ": "ИКТ",
}

SUBJECT_CANON_BY_KEY: Dict[str, str] = {
    "РУССКИЙ ЯЗЫК": "Русский язык",
    "МАТЕМАТИКА": "Математика",
    "ФИЗИКА": "Физика",
    "ОБЩЕСТВОЗНАНИЕ": "Обществознание",
    "ИСТОРИЯ": "История",
    "ИКТ": "ИКТ",
    "ИНОСТРАННЫЙ ЯЗЫК": "Иностранный язык",
    "ЛИТЕРАТУРА": "Литература",
    "БИОЛОГИЯ": "Биология",
    "ГЕОГРАФИЯ": "География",
    "ХИМИЯ": "Химия",
}

INSTITUTE_CANON_BY_KEY: Dict[str, str] = {normalize_upper_key(name): name for _, name in INSTITUTE_SEED}


def canonical_institute(name: str) -> str:
    k = normalize_upper_key(name)
    k = INSTITUTE_ALIASES.get(k, k)
    return INSTITUTE_CANON_BY_KEY.get(k, normalize_text(name))


def canonical_subject(name: str) -> str:
    k = normalize_upper_key(name)
    k = SUBJECT_ALIASES.get(k, k)
    return SUBJECT_CANON_BY_KEY.get(k, normalize_text(name))


@dataclass
class ProgramRow:
    institute: str
    code: str
    name: str
    vi1: Optional[str]
    vi2_choices: List[str]
    vi3: Optional[str]


def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    required = {
        "Институт": "Институт",
        "Код": "Код",
        "Направление подготовки": "Направление подготовки",
        "Вступительное испытание 1": "Вступительное испытание 1",
        "Вступительное испытание 2 (на выбор)": "Вступительное испытание 2 (на выбор)",
        "Вступительное испытание 3": "Вступительное испытание 3",
    }
    norm_map = {norm_spaces(str(c).replace("\n", " ")).casefold(): str(c) for c in df.columns}
    out: Dict[str, str] = {}
    for logical, expected in required.items():
        key = norm_spaces(expected).casefold()
        if key not in norm_map:
            raise ValueError(f"В файле не найдена колонка: {expected}")
        out[logical] = norm_map[key]
    return out


def parse_programs(df: pd.DataFrame) -> List[ProgramRow]:
    cols = _resolve_columns(df)
    col_inst = cols["Институт"]
    col_code = cols["Код"]
    col_name = cols["Направление подготовки"]
    col_vi1 = cols["Вступительное испытание 1"]
    col_vi2 = cols["Вступительное испытание 2 (на выбор)"]
    col_vi3 = cols["Вступительное испытание 3"]

    starts = [i for i, v in df[col_code].items() if pd.notna(v) and normalize_text(v)]
    if not starts:
        return []

    programs: List[ProgramRow] = []
    last_idx = int(df.index[-1])

    for i, start in enumerate(starts):
        end = starts[i + 1] - 1 if i + 1 < len(starts) else last_idx
        block = df.loc[start:end]
        row0 = df.loc[start]

        inst_raw = row0.get(col_inst)
        if pd.isna(inst_raw) or not normalize_text(inst_raw):
            raise ValueError(f"Не найден институт в строке {start + 2} (Excel row)")
        inst = canonical_institute(str(inst_raw))

        code = normalize_text(str(row0.get(col_code, "")))
        name = normalize_text(str(row0.get(col_name, "")))
        vi1 = normalize_text(str(row0.get(col_vi1, ""))) if pd.notna(row0.get(col_vi1)) else None
        vi3 = normalize_text(str(row0.get(col_vi3, ""))) if pd.notna(row0.get(col_vi3)) else None

        vi2_list: List[str] = []
        for v in block[col_vi2].dropna().astype(str):
            vv = normalize_text(v)
            if not vv:
                continue
            if vv.casefold() == "или":
                continue
            vi2_list.append(vv)

        seen: set[str] = set()
        vi2_unique: List[str] = []
        for s in vi2_list:
            k = normalize_upper_key(s)
            if k in seen:
                continue
            seen.add(k)
            vi2_unique.append(s)

        programs.append(
            ProgramRow(
                institute=inst,
                code=code,
                name=name,
                vi1=vi1 or None,
                vi2_choices=vi2_unique,
                vi3=vi3 or None,
            )
        )

    return programs


def seed_institutes(cur) -> None:
    empty = table_is_empty(cur, "edu.institute")
    if empty:
        sql = """
            INSERT INTO edu.institute (institute_id, name)
            VALUES %s
            ON CONFLICT (institute_id) DO UPDATE SET name = EXCLUDED.name
        """
        execute_values(cur, sql, INSTITUTE_SEED)
    else:
        sql = """
            INSERT INTO edu.institute (name)
            VALUES %s
            ON CONFLICT (name) DO NOTHING
        """
        execute_values(cur, sql, [(name,) for _, name in INSTITUTE_SEED])


def seed_subjects(cur) -> None:
    empty = table_is_empty(cur, "edu.ege_subject")
    if empty:
        sql = """
            INSERT INTO edu.ege_subject (subject_id, name, min_passing_score)
            VALUES %s
            ON CONFLICT (subject_id) DO UPDATE
            SET name = EXCLUDED.name,
                min_passing_score = EXCLUDED.min_passing_score
        """
        execute_values(cur, sql, SUBJECT_SEED)
    else:
        sql = """
            INSERT INTO edu.ege_subject (name, min_passing_score)
            VALUES %s
            ON CONFLICT (name) DO NOTHING
        """
        execute_values(cur, sql, [(name, score) for _, name, score in SUBJECT_SEED])


def ensure_subjects_exist(cur, subject_names: Iterable[str], create_missing: bool) -> None:
    names = sorted({canonical_subject(s) for s in subject_names if s and normalize_text(s)})
    if not names:
        return

    cur.execute("SELECT name FROM edu.ege_subject WHERE name = ANY(%s)", (names,))
    existing = {str(r[0]) for r in cur.fetchall()}
    missing = [n for n in names if n not in existing]

    if not missing:
        return

    if not create_missing:
        raise ValueError("В edu.ege_subject отсутствуют предметы: " + ", ".join(missing))

    sql = """
        INSERT INTO edu.ege_subject (name, min_passing_score)
        VALUES %s
        ON CONFLICT (name) DO NOTHING
    """
    execute_values(cur, sql, [(n, None) for n in missing])


def upsert_programs(cur, programs: List[ProgramRow], inst_name_to_id: Dict[str, int]) -> None:
    inst_name_to_id_upper = {normalize_upper_key(k): v for k, v in inst_name_to_id.items()}
    rows_by_code: Dict[str, Tuple[int, str, str, bool]] = {}
    for p in programs:
        inst_id = inst_name_to_id.get(p.institute)
        if inst_id is None:
            inst_id = inst_name_to_id_upper.get(normalize_upper_key(p.institute))
        if inst_id is None:
            raise ValueError(f"Не найден institute_id для института: {p.institute}")
        if p.code in rows_by_code:
            # В файле код может повторяться (варианты одной группы направлений).
            # Для таблицы edu.study_program код уникален, поэтому оставляем первую запись.
            continue
        rows_by_code[p.code] = (inst_id, p.code, p.name, True)

    rows = list(rows_by_code.values())

    sql = """
        INSERT INTO edu.study_program (institute_id, code, name, is_active)
        VALUES %s
        ON CONFLICT (code) DO UPDATE
        SET institute_id = EXCLUDED.institute_id,
            name = EXCLUDED.name,
            is_active = TRUE
    """
    execute_values(cur, sql, rows)


def upsert_requirements(
    cur,
    programs: List[ProgramRow],
    code_to_program_id: Dict[str, int],
    subject_name_to_id: Dict[str, int],
) -> None:
    req_map: Dict[Tuple[int, int], Tuple[str, int]] = {}

    for p in programs:
        program_id = code_to_program_id.get(p.code)
        if program_id is None:
            raise ValueError(f"Не найден program_id для кода: {p.code}")

        def add(subject: Optional[str], role: str, weight: int) -> None:
            if not subject:
                return
            sname = canonical_subject(subject)
            sid = subject_name_to_id.get(sname)
            if sid is None:
                raise ValueError(f"Не найден subject_id для предмета: {sname}")
            key = (program_id, sid)
            cur_val = req_map.get(key)
            if cur_val is None:
                req_map[key] = (role, weight)
                return

            cur_role, cur_weight = cur_val
            # Если один и тот же предмет встретился и как required, и как choice,
            # приоритет у required из-за бизнес-ограничений.
            if cur_role == "required" or role == "required":
                req_map[key] = ("required", 1)
            else:
                req_map[key] = ("choice", max(cur_weight, weight))

        add(p.vi1, "required", 1)
        add(p.vi3, "required", 1)
        for s in p.vi2_choices:
            add(s, "choice", 2)

    if not req_map:
        return

    req_rows = [(pid, sid, role, weight) for (pid, sid), (role, weight) in req_map.items()]

    sql = """
        INSERT INTO edu.program_ege_requirement (program_id, subject_id, role, weight)
        VALUES %s
        ON CONFLICT (program_id, subject_id) DO UPDATE
        SET role = EXCLUDED.role,
            weight = EXCLUDED.weight
    """
    execute_values(cur, sql, req_rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Загрузка направлений и требований ЕГЭ из Excel")
    p.add_argument("excel", nargs="?", help="Путь к Excel-файлу")
    p.add_argument("--sheet", default="", help="Имя листа или его индекс (0-based). По умолчанию первый лист")
    p.add_argument("--pick", action="store_true", help="Открыть окно выбора файла с рабочего стола")
    p.add_argument("--dry-run", action="store_true", help="Только разбор файла и валидация, без записи в БД")
    p.add_argument(
        "--create-missing-subjects",
        action="store_true",
        help="Создавать отсутствующие предметы в edu.ege_subject (min_passing_score = NULL)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    path: Optional[Path] = None
    if args.pick or not args.excel:
        path = pick_file_dialog_desktop(title="Выберите Excel с направлениями")
        if path is None and not args.excel:
            raise SystemExit("Окно выбора файла недоступно. Укажи путь: python load_programs_requirements.py <файл.xlsx>")
    if path is None:
        path = resolve_user_path(args.excel)

    sheet_name = resolve_excel_sheet_name(path, args.sheet, engine="openpyxl")
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl", dtype=str)
    df = df.dropna(how="all").reset_index(drop=True)

    programs = parse_programs(df)
    if not programs:
        raise SystemExit("В файле не найдено направлений (нет заполненных значений в колонке 'Код').")

    all_subjects: List[str] = []
    for p in programs:
        if p.vi1:
            all_subjects.append(p.vi1)
        if p.vi3:
            all_subjects.append(p.vi3)
        all_subjects.extend(p.vi2_choices)

    if args.dry_run:
        print(f"Файл: {path}")
        print(f"Лист: {sheet_name}")
        print(f"Направлений: {len(programs)}")
        for p in programs[:5]:
            print(f"- {p.institute} | {p.code} | {p.name}")
            print(f"  required: {[canonical_subject(x) for x in [p.vi1, p.vi3] if x]}")
            print(f"  choice: {[canonical_subject(x) for x in p.vi2_choices]}")
        return 0

    db_cfg = get_db_config(search_from=Path(__file__))

    with psycopg2.connect(**db_cfg) as conn:
        with conn.cursor() as cur:
            seed_institutes(cur)
            seed_subjects(cur)

            inst_names = sorted({p.institute for p in programs})
            inst_name_to_id = fetch_map(
                cur,
                "SELECT name, institute_id FROM edu.institute WHERE name = ANY(%s)",
                (inst_names,),
            )

            ensure_subjects_exist(cur, all_subjects, create_missing=args.create_missing_subjects)

            subj_names = sorted({canonical_subject(s) for s in all_subjects if s})
            subject_name_to_id = fetch_map(
                cur,
                "SELECT name, subject_id FROM edu.ege_subject WHERE name = ANY(%s)",
                (subj_names,),
            )

            upsert_programs(cur, programs, inst_name_to_id)

            code_list = [p.code for p in programs]
            code_to_program_id = fetch_map(
                cur,
                "SELECT code, program_id FROM edu.study_program WHERE code = ANY(%s)",
                (code_list,),
            )

            upsert_requirements(cur, programs, code_to_program_id, subject_name_to_id)

        conn.commit()

    print(f"Готово: загружено/обновлено направлений: {len(programs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
