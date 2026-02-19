# -*- coding: utf-8 -*-
"""
Загрузка школ, муниципалитетов и профильных связей из Excel в PostgreSQL.

Скрипт нормализует названия, подготавливает профили
и записывает данные в таблицы edu.school* и справочники.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from load_common import (
    clean_inner_quotes,
    get_db_config,
    normalize_municipality_name,
    norm_spaces,
    pick_file_dialog_desktop,
    resolve_user_path,
    standardize_school_name,
    strip_outer_quotes,
)


warnings.filterwarnings("ignore", message=r"Print area cannot be set.*", category=UserWarning, module="openpyxl")


# Нормализация профилей.
PROFILE_ABSENT_RE = re.compile(r"(^|\b)(х|x|нет|отсутствует|не\s*имеется)\b", flags=re.IGNORECASE)
PROFILE_TAIL_NOTE_RE = re.compile(r"(\b[хx]\b\s*[-–—]?\s*отсутствует.*)$", flags=re.IGNORECASE)
HYPHEN_FIX_RE = re.compile(r"\s*[-–—]\s*")

PROFILE_SYNONYMS: Dict[str, str] = {
    "тенологический": "технологический",
    "технолгический": "технологический",

    "социально экономический": "социально-экономический",
    "социально-экономический": "социально-экономический",

    "социально гуманитарный": "социально-гуманитарный",
    "социально-гуманитарный": "социально-гуманитарный",

    "физико математический": "физико-математический",
    "физико-математический": "физико-математический",

    "химико биологический": "химико-биологический",
    "химико-биологический": "химико-биологический",

    "естественно научный": "естественно-научный",
    "естественно-научный": "естественно-научный",

    "оборонно спортивный": "оборонно-спортивный",
    "оборонно-спортивный": "оборонно-спортивный",

    "универсальный": "универсальный",
    "гуманитарный": "гуманитарный",
    "филологический": "филологический",
}


def normalize_profile_token(token: str) -> str:
    if not token:
        return ""

    s = norm_spaces(token)
    s = clean_inner_quotes(s)
    s = strip_outer_quotes(s)
    s = norm_spaces(s)
    s = s.strip(" .;,:|/\\")
    s = s.rstrip(".")

    if not s:
        return ""

    if PROFILE_ABSENT_RE.fullmatch(s.casefold()):
        return ""

    s = HYPHEN_FIX_RE.sub("-", s)

    key = s.casefold()
    key_space = norm_spaces(key.replace("-", " "))

    if key in PROFILE_SYNONYMS:
        return PROFILE_SYNONYMS[key]
    if key_space in PROFILE_SYNONYMS:
        return PROFILE_SYNONYMS[key_space]

    return s


def split_profiles(cell: str) -> List[str]:
    if cell is None:
        return []

    raw = str(cell).replace("\u00A0", " ")
    raw = PROFILE_TAIL_NOTE_RE.sub("", raw)

    if not norm_spaces(raw):
        return []
    if PROFILE_ABSENT_RE.fullmatch(norm_spaces(raw).casefold()):
        return []

    raw = raw.replace("\n", ";").replace("\r", ";")
    parts = re.split(r"[;,]+", raw)

    out: List[str] = []
    seen: Set[str] = set()

    for p in parts:
        np = normalize_profile_token(p)
        if not np:
            continue
        k = np.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(np)

    return out


# Выбор листа и чтение Excel.
def get_sheet_names(path: Path) -> List[str]:
    xls = pd.ExcelFile(path, engine="openpyxl")
    return list(xls.sheet_names)


def pick_sheet_console(sheet_names: List[str], default: Optional[str] = None) -> str:
    if default and default in sheet_names:
        prompt_default = default
    else:
        prompt_default = sheet_names[0] if sheet_names else "0"

    print("Доступные листы в файле:")
    for i, s in enumerate(sheet_names, 1):
        print(f"{i}. {s}")

    ans = input(f"Выбери лист по номеру или имени. По умолчанию {prompt_default}: ").strip()

    if not ans:
        return prompt_default

    if ans.isdigit():
        idx = int(ans)
        if 1 <= idx <= len(sheet_names):
            return sheet_names[idx - 1]

    for s in sheet_names:
        if s.casefold() == ans.casefold():
            return s

    print("Лист не распознан, используется значение по умолчанию")
    return prompt_default


def resolve_sheet_argument(sheet_arg: str, sheet_names: List[str]) -> object:
    value = sheet_arg.strip()
    if not value:
        raise ValueError("Параметр --sheet не должен быть пустым.")

    # Сначала пытаемся сопоставить по имени листа (включая числовые имена: "2024").
    for sheet_name in sheet_names:
        if sheet_name.casefold() == value.casefold():
            return sheet_name

    if value.isdigit():
        idx = int(value)
        # Поддерживаем ввод индекса листа в человеко-ориентированном формате (1..N).
        if 1 <= idx <= len(sheet_names):
            return sheet_names[idx - 1]
        if idx == 0 and sheet_names:
            return sheet_names[0]
        # Сохраняем обратную совместимость со старым 0-based поведением.
        if 0 <= idx < len(sheet_names):
            return sheet_names[idx]

    raise ValueError(f"Лист '{value}' не найден. Доступные листы: {', '.join(sheet_names)}")


def find_header_row(path: Path, sheet: object, max_scan: int = 50) -> int:
    raw = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl", dtype=str, nrows=max_scan)
    for i in range(min(max_scan, len(raw))):
        row = " ".join([str(x).lower() for x in raw.iloc[i].fillna("").tolist()])
        if "муницип" in row and "образоват" in row and ("школ" in row or "организац" in row):
            return i
    return 0


def infer_region_from_filename(path: Path) -> str:
    region = norm_spaces(path.stem.replace("_", " "))
    cleaned = re.sub(r"^\s*directories(?:\s+|[-_]+)", "", region, flags=re.IGNORECASE)
    cleaned = norm_spaces(cleaned)
    return cleaned or region


def _norm_col_name(x: object) -> str:
    s = str(x).replace("\n", " ")
    s = clean_inner_quotes(s)
    return norm_spaces(s).casefold()


def resolve_excel_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    wanted_mun = _norm_col_name("Муниципальное образование")
    wanted_school = _norm_col_name("Образовательная организация (школа)")
    wanted_profile = _norm_col_name("Профиль (при наличии)")

    col_map = {_norm_col_name(c): c for c in df.columns}

    def fallback(find_words: List[str]) -> Optional[str]:
        for norm, orig in col_map.items():
            ok = True
            for w in find_words:
                if w not in norm:
                    ok = False
                    break
            if ok:
                return orig
        return None

    mun_col = col_map.get(wanted_mun) or fallback(["муницип"])
    school_col = col_map.get(wanted_school) or fallback(["образоват", "организац"]) or fallback(["школ"])
    profile_col = col_map.get(wanted_profile) or fallback(["профил"])

    if not mun_col or not school_col:
        raise ValueError(
            "Не найдены обязательные столбцы. Нужны: Муниципальное образование и Образовательная организация (школа)."
        )

    return mun_col, school_col, profile_col or ""


def read_excel_fixed(path: Path, sheet: object) -> pd.DataFrame:
    header_row = find_header_row(path, sheet=sheet)
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl", dtype=str, header=header_row)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    mun_col, school_col, profile_col = resolve_excel_columns(df)
    region = infer_region_from_filename(path)

    # Если в Excel объединены ячейки, часть строк будет пустой.
    df[mun_col] = df[mun_col].ffill()
    df[school_col] = df[school_col].ffill()

    out = pd.DataFrame(
        {
            "municipality": df[mun_col],
            "school": df[school_col],
            "profile_raw": df[profile_col] if profile_col else "",
        }
    )
    out.insert(0, "region", region)

    for c in ["municipality", "school", "profile_raw"]:
        out[c] = out[c].map(lambda x: "" if str(x).strip().lower() == "nan" else str(x))
        out[c] = out[c].map(norm_spaces)

    out["municipality"] = out["municipality"].map(normalize_municipality_name)
    out = out[(out["municipality"] != "") & (out["school"] != "")]
    out["school"] = out["school"].map(standardize_school_name)

    out = out[~out["municipality"].str.contains(r"\bвсего\b", case=False, na=False)]
    out = out[~out["school"].str.contains(r"\bвсего\b", case=False, na=False)]

    out = out.drop_duplicates(subset=["region", "municipality", "school", "profile_raw"]).reset_index(drop=True)
    return out


# Вставки в БД.
def insert_regions(cur, regions: Sequence[str]) -> None:
    q = """
        INSERT INTO edu.region (name)
        VALUES %s
        ON CONFLICT (name) DO NOTHING
    """
    execute_values(cur, q, [(r,) for r in regions], page_size=1000)


def fetch_region_ids(cur, regions: Sequence[str]) -> Dict[str, int]:
    cur.execute("SELECT region_id, name FROM edu.region WHERE name = ANY(%s)", (list(regions),))
    return {name: rid for rid, name in cur.fetchall()}


def _municipality_norm_key(name: str) -> str:
    return normalize_municipality_name(name).casefold().replace("ё", "е")


def _municipality_sql_norm(expr: str) -> str:
    squeezed = f"regexp_replace(btrim({expr}), '[[:space:]]+', ' ', 'g')"
    with_city_prefix = f"regexp_replace({squeezed}, '^[[:space:]]*г(\\.|[[:space:]]+)', 'город ', 'i')"
    with_city_prefix = f"regexp_replace({with_city_prefix}, '^[[:space:]]*город[[:space:]]+', 'город ', 'i')"
    return f"replace(lower({with_city_prefix}), 'ё', 'е')"


def _municipality_code3_sql(expr: str) -> str:
    return (
        "CASE "
        f"WHEN {expr} ~ '^[[:space:]]*[(][[:space:]]*[0-9]+' "
        f"THEN lpad(substr((regexp_match({expr}, '^[[:space:]]*[(][[:space:]]*([0-9]+)'))[1], 1, 3), 3, '0') "
        "ELSE NULL "
        "END"
    )


def _municipality_key_sql(expr: str) -> str:
    return f"COALESCE({_municipality_code3_sql(expr)}, {_municipality_sql_norm(expr)})"


PRIMORSKY_REGION_NORM = "приморский край"


def _region_sql_norm(expr: str) -> str:
    squeezed = f"regexp_replace(btrim({expr}), '[[:space:]]+', ' ', 'g')"
    return f"replace(lower({squeezed}), 'ё', 'е')"


def _school_code3_sql(expr: str) -> str:
    return (
        f"CASE "
        f"WHEN {expr} ~ '^[[:space:]]*[(]?[[:space:]]*[0-9]+' "
        f"THEN lpad(substr((regexp_match({expr}, '^[[:space:]]*[(]?[[:space:]]*([0-9]+)'))[1], 1, 3), 3, '0') "
        f"ELSE NULL "
        f"END"
    )


def _school_sql_norm(expr: str) -> str:
    squeezed = f"regexp_replace(btrim({expr}), '[[:space:]]+', ' ', 'g')"
    folded = f"replace(lower({squeezed}), 'ё', 'е')"
    return f"regexp_replace({folded}, 'ого(?=$|[[:space:][:punct:]])', 'ий', 'g')"


def insert_municipalities(cur, rows: Sequence[Tuple[int, str]]) -> None:
    if not rows:
        return

    dedup_rows: List[Tuple[int, str]] = []
    seen: Set[Tuple[int, str]] = set()
    for region_id, name in rows:
        canon = normalize_municipality_name(name)
        if not canon:
            continue
        key = (region_id, _municipality_norm_key(canon))
        if key in seen:
            continue
        seen.add(key)
        dedup_rows.append((region_id, canon))

    if not dedup_rows:
        return

    d_name_norm = _municipality_sql_norm("d.name")
    m_name_norm = _municipality_sql_norm("m.name")
    q = f"""
        WITH v(region_id, name) AS (VALUES %s),
        dedup AS (
            SELECT DISTINCT region_id, name
            FROM v
        )
        INSERT INTO edu.municipality (region_id, name)
        SELECT d.region_id, d.name
        FROM dedup d
        WHERE NOT EXISTS (
            SELECT 1
            FROM edu.municipality m
            WHERE m.region_id = d.region_id
              AND {m_name_norm} = {d_name_norm}
        )
    """
    execute_values(cur, q, dedup_rows, page_size=1000)


def fetch_municipality_ids(cur, rows: Sequence[Tuple[int, str]]) -> Dict[Tuple[int, str], int]:
    if not rows:
        return {}

    dedup_rows: List[Tuple[int, str]] = []
    seen: Set[Tuple[int, str]] = set()
    for region_id, name in rows:
        canon = normalize_municipality_name(name)
        if not canon:
            continue
        key = (region_id, _municipality_norm_key(canon))
        if key in seen:
            continue
        seen.add(key)
        dedup_rows.append((region_id, canon))

    if not dedup_rows:
        return {}

    v_name_norm = _municipality_sql_norm("v.name")
    m_name_norm = _municipality_sql_norm("m.name")
    query = f"""
        WITH v(region_id, name) AS (VALUES %s)
        SELECT MIN(m.municipality_id) AS municipality_id, v.region_id, v.name
        FROM v
        JOIN edu.municipality m
          ON m.region_id = v.region_id
         AND {m_name_norm} = {v_name_norm}
        GROUP BY v.region_id, v.name
    """
    fetched = execute_values(cur, query, dedup_rows, fetch=True)

    by_norm: Dict[Tuple[int, str], int] = {}
    for mid, rid, name in fetched:
        key = (rid, _municipality_norm_key(name))
        curr = by_norm.get(key)
        if curr is None or mid < curr:
            by_norm[key] = mid

    out: Dict[Tuple[int, str], int] = {}
    for rid, name in rows:
        canon = normalize_municipality_name(name)
        if not canon:
            continue
        key = (rid, _municipality_norm_key(canon))
        mid = by_norm.get(key)
        if mid is None:
            continue
        out[(rid, name)] = mid
        out[(rid, canon)] = mid
    return out


def insert_schools(cur, rows: Sequence[Tuple[int, str]]) -> None:
    src_mun_key = _municipality_key_sql("m.name")
    m2_mun_key = _municipality_key_sql("m2.name")
    r_name_norm = _region_sql_norm("r.name")
    src_code3 = _school_code3_sql("v.full_name")
    s_code3 = _school_code3_sql("s.full_name")
    src_school_norm = _school_sql_norm("v.full_name")
    s_school_norm = _school_sql_norm("s.full_name")
    q = f"""
        WITH v(municipality_id, full_name, is_active) AS (VALUES %s),
        src AS (
            SELECT
                v.municipality_id,
                v.full_name,
                v.is_active,
                m.region_id,
                {src_mun_key} AS municipality_key,
                {r_name_norm} AS region_name_norm,
                {src_code3} AS school_code3,
                {src_school_norm} AS school_name_norm
            FROM v
            JOIN edu.municipality m ON m.municipality_id = v.municipality_id
            JOIN edu.region r ON r.region_id = m.region_id
        ),
        dedup AS (
            SELECT
                MIN(municipality_id) AS municipality_id,
                MIN(full_name) AS full_name,
                BOOL_OR(is_active) AS is_active,
                region_id,
                municipality_key,
                region_name_norm,
                school_code3,
                school_name_norm
            FROM src
            GROUP BY region_id, municipality_key, region_name_norm, school_code3, school_name_norm
        )
        INSERT INTO edu.school (municipality_id, full_name, is_active)
        SELECT d.municipality_id, d.full_name, d.is_active
        FROM dedup d
        WHERE NOT EXISTS (
            SELECT 1
            FROM edu.school s
            JOIN edu.municipality m2 ON m2.municipality_id = s.municipality_id
            WHERE m2.region_id = d.region_id
              AND {m2_mun_key} = d.municipality_key
              AND (
                    (
                        d.region_name_norm = '{PRIMORSKY_REGION_NORM}'
                        AND d.school_code3 IS NOT NULL
                        AND {s_code3} = d.school_code3
                    )
                    OR {s_school_norm} = d.school_name_norm
              )
        )
        ON CONFLICT (municipality_id, full_name) DO NOTHING
    """
    execute_values(cur, q, [(mid, nm, True) for mid, nm in rows], page_size=1000)


def fetch_school_ids(cur, rows: Sequence[Tuple[int, str]]) -> Dict[Tuple[int, str], int]:
    if not rows:
        return {}

    src_mun_key = _municipality_key_sql("m.name")
    m2_mun_key = _municipality_key_sql("m2.name")
    r_name_norm = _region_sql_norm("r.name")
    src_code3 = _school_code3_sql("v.full_name")
    s_code3 = _school_code3_sql("s.full_name")
    src_school_norm = _school_sql_norm("v.full_name")
    s_school_norm = _school_sql_norm("s.full_name")
    query = f"""
        WITH v(municipality_id, full_name) AS (VALUES %s),
        v_norm AS (
            SELECT
                v.municipality_id AS src_mid,
                v.full_name AS src_full_name,
                m.region_id,
                {src_mun_key} AS municipality_key,
                {src_code3} AS school_code3,
                {r_name_norm} AS region_name_norm,
                {src_school_norm} AS school_name_norm
            FROM v
            JOIN edu.municipality m ON m.municipality_id = v.municipality_id
            JOIN edu.region r ON r.region_id = m.region_id
        ),
        unique_code_match AS (
            SELECT
                m2.region_id,
                {m2_mun_key} AS municipality_key,
                {s_code3} AS school_code3,
                MIN(s.school_id) AS school_id
            FROM edu.school s
            JOIN edu.municipality m2 ON m2.municipality_id = s.municipality_id
            JOIN edu.region r2 ON r2.region_id = m2.region_id
            WHERE {_region_sql_norm("r2.name")} = '{PRIMORSKY_REGION_NORM}'
              AND {s_code3} IS NOT NULL
            GROUP BY m2.region_id, {m2_mun_key}, {s_code3}
        ),
        candidates AS (
            SELECT
                v.src_mid,
                v.src_full_name,
                ucm.school_id,
                1 AS match_priority
            FROM v_norm v
            JOIN unique_code_match ucm
              ON ucm.region_id = v.region_id
             AND ucm.municipality_key = v.municipality_key
             AND ucm.school_code3 = v.school_code3
            WHERE v.region_name_norm = '{PRIMORSKY_REGION_NORM}'
              AND v.school_code3 IS NOT NULL

            UNION ALL

            SELECT
                v.src_mid,
                v.src_full_name,
                s.school_id,
                2 AS match_priority
            FROM v_norm v
            JOIN edu.municipality m2
              ON m2.region_id = v.region_id
             AND {m2_mun_key} = v.municipality_key
            JOIN edu.school s
              ON s.municipality_id = m2.municipality_id
             AND {s_school_norm} = v.school_name_norm
        )
        SELECT DISTINCT ON (c.src_mid, c.src_full_name)
            c.school_id,
            c.src_mid,
            c.src_full_name
        FROM candidates c
        ORDER BY c.src_mid, c.src_full_name, c.match_priority, c.school_id
    """
    fetched = execute_values(cur, query, list(rows), fetch=True)
    return {(mid, nm): sid for (sid, mid, nm) in fetched}


def insert_profiles(cur, profiles: Sequence[str]) -> None:
    if not profiles:
        return
    q = """
        INSERT INTO edu.school_profile (name)
        VALUES %s
        ON CONFLICT (name) DO NOTHING
    """
    execute_values(cur, q, [(p,) for p in profiles], page_size=1000)


def fetch_profile_ids(cur, profiles: Sequence[str]) -> Dict[str, int]:
    if not profiles:
        return {}
    cur.execute("SELECT profile_id, name FROM edu.school_profile WHERE name = ANY(%s)", (list(profiles),))
    return {name: pid for pid, name in cur.fetchall()}


def insert_profile_links(cur, links: Iterable[Tuple[int, int]]) -> int:
    data = list(set(links))
    if not data:
        return 0
    q = """
        INSERT INTO edu.school_profile_link (school_id, profile_id)
        VALUES %s
        ON CONFLICT (school_id, profile_id) DO NOTHING
    """
    execute_values(cur, q, data, page_size=2000)
    return len(data)


# Основной процесс загрузки.
def load_to_db(df: pd.DataFrame, db_cfg: Dict[str, object], dry_run: bool = False) -> None:
    if df.empty:
        print("Нет данных для загрузки")
        return

    regions = sorted(df["region"].dropna().unique().tolist())

    with psycopg2.connect(**db_cfg) as conn:
        with conn.cursor() as cur:
            if dry_run:
                print("Режим dry run – без записи в базу")
                print(df.head(50).to_string(index=False))
                return

            insert_regions(cur, regions)
            region_ids = fetch_region_ids(cur, regions)

            mun_pairs = df[["region", "municipality"]].drop_duplicates()
            mun_rows: List[Tuple[int, str]] = []
            for _, r in mun_pairs.iterrows():
                rid = region_ids.get(r["region"])
                if rid and r["municipality"]:
                    mun_rows.append((rid, r["municipality"]))
            mun_rows = sorted(set(mun_rows))

            insert_municipalities(cur, mun_rows)
            mun_ids = fetch_municipality_ids(cur, mun_rows)

            school_rows: List[Tuple[int, str]] = []
            for _, r in df.iterrows():
                rid = region_ids.get(r["region"])
                if not rid:
                    continue
                mid = mun_ids.get((rid, r["municipality"]))
                if not mid:
                    continue
                if r["school"]:
                    school_rows.append((mid, r["school"]))
            school_rows = sorted(set(school_rows))

            insert_schools(cur, school_rows)
            school_ids = fetch_school_ids(cur, school_rows)

            all_profiles: Set[str] = set()
            per_school_profiles: List[Tuple[int, List[str]]] = []

            for _, r in df.iterrows():
                rid = region_ids.get(r["region"])
                if not rid:
                    continue
                mid = mun_ids.get((rid, r["municipality"]))
                if not mid:
                    continue
                sid = school_ids.get((mid, r["school"]))
                if not sid:
                    continue

                profiles = split_profiles(r.get("profile_raw", ""))
                if not profiles:
                    continue

                for p in profiles:
                    all_profiles.add(p)
                per_school_profiles.append((sid, profiles))

            profiles_sorted = sorted(all_profiles)
            insert_profiles(cur, profiles_sorted)
            profile_ids = fetch_profile_ids(cur, profiles_sorted)

            links: List[Tuple[int, int]] = []
            for sid, prof_list in per_school_profiles:
                for p in prof_list:
                    pid = profile_ids.get(p)
                    if pid:
                        links.append((sid, pid))

            link_count = insert_profile_links(cur, links)

    print("Загрузка завершена")
    print(f"Регионов – {len(regions)}")
    print(f"Муниципалитетов – {len(mun_rows)}")
    print(f"Школ – {len(school_rows)}")
    print(f"Профилей – {len(profiles_sorted)}")
    print(f"Связей школа–профиль – {link_count}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Загрузка школ и профилей из Excel в PostgreSQL")
    p.add_argument("file", nargs="?", help="Путь к Excel или имя файла в текущей папке или на рабочем столе")
    p.add_argument(
        "--sheet",
        default="",
        help="Имя листа или индекс (1..N/0). Если число совпадает с именем листа, оно трактуется как имя.",
    )
    p.add_argument("--dry-run", action="store_true", help="Только чтение и обработка, без записи в базу")
    p.add_argument("--pick", action="store_true", help="Открыть окно выбора файла")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    path: Optional[Path] = None

    if args.pick or not args.file:
        path = pick_file_dialog_desktop()
        if path is None:
            if not args.file:
                print('Окно выбора файла недоступно. Укажи файл так: python load_schools_to_db.py "Приморский край.xlsx"')
                return 2

    if path is None:
        path = resolve_user_path(args.file)

    sheet_names = get_sheet_names(path)

    if args.sheet.strip():
        sheet = resolve_sheet_argument(args.sheet, sheet_names)
    else:
        default = "2024" if "2024" in sheet_names else (sheet_names[0] if sheet_names else "0")
        sheet = pick_sheet_console(sheet_names, default=default)

    df = read_excel_fixed(path, sheet=sheet)
    db_cfg = get_db_config(search_from=Path(__file__))
    load_to_db(df, db_cfg, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
