# -*- coding: utf-8 -*-
"""
Загрузка школ и профилей из Excel в PostgreSQL.

Ожидаемые столбцы Excel
- Муниципальное образование
- Образовательная организация (школа)
- Профиль (при наличии)

Что делает
- Регион берёт из имени файла
- Записывает регион, муниципалитет, школу
- Приводит полные типовые названия школ к сокращенному виду
- Нормализует профили и записывает в edu.school_profile и edu.school_profile_link
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
    norm_spaces,
    pick_file_dialog_desktop,
    resolve_user_path,
    standardize_school_name,
    strip_outer_quotes,
)


warnings.filterwarnings("ignore", message=r"Print area cannot be set.*", category=UserWarning, module="openpyxl")


# Нормализация профилей 
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


#  Выбор листа и чтение Excel 
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
        # Поддержка ввода индекса листа в человеко-ориентированном формате (1..N).
        if 1 <= idx <= len(sheet_names):
            return sheet_names[idx - 1]
        if idx == 0 and sheet_names:
            return sheet_names[0]
        # Обратная совместимость со старым 0-based поведением.
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


def normalize_municipality_name(value: str) -> str:
    text = norm_spaces(value)
    if not text:
        return ""
    # Избегаем дублей вида "Город X" и "город X".
    return text[0].upper() + text[1:] if text else text


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

    # если в Excel объединены ячейки, то часть строк будет пустой
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


#  Вставки в БД 
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


def insert_municipalities(cur, rows: Sequence[Tuple[int, str]]) -> None:
    q = """
        INSERT INTO edu.municipality (region_id, name)
        VALUES %s
        ON CONFLICT (region_id, name) DO NOTHING
    """
    execute_values(cur, q, list(rows), page_size=1000)


def fetch_municipality_ids(cur, rows: Sequence[Tuple[int, str]]) -> Dict[Tuple[int, str], int]:
    if not rows:
        return {}

    query = """
        WITH v(region_id, name) AS (VALUES %s)
        SELECT m.municipality_id, m.region_id, m.name
        FROM edu.municipality m
        JOIN v ON v.region_id = m.region_id AND v.name = m.name
    """
    fetched = execute_values(cur, query, list(rows), fetch=True)
    return {(rid, name): mid for (mid, rid, name) in fetched}


def insert_schools(cur, rows: Sequence[Tuple[int, str]]) -> None:
    q = """
        INSERT INTO edu.school (municipality_id, full_name, is_active)
        VALUES %s
        ON CONFLICT (municipality_id, full_name) DO NOTHING
    """
    execute_values(cur, q, [(mid, nm, True) for mid, nm in rows], page_size=1000)


def fetch_school_ids(cur, rows: Sequence[Tuple[int, str]]) -> Dict[Tuple[int, str], int]:
    if not rows:
        return {}

    query = """
        WITH v(municipality_id, full_name) AS (VALUES %s)
        SELECT s.school_id, s.municipality_id, s.full_name
        FROM edu.school s
        JOIN v ON v.municipality_id = s.municipality_id AND v.full_name = s.full_name
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


#  Основная загрузка 
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
