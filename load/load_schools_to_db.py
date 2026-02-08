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
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


# Эти предупреждения openpyxl обычно не мешают работе, но засоряют вывод
warnings.filterwarnings("ignore", message=r"Print area cannot be set.*", category=UserWarning, module="openpyxl")


# -------------------- Базовые утилиты --------------------

def norm_spaces(s: str) -> str:
    s = str(s).replace("\u00A0", " ")
    return re.sub(r"\s+", " ", s.strip())


def clean_inner_quotes(s: str) -> str:
    return str(s).replace("«", "").replace("»", "").replace('"', "").replace("'", "")


def strip_outer_quotes(s: str) -> str:
    s = str(s).strip()
    while True:
        changed = False
        for a, b in [('"', '"'), ("'", "'"), ("«", "»")]:
            if len(s) >= 2 and s.startswith(a) and s.endswith(b):
                s = s[1:-1].strip()
                changed = True
                break
        if not changed:
            break
    return s.strip().strip('"').strip("'").strip("«").strip("»").strip()


# -------------------- Нормализация школ --------------------
import re
from typing import List, Tuple

# скобки убираем, текст внутри сохраняем
_PARENS_RE = re.compile(r"\(([^)]*)\)")

# убираем лидирующий "138049 - "
_LEADING_NUM_RE = re.compile(r"^\s*\d+\s*[-–—]\s*")

# разные тире приводим к "-"
_DASHES_RE = re.compile(r"[–—]")

# склеенные слова: "...краяМБОУ..." -> "...края МБОУ..."
_GLUE_RE = re.compile(r"(?<=[а-яё])(?=[А-ЯЁ])")

# точка между буквами: "АП.Светогорова" -> "АПСветогорова"
_DOT_BETWEEN_LETTERS_RE = re.compile(r"(?<=[A-Za-zА-Яа-яЁё])\.(?=[A-Za-zА-Яа-яЁё])")

# точки после заглавной буквы в конце токена: "ВД." -> "ВД"
_DOT_AFTER_CAP_RE = re.compile(r"(?<=[A-ZА-ЯЁ])\.(?=\s|$)")

# хвост "имени ..." / "им ..." – для схлопывания дублей
_HONORIFIC_RE = re.compile(r"\bИМ(?:ЕНИ)?\b.*$", flags=re.IGNORECASE)


def norm_spaces(s: str) -> str:
    s = str(s).replace("\u00A0", " ")
    return re.sub(r"\s+", " ", s.strip())


def clean_inner_quotes(s: str) -> str:
    return str(s).replace("«", "").replace("»", "").replace('"', "").replace("'", "")


def strip_outer_quotes(s: str) -> str:
    s = str(s).strip()
    while True:
        changed = False
        for a, b in [('"', '"'), ("'", "'"), ("«", "»")]:
            if len(s) >= 2 and s.startswith(a) and s.endswith(b):
                s = s[1:-1].strip()
                changed = True
                break
        if not changed:
            break
    return s.strip().strip('"').strip("'").strip("«").strip("»").strip()


def remove_parens_keep_text(s: str) -> str:
    return _PARENS_RE.sub(lambda m: f" {m.group(1)} ", s)


def strip_leading_num_prefix(s: str) -> str:
    return _LEADING_NUM_RE.sub("", s).strip()


SCHOOL_REPLACEMENTS: List[Tuple[str, str]] = [
    # Организационно-правовая форма
    (r"\bФедеральн\w*\s+государственн\w*\s+каз[её]нн\w*\s+общеобразовательн\w*\s+учрежден\w*\b", "ФГКОУ"),
    (r"\bКраев\w*\s+государственн\w*\s+автономн\w*\s+нетипов\w*\s+образовательн\w*\s+учрежден\w*\b", "КГАНОУ"),
    (r"\bКраев\w*\s+государственн\w*\s+бюджетн\w*\s+общеобразовательн\w*\s+учрежден\w*\b", "КГБОУ"),
    (r"\bМуниципальн\w*\s+автономн\w*\s+общеобразовательн\w*\s+учрежден\w*\b", "МАОУ"),
    (r"\bМуниципальн\w*\s+бюджетн\w*\s+общеобразовательн\w*\s+учрежден\w*\b", "МБОУ"),
    (r"\bМуниципальн\w*\s+каз[её]нн\w*\s+общеобразовательн\w*\s+учрежден\w*\b", "МКОУ"),
    (r"\bМуниципальн\w*\s+общеобразовательн\w*\s+учрежден\w*\b", "МОУ"),
    (r"\bЧастн\w*\s+общеобразовательн\w*\s+учрежден\w*\b", "ЧОУ"),

    # Тип учреждения
    (r"\bсредн\w*\s+общеобразовательн\w*\s+школ\w*\b", "СОШ"),
    (r"\bобщеобразовательн\w*\s+школ\w*\b", "ОШ"),
    (r"\bсредн\w*\s+школ\w*\b", "СШ"),
    (r"\bвечерн\w*\s*\(сменн\w*\)\s*школ\w*\b", "ВСШ"),
    (r"\bшкола[-\s]?интернат\b", "ШКОЛА-ИНТЕРНАТ"),
    (r"\bгимназ\w*\b", "ГИМНАЗИЯ"),
    (r"\bлице\w*\b", "ЛИЦЕЙ"),
    (r"\bцентр\s+образован\w*\b", "ЦЕНТР ОБРАЗОВАНИЯ"),
    (r"\bкадетск\w*\s+школ\w*\b", "КАДЕТСКАЯ ШКОЛА"),
]

# унификация "поселок/поселка" -> "пос", "село/села" -> "с", "город" -> "г"
LOCATION_REPLACEMENTS: List[Tuple[str, str]] = [
    (r"\bпос[её]л[её]к(?:а)?\b", "пос"),
    (r"\bсел[оа]\b", "с"),
    (r"\bгород\b", "г"),
]


def _to_e_upper(s: str) -> str:
    """
    Верхний регистр + приведение Ё->Е.
    Делать в конце, после замен по regex, где встречаются [её].
    """
    s = s.upper()
    return s.replace("Ё", "Е")


def strip_honorific_tail(s: str) -> str:
    # "… ИМЕНИ …" / "… ИМ …" – убрать хвост
    return _HONORIFIC_RE.sub("", s).strip(" ,;.")


def standardize_school_name(name: str, *, drop_honorific: bool = False) -> str:
    if not name:
        return ""

    s = str(name).replace("\u00A0", " ")

    # чинит "краяМБОУ" и подобные склейки
    s = _GLUE_RE.sub(" ", s)

    # единое тире
    s = _DASHES_RE.sub("-", s)

    # убираем префикс "123 - "
    s = strip_leading_num_prefix(s)

    # скобки сохраняем, кавычки чистим
    s = remove_parens_keep_text(s)
    s = clean_inner_quotes(s)
    s = strip_outer_quotes(s)

    # точки в инициалах/склейках
    s = _DOT_BETWEEN_LETTERS_RE.sub("", s)

    s = norm_spaces(s)

    # сокращаем форму и тип (до приведения Ё->Е!)
    for pat, repl in SCHOOL_REPLACEMENTS:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)

    # унификация "поселок/село/город"
    for pat, repl in LOCATION_REPLACEMENTS:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)

    # нормализуем № (всегда пробел после №)
    s = re.sub(r"№\s*(\d)", r"№ \1", s)

    # при желании – обрезаем хвост "имени ..."
    if drop_honorific:
        s = strip_honorific_tail(s)

    # верхний регистр + Ё->Е
    s = _to_e_upper(s)

    # убираем точки после заглавных букв (Г., С., А.С. и т.п.)
    s = _DOT_AFTER_CAP_RE.sub("", s)

    # финальная чистка пробелов и лишней пунктуации
    s = norm_spaces(s).strip(" ,;.")
    return s


def make_key(s: str, *, drop_honorific: bool = False) -> str:
    """
    Ключ для сопоставления школ (устойчивый к пробелам/знакам/регистру).
    drop_honorific=True – схлопывает 'СШ № 10' и 'СШ № 10 ИМЕНИ ...' в один ключ.
    """
    s = standardize_school_name(s, drop_honorific=drop_honorific)
    return re.sub(r"[^0-9A-ZА-Я]+", "", s)  # Ё уже нет, она приведена к Е


def make_muni_key(s: str) -> str:
    s = norm_spaces(str(s))
    s = _GLUE_RE.sub(" ", s)
    s = _DASHES_RE.sub("-", s)
    s = clean_inner_quotes(s)
    s = strip_outer_quotes(s)
    s = norm_spaces(s)
    s = _to_e_upper(s)
    s = _DOT_AFTER_CAP_RE.sub("", s)
    return re.sub(r"[^0-9A-ZА-Я]+", "", s)


# -------------------- Нормализация профилей --------------------

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


# -------------------- Выбор листа и чтение Excel --------------------

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


def find_header_row(path: Path, sheet: object, max_scan: int = 50) -> int:
    raw = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl", dtype=str, nrows=max_scan)
    for i in range(min(max_scan, len(raw))):
        row = " ".join([str(x).lower() for x in raw.iloc[i].fillna("").tolist()])
        if "муницип" in row and "образоват" in row and ("школ" in row or "организац" in row):
            return i
    return 0


def infer_region_from_filename(path: Path) -> str:
    return norm_spaces(path.stem.replace("_", " "))


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

    out = out[(out["municipality"] != "") & (out["school"] != "")]
    out["school"] = out["school"].map(standardize_school_name)

    out = out[~out["municipality"].str.contains(r"\bвсего\b", case=False, na=False)]
    out = out[~out["school"].str.contains(r"\bвсего\b", case=False, na=False)]

    out = out.drop_duplicates(subset=["region", "municipality", "school", "profile_raw"]).reset_index(drop=True)
    return out


# -------------------- Переменные окружения --------------------

def load_env() -> None:
    if load_dotenv is None:
        return

    # сначала пробуем из текущей папки
    if (Path.cwd() / ".env").exists():
        load_dotenv(dotenv_path=Path.cwd() / ".env")
        return

    # затем ищем рядом со скриптом и выше
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        env_path = p / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            return

    # последний шанс
    load_dotenv()


def get_db_config() -> Dict[str, object]:
    load_env()

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    missing = [k for k, v in [("POSTGRES_DB", dbname), ("POSTGRES_USER", user), ("POSTGRES_PASSWORD", password)] if not v]
    if missing:
        raise ValueError(f"Не заданы переменные окружения: {', '.join(missing)}")

    return {"host": host, "port": port, "dbname": dbname, "user": user, "password": password}


# -------------------- Вставки в БД --------------------

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


# -------------------- Выбор файла пользователем --------------------

def pick_file_dialog() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Выберите Excel файл",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


def resolve_user_path(s: str) -> Path:
    p = Path(s)
    if p.exists():
        return p

    cur = Path.cwd() / s
    if cur.exists():
        return cur

    desktop = Path.home() / "Desktop" / s
    if desktop.exists():
        return desktop

    if not s.lower().endswith((".xlsx", ".xls")):
        for base in [Path.cwd(), Path.home() / "Desktop"]:
            cand = base / f"{s}.xlsx"
            if cand.exists():
                return cand

    raise FileNotFoundError(
        f"Файл не найден: {s}. Укажи полный путь или положи файл в текущую папку или на рабочий стол."
    )


# -------------------- Основная загрузка --------------------

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
    p.add_argument("--sheet", default="", help="Имя листа или индекс. Если не задано – предложит выбрать")
    p.add_argument("--dry-run", action="store_true", help="Только чтение и обработка, без записи в базу")
    p.add_argument("--pick", action="store_true", help="Открыть окно выбора файла")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    path: Optional[Path] = None

    if args.pick or not args.file:
        path = pick_file_dialog()
        if path is None:
            if not args.file:
                print('Окно выбора файла недоступно. Укажи файл так: python load_schools_to_db.py "Приморский край.xlsx"')
                return 2

    if path is None:
        path = resolve_user_path(args.file)

    sheet_names = get_sheet_names(path)

    if args.sheet.strip():
        s = args.sheet.strip()
        sheet: object = int(s) if s.isdigit() else s
    else:
        default = "2024" if "2024" in sheet_names else (sheet_names[0] if sheet_names else "0")
        sheet = pick_sheet_console(sheet_names, default=default)

    df = read_excel_fixed(path, sheet=sheet)
    db_cfg = get_db_config()
    load_to_db(df, db_cfg, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())