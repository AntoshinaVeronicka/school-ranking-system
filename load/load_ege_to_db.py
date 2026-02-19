# -*- coding: utf-8 -*-
"""
Загрузка статистики ЕГЭ из Excel в таблицы edu.*.

Скрипт читает лист, сопоставляет школы со справочником,
агрегирует показатели и выполняет upsert в БД.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from load_common import (
    clean_inner_quotes,
    get_db_config,
    load_env_file,
    make_muni_key,
    make_school_key,
    normalize_municipality_name,
    norm_spaces,
    pick_file_dialog_desktop,
    resolve_user_path,
    standardize_school_name,
)

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None  # type: ignore
    execute_values = None  # type: ignore

warnings.filterwarnings(
    "ignore", message=r"Print area cannot be set.*", category=UserWarning, module="openpyxl"
)


# Логирование.

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# Справочник предметов.

SUBJECT_CANON: List[str] = [
    "Русский язык",
    "Математика",
    "Физика",
    "Обществознание",
    "История",
    "ИКТ",
    "Иностранный язык",
    "Литература",
    "Биология",
    "География",
    "Химия",
]


def normalize_subject_title(x: Any) -> str:
    s0 = norm_spaces(str(x).replace("\n", " "))
    s = s0.casefold().replace("ё", "е")
    s = re.sub(r",?\s*чел\.?$", "", s).strip()
    s = re.sub(r"\(.*?\)", "", s).strip()

    if "русск" in s:
        return "Русский язык"
    if "математ" in s:
        return "Математика"
    if "физик" in s:
        return "Физика"
    if "обществ" in s:
        return "Обществознание"
    if "истор" in s:
        return "История"
    if "икт" in s or "информ" in s:
        return "ИКТ"
    if "иностран" in s or "англий" in s:
        return "Иностранный язык"
    if "литерат" in s:
        return "Литература"
    if "биолог" in s:
        return "Биология"
    if "географ" in s:
        return "География"
    if "хими" in s:
        return "Химия"

    return s0


# Преобразование значений.

def to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = norm_spaces(v)
    if not s or s.lower() == "nan" or s == "-":
        return None
    try:
        return int(float(s.replace(",", ".")))
    except Exception:
        return None


def to_decimal_2(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = norm_spaces(v)
    if not s or s.lower() == "nan" or s == "-":
        return None
    s = s.replace(",", ".")
    try:
        d = Decimal(s)
        return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except Exception:
        return None


# Чтение и подготовка Excel.

def choose_sheet(path: Path, default_sheet: Optional[str] = None) -> str:
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names
    default_sheet = default_sheet or sheets[0]

    logging.info("Доступные листы:")
    for i, s in enumerate(sheets, 1):
        logging.info("%s. %s", i, s)

    ans = input(f"Выбери лист (номер или имя). По умолчанию {default_sheet}: ").strip()
    if not ans:
        return default_sheet

    if ans.isdigit():
        i = int(ans)
        if 1 <= i <= len(sheets):
            return sheets[i - 1]

    for s in sheets:
        if s.casefold() == ans.casefold():
            return s

    logging.warning("Лист не распознан, использую по умолчанию: %s", default_sheet)
    return default_sheet


def infer_year_from_sheet(sheet: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", str(sheet))
    return int(m.group(1)) if m else None


def find_header_row(path: Path, sheet: str, max_scan: int = 60) -> int:
    raw = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl", dtype=str, nrows=max_scan)
    for i in range(min(max_scan, len(raw))):
        row = " ".join(str(x).lower() for x in raw.iloc[i].fillna("").tolist())
        if "муницип" in row and "образоват" in row and ("школ" in row or "организац" in row):
            return i
    return 0


def _norm_col(x: Any) -> str:
    return norm_spaces(clean_inner_quotes(str(x)).replace("\n", " ")).casefold()


def resolve_fixed_cols_flat(df: pd.DataFrame) -> Tuple[str, str, Optional[str], str]:
    col_map = {_norm_col(c): c for c in df.columns}

    def find_any(words: Sequence[str]) -> Optional[str]:
        for k, orig in col_map.items():
            if all(w in k for w in words):
                return orig
        return None

    mun = find_any(["муницип"])
    school = find_any(["образоват"]) or find_any(["школ"])
    profile = find_any(["профил"])
    grads = None

    for k, orig in col_map.items():
        if "всего" in k and "выпуск" in k and ("егэ" in k or "участ" in k):
            grads = orig
            break
    if grads is None:
        grads = find_any(["всего", "выпуск"])

    if not mun or not school or not grads:
        raise ValueError("Не найдены обязательные столбцы (муниципалитет/школа/выпускники).")

    return mun, school, profile, grads


def clean_common_rows(out: pd.DataFrame) -> pd.DataFrame:
    # Приводим NaN/None к пустому значению до текстовой нормализации.
    out["municipality"] = out["municipality"].map(normalize_municipality_name)
    out["school"] = out["school"].map(
        lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else norm_spaces(v)
    )
    out["school"] = out["school"].replace({"nan": "", "none": "", "nat": ""})
    out["school"] = out["school"].map(standardize_school_name)

    # Фильтры мусора.
    out = out[(out["municipality"] != "") & (out["school"] != "")]
    out = out[~out["municipality"].str.contains(r"\bвсего\b", case=False, na=False)]
    out = out[~out["school"].str.contains(r"\bвсего\b", case=False, na=False)]
    out = out[~out["school"].astype(str).str.fullmatch(r"\d+")]
    return out.reset_index(drop=True)


def read_plan_excel(path: Path, sheet: str) -> pd.DataFrame:
    header_row = find_header_row(path, sheet)
    header_variants: List[Any] = [
        [header_row, header_row + 1],
        header_row,
    ]

    def _raw_levels(col: Any) -> List[str]:
        if isinstance(col, tuple):
            levels = list(col)
        else:
            levels = [col]
        return [norm_spaces(str(v).replace("\n", " ")).casefold() for v in levels]

    def _subject_from_levels(levels: Sequence[str]) -> Optional[str]:
        for lvl in levels:
            subj = normalize_subject_title(lvl)
            if subj in SUBJECT_CANON:
                return subj
        return None

    def _find_cols(df_local: pd.DataFrame, words: Sequence[str]) -> List[Any]:
        cols: List[Any] = []
        for c in df_local.columns:
            joined = " | ".join(_raw_levels(c))
            if all(w in joined for w in words):
                cols.append(c)
        return cols

    def _pick_first(df_local: pd.DataFrame, words: Sequence[str]) -> Optional[Any]:
        cols = _find_cols(df_local, words)
        return cols[0] if cols else None

    def _pick_grads_col(df_local: pd.DataFrame) -> Optional[Any]:
        candidates = _find_cols(df_local, ["всего", "выпуск"])
        if not candidates:
            return None
        for c in candidates:
            if _subject_from_levels(_raw_levels(c)) is None:
                return c
        return candidates[0]

    def _parse_plan_frame(df_local: pd.DataFrame) -> pd.DataFrame:
        mun_col = _pick_first(df_local, ["муницип"])
        school_col = _pick_first(df_local, ["образоват"]) or _pick_first(df_local, ["школ"])
        grads_col = _pick_grads_col(df_local)
        if not mun_col or not school_col or not grads_col:
            raise ValueError("Не найдены обязательные столбцы (муниципалитет/школа/выпускники).")

        out = pd.DataFrame(
            {
                "municipality": df_local[mun_col],
                "school": df_local[school_col],
                "graduates_total": df_local[grads_col],
            }
        )

        # В таких таблицах муниципалитет и школа часто идут "сверху вниз" — протягиваем значения.
        out["municipality"] = out["municipality"].ffill()
        out["school"] = out["school"].ffill()

        for c in df_local.columns:
            if c in (mun_col, school_col, grads_col):
                continue
            subj = _subject_from_levels(_raw_levels(c))
            if subj:
                out[f"chosen__{subj}"] = df_local[c]

        return out

    last_error: Optional[Exception] = None
    best_df: Optional[pd.DataFrame] = None
    best_subject_cols = -1

    for header_spec in header_variants:
        try:
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl", header=header_spec)
            df = df.dropna(how="all").dropna(axis=1, how="all")
            parsed = _parse_plan_frame(df)
            subj_cols = sum(1 for c in parsed.columns if str(c).startswith("chosen__"))
            if subj_cols > best_subject_cols:
                best_subject_cols = subj_cols
                best_df = parsed
            if subj_cols > 0:
                return clean_common_rows(parsed)
        except Exception as exc:
            last_error = exc

    if best_df is not None:
        return clean_common_rows(best_df)

    if last_error is not None:
        raise last_error

    raise ValueError("Не удалось распознать формат предварительных данных ЕГЭ.")


def read_actual_excel(path: Path, sheet: str) -> pd.DataFrame:
    header_row = find_header_row(path, sheet)
    header_variants: List[List[int]] = [
        [header_row, header_row + 1, header_row + 2],
        [header_row, header_row + 1],
    ]

    def _raw_levels(col: Any) -> List[str]:
        if isinstance(col, tuple):
            levels = list(col)
        else:
            levels = [col]
        return [norm_spaces(str(v).replace("\n", " ")).casefold() for v in levels]

    def _pick_fixed(df_local: pd.DataFrame, words: Sequence[str]) -> Optional[Any]:
        for c in df_local.columns:
            joined = " | ".join(_raw_levels(c))
            if all(w in joined for w in words):
                return c
        return None

    def _detect_metric(levels: Sequence[str]) -> Optional[str]:
        joined = " | ".join(levels)
        if "\u043f\u0440\u0438\u043d\u044f\u043b" in joined and "\u0443\u0447\u0430\u0441\u0442" in joined:
            return "participants_cnt"
        if "\u043d\u0435 \u043f\u0440\u0435\u043e\u0434\u043e\u043b" in joined or "min" in joined or "\u043f\u043e\u0440\u043e\u0433" in joined:
            return "not_passed_cnt"
        if "\u0432\u044b\u0441\u043e\u043a" in joined or "80-99" in joined:
            return "high_80_99_cnt"
        if "\u0441\u0440\u0435\u0434\u043d" in joined and "\u0431\u0430\u043b\u043b" in joined:
            return "avg_score"
        if "100" in joined:
            return "score_100_cnt"
        return None

    def _parse_actual_frame(df_local: pd.DataFrame) -> pd.DataFrame:
        mun_t = _pick_fixed(df_local, ["\u043c\u0443\u043d\u0438\u0446\u0438\u043f"])
        school_t = _pick_fixed(df_local, ["\u043e\u0431\u0440\u0430\u0437\u043e\u0432\u0430\u0442"]) or _pick_fixed(df_local, ["\u0448\u043a\u043e\u043b"])
        grads_t = _pick_fixed(df_local, ["\u0432\u0441\u0435\u0433\u043e", "\u0432\u044b\u043f\u0443\u0441\u043a", "\u0435\u0433\u044d"]) or _pick_fixed(
            df_local, ["\u0432\u0441\u0435\u0433\u043e", "\u0432\u044b\u043f\u0443\u0441\u043a"]
        )

        if not mun_t or not school_t or not grads_t:
            raise ValueError("Не найдены фиксированные колонки (муниципалитет/школа/выпускники).")

        out = pd.DataFrame(
            {
                "municipality": df_local[mun_t],
                "school": df_local[school_t],
                "graduates_total": df_local[grads_t],
            }
        )
        out["municipality"] = out["municipality"].ffill()

        for col in df_local.columns:
            if col in (mun_t, school_t, grads_t):
                continue

            levels = _raw_levels(col)
            subj: Optional[str] = None
            for lvl in levels:
                candidate = normalize_subject_title(lvl)
                if candidate in SUBJECT_CANON:
                    subj = candidate
                    break
            if subj is None:
                continue

            metric = _detect_metric(levels)
            if metric is None:
                # Упрощенный формат: есть только колонка предмета без разбивки по метрикам.
                # Интерпретируем как число участников по предмету.
                metric = "participants_cnt"

            out[f"{metric}__{subj}"] = df_local[col]

        return out

    last_error: Optional[Exception] = None
    best_df: Optional[pd.DataFrame] = None
    best_subject_cols = -1

    for header_spec in header_variants:
        try:
            df = pd.read_excel(
                path,
                sheet_name=sheet,
                engine="openpyxl",
                header=header_spec,
            )
            df = df.dropna(how="all").dropna(axis=1, how="all")
            parsed = _parse_actual_frame(df)
            subj_cols = sum(1 for c in parsed.columns if "__" in str(c))
            if subj_cols > best_subject_cols:
                best_subject_cols = subj_cols
                best_df = parsed
            if subj_cols > 0:
                return clean_common_rows(parsed)
        except Exception as exc:
            last_error = exc

    if best_df is not None:
        return clean_common_rows(best_df)

    if last_error is not None:
        raise last_error

    raise ValueError("Не удалось распознать формат фактического листа ЕГЭ.")
def fetch_allowed_kinds(cur) -> List[str]:
    cur.execute("SELECT unnest(enum_range(NULL::edu.ege_dataset_kind))::text")
    return [r[0] for r in cur.fetchall()]


def normalize_kind_for_db(kind_ui: str, allowed: Optional[Sequence[str]] = None) -> str:
    """
    Приводим пользовательский ввод к enum в БД:
    - plan: предварительные (ранее у тебя называлось preliminary)
    - actual: фактические
    """
    k = (kind_ui or "").strip().casefold()

    if k in {"1", "plan", "предварительные", "предварительные данные", "план", "preliminary"}:
        k_db = "plan"
    elif k in {"2", "actual", "фактические", "фактические данные", "акт"}:
        k_db = "actual"
    else:
        k_db = kind_ui.strip()

    if allowed is not None and k_db not in set(allowed):
        raise ValueError(
            f"Недопустимое значение kind='{k_db}' для edu.ege_dataset_kind. "
            f"Допустимые значения в БД: {', '.join(allowed)}."
        )
    return k_db


def choose_kind_ui() -> str:
    print("Выбор типа данных:")
    print("1 – Предварительные данные (kind = plan)")
    print("2 – Фактические данные (kind = actual)")
    ans = input("Выбери 1 или 2. По умолчанию 2: ").strip()
    return "plan" if ans == "1" else "actual"


def fetch_subject_map(cur) -> Dict[str, int]:
    cur.execute("SELECT subject_id, name FROM edu.ege_subject")
    return {str(name): int(sid) for sid, name in cur.fetchall()}


@dataclass(frozen=True)
class SchoolRow:
    school_id: int
    full_name: str
    municipality_name: Optional[str]


PRIMORSKY_REGION_KEY = "приморский край"


def _normalize_region_key(value: Any) -> str:
    text = norm_spaces(value).casefold()
    return re.sub(r"[^0-9a-zа-я]+", " ", text).strip()


def is_primorsky_region(region_name: Optional[str]) -> bool:
    if not region_name:
        return False
    region_key = _normalize_region_key(region_name)
    return region_key == PRIMORSKY_REGION_KEY or ("приморск" in region_key and "край" in region_key)


def resolve_region_name(cur, region_name: Optional[str]) -> Optional[str]:
    """
    Возвращает каноническое имя региона из edu.region.
    Поддерживает нестрогий ввод (например, имя файла с добавленными словами/годом).
    """
    if not region_name:
        return None
    raw = norm_spaces(region_name)
    if not raw:
        return None

    cur.execute(
        """
        SELECT name
        FROM edu.region
        WHERE lower(name) = lower(%s)
        LIMIT 1
        """,
        (raw,),
    )
    exact = cur.fetchone()
    if exact and exact[0]:
        return str(exact[0])

    cur.execute(
        """
        SELECT name
        FROM edu.region
        WHERE lower(%s) LIKE '%%' || lower(name) || '%%'
           OR lower(name) LIKE '%%' || lower(%s) || '%%'
        ORDER BY char_length(name) DESC, name
        """,
        (raw, raw),
    )
    seen: set[str] = set()
    matches: list[str] = []
    for row in cur.fetchall():
        if not row or not row[0]:
            continue
        name = str(row[0])
        if name in seen:
            continue
        seen.add(name)
        matches.append(name)

    if len(matches) == 1:
        return matches[0]
    return None


def extract_school_code3(name: Any) -> Optional[str]:
    """
    Для строк вида "(286) МБОУ СОШ № 14 ..." извлекает "286".
    Для коротких кодов "(8) ..." делает zero-pad до 3 знаков: "008".
    Для длинных кодов берет первые 3 цифры.
    """
    s = norm_spaces(name)
    if not s:
        return None

    m = re.match(r"^\s*\(?\s*(\d+)\s*\)?", s)
    if not m:
        return None

    digits = m.group(1)
    return digits[:3].zfill(3)


def fetch_schools(cur, region_name: Optional[str]) -> List[SchoolRow]:
    """
    Загружаем школы вместе с муниципалитетом.
    region_name должен быть каноническим именем региона из edu.region.
    """
    if region_name:
        cur.execute(
            """
            SELECT s.school_id, s.full_name, m.name
            FROM edu.school s
            JOIN edu.municipality m ON m.municipality_id = s.municipality_id
            JOIN edu.region r ON r.region_id = m.region_id
            WHERE lower(r.name) = lower(%s)
            """,
            (region_name,),
        )
    else:
        cur.execute(
            """
            SELECT s.school_id, s.full_name, m.name
            FROM edu.school s
            JOIN edu.municipality m ON m.municipality_id = s.municipality_id
            """
        )

    rows = cur.fetchall()
    return [SchoolRow(int(r[0]), str(r[1]), str(r[2]) if r[2] is not None else None) for r in rows]


def build_school_code3_index(schools: List[SchoolRow]) -> Tuple[Dict[str, int], set[str]]:
    code_ids: Dict[str, List[int]] = {}
    for s in schools:
        code3 = extract_school_code3(s.full_name)
        if not code3:
            continue
        code_ids.setdefault(code3, []).append(s.school_id)

    code_first: Dict[str, int] = {}
    code_amb: set[str] = set()
    for code3, ids in code_ids.items():
        uniq = sorted(set(ids))
        code_first[code3] = uniq[0]
        if len(uniq) > 1:
            code_amb.add(code3)

    return code_first, code_amb


def build_school_index(
    schools: List[SchoolRow],
) -> Tuple[
    Dict[Tuple[str, str], int],
    Dict[Tuple[str, str], int],
    Dict[str, int],
    set[Tuple[str, str]],
    set[Tuple[str, str]],
    set[str],
]:
    """
    Возвращает:
    - pair_exact_unique[(municipality_text, school_key)] = school_id (если уникально)
    - pair_unique[(muni_key, school_key)] = school_id (если уникально)
    - school_unique[school_key] = school_id (если уникально по всей выборке)
    - pair_exact_ambiguous: точные пары municipality_text + school_key с несколькими school_id
    - pair_ambiguous: пары с несколькими school_id
    - school_ambiguous: school_key с несколькими school_id
    """
    pair_exact_ids: Dict[Tuple[str, str], List[int]] = {}
    pair_ids: Dict[Tuple[str, str], List[int]] = {}
    school_ids: Dict[str, List[int]] = {}

    for s in schools:
        sk = make_school_key(s.full_name)
        mk_exact = normalize_municipality_name(s.municipality_name or "")
        mk = make_muni_key(s.municipality_name or "")
        pair_exact_ids.setdefault((mk_exact, sk), []).append(s.school_id)
        pair_ids.setdefault((mk, sk), []).append(s.school_id)
        school_ids.setdefault(sk, []).append(s.school_id)

    pair_exact_unique: Dict[Tuple[str, str], int] = {}
    pair_exact_amb: set[Tuple[str, str]] = set()
    for k, ids in pair_exact_ids.items():
        uniq = sorted(set(ids))
        # Если дубли возникли только из-за разного регистра муниципалитета в справочнике,
        # выбираем детерминированно минимальный school_id.
        pair_exact_unique[k] = uniq[0]
        if len(uniq) > 1:
            pair_exact_amb.add(k)

    pair_unique: Dict[Tuple[str, str], int] = {}
    pair_amb: set[Tuple[str, str]] = set()
    for k, ids in pair_ids.items():
        uniq = sorted(set(ids))
        if len(uniq) == 1:
            pair_unique[k] = uniq[0]
        else:
            pair_amb.add(k)

    school_unique: Dict[str, int] = {}
    school_amb: set[str] = set()
    for k, ids in school_ids.items():
        uniq = sorted(set(ids))
        if len(uniq) == 1:
            school_unique[k] = uniq[0]
        else:
            school_amb.add(k)

    return pair_exact_unique, pair_unique, school_unique, pair_exact_amb, pair_amb, school_amb


# Запись данных в БД (upsert).

def upsert_ege_school_year(cur, rows: List[Tuple[int, int, str, int]]) -> Dict[int, int]:
    """
    rows: (school_id, year, kind, graduates_total)
    return: {school_id: ege_school_year_id}
    """
    q = """
        INSERT INTO edu.ege_school_year (school_id, "year", kind, graduates_total)
        VALUES %s
        ON CONFLICT (school_id, "year", kind) DO UPDATE
        SET graduates_total = EXCLUDED.graduates_total
        RETURNING ege_school_year_id, school_id
    """
    returned = execute_values(cur, q, rows, page_size=1000, fetch=True)
    return {int(sid): int(yid) for (yid, sid) in returned}


def upsert_ege_subject_stats(
    cur,
    rows: List[Tuple[int, int, Optional[int], Optional[int], Optional[int], Optional[int], Optional[Decimal], Optional[int]]],
) -> None:
    q = """
        INSERT INTO edu.ege_school_subject_stat (
            ege_school_year_id,
            subject_id,
            participants_cnt,
            not_passed_cnt,
            high_80_99_cnt,
            score_100_cnt,
            avg_score,
            chosen_cnt
        )
        VALUES %s
        ON CONFLICT (ege_school_year_id, subject_id) DO UPDATE SET
            participants_cnt = EXCLUDED.participants_cnt,
            not_passed_cnt = EXCLUDED.not_passed_cnt,
            high_80_99_cnt = EXCLUDED.high_80_99_cnt,
            score_100_cnt = EXCLUDED.score_100_cnt,
            avg_score = EXCLUDED.avg_score,
            chosen_cnt = EXCLUDED.chosen_cnt
    """
    execute_values(cur, q, rows, page_size=2000)


# Агрегация, если в Excel несколько строк на одну школу.

def _sum_int(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def _quantize_avg(sum_val: Decimal, weight: int) -> Optional[Decimal]:
    if weight <= 0:
        return None
    return (sum_val / Decimal(weight)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def merge_subject_acc(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """
    dst аккумулирует:
    - counts: суммируем
    - avg_score: считаем среднее взвешенное по participants_cnt (если есть), иначе по 1
    """
    # Счётчики.
    for k in ("participants_cnt", "not_passed_cnt", "high_80_99_cnt", "score_100_cnt", "chosen_cnt"):
        dst[k] = _sum_int(dst.get(k), src.get(k))

    # Средние значения.
    a = src.get("avg_score")
    if a is None:
        return

    w = src.get("participants_cnt")
    w = int(w) if isinstance(w, int) and w > 0 else 1

    dst["_avg_sum"] = (dst.get("_avg_sum") or Decimal("0")) + (a * Decimal(w))
    dst["_avg_w"] = (dst.get("_avg_w") or 0) + w


def sanitize_subject_metrics(
    metrics: Dict[str, Any],
    *,
    school_id: int,
    subject: str,
    correction_stats: Optional[Dict[str, int]] = None,
    log_limit: int = 30,
) -> None:
    """
    Приводим метрики к ограничениям БД:
    - если participants_cnt отсутствует, восстанавливаем по максимуму счетчиков;
    - счетчики не могут превышать participants_cnt.
    """
    participants = metrics.get("participants_cnt")
    not_passed = metrics.get("not_passed_cnt")
    high = metrics.get("high_80_99_cnt")
    score_100 = metrics.get("score_100_cnt")
    chosen = metrics.get("chosen_cnt")
    avg_score = metrics.get("avg_score")

    if participants is None:
        derived = max((v for v in (not_passed, high, score_100) if isinstance(v, int)), default=None)
        if derived is not None:
            metrics["participants_cnt"] = derived
            participants = derived
            if correction_stats is not None:
                correction_stats["restored"] = correction_stats.get("restored", 0) + 1
                if correction_stats.get("logged", 0) < log_limit:
                    logging.warning(
                        "Восстановлен participants_cnt: school_id=%s subject='%s' value=%s",
                        school_id,
                        subject,
                        derived,
                    )
                    correction_stats["logged"] = correction_stats.get("logged", 0) + 1

    if participants is None:
        has_positive_data = any(isinstance(v, int) and v > 0 for v in (not_passed, high, score_100, chosen))
        if avg_score is not None and not has_positive_data:
            try:
                if float(avg_score) == 0.0:
                    metrics["avg_score"] = None
                    if correction_stats is not None:
                        correction_stats["avg_nullified"] = correction_stats.get("avg_nullified", 0) + 1
            except (TypeError, ValueError):
                pass
        return

    has_positive_data = any(isinstance(v, int) and v > 0 for v in (participants, not_passed, high, score_100, chosen))
    if avg_score is not None and not has_positive_data:
        try:
            if float(avg_score) == 0.0:
                metrics["avg_score"] = None
                if correction_stats is not None:
                    correction_stats["avg_nullified"] = correction_stats.get("avg_nullified", 0) + 1
        except (TypeError, ValueError):
            pass

    for key in ("not_passed_cnt", "high_80_99_cnt", "score_100_cnt", "chosen_cnt"):
        value = metrics.get(key)
        if isinstance(value, int) and value > participants:
            if correction_stats is not None:
                correction_stats["clamped"] = correction_stats.get("clamped", 0) + 1
                if correction_stats.get("logged", 0) < log_limit:
                    logging.warning(
                        "Скорректировано %s: school_id=%s subject='%s' %s -> %s (participants_cnt=%s)",
                        key,
                        school_id,
                        subject,
                        value,
                        participants,
                        participants,
                    )
                    correction_stats["logged"] = correction_stats.get("logged", 0) + 1
            metrics[key] = participants


# Основной процесс загрузки.

def load_ege_to_db(
    path: Path,
    kind_ui: str,
    sheet: str,
    year: int,
    region_name: Optional[str],
    dry_run: bool,
) -> None:
    if psycopg2 is None or execute_values is None:
        raise RuntimeError("psycopg2 не установлен. Установи зависимости и повтори запуск.")

    db_cfg = get_db_config(search_from=Path(__file__))
    logging.info("Файл: %s", path)
    logging.info("Лист: %s", sheet)
    logging.info("Год: %s", year)
    logging.info("Регион: %s", region_name or "(без фильтра)")
    logging.info("Подключение: host=%s port=%s db=%s user=%s", db_cfg["host"], db_cfg["port"], db_cfg["dbname"], db_cfg["user"])

    with psycopg2.connect(**db_cfg) as conn:
        with conn.cursor() as cur:
            allowed_kinds = fetch_allowed_kinds(cur)
            kind_db = normalize_kind_for_db(kind_ui, allowed=allowed_kinds)

            # Читаем Excel для выбранного типа данных.
            if kind_db == "plan":
                df = read_plan_excel(path, sheet)
            else:
                df = read_actual_excel(path, sheet)

            logging.info("Прочитано строк (после чистки): %s", len(df))
            if kind_db == "plan":
                subject_metric_cols = [c for c in df.columns if str(c).startswith("chosen__")]
                if not subject_metric_cols:
                    raise ValueError(
                        "Для предварительных данных на выбранном листе не найдены предметные колонки. "
                        "Проверьте тип данных/лист файла."
                    )
            if kind_db == "actual":
                subject_metric_cols = [c for c in df.columns if "__" in str(c)]
                if not subject_metric_cols:
                    raise ValueError(
                        "Для фактических данных на выбранном листе не найдены предметные колонки. "
                        "Проверьте тип данных/лист файла."
                    )

            # Загружаем справочники.
            subject_name_to_id = fetch_subject_map(cur)
            resolved_region_name = resolve_region_name(cur, region_name)
            if region_name and not resolved_region_name:
                logging.warning(
                    "Регион '%s' не распознан, сопоставление школ выполняется без фильтра по региону.",
                    region_name,
                )
            elif (
                region_name
                and resolved_region_name
                and norm_spaces(region_name).casefold() != norm_spaces(resolved_region_name).casefold()
            ):
                logging.info("Регион '%s' сопоставлен как '%s'.", region_name, resolved_region_name)

            schools = fetch_schools(cur, region_name=resolved_region_name)
            pair_exact_unique, pair_unique, school_unique, pair_exact_amb, pair_amb, school_amb = build_school_index(schools)
            use_code3_fallback = is_primorsky_region(resolved_region_name)
            code3_first: Dict[str, int] = {}
            code3_amb: set[str] = set()
            if use_code3_fallback:
                code3_first, code3_amb = build_school_code3_index(schools)
                logging.info(
                    "Включен fallback по коду школы для региона '%s': mapped=%s ambiguous=%s",
                    resolved_region_name,
                    len(code3_first),
                    len(code3_amb),
                )

            # Агрегируем данные по school_id.
            per_school_grads: Dict[int, int] = {}
            per_school_subjects: Dict[int, Dict[str, Dict[str, Any]]] = {}

            skipped_not_found = 0
            skipped_ambiguous = 0
            matched = 0

            for _, r in df.iterrows():
                mun = normalize_municipality_name(r.get("municipality"))
                school_raw = r.get("school")
                sch = standardize_school_name(school_raw)
                if not mun or not sch:
                    continue

                mk = make_muni_key(mun)
                sk = make_school_key(sch)
                code3 = extract_school_code3(school_raw) if use_code3_fallback else None

                sid = pair_exact_unique.get((mun, sk))
                if sid is None:
                    sid = pair_unique.get((mk, sk))
                if sid is None and code3 is not None:
                    sid = code3_first.get(code3)
                if sid is None:
                    sid = school_unique.get(sk)

                if sid is None:
                    # Различаем случаи "не нашли" и "неоднозначно".
                    code3_is_amb = bool(code3 and code3 in code3_amb)
                    if (mun, sk) in pair_exact_amb or (mk, sk) in pair_amb or sk in school_amb or code3_is_amb:
                        skipped_ambiguous += 1
                        logging.warning(
                            "Неоднозначная школа: municipality='%s' school='%s' code3='%s'",
                            mun,
                            sch,
                            code3 or "",
                        )
                    else:
                        skipped_not_found += 1
                        logging.warning(
                            "Школа не найдена: municipality='%s' school='%s' code3='%s'",
                            mun,
                            sch,
                            code3 or "",
                        )
                    continue

                matched += 1

                grads = to_int(r.get("graduates_total")) or 0
                per_school_grads[sid] = per_school_grads.get(sid, 0) + grads

                per_school_subjects.setdefault(sid, {})

                if kind_db == "plan":
                    for subj in SUBJECT_CANON:
                        col = f"chosen__{subj}"
                        if col not in df.columns:
                            continue
                        chosen = to_int(r.get(col))
                        if chosen is None:
                            continue

                        acc = per_school_subjects[sid].setdefault(
                            subj,
                            {
                                "participants_cnt": None,
                                "not_passed_cnt": None,
                                "high_80_99_cnt": None,
                                "score_100_cnt": None,
                                "avg_score": None,
                                "chosen_cnt": 0,
                            },
                        )
                        acc["chosen_cnt"] = (acc.get("chosen_cnt") or 0) + chosen
                else:
                    for subj in SUBJECT_CANON:
                        p = to_int(r.get(f"participants_cnt__{subj}"))
                        n = to_int(r.get(f"not_passed_cnt__{subj}"))
                        h = to_int(r.get(f"high_80_99_cnt__{subj}"))
                        s100 = to_int(r.get(f"score_100_cnt__{subj}"))
                        a = to_decimal_2(r.get(f"avg_score__{subj}"))

                        if p is None and n is None and h is None and s100 is None and a is None:
                            continue

                        src = {
                            "participants_cnt": p,
                            "not_passed_cnt": n,
                            "high_80_99_cnt": h,
                            "score_100_cnt": s100,
                            "avg_score": a,
                            "chosen_cnt": p,  # Для actual храним выбранных в participants_cnt.
                        }
                        dst = per_school_subjects[sid].setdefault(
                            subj,
                            {
                                "participants_cnt": None,
                                "not_passed_cnt": None,
                                "high_80_99_cnt": None,
                                "score_100_cnt": None,
                                "avg_score": None,
                                "chosen_cnt": None,
                                "_avg_sum": Decimal("0"),
                                "_avg_w": 0,
                            },
                        )
                        merge_subject_acc(dst, src)

            # Приводим avg_score к финальному виду.
            if kind_db == "actual":
                for sid, subj_map in per_school_subjects.items():
                    for subj, m in subj_map.items():
                        if m.get("_avg_w"):
                            m["avg_score"] = _quantize_avg(m.get("_avg_sum", Decimal("0")), int(m["_avg_w"]))
                        m.pop("_avg_sum", None)
                        m.pop("_avg_w", None)

            logging.info(
                "Сопоставление: matched=%s, skipped_not_found=%s, skipped_ambiguous=%s",
                matched,
                skipped_not_found,
                skipped_ambiguous,
            )

            school_year_rows = [(sid, year, kind_db, grads) for sid, grads in per_school_grads.items()]

            if dry_run:
                logging.info("dry-run: к вставке ege_school_year=%s", len(school_year_rows))
                logging.info("dry-run: пример 3 строк: %s", school_year_rows[:3])
                return

            if not school_year_rows:
                logging.warning("Нет данных для вставки в edu.ege_school_year")
                return

            school_id_to_year_id = upsert_ege_school_year(cur, school_year_rows)

            stats_rows: List[
                Tuple[int, int, Optional[int], Optional[int], Optional[int], Optional[int], Optional[Decimal], Optional[int]]
            ] = []
            correction_stats: Dict[str, int] = {"restored": 0, "clamped": 0, "logged": 0}
            for sid, subj_map in per_school_subjects.items():
                yid = school_id_to_year_id.get(sid)
                if not yid:
                    continue

                for subj, m in subj_map.items():
                    sanitize_subject_metrics(
                        m,
                        school_id=sid,
                        subject=subj,
                        correction_stats=correction_stats,
                    )
                    subject_id = subject_name_to_id.get(subj)
                    if subject_id is None:
                        logging.warning("Не найден subject_id для предмета: %s", subj)
                        continue

                    stats_rows.append(
                        (
                            yid,
                            subject_id,
                            m.get("participants_cnt"),
                            m.get("not_passed_cnt"),
                            m.get("high_80_99_cnt"),
                            m.get("score_100_cnt"),
                            m.get("avg_score"),
                            m.get("chosen_cnt"),
                        )
                    )

            if stats_rows:
                upsert_ege_subject_stats(cur, stats_rows)

            if correction_stats["restored"] or correction_stats["clamped"] or correction_stats.get("avg_nullified", 0):
                suppressed = max(
                    0,
                    correction_stats["restored"] + correction_stats["clamped"] - correction_stats["logged"],
                )
                logging.warning(
                    "Автокоррекция метрик ЕГЭ: restored_participants=%s, clamped_counts=%s, avg_nullified=%s, suppressed_logs=%s",
                    correction_stats["restored"],
                    correction_stats["clamped"],
                    correction_stats.get("avg_nullified", 0),
                    suppressed,
                )

            logging.info(
                "Готово: ege_school_year=%s, ege_school_subject_stat=%s",
                len(school_year_rows),
                len(stats_rows),
            )


# Интерфейс командной строки.

def main() -> None:
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    load_env_file(search_from=Path(__file__))

    ap = argparse.ArgumentParser(description="Загрузка данных ЕГЭ из Excel в PostgreSQL")
    ap.add_argument("--file", help="Путь к Excel. Если не указан – откроется диалог выбора файла.")
    ap.add_argument("--sheet", help="Имя листа (если не указано – будет интерактивный выбор).")
    ap.add_argument("--kind", help="plan (предварительные) или actual (фактические). Можно 1/2 или preliminary/actual.")
    ap.add_argument("--year", type=int, help="Год данных. Если не указан – будет взят из названия листа или запрошен.")
    ap.add_argument("--region", help="Название региона для фильтрации школ (опционально). По умолчанию – имя файла.")
    ap.add_argument("--dry-run", action="store_true", help="Без записи в БД (проверка чтения и сопоставления).")

    args = ap.parse_args()

    if args.file:
        path = resolve_user_path(args.file)
    else:
        path = pick_file_dialog_desktop()
        if path is None:
            s = input("Диалог недоступен. Введи путь или имя файла: ").strip()
            path = resolve_user_path(s)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    kind_ui = args.kind or choose_kind_ui()
    sheet = args.sheet or choose_sheet(path)
    year = args.year or infer_year_from_sheet(sheet) or int(input("Не удалось определить год. Введи год (например 2023): ").strip())
    region_name = args.region or norm_spaces(path.stem.replace("_", " "))

    load_ege_to_db(
        path=path,
        kind_ui=kind_ui,
        sheet=sheet,
        year=year,
        region_name=region_name,
        dry_run=args.dry_run,
    )
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем")
        sys.exit(1)
