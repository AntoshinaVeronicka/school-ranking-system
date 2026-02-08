# -*- coding: utf-8 -*-
"""
Загрузка данных ЕГЭ из Excel в PostgreSQL (edu.*).

Поддерживает 2 формата:
A) Предварительные данные (kind=plan) – фиксированные колонки + 1 колонка "выбрали" на предмет.
B) Фактические данные (kind=actual) – фиксированные колонки + многоуровневые заголовки по метрикам на предмет.

Таблицы:
- edu.ege_school_year (school_id, year, kind)
- edu.ege_school_subject_stat (ege_school_year_id, subject_id)

Сопоставление школ:
- Нормализуем school.full_name из БД и school из Excel одной функцией
- Основной матч: (муниципалитет, школа)
- Fallback: по школе, если имя уникально в выбранном регионе
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
    norm_spaces,
    pick_file_dialog_desktop,
    resolve_user_path,
    standardize_school_name,
)

try:
    import psycopg2
    from psycopg2.extras import execute_values
except Exception:
    psycopg2 = None  # type: ignore
    execute_values = None  # type: ignore

warnings.filterwarnings(
    "ignore", message=r"Print area cannot be set.*", category=UserWarning, module="openpyxl"
)


# -------------------- logging --------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -------------------- subjects --------------------

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


# -------------------- converters --------------------

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


# -------------------- excel reading helpers --------------------

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
    out["municipality"] = out["municipality"].map(norm_spaces)
    out["school"] = out["school"].map(norm_spaces).map(standardize_school_name)

    # фильтры мусора
    out = out[(out["municipality"] != "") & (out["school"] != "")]
    out = out[~out["municipality"].str.contains(r"\bвсего\b", case=False, na=False)]
    out = out[~out["school"].str.contains(r"\bвсего\b", case=False, na=False)]
    out = out[~out["school"].astype(str).str.fullmatch(r"\d+")]
    return out.reset_index(drop=True)


def read_plan_excel(path: Path, sheet: str) -> pd.DataFrame:
    header_row = find_header_row(path, sheet)
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl", header=header_row)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    mun_col, school_col, _, grads_col = resolve_fixed_cols_flat(df)

    # часто в таких таблицах муниципалитет и школа "сверху вниз" – протягиваем
    df[mun_col] = df[mun_col].ffill()
    df[school_col] = df[school_col].ffill()

    out = pd.DataFrame(
        {
            "municipality": df[mun_col],
            "school": df[school_col],
            "graduates_total": df[grads_col],
        }
    )

    for c in df.columns:
        if c in (mun_col, school_col, grads_col):
            continue
        subj = normalize_subject_title(c)
        if subj in SUBJECT_CANON:
            out[f"chosen__{subj}"] = df[c]

    return clean_common_rows(out)


def read_actual_excel(path: Path, sheet: str) -> pd.DataFrame:
    header_row = find_header_row(path, sheet)
    df = pd.read_excel(
        path,
        sheet_name=sheet,
        engine="openpyxl",
        header=[header_row, header_row + 1, header_row + 2],
    )
    df = df.dropna(how="all").dropna(axis=1, how="all")

    def pick_fixed(prefix: str) -> Optional[Tuple[Any, Any, Any]]:
        for c in df.columns:
            if str(c[0]).casefold().startswith(prefix):
                return c
        return None

    mun_t = pick_fixed("муницип")
    school_t = pick_fixed("образоват")
    prof_t = pick_fixed("профил")  # not used, but may exist

    grads_t = None
    for c in df.columns:
        c0 = str(c[0]).casefold()
        if "всего" in c0 and "выпуск" in c0:
            grads_t = c
            break

    if not mun_t or not school_t or not grads_t:
        raise ValueError("Не найдены фиксированные колонки (муниципалитет/школа/выпускники).")

    out = pd.DataFrame(
        {
            "municipality": df[mun_t],
            "school": df[school_t],
            "graduates_total": df[grads_t],
        }
    )

    for col in df.columns:
        if str(col[0]).strip().casefold() != "из них":
            continue

        subj = normalize_subject_title(col[1])
        if subj not in SUBJECT_CANON:
            continue

        metric_raw = norm_spaces(str(col[2]).replace("\n", " ")).casefold()

        if "принял" in metric_raw and "участ" in metric_raw:
            metric = "participants_cnt"
        elif "не преодол" in metric_raw or "min" in metric_raw or "порог" in metric_raw:
            metric = "not_passed_cnt"
        elif "высок" in metric_raw or "80-99" in metric_raw:
            metric = "high_80_99_cnt"
        elif "100" in metric_raw:
            metric = "score_100_cnt"
        elif "средн" in metric_raw and "балл" in metric_raw:
            metric = "avg_score"
        else:
            continue

        out[f"{metric}__{subj}"] = df[col]

    return clean_common_rows(out)


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


def fetch_schools(cur, region_name: Optional[str]) -> List[SchoolRow]:
    """
    Пытаемся подтянуть школы вместе с муниципалитетом и фильтром по региону.
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


def build_school_index(
    schools: List[SchoolRow],
) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int], set[Tuple[str, str]], set[str]]:
    """
    Возвращает:
    - pair_unique[(muni_key, school_key)] = school_id (если уникально)
    - school_unique[school_key] = school_id (если уникально по всей выборке)
    - pair_ambiguous: пары с несколькими school_id
    - school_ambiguous: school_key с несколькими school_id
    """
    pair_ids: Dict[Tuple[str, str], List[int]] = {}
    school_ids: Dict[str, List[int]] = {}

    for s in schools:
        sk = make_school_key(s.full_name)
        mk = make_muni_key(s.municipality_name or "")
        pair_ids.setdefault((mk, sk), []).append(s.school_id)
        school_ids.setdefault(sk, []).append(s.school_id)

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

    return pair_unique, school_unique, pair_amb, school_amb


# -------------------- upsert --------------------

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


# -------------------- aggregation (если в Excel несколько строк на одну школу) --------------------

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
    # counts
    for k in ("participants_cnt", "not_passed_cnt", "high_80_99_cnt", "score_100_cnt", "chosen_cnt"):
        dst[k] = _sum_int(dst.get(k), src.get(k))

    # avg
    a = src.get("avg_score")
    if a is None:
        return

    w = src.get("participants_cnt")
    w = int(w) if isinstance(w, int) and w > 0 else 1

    dst["_avg_sum"] = (dst.get("_avg_sum") or Decimal("0")) + (a * Decimal(w))
    dst["_avg_w"] = (dst.get("_avg_w") or 0) + w


# -------------------- main load --------------------

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

            # читаем Excel под kind
            if kind_db == "plan":
                df = read_plan_excel(path, sheet)
            else:
                df = read_actual_excel(path, sheet)

            logging.info("Прочитано строк (после чистки): %s", len(df))

            # справочники
            subject_name_to_id = fetch_subject_map(cur)
            schools = fetch_schools(cur, region_name=region_name)
            pair_unique, school_unique, pair_amb, school_amb = build_school_index(schools)

            # аккумулируем по school_id
            per_school_grads: Dict[int, int] = {}
            per_school_subjects: Dict[int, Dict[str, Dict[str, Any]]] = {}

            skipped_not_found = 0
            skipped_ambiguous = 0
            matched = 0

            for _, r in df.iterrows():
                mun = norm_spaces(r.get("municipality"))
                sch = standardize_school_name(r.get("school"))
                if not mun or not sch:
                    continue

                mk = make_muni_key(mun)
                sk = make_school_key(sch)

                sid = pair_unique.get((mk, sk))
                if sid is None:
                    sid = school_unique.get(sk)

                if sid is None:
                    # различаем "не нашли" и "неоднозначно"
                    if (mk, sk) in pair_amb or sk in school_amb:
                        skipped_ambiguous += 1
                        logging.warning("Неоднозначная школа: municipality='%s' school='%s'", mun, sch)
                    else:
                        skipped_not_found += 1
                        logging.warning("Школа не найдена: municipality='%s' school='%s'", mun, sch)
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
                            "chosen_cnt": p,  # выбранных в actual храним как participants_cnt
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

            # приводим avg_score к финальному виду
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
            for sid, subj_map in per_school_subjects.items():
                yid = school_id_to_year_id.get(sid)
                if not yid:
                    continue

                for subj, m in subj_map.items():
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

            logging.info(
                "Готово: ege_school_year=%s, ege_school_subject_stat=%s",
                len(school_year_rows),
                len(stats_rows),
            )


# -------------------- CLI --------------------

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
