# -*- coding: utf-8 -*-
"""
Общие утилиты для загрузчиков:
- нормализация строк и названий школ;
- выбор файла (по умолчанию с рабочего стола);
- разрешение путей к файлу;
- загрузка .env и получение конфигурации БД.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # type: ignore


def norm_spaces(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00A0", " ")
    return re.sub(r"\s+", " ", s.strip())


def normalize_municipality_name(s: Any) -> str:
    """
    Каноническая форма муниципалитета:
    - схлопываем пробелы;
    - убираем пустые/nan-like значения;
    - нормализуем префикс города к виду "город ".
    """
    text = norm_spaces(s)
    if not text:
        return ""
    if text.casefold() in {"nan", "none", "nat"}:
        return ""
    text = re.sub(r"^\s*г(?:\.|\s+)", "город ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*город\s+", "город ", text, flags=re.IGNORECASE)
    return text


def normalize_text(s: Any) -> str:
    return re.sub(r"\s+", " ", ("" if s is None else str(s)).strip())


def normalize_upper_key(s: Any) -> str:
    return normalize_text(s).upper().replace("Ё", "Е")


def clean_inner_quotes(s: Any) -> str:
    s = "" if s is None else str(s)
    return s.replace("«", "").replace("»", "").replace('"', "").replace("'", "")


def strip_outer_quotes(s: Any) -> str:
    s = norm_spaces(s)
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


# -------------------- Regex (техническая нормализация) --------------------

_PARENS_RE = re.compile(r"\(([^)]*)\)")                         # (текст) -> текст
_LEADING_NUM_RE = re.compile(r"^\s*\d+\s*[-–—]\s*")             # "138049 - "
_DASHES_RE = re.compile(r"[–—]")                                # –/— -> -
_GLUE_RE = re.compile(r"(?<=[а-яё])(?=[А-ЯЁ])")                 # "...краяМБОУ..." -> "...края МБОУ..."
_DOT_NO_SPACE_RE = re.compile(r"(?<=[A-Za-zА-Яа-яЁё])\.(?=[A-Za-zА-Яа-яЁё])")  # "г.Х" -> "г. Х"
_EXCEL_REF_TAIL_RE = re.compile(r"\s*\+\s*[A-ZА-Я]+\s*\d+\s*$", flags=re.IGNORECASE)  # "+B116" в конце

# -------------------- Опечатки (дополняй по мере встречаемости) --------------------

TYPO_REPLACEMENTS: List[Tuple[str, str]] = [
    (r"\bгосудаственн", "государственн"),
    (r"\bгосудаственого\b", "государственного"),
    (r"\bгосудаственного\b", "государственного"),
    (r"\bобщеоразовательн", "общеобразовательн"),
    (r"\bобщео?бразовательн", "общеобразовательн"),
    (r"\bмуниципального\s+райна\b", "муниципального района"),
    (r"\bмуниципальн\w*\s+райна\b", "муниципального района"),
    (r"\bрайна\b", "района"),
    (r"\bимни\b", "имени"),
    (r"\bмуниципального\s+райна\b", "муниципального района"),
]

# -------------------- Сокращения правовой формы (только это сокращаем) --------------------

LEGAL_FORM_RULES: List[Tuple[re.Pattern, str]] = [

    (re.compile(
        r"\bМуниципальн\w*\s+каз[её]нн\w*\s*(?P<mid>(?:[\w-]+\s+){0,10})"
        r"общеобразовательн\w*\s+учрежден\w*\b",
        flags=re.IGNORECASE
    ), "МКОУ"),

    (re.compile(
        r"\bМуниципальн\w*\s+бюджетн\w*\s*(?P<mid>(?:[\w-]+\s+){0,10})"
        r"общеобразовательн\w*\s+учрежден\w*\b",
        flags=re.IGNORECASE
    ), "МБОУ"),

    (re.compile(
        r"\bМуниципальн\w*\s+автономн\w*\s*(?P<mid>(?:[\w-]+\s+){0,10})"
        r"общеобразовательн\w*\s+учрежден\w*\b",
        flags=re.IGNORECASE
    ), "МАОУ"),

    (re.compile(
        r"\bМуниципальн\w*\s*(?P<mid>(?:[\w-]+\s+){0,10})"
        r"общеобразовательн\w*\s+учрежден\w*\b",
        flags=re.IGNORECASE
    ), "МОУ"),

    (re.compile(
        r"\bЧастн\w*\s*(?P<mid>(?:[\w-]+\s+){0,10})"
        r"общеобразовательн\w*\s+учрежден\w*\b",
        flags=re.IGNORECASE
    ), "ЧОУ"),

    (re.compile(
        r"\bКраев\w*\s+государственн\w*\s+бюджетн\w*\s*(?P<mid>(?:[\w-]+\s+){0,10})"
        r"общеобразовательн\w*\s+учрежден\w*\b",
        flags=re.IGNORECASE
    ), "КГБОУ"),

    (re.compile(
        r"\bКраев\w*\s+государственн\w*\s+автономн\w*\s+нетипов\w*\s*(?P<mid>(?:[\w-]+\s+){0,10})"
        r"образовательн\w*\s+учрежден\w*\b",
        flags=re.IGNORECASE
    ), "КГАНОУ"),

    (re.compile(
        r"\bФедеральн\w*\s+государственн\w*\s+каз[её]нн\w*\s*(?P<mid>(?:[\w-]+\s+){0,10})"
        r"общеобразовательн\w*\s+учрежден\w*\b",
        flags=re.IGNORECASE
    ), "ФГКОУ"),
]


def _norm_spaces(s: str) -> str:
    s = str(s).replace("\u00A0", " ")
    return re.sub(r"\s+", " ", s).strip()


def _clean_quotes(s: str) -> str:
    return str(s).replace("«", "").replace("»", "").replace('"', "").replace("'", "").replace(">", "").replace("<", "")


def _strip_outer_quotes(s: str) -> str:
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
    return s.strip()


def _remove_parens_keep_text(s: str) -> str:
    return _PARENS_RE.sub(lambda m: f" {m.group(1)} ", s)


def _apply_typos(s: str) -> str:
    for pat, repl in TYPO_REPLACEMENTS:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    return s


def _apply_legal_form_abbr(s: str) -> str:
    for rx, abbr in LEGAL_FORM_RULES:
        def _repl(m: re.Match) -> str:
            mid = (m.groupdict().get("mid") or "").strip()
            return f"{abbr} {mid}".strip()
        s = rx.sub(_repl, s)
    return s


def standardize_school_name(name: str) -> str:
    if name is None:
        return ""

    s = str(name).replace("\u00A0", " ")

    # 1) минимальная техн. чистка
    s = _LEADING_NUM_RE.sub("", s)
    s = _DASHES_RE.sub("-", s)
    s = _GLUE_RE.sub(" ", s)
    s = _remove_parens_keep_text(s)

    s = _clean_quotes(s)
    s = _strip_outer_quotes(s)

    # "г.Хабаровска" -> "г. Хабаровска", "АП.Светогорова" -> "АП. Светогорова"
    s = _DOT_NO_SPACE_RE.sub(". ", s)

    # мусор Excel в конце: "+B116"
    s = _EXCEL_REF_TAIL_RE.sub("", s)

    # "г.Хабаровска" / "с.Нелькан" / "п.Быстринск" -> "г. Хабаровска" / "с. Нелькан" / "п. Быстринск"
    s = re.sub(r"\b([гпс])\.\s*(?=\S)", r"\1. ", s, flags=re.IGNORECASE)

    # "№2" -> "№ 2" (ничего дальше не режем)
    s = re.sub(r"№\s*(\d)", r"№ \1", s)

    s = _norm_spaces(s)

    # 2) опечатки
    s = _apply_typos(s)
    s = _norm_spaces(s)

    # 3) сокращение правовой формы (и только её)
    s = _apply_legal_form_abbr(s)
    s = _norm_spaces(s).strip(" ,;.")

    # 4) верхний регистр + Ё->Е
    s = s.upper().replace("Ё", "Е")
    return _norm_spaces(s).strip(" ,;.")


def make_school_key(name: str) -> str:
    s = standardize_school_name(name)
    return re.sub(r"[^0-9A-ZА-Я]+", "", s)


def make_muni_key(name: Any) -> str:
    s = normalize_municipality_name(name)
    s = s.upper().replace("Ё", "Е")
    return re.sub(r"[^0-9A-ZА-Я]+", "", s)


def pick_file_dialog_desktop(title: str = "Выберите Excel файл") -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()

    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        desktop = Path.home()

    selected = filedialog.askopenfilename(
        title=title,
        initialdir=str(desktop),
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(selected) if selected else None


def resolve_user_path(s: str) -> Path:
    p = Path(s)
    if p.exists():
        return p
    cand = Path.cwd() / s
    if cand.exists():
        return cand
    cand = Path.home() / "Desktop" / s
    if cand.exists():
        return cand

    if not s.lower().endswith((".xlsx", ".xls")):
        for base in (Path.cwd(), Path.home() / "Desktop"):
            cand2 = base / f"{s}.xlsx"
            if cand2.exists():
                return cand2

    raise FileNotFoundError(
        f"Файл не найден: {s}. Укажи полный путь или положи файл в текущую папку или на рабочий стол."
    )


def resolve_excel_sheet_name(path: Path, requested: str = "", engine: str = "openpyxl") -> str:
    import pandas as pd

    xls = pd.ExcelFile(path, engine=engine)
    sheets = list(xls.sheet_names)
    if not sheets:
        raise ValueError(f"В файле нет листов: {path}")

    if not requested:
        return sheets[0]

    value = str(requested).strip()
    if value.isdigit():
        idx = int(value)
        if 0 <= idx < len(sheets):
            return sheets[idx]
        raise ValueError(f"Неверный индекс листа: {idx}. Листов: {len(sheets)}")

    for s in sheets:
        if s.casefold() == value.casefold():
            return s
    raise ValueError(f"Лист '{value}' не найден. Доступно: {', '.join(sheets)}")


def load_env_file(search_from: Optional[Path] = None) -> None:
    if load_dotenv is None:
        return

    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(dotenv_path=cwd_env)
        return

    if search_from is None:
        base = Path(__file__).resolve().parent
    else:
        p = Path(search_from).resolve()
        base = p if p.is_dir() else p.parent

    for folder in [base] + list(base.parents):
        env_path = folder / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            return

    load_dotenv()


def get_db_config(search_from: Optional[Path] = None) -> Dict[str, Any]:
    load_env_file(search_from=search_from)

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    missing = [k for k, v in [("POSTGRES_DB", dbname), ("POSTGRES_USER", user), ("POSTGRES_PASSWORD", password)] if not v]
    if missing:
        raise ValueError(f"Не заданы переменные окружения: {', '.join(missing)}")

    return {"host": host, "port": port, "dbname": dbname, "user": user, "password": password}


def table_is_empty(cur, full_name: str) -> bool:
    # full_name передаем только из trusted-кода (например, "edu.table").
    cur.execute(f"SELECT COUNT(*) FROM {full_name}")
    return int(cur.fetchone()[0]) == 0


def fetch_map(cur, sql: str, params: Sequence[Any]) -> Dict[str, int]:
    cur.execute(sql, params)
    return {str(k): int(v) for (k, v) in cur.fetchall()}
