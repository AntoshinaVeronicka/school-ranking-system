# -*- coding: utf-8 -*-
"""
Применение SQL-оптимизаций для поиска школ.

Запускает скрипт индексов/расширений из db/sql,
нужный для ускорения текстового поиска по школам.
"""

from __future__ import annotations

from pathlib import Path

import psycopg2

from load_common import get_db_config


def main() -> int:
    script_path = Path(__file__).resolve().parents[1] / "db" / "sql" / "search_school_optimization.sql"
    if not script_path.exists():
        raise FileNotFoundError(f"SQL-файл не найден: {script_path}")

    sql_text = script_path.read_text(encoding="utf-8")
    db_cfg = get_db_config(search_from=Path(__file__))

    with psycopg2.connect(**db_cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_text)
        conn.commit()

    print(f"Оптимизация поиска применена: {script_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
