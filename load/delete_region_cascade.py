# -*- coding: utf-8 -*-
"""
Каскадная очистка региона и связанных данных.

Удаляет:
- запись в edu.region;
- все муниципалитеты региона (edu.municipality);
- все школы этих муниципалитетов (edu.school);
- записи в аналитических таблицах с FK ON DELETE RESTRICT, которые иначе блокируют удаление.

По умолчанию работает в dry-run (с откатом). Для фактического удаления укажи --apply.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import psycopg2

from load_common import get_db_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Удаление региона и связанных муниципалитетов/школ")
    p.add_argument(
        "--region",
        default="Еврейская автономная область",
        help="Точное имя региона в таблице edu.region",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Применить удаление (без этого выполняется dry-run и откат)",
    )
    return p.parse_args()


def scalar(cur, sql: str, params=()) -> int:
    cur.execute(sql, params)
    row = cur.fetchone()
    return int(row[0]) if row else 0


def delete_with_count(cur, sql: str, params=()) -> int:
    cur.execute(sql, params)
    return int(cur.rowcount)


def main() -> int:
    args = parse_args()
    db_cfg = get_db_config(search_from=Path(__file__))

    with psycopg2.connect(**db_cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT region_id, name
                FROM edu.region
                WHERE name = %s
                """,
                (args.region,),
            )
            row = cur.fetchone()
            if row is None:
                print(f"Регион не найден: {args.region}")
                return 2

            region_id = int(row[0])
            region_name = str(row[1])

            cur.execute(
                """
                CREATE TEMP TABLE tmp_target_municipality ON COMMIT DROP AS
                SELECT municipality_id
                FROM edu.municipality
                WHERE region_id = %s
                """,
                (region_id,),
            )
            cur.execute(
                """
                CREATE TEMP TABLE tmp_target_school ON COMMIT DROP AS
                SELECT school_id
                FROM edu.school
                WHERE municipality_id IN (SELECT municipality_id FROM tmp_target_municipality)
                """
            )

            scope_municipalities = scalar(cur, "SELECT COUNT(*) FROM tmp_target_municipality")
            scope_schools = scalar(cur, "SELECT COUNT(*) FROM tmp_target_school")

            print("Область удаления:")
            print(f"- region_id: {region_id}")
            print(f"- region_name: {region_name}")
            print(f"- муниципалитетов: {scope_municipalities}")
            print(f"- школ: {scope_schools}")

            deleted: Dict[str, int] = {}

            # Сначала удаляем строки из таблиц с ON DELETE RESTRICT.
            deleted["analytics_request_filter"] = delete_with_count(
                cur,
                """
                DELETE FROM edu.analytics_request_filter
                WHERE region_id = %s
                   OR municipality_id IN (SELECT municipality_id FROM tmp_target_municipality)
                   OR school_id IN (SELECT school_id FROM tmp_target_school)
                """,
                (region_id,),
            )
            deleted["analytics_school_selection"] = delete_with_count(
                cur,
                """
                DELETE FROM edu.analytics_school_selection
                WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )
            deleted["analytics_school_metric_value"] = delete_with_count(
                cur,
                """
                DELETE FROM edu.analytics_school_metric_value
                WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )
            deleted["analytics_school_rating"] = delete_with_count(
                cur,
                """
                DELETE FROM edu.analytics_school_rating
                WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )
            deleted["analytics_school_card"] = delete_with_count(
                cur,
                """
                DELETE FROM edu.analytics_school_card
                WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )
            deleted["analytics_school_report"] = delete_with_count(
                cur,
                """
                DELETE FROM edu.analytics_school_report
                WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )

            # Основные справочники. Остальные связанные таблицы удалятся каскадно через FK от school.
            deleted["school"] = delete_with_count(
                cur,
                """
                DELETE FROM edu.school
                WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )
            deleted["municipality"] = delete_with_count(
                cur,
                """
                DELETE FROM edu.municipality
                WHERE municipality_id IN (SELECT municipality_id FROM tmp_target_municipality)
                """,
            )
            deleted["region"] = delete_with_count(
                cur,
                """
                DELETE FROM edu.region
                WHERE region_id = %s
                """,
                (region_id,),
            )

            print("Удалено строк:")
            for table_name, count in deleted.items():
                print(f"- {table_name}: {count}")

            if args.apply:
                conn.commit()
                print("Изменения применены.")
            else:
                conn.rollback()
                print("Dry-run завершён: изменения откатаны. Для применения добавьте --apply.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
