# -*- coding: utf-8 -*-
"""
Каскадная очистка региона и связанных данных.

Удаляет:
- запись в edu.region;
- все муниципалитеты региона (edu.municipality);
- все школы этих муниципалитетов (edu.school);
- данные ЕГЭ по этим школам;
- данные приёма и профориентации по этим школам;
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
        default="Камчатский край",
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


def table_exists(cur, table_name: str, schema: str = "edu") -> bool:
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"{schema}.{table_name}",))
    row = cur.fetchone()
    return bool(row and row[0])


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

            existing_tables = {
                name: table_exists(cur, name)
                for name in [
                    "region",
                    "municipality",
                    "school",
                    "school_profile_link",
                    "school_external_key",
                    "ege_school_year",
                    "ege_school_subject_stat",
                    "school_admission_stat",
                    "school_admission_detail",
                    "prof_event",
                    "analytics_request",
                    "analytics_request_filter",
                    "analytics_school_selection",
                    "analytics_school_metric_value",
                    "analytics_school_rating",
                    "analytics_school_card",
                    "analytics_school_report",
                    "analytics_rating_run",
                    "analytics_run_metric",
                ]
            }

            cur.execute("CREATE TEMP TABLE tmp_target_ege_school_year (ege_school_year_id BIGINT) ON COMMIT DROP")
            if existing_tables["ege_school_year"]:
                cur.execute(
                    """
                    INSERT INTO tmp_target_ege_school_year (ege_school_year_id)
                    SELECT y.ege_school_year_id
                    FROM edu.ege_school_year y
                    WHERE y.school_id IN (SELECT school_id FROM tmp_target_school)
                    """
                )

            cur.execute("CREATE TEMP TABLE tmp_target_adm_stat (adm_stat_id BIGINT) ON COMMIT DROP")
            if existing_tables["school_admission_stat"]:
                cur.execute(
                    """
                    INSERT INTO tmp_target_adm_stat (adm_stat_id)
                    SELECT s.adm_stat_id
                    FROM edu.school_admission_stat s
                    WHERE s.school_id IN (SELECT school_id FROM tmp_target_school)
                    """
                )

            scope_ege_year = scalar(cur, "SELECT COUNT(*) FROM tmp_target_ege_school_year")
            scope_adm_stat = scalar(cur, "SELECT COUNT(*) FROM tmp_target_adm_stat")

            print("Область удаления:")
            print(f"- region_id: {region_id}")
            print(f"- region_name: {region_name}")
            print(f"- муниципалитетов: {scope_municipalities}")
            print(f"- школ: {scope_schools}")
            print(f"- записей ege_school_year: {scope_ege_year}")
            print(f"- записей school_admission_stat: {scope_adm_stat}")

            deleted: Dict[str, int] = {}

            def delete_if_exists(key: str, table_name: str, sql: str, params=()) -> None:
                if not existing_tables.get(table_name, False):
                    deleted[key] = 0
                    return
                deleted[key] = delete_with_count(cur, sql, params)

            # Сначала удаляем строки из таблиц с ON DELETE RESTRICT.
            delete_if_exists(
                "analytics_request_filter",
                "analytics_request_filter",
                """
                    DELETE FROM edu.analytics_request_filter
                    WHERE region_id = %s
                       OR municipality_id IN (SELECT municipality_id FROM tmp_target_municipality)
                       OR school_id IN (SELECT school_id FROM tmp_target_school)
                """,
                (region_id,),
            )
            delete_if_exists(
                "analytics_school_selection",
                "analytics_school_selection",
                """
                    DELETE FROM edu.analytics_school_selection
                    WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )
            delete_if_exists(
                "analytics_school_metric_value",
                "analytics_school_metric_value",
                """
                    DELETE FROM edu.analytics_school_metric_value
                    WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )
            delete_if_exists(
                "analytics_school_rating",
                "analytics_school_rating",
                """
                    DELETE FROM edu.analytics_school_rating
                    WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )
            delete_if_exists(
                "analytics_school_card",
                "analytics_school_card",
                """
                    DELETE FROM edu.analytics_school_card
                    WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )
            delete_if_exists(
                "analytics_school_report",
                "analytics_school_report",
                """
                    DELETE FROM edu.analytics_school_report
                    WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )

            # Данные ЕГЭ.
            delete_if_exists(
                "ege_school_subject_stat",
                "ege_school_subject_stat",
                """
                    DELETE FROM edu.ege_school_subject_stat
                    WHERE ege_school_year_id IN (SELECT ege_school_year_id FROM tmp_target_ege_school_year)
                """,
            )
            delete_if_exists(
                "ege_school_year",
                "ege_school_year",
                """
                    DELETE FROM edu.ege_school_year
                    WHERE ege_school_year_id IN (SELECT ege_school_year_id FROM tmp_target_ege_school_year)
                """,
            )

            # Данные приёма.
            delete_if_exists(
                "school_admission_detail",
                "school_admission_detail",
                """
                    DELETE FROM edu.school_admission_detail
                    WHERE adm_stat_id IN (SELECT adm_stat_id FROM tmp_target_adm_stat)
                """,
            )
            delete_if_exists(
                "school_admission_stat",
                "school_admission_stat",
                """
                    DELETE FROM edu.school_admission_stat
                    WHERE adm_stat_id IN (SELECT adm_stat_id FROM tmp_target_adm_stat)
                """,
            )

            # Прочие связи школы.
            delete_if_exists(
                "prof_event",
                "prof_event",
                """
                    DELETE FROM edu.prof_event
                    WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )
            delete_if_exists(
                "school_external_key",
                "school_external_key",
                """
                    DELETE FROM edu.school_external_key
                    WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )
            delete_if_exists(
                "school_profile_link",
                "school_profile_link",
                """
                    DELETE FROM edu.school_profile_link
                    WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )

            # Основные справочники.
            delete_if_exists(
                "school",
                "school",
                """
                    DELETE FROM edu.school
                    WHERE school_id IN (SELECT school_id FROM tmp_target_school)
                """,
            )
            delete_if_exists(
                "municipality",
                "municipality",
                """
                    DELETE FROM edu.municipality
                    WHERE municipality_id IN (SELECT municipality_id FROM tmp_target_municipality)
                """,
            )
            delete_if_exists(
                "region",
                "region",
                """
                    DELETE FROM edu.region
                    WHERE region_id = %s
                """,
                (region_id,),
            )

            # Удаляем "пустые" аналитические запуски/запросы после очистки.
            if all(
                existing_tables.get(name, False)
                for name in [
                    "analytics_rating_run",
                    "analytics_school_rating",
                    "analytics_school_metric_value",
                    "analytics_school_card",
                    "analytics_school_report",
                ]
            ):
                deleted["analytics_rating_run(empty)"] = delete_with_count(
                    cur,
                    """
                    DELETE FROM edu.analytics_rating_run rr
                    WHERE NOT EXISTS (SELECT 1 FROM edu.analytics_school_rating r WHERE r.run_id = rr.run_id)
                      AND NOT EXISTS (SELECT 1 FROM edu.analytics_school_metric_value mv WHERE mv.run_id = rr.run_id)
                      AND NOT EXISTS (SELECT 1 FROM edu.analytics_school_card c WHERE c.run_id = rr.run_id)
                      AND NOT EXISTS (SELECT 1 FROM edu.analytics_school_report rp WHERE rp.run_id = rr.run_id)
                    """,
                )
            else:
                deleted["analytics_rating_run(empty)"] = 0

            if all(
                existing_tables.get(name, False)
                for name in [
                    "analytics_request",
                    "analytics_request_filter",
                    "analytics_school_selection",
                    "analytics_rating_run",
                ]
            ):
                deleted["analytics_request(empty)"] = delete_with_count(
                    cur,
                    """
                    DELETE FROM edu.analytics_request rq
                    WHERE NOT EXISTS (SELECT 1 FROM edu.analytics_request_filter rf WHERE rf.request_id = rq.request_id)
                      AND NOT EXISTS (SELECT 1 FROM edu.analytics_school_selection ss WHERE ss.request_id = rq.request_id)
                      AND NOT EXISTS (SELECT 1 FROM edu.analytics_rating_run rr WHERE rr.request_id = rq.request_id)
                    """,
                )
            else:
                deleted["analytics_request(empty)"] = 0

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
