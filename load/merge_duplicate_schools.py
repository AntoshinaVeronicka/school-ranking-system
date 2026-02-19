# -*- coding: utf-8 -*-
"""
??????????? ?????? ???? ? ??????? ????????? ?????? ?? ???????????? ??????.

????? ?????? ?????? ???????; ??? ?????? ?????? ??????????
???? ?????-?????, ????????? ?????? ??????????? ? ?????????.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import psycopg2

from load_common import get_db_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Схлопывание дублей школ")
    p.add_argument(
        "--region",
        default="",
        help="Ограничить обработку одним регионом (точное имя, регистр не важен). По умолчанию: все регионы.",
    )
    p.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Сколько групп дублей показать в превью (по умолчанию 20).",
    )
    p.add_argument(
        "--limit-groups",
        type=int,
        default=0,
        help="Ограничить число обрабатываемых групп (0 = без ограничения).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Применить изменения. Без флага выполняется dry-run с rollback.",
    )
    return p.parse_args()


def _norm_spaces_sql(expr: str) -> str:
    return f"regexp_replace(btrim({expr}), '[[:space:]]+', ' ', 'g')"


def _fold_ru_sql(expr: str) -> str:
    return f"replace(lower({expr}), 'ё', 'е')"


def _region_norm_sql(expr: str) -> str:
    return _fold_ru_sql(_norm_spaces_sql(expr))


def _municipality_norm_sql(expr: str) -> str:
    squeezed = _norm_spaces_sql(expr)
    with_city_prefix = f"regexp_replace({squeezed}, '^[[:space:]]*г(\\.|[[:space:]]+)', 'город ', 'i')"
    with_city_prefix = f"regexp_replace({with_city_prefix}, '^[[:space:]]*город[[:space:]]+', 'город ', 'i')"
    return _fold_ru_sql(with_city_prefix)


def _municipality_code3_sql(expr: str) -> str:
    return (
        "CASE "
        f"WHEN {expr} ~ '^[[:space:]]*[(][[:space:]]*[0-9]+' "
        f"THEN lpad(substr((regexp_match({expr}, '^[[:space:]]*[(][[:space:]]*([0-9]+)'))[1], 1, 3), 3, '0') "
        "ELSE NULL "
        "END"
    )


def _school_code3_sql(expr: str) -> str:
    return (
        "CASE "
        f"WHEN {expr} ~ '^[[:space:]]*[(]?[[:space:]]*[0-9]+' "
        f"THEN lpad(substr((regexp_match({expr}, '^[[:space:]]*[(]?[[:space:]]*([0-9]+)'))[1], 1, 3), 3, '0') "
        "ELSE NULL "
        "END"
    )


def _school_name_norm_sql(expr: str) -> str:
    squeezed = _norm_spaces_sql(expr)
    folded = _fold_ru_sql(squeezed)
    # ...ОГО -> ...ИЙ
    folded = f"regexp_replace({folded}, 'ого(?=$|[[:space:][:punct:]])', 'ий', 'g')"
    # убираем ведущий цифровой код школы: "(142) " / "142 "
    no_lead_code = f"regexp_replace({folded}, '^[[:space:]]*[(]?[[:space:]]*[0-9]+[)]?[[:space:]]*', '', 'g')"
    # сравнение в ключе без разделителей
    return f"regexp_replace({no_lead_code}, '[^0-9a-zа-я]+', '', 'g')"


def scalar(cur, sql: str, params=()) -> int:
    cur.execute(sql, params)
    row = cur.fetchone()
    return int(row[0]) if row else 0


def create_dup_groups(cur, region_name: Optional[str]) -> None:
    region_norm = _region_norm_sql("r.name")
    mun_key = f"COALESCE({_municipality_code3_sql('m.name')}, {_municipality_norm_sql('m.name')})"
    school_code3 = _school_code3_sql("s.full_name")
    school_norm = _school_name_norm_sql("s.full_name")

    cur.execute(
        f"""
        CREATE TEMP TABLE tmp_school_dup_group ON COMMIT DROP AS
        WITH school_base AS (
            SELECT
                s.school_id,
                s.is_active,
                r.region_id,
                r.name AS region_name,
                {mun_key} AS municipality_key,
                {school_code3} AS school_code3,
                {school_norm} AS school_name_norm
            FROM edu.school s
            JOIN edu.municipality m ON m.municipality_id = s.municipality_id
            JOIN edu.region r ON r.region_id = m.region_id
            WHERE (%s IS NULL OR {region_norm} = { _region_norm_sql('%s') })
        ),
        ranked AS (
            SELECT
                school_id,
                is_active,
                region_id,
                region_name,
                municipality_key,
                school_code3,
                school_name_norm,
                ROW_NUMBER() OVER (
                    PARTITION BY region_id, municipality_key, school_code3, school_name_norm
                    ORDER BY is_active DESC, school_id
                )::int AS rn,
                COUNT(*) OVER (
                    PARTITION BY region_id, municipality_key, school_code3, school_name_norm
                )::int AS group_size
            FROM school_base
            WHERE school_name_norm <> ''
        ),
        grouped AS (
            SELECT
                region_id,
                MIN(region_name) AS region_name,
                municipality_key,
                school_code3,
                school_name_norm,
                MIN(school_id) FILTER (WHERE rn = 1) AS keep_school_id,
                ARRAY_AGG(school_id ORDER BY rn) FILTER (WHERE rn > 1) AS dup_school_ids,
                MAX(group_size)::int AS group_size
            FROM ranked
            GROUP BY region_id, municipality_key, school_code3, school_name_norm
            HAVING MAX(group_size) > 1
        )
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY region_id, municipality_key, COALESCE(school_code3, ''), school_name_norm, keep_school_id
            )::int AS group_id,
            region_id,
            region_name,
            municipality_key,
            school_code3,
            school_name_norm,
            keep_school_id,
            dup_school_ids,
            group_size
        FROM grouped
        """,
        (region_name, region_name),
    )


def create_merge_map(cur, limit_groups: int) -> None:
    if limit_groups > 0:
        cur.execute(
            """
            CREATE TEMP TABLE tmp_school_merge_map ON COMMIT DROP AS
            SELECT
                g.group_id,
                old_school_id::int AS old_school_id,
                g.keep_school_id::int AS new_school_id
            FROM tmp_school_dup_group g
            CROSS JOIN LATERAL unnest(g.dup_school_ids) AS old_school_id
            WHERE g.group_id <= %s
            """,
            (limit_groups,),
        )
    else:
        cur.execute(
            """
            CREATE TEMP TABLE tmp_school_merge_map ON COMMIT DROP AS
            SELECT
                g.group_id,
                old_school_id::int AS old_school_id,
                g.keep_school_id::int AS new_school_id
            FROM tmp_school_dup_group g
            CROSS JOIN LATERAL unnest(g.dup_school_ids) AS old_school_id
            """,
        )

    cur.execute("CREATE UNIQUE INDEX idx_tmp_school_merge_old ON tmp_school_merge_map (old_school_id)")
    cur.execute("CREATE INDEX idx_tmp_school_merge_new ON tmp_school_merge_map (new_school_id)")
    cur.execute("ANALYZE tmp_school_merge_map")


def print_samples(cur, sample_limit: int) -> None:
    cur.execute(
        """
        SELECT
            g.group_id,
            g.region_name,
            g.municipality_key,
            COALESCE(g.school_code3, '') AS school_code3,
            g.keep_school_id,
            k.is_active AS keep_is_active,
            k.full_name AS keep_full_name,
            COALESCE(array_length(g.dup_school_ids, 1), 0) AS dup_count,
            (
                SELECT string_agg(
                    d.school_id::text || ' [' || CASE WHEN d.is_active THEN 'active' ELSE 'inactive' END || ']: ' || m.name || ' | ' || d.full_name,
                    E'\n' ORDER BY d.school_id
                )
                FROM unnest(g.dup_school_ids) AS did(school_id)
                JOIN edu.school d ON d.school_id = did.school_id
                JOIN edu.municipality m ON m.municipality_id = d.municipality_id
            ) AS dup_items
        FROM tmp_school_dup_group g
        JOIN edu.school k ON k.school_id = g.keep_school_id
        ORDER BY g.group_size DESC, g.group_id
        LIMIT %s
        """,
        (sample_limit,),
    )
    rows = cur.fetchall()
    if not rows:
        print("Превью: дублей не найдено.")
        return

    print("Превью групп дублей:")
    for row in rows:
        group_id, region_name, mun_key, code3, keep_id, keep_is_active, keep_name, dup_count, dup_items = row
        print(f"- group_id={group_id}, region='{region_name}', mun_key='{mun_key}', code3='{code3}'")
        print(f"  keep: {keep_id} [{'active' if keep_is_active else 'inactive'}]: {keep_name}")
        print(f"  duplicates: {dup_count}")
        if dup_items:
            print(dup_items)


def _merge_nullable_max_expr(table: str, col: str) -> str:
    return (
        f"CASE "
        f"WHEN {table}.{col} IS NULL THEN EXCLUDED.{col} "
        f"WHEN EXCLUDED.{col} IS NULL THEN {table}.{col} "
        f"ELSE GREATEST({table}.{col}, EXCLUDED.{col}) "
        f"END"
    )


def apply_merge(cur) -> Dict[str, int]:
    stats: Dict[str, int] = {}

    def exec_count(name: str, sql: str) -> None:
        cur.execute(sql)
        stats[name] = int(cur.rowcount) if cur.rowcount != -1 else 0

    # --- Лёгкие связи (insert+delete old)
    exec_count(
        "school_profile_link_insert",
        """
        INSERT INTO edu.school_profile_link (school_id, profile_id)
        SELECT m.new_school_id, spl.profile_id
        FROM edu.school_profile_link spl
        JOIN tmp_school_merge_map m ON m.old_school_id = spl.school_id
        ON CONFLICT (school_id, profile_id) DO NOTHING
        """,
    )
    exec_count(
        "school_profile_link_delete_old",
        """
        DELETE FROM edu.school_profile_link spl
        USING tmp_school_merge_map m
        WHERE spl.school_id = m.old_school_id
        """,
    )

    exec_count(
        "school_external_key_insert",
        """
        INSERT INTO edu.school_external_key (school_id, source_name, external_key, normalized_name)
        SELECT m.new_school_id, sek.source_name, sek.external_key, sek.normalized_name
        FROM edu.school_external_key sek
        JOIN tmp_school_merge_map m ON m.old_school_id = sek.school_id
        ON CONFLICT DO NOTHING
        """,
    )
    exec_count(
        "school_external_key_delete_old",
        """
        DELETE FROM edu.school_external_key sek
        USING tmp_school_merge_map m
        WHERE sek.school_id = m.old_school_id
        """,
    )

    # --- ЕГЭ (с мерджем по (school_id, year, kind))
    exec_count(
        "ege_school_year_upsert",
        f"""
        INSERT INTO edu.ege_school_year (school_id, "year", kind, graduates_total)
        SELECT m.new_school_id, y."year", y.kind, y.graduates_total
        FROM edu.ege_school_year y
        JOIN tmp_school_merge_map m ON m.old_school_id = y.school_id
        ON CONFLICT (school_id, "year", kind) DO UPDATE
        SET graduates_total = GREATEST(edu.ege_school_year.graduates_total, EXCLUDED.graduates_total)
        """,
    )

    cur.execute(
        """
        CREATE TEMP TABLE tmp_ege_year_merge_map ON COMMIT DROP AS
        SELECT
            y_old.ege_school_year_id AS old_ege_school_year_id,
            y_new.ege_school_year_id AS new_ege_school_year_id
        FROM edu.ege_school_year y_old
        JOIN tmp_school_merge_map m ON m.old_school_id = y_old.school_id
        JOIN edu.ege_school_year y_new
          ON y_new.school_id = m.new_school_id
         AND y_new."year" = y_old."year"
         AND y_new.kind = y_old.kind
        """
    )
    cur.execute("CREATE UNIQUE INDEX idx_tmp_ege_year_old ON tmp_ege_year_merge_map (old_ege_school_year_id)")
    cur.execute("ANALYZE tmp_ege_year_merge_map")

    exec_count(
        "ege_school_subject_stat_upsert",
        f"""
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
        SELECT
            map.new_ege_school_year_id,
            ss.subject_id,
            ss.participants_cnt,
            ss.not_passed_cnt,
            ss.high_80_99_cnt,
            ss.score_100_cnt,
            ss.avg_score,
            ss.chosen_cnt
        FROM edu.ege_school_subject_stat ss
        JOIN tmp_ege_year_merge_map map
          ON map.old_ege_school_year_id = ss.ege_school_year_id
        ON CONFLICT (ege_school_year_id, subject_id) DO UPDATE
        SET
            participants_cnt = {_merge_nullable_max_expr("edu.ege_school_subject_stat", "participants_cnt")},
            not_passed_cnt = {_merge_nullable_max_expr("edu.ege_school_subject_stat", "not_passed_cnt")},
            high_80_99_cnt = {_merge_nullable_max_expr("edu.ege_school_subject_stat", "high_80_99_cnt")},
            score_100_cnt = {_merge_nullable_max_expr("edu.ege_school_subject_stat", "score_100_cnt")},
            avg_score = {_merge_nullable_max_expr("edu.ege_school_subject_stat", "avg_score")},
            chosen_cnt = {_merge_nullable_max_expr("edu.ege_school_subject_stat", "chosen_cnt")}
        """,
    )

    exec_count(
        "ege_school_year_delete_old",
        """
        DELETE FROM edu.ege_school_year y
        USING tmp_school_merge_map m
        WHERE y.school_id = m.old_school_id
        """,
    )

    # --- Прием (school_admission_stat + school_admission_detail)
    exec_count(
        "school_admission_stat_upsert",
        f"""
        INSERT INTO edu.school_admission_stat (
            school_id,
            "year",
            applicants_cnt,
            enrolled_cnt,
            enrolled_avg_score,
            external_key
        )
        SELECT
            m.new_school_id,
            sas."year",
            sas.applicants_cnt,
            sas.enrolled_cnt,
            sas.enrolled_avg_score,
            sas.external_key
        FROM edu.school_admission_stat sas
        JOIN tmp_school_merge_map m ON m.old_school_id = sas.school_id
        ON CONFLICT (school_id, "year") DO UPDATE
        SET
            applicants_cnt = GREATEST(edu.school_admission_stat.applicants_cnt, EXCLUDED.applicants_cnt),
            enrolled_cnt = GREATEST(edu.school_admission_stat.enrolled_cnt, EXCLUDED.enrolled_cnt),
            enrolled_avg_score = {_merge_nullable_max_expr("edu.school_admission_stat", "enrolled_avg_score")}
        """,
    )

    cur.execute(
        """
        CREATE TEMP TABLE tmp_adm_stat_merge_map ON COMMIT DROP AS
        SELECT
            old_sas.adm_stat_id AS old_adm_stat_id,
            new_sas.adm_stat_id AS new_adm_stat_id
        FROM edu.school_admission_stat old_sas
        JOIN tmp_school_merge_map m ON m.old_school_id = old_sas.school_id
        JOIN edu.school_admission_stat new_sas
          ON new_sas.school_id = m.new_school_id
         AND new_sas."year" = old_sas."year"
        """
    )
    cur.execute("CREATE UNIQUE INDEX idx_tmp_adm_stat_old ON tmp_adm_stat_merge_map (old_adm_stat_id)")
    cur.execute("ANALYZE tmp_adm_stat_merge_map")

    exec_count(
        "school_admission_detail_upsert",
        f"""
        INSERT INTO edu.school_admission_detail (
            adm_stat_id,
            program_id,
            applicants_cnt,
            enrolled_cnt,
            avg_score
        )
        SELECT
            map.new_adm_stat_id,
            sad.program_id,
            sad.applicants_cnt,
            sad.enrolled_cnt,
            sad.avg_score
        FROM edu.school_admission_detail sad
        JOIN tmp_adm_stat_merge_map map
          ON map.old_adm_stat_id = sad.adm_stat_id
        ON CONFLICT (adm_stat_id, program_id) DO UPDATE
        SET
            applicants_cnt = GREATEST(edu.school_admission_detail.applicants_cnt, EXCLUDED.applicants_cnt),
            enrolled_cnt = GREATEST(edu.school_admission_detail.enrolled_cnt, EXCLUDED.enrolled_cnt),
            avg_score = {_merge_nullable_max_expr("edu.school_admission_detail", "avg_score")}
        """,
    )

    exec_count(
        "school_admission_stat_delete_old",
        """
        DELETE FROM edu.school_admission_stat sas
        USING tmp_school_merge_map m
        WHERE sas.school_id = m.old_school_id
        """,
    )

    # --- Профориентация
    exec_count(
        "prof_event_update_school",
        """
        UPDATE edu.prof_event pe
        SET school_id = m.new_school_id
        FROM tmp_school_merge_map m
        WHERE pe.school_id = m.old_school_id
        """,
    )

    # --- Аналитика
    exec_count(
        "analytics_run_school_insert",
        """
        INSERT INTO edu.analytics_run_school (run_id, school_id, selected_at)
        SELECT ars.run_id, m.new_school_id, ars.selected_at
        FROM edu.analytics_run_school ars
        JOIN tmp_school_merge_map m ON m.old_school_id = ars.school_id
        ON CONFLICT (run_id, school_id) DO NOTHING
        """,
    )
    exec_count(
        "analytics_run_school_delete_old",
        """
        DELETE FROM edu.analytics_run_school ars
        USING tmp_school_merge_map m
        WHERE ars.school_id = m.old_school_id
        """,
    )

    exec_count(
        "analytics_school_card_insert",
        """
        INSERT INTO edu.analytics_school_card (
            run_id,
            school_id,
            generated_at,
            school_info,
            ege_summary,
            admission_summary,
            prof_summary,
            conclusions
        )
        SELECT
            ascard.run_id,
            m.new_school_id,
            ascard.generated_at,
            ascard.school_info,
            ascard.ege_summary,
            ascard.admission_summary,
            ascard.prof_summary,
            ascard.conclusions
        FROM edu.analytics_school_card ascard
        JOIN tmp_school_merge_map m ON m.old_school_id = ascard.school_id
        ON CONFLICT DO NOTHING
        """,
    )
    exec_count(
        "analytics_school_card_delete_old",
        """
        DELETE FROM edu.analytics_school_card ascard
        USING tmp_school_merge_map m
        WHERE ascard.school_id = m.old_school_id
        """,
    )

    exec_count(
        "analytics_school_metric_value_upsert",
        f"""
        INSERT INTO edu.analytics_school_metric_value (
            run_id,
            school_id,
            metric_code,
            raw_value,
            norm_value,
            weighted_value
        )
        SELECT
            asmv.run_id,
            m.new_school_id,
            asmv.metric_code,
            asmv.raw_value,
            asmv.norm_value,
            asmv.weighted_value
        FROM edu.analytics_school_metric_value asmv
        JOIN tmp_school_merge_map m ON m.old_school_id = asmv.school_id
        ON CONFLICT (run_id, school_id, metric_code) DO UPDATE
        SET
            raw_value = {_merge_nullable_max_expr("edu.analytics_school_metric_value", "raw_value")},
            norm_value = {_merge_nullable_max_expr("edu.analytics_school_metric_value", "norm_value")},
            weighted_value = {_merge_nullable_max_expr("edu.analytics_school_metric_value", "weighted_value")}
        """,
    )
    exec_count(
        "analytics_school_metric_value_delete_old",
        """
        DELETE FROM edu.analytics_school_metric_value asmv
        USING tmp_school_merge_map m
        WHERE asmv.school_id = m.old_school_id
        """,
    )

    exec_count(
        "analytics_school_rating_insert",
        """
        INSERT INTO edu.analytics_school_rating (
            run_id,
            school_id,
            rank_pos,
            total_score,
            year_basis,
            students_cnt,
            ege_avg_all
        )
        SELECT
            asr.run_id,
            m.new_school_id,
            asr.rank_pos,
            asr.total_score,
            asr.year_basis,
            asr.students_cnt,
            asr.ege_avg_all
        FROM edu.analytics_school_rating asr
        JOIN tmp_school_merge_map m ON m.old_school_id = asr.school_id
        ON CONFLICT DO NOTHING
        """,
    )
    exec_count(
        "analytics_school_rating_delete_old",
        """
        DELETE FROM edu.analytics_school_rating asr
        USING tmp_school_merge_map m
        WHERE asr.school_id = m.old_school_id
        """,
    )

    exec_count(
        "analytics_school_selection_insert",
        """
        INSERT INTO edu.analytics_school_selection (
            request_id,
            school_id,
            selected_at,
            match_score
        )
        SELECT
            ass.request_id,
            m.new_school_id,
            ass.selected_at,
            ass.match_score
        FROM edu.analytics_school_selection ass
        JOIN tmp_school_merge_map m ON m.old_school_id = ass.school_id
        ON CONFLICT (request_id, school_id) DO NOTHING
        """,
    )
    exec_count(
        "analytics_school_selection_delete_old",
        """
        DELETE FROM edu.analytics_school_selection ass
        USING tmp_school_merge_map m
        WHERE ass.school_id = m.old_school_id
        """,
    )

    exec_count(
        "analytics_school_report_update_school",
        """
        UPDATE edu.analytics_school_report asr
        SET school_id = m.new_school_id
        FROM tmp_school_merge_map m
        WHERE asr.school_id = m.old_school_id
        """,
    )

    exec_count(
        "analytics_request_filter_insert",
        """
        INSERT INTO edu.analytics_request_filter (
            request_id,
            filter_type,
            region_id,
            municipality_id,
            institute_id,
            profile_id,
            program_id,
            subject_id,
            school_id,
            min_score
        )
        SELECT
            arf.request_id,
            arf.filter_type,
            arf.region_id,
            arf.municipality_id,
            arf.institute_id,
            arf.profile_id,
            arf.program_id,
            arf.subject_id,
            m.new_school_id,
            arf.min_score
        FROM edu.analytics_request_filter arf
        JOIN tmp_school_merge_map m ON m.old_school_id = arf.school_id
        ON CONFLICT DO NOTHING
        """,
    )
    exec_count(
        "analytics_request_filter_delete_old",
        """
        DELETE FROM edu.analytics_request_filter arf
        USING tmp_school_merge_map m
        WHERE arf.school_id = m.old_school_id
        """,
    )

    # --- Финально удаляем дубли в edu.school
    exec_count(
        "school_delete_old",
        """
        DELETE FROM edu.school s
        USING tmp_school_merge_map m
        WHERE s.school_id = m.old_school_id
        """,
    )

    return stats


def main() -> int:
    args = parse_args()
    region_name = args.region.strip() or None
    sample_limit = max(0, int(args.sample_limit))
    limit_groups = max(0, int(args.limit_groups))

    db_cfg = get_db_config(search_from=Path(__file__))

    with psycopg2.connect(**db_cfg) as conn:
        with conn.cursor() as cur:
            create_dup_groups(cur, region_name=region_name)
            total_groups = scalar(cur, "SELECT COUNT(*) FROM tmp_school_dup_group")
            total_pairs = scalar(cur, "SELECT COALESCE(SUM(array_length(dup_school_ids, 1)), 0) FROM tmp_school_dup_group")

            print("Найдено групп дублей:", total_groups)
            print("Найдено школ-дублей к переносу:", total_pairs)
            if region_name:
                print("Фильтр по региону:", region_name)
            if limit_groups > 0:
                print("Ограничение по группам:", limit_groups)

            if sample_limit > 0:
                print_samples(cur, sample_limit=sample_limit)

            if total_pairs == 0:
                if args.apply:
                    conn.commit()
                else:
                    conn.rollback()
                print("Дубликатов для схлопывания нет.")
                return 0

            create_merge_map(cur, limit_groups=limit_groups)
            selected_pairs = scalar(cur, "SELECT COUNT(*) FROM tmp_school_merge_map")
            selected_groups = scalar(cur, "SELECT COUNT(DISTINCT group_id) FROM tmp_school_merge_map")
            print("К обработке групп:", selected_groups)
            print("К обработке дублей:", selected_pairs)

            stats = apply_merge(cur)
            print("Операции:")
            for k in sorted(stats.keys()):
                print(f"- {k}: {stats[k]}")

            still_old = scalar(
                cur,
                """
                SELECT COUNT(*)
                FROM edu.school s
                JOIN tmp_school_merge_map m ON m.old_school_id = s.school_id
                """,
            )
            print("Осталось дублей в edu.school после мерджа:", still_old)

            if args.apply:
                conn.commit()
                print("Готово: изменения применены.")
            else:
                conn.rollback()
                print("Dry-run: изменения откатены.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
