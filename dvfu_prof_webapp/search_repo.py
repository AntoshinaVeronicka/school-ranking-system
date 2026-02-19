from __future__ import annotations

import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

try:
    from .filter_options_repo import fetch_common_filter_options
except ImportError:
    from filter_options_repo import fetch_common_filter_options


_SPACE_RE = re.compile(r"\s+")
_NORM_FULL_NAME_SQL = (
    "lower(replace(replace(regexp_replace(coalesce(s.full_name, ''), '\\s+', ' ', 'g'), chr(1025), chr(1045)), chr(1105), chr(1077)))"
)


@dataclass(frozen=True)
class SchoolSearchFilters:
    q: str = ""
    region_id: int | None = None
    municipality_id: int | None = None
    profile_ids: tuple[int, ...] = ()
    year: int | None = None
    kind: str | None = None
    subject_ids: tuple[int, ...] = ()


def normalize_search_text(text: str) -> str:
    return _SPACE_RE.sub(" ", (text or "").replace("\u00A0", " ").strip()).lower().replace(chr(1105), chr(1077))


def _load_env() -> None:
    if load_dotenv is None:
        return
    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


def _get_pg_config() -> dict[str, Any]:
    _load_env()
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    missing = [k for k, v in [("POSTGRES_DB", dbname), ("POSTGRES_USER", user), ("POSTGRES_PASSWORD", password)] if not v]
    if missing:
        raise ValueError(f"Не заданы параметры подключения к PostgreSQL: {', '.join(missing)}")
    return {
        "host": host,
        "port": port,
        "dbname": dbname,
        "user": user,
        "password": password,
    }

_PG_POOL: ThreadedConnectionPool | None = None
_PG_POOL_LOCK = threading.Lock()


def _get_pool_bounds() -> tuple[int, int]:
    raw_min = os.getenv("PG_POOL_MIN_SIZE", "1")
    raw_max = os.getenv("PG_POOL_MAX_SIZE", "10")
    try:
        min_size = int(raw_min)
    except ValueError:
        min_size = 1
    try:
        max_size = int(raw_max)
    except ValueError:
        max_size = 10
    min_size = max(1, min(min_size, 32))
    max_size = max(min_size, min(max_size, 64))
    return min_size, max_size


def _get_pg_pool() -> ThreadedConnectionPool:
    global _PG_POOL
    pool = _PG_POOL
    if pool is not None:
        return pool

    with _PG_POOL_LOCK:
        pool = _PG_POOL
        if pool is None:
            min_size, max_size = _get_pool_bounds()
            pool = ThreadedConnectionPool(minconn=min_size, maxconn=max_size, **_get_pg_config())
            _PG_POOL = pool
    return pool


@contextmanager
def get_pg_connection():
    pool = _get_pg_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        pool.putconn(conn)


def close_pg_pool() -> None:
    global _PG_POOL
    with _PG_POOL_LOCK:
        pool = _PG_POOL
        _PG_POOL = None
    if pool is not None:
        pool.closeall()



def _build_school_filters(filters: SchoolSearchFilters) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    q_norm = normalize_search_text(filters.q)
    if q_norm:
        clauses.append(f"{_NORM_FULL_NAME_SQL} LIKE %s")
        params.append(f"%{q_norm}%")

    if filters.region_id is not None:
        clauses.append("r.region_id = %s")
        params.append(filters.region_id)

    if filters.municipality_id is not None:
        clauses.append("m.municipality_id = %s")
        params.append(filters.municipality_id)

    if filters.profile_ids:
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM edu.school_profile_link spl
                WHERE spl.school_id = s.school_id
                  AND spl.profile_id = ANY(%s)
            )
            """
        )
        params.append(list(filters.profile_ids))

    if filters.year is not None or filters.kind or filters.subject_ids:
        ege_conditions: list[str] = ["y.school_id = s.school_id"]
        if filters.year is not None:
            ege_conditions.append('y."year" = %s')
            params.append(filters.year)
        if filters.kind:
            ege_conditions.append("y.kind::text = %s")
            params.append(filters.kind)

        if filters.subject_ids:
            unique_subject_ids = sorted(set(filters.subject_ids))
            ege_sql = f"""
                EXISTS (
                    SELECT 1
                    FROM edu.ege_school_year y
                    JOIN edu.ege_school_subject_stat ss ON ss.ege_school_year_id = y.ege_school_year_id
                    WHERE {" AND ".join(ege_conditions)}
                      AND ss.subject_id = ANY(%s)
                    GROUP BY y.ege_school_year_id
                    HAVING COUNT(DISTINCT ss.subject_id) = %s
                )
            """
            clauses.append(ege_sql)
            params.append(unique_subject_ids)
            params.append(len(unique_subject_ids))
        else:
            ege_sql = f"""
                EXISTS (
                    SELECT 1
                    FROM edu.ege_school_year y
                    WHERE {" AND ".join(ege_conditions)}
                )
            """
            clauses.append(ege_sql)

    where_sql = " AND ".join(f"({c.strip()})" for c in clauses) if clauses else "TRUE"
    return where_sql, params


def fetch_filter_options(region_id: int | None = None) -> dict[str, list[dict[str, Any]]]:
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            return fetch_common_filter_options(cur, region_id=region_id, include_subject_min_score=False)


def fetch_municipalities(region_id: int) -> list[dict[str, Any]]:
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT municipality_id, name
                FROM edu.municipality
                WHERE region_id = %s
                ORDER BY name
                """,
                (region_id,),
            )
            return [dict(r) for r in cur.fetchall()]


def search_schools(
    filters: SchoolSearchFilters,
    *,
    page: int = 1,
    per_page: int = 20,
    apply_pagination: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    where_sql, where_params = _build_school_filters(filters)
    base_from = """
        FROM edu.school s
        JOIN edu.municipality m ON m.municipality_id = s.municipality_id
        JOIN edu.region r ON r.region_id = m.region_id
    """

    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            count_sql = f"SELECT COUNT(*) AS total {base_from} WHERE {where_sql}"
            cur.execute(count_sql, where_params)
            total = int(cur.fetchone()["total"])

            order_clause = "ORDER BY s.full_name ASC"
            query_sql = f"""
                SELECT
                    s.school_id,
                    s.full_name,
                    NULL::text AS short_name,
                    s.is_active,
                    r.region_id,
                    r.name AS region_name,
                    m.municipality_id,
                    m.name AS municipality_name,
                    COALESCE(p.profile_names, '') AS profile_names,
                    ys.last_year,
                    ys.ege_years,
                    COALESCE(ys.avg_graduates, 0)::numeric(10, 1) AS avg_graduates,
                    ssm.avg_score
                {base_from}
                LEFT JOIN LATERAL (
                    SELECT string_agg(sp.name, ', ' ORDER BY sp.name) AS profile_names
                    FROM edu.school_profile_link spl
                    JOIN edu.school_profile sp ON sp.profile_id = spl.profile_id
                    WHERE spl.school_id = s.school_id
                ) p ON TRUE
                LEFT JOIN (
                    SELECT
                        y.school_id,
                        MAX(y."year") AS last_year,
                        COUNT(DISTINCT y."year") AS ege_years,
                        ROUND(AVG(y.graduates_total)::numeric, 1) AS avg_graduates
                    FROM edu.ege_school_year y
                    GROUP BY y.school_id
                ) ys ON ys.school_id = s.school_id
                LEFT JOIN (
                    SELECT
                        y.school_id,
                        ROUND(
                            (
                                AVG(ss.avg_score) FILTER (
                                    WHERE ss.avg_score IS NOT NULL
                                      AND COALESCE(ss.participants_cnt, ss.chosen_cnt, 0) > 0
                                )
                            )::numeric,
                            2
                        ) AS avg_score
                    FROM edu.ege_school_year y
                    LEFT JOIN edu.ege_school_subject_stat ss ON ss.ege_school_year_id = y.ege_school_year_id
                    GROUP BY y.school_id
                ) ssm ON ssm.school_id = s.school_id
                WHERE {where_sql}
                {order_clause}
            """

            params = list(where_params)
            if apply_pagination:
                safe_page = max(1, int(page))
                safe_per_page = max(1, min(int(per_page), 200))
                offset = (safe_page - 1) * safe_per_page
                query_sql += " LIMIT %s OFFSET %s"
                params.extend([safe_per_page, offset])

            cur.execute(query_sql, params)
            rows = [dict(r) for r in cur.fetchall()]

    return rows, total


def fetch_school_card(school_id: int) -> dict[str, Any] | None:
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    s.school_id,
                    s.full_name,
                    NULL::text AS short_name,
                    NULL::text AS address,
                    NULL::text AS website,
                    s.is_active,
                    m.municipality_id,
                    m.name AS municipality_name,
                    r.region_id,
                    r.name AS region_name,
                    COALESCE(array_agg(sp.name ORDER BY sp.name) FILTER (WHERE sp.name IS NOT NULL), '{}') AS profiles
                FROM edu.school s
                JOIN edu.municipality m ON m.municipality_id = s.municipality_id
                JOIN edu.region r ON r.region_id = m.region_id
                LEFT JOIN edu.school_profile_link spl ON spl.school_id = s.school_id
                LEFT JOIN edu.school_profile sp ON sp.profile_id = spl.profile_id
                WHERE s.school_id = %s
                GROUP BY
                    s.school_id,
                    s.full_name,
                    s.is_active,
                    m.municipality_id,
                    m.name,
                    r.region_id,
                    r.name
                """,
                (school_id,),
            )
            school = cur.fetchone()
            if school is None:
                return None

            cur.execute(
                """
                SELECT source_name::text AS source_name, external_key, normalized_name
                FROM edu.school_external_key
                WHERE school_id = %s
                ORDER BY source_name::text
                """,
                (school_id,),
            )
            external_keys = [dict(r) for r in cur.fetchall()]

            cur.execute(
                """
                SELECT
                    "year",
                    applicants_cnt,
                    enrolled_cnt,
                    enrolled_avg_score
                FROM edu.school_admission_stat
                WHERE school_id = %s
                ORDER BY "year" DESC
                """,
                (school_id,),
            )
            admission_stats = [dict(r) for r in cur.fetchall()]

            cur.execute(
                """
                SELECT
                    EXTRACT(YEAR FROM event_date)::int AS year,
                    COUNT(*)::int AS events_cnt,
                    COALESCE(SUM(coverage_cnt), 0)::int AS coverage_cnt
                FROM edu.prof_event
                WHERE school_id = %s
                GROUP BY EXTRACT(YEAR FROM event_date)
                ORDER BY year DESC
                """,
                (school_id,),
            )
            prof_events = [dict(r) for r in cur.fetchall()]

            cur.execute(
                """
                SELECT
                    y."year",
                    y.kind::text AS kind,
                    y.graduates_total,
                    subj.subject_id,
                    subj.name AS subject_name,
                    subj.min_passing_score,
                    ss.participants_cnt,
                    ss.not_passed_cnt,
                    ss.high_80_99_cnt,
                    ss.score_100_cnt,
                    ss.avg_score,
                    ss.chosen_cnt
                FROM edu.ege_school_year y
                LEFT JOIN edu.ege_school_subject_stat ss ON ss.ege_school_year_id = y.ege_school_year_id
                LEFT JOIN edu.ege_subject subj ON subj.subject_id = ss.subject_id
                WHERE y.school_id = %s
                ORDER BY y."year" DESC, y.kind::text, subj.name
                """,
                (school_id,),
            )
            ege_rows = [dict(r) for r in cur.fetchall()]

    ege_by_period: dict[tuple[int, str], dict[str, Any]] = {}
    for row in ege_rows:
        key = (int(row["year"]), str(row["kind"]))
        bucket = ege_by_period.get(key)
        if bucket is None:
            bucket = {
                "year": int(row["year"]),
                "kind": str(row["kind"]),
                "graduates_total": row["graduates_total"],
                "subjects": [],
            }
            ege_by_period[key] = bucket

        if row["subject_id"] is None:
            continue

        participants_cnt = row["participants_cnt"]
        not_passed_cnt = row["not_passed_cnt"]
        high_80_99_cnt = row["high_80_99_cnt"]
        score_100_cnt = row["score_100_cnt"]
        chosen_cnt = row["chosen_cnt"]
        avg_score = row["avg_score"]

        has_subject_data = any(
            (isinstance(v, int) and v > 0)
            for v in (participants_cnt, not_passed_cnt, high_80_99_cnt, score_100_cnt, chosen_cnt)
        )
        if not has_subject_data and avg_score is not None:
            try:
                if float(avg_score) == 0.0:
                    avg_score = None
            except (TypeError, ValueError):
                pass

        bucket["subjects"].append(
            {
                "subject_id": int(row["subject_id"]),
                "subject_name": str(row["subject_name"]),
                "min_passing_score": row["min_passing_score"],
                "participants_cnt": participants_cnt,
                "not_passed_cnt": not_passed_cnt,
                "high_80_99_cnt": high_80_99_cnt,
                "score_100_cnt": score_100_cnt,
                "avg_score": avg_score,
                "chosen_cnt": chosen_cnt,
            }
        )

    ege_timeline = sorted(ege_by_period.values(), key=lambda x: (x["year"], x["kind"]), reverse=True)
    for period in ege_timeline:
        period["subjects"].sort(key=lambda x: x["subject_name"])

    return {
        "school": dict(school),
        "external_keys": external_keys,
        "admission_stats": admission_stats,
        "prof_events": prof_events,
        "ege_timeline": ege_timeline,
    }
