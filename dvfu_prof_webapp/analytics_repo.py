from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values

try:
    from .search_repo import _get_pg_config
except ImportError:
    from search_repo import _get_pg_config


METRIC_CODES = (
    "avg_graduates",
    "avg_score_all",
    "match_share",
    "threshold_share",
)

METRIC_DEFS = {
    "avg_graduates": {
        "name_ru": "Среднее число выпускников",
        "category": "ege",
        "unit": "чел",
        "description": "Среднее число выпускников по доступным периодам ЕГЭ",
    },
    "avg_score_all": {
        "name_ru": "Средний балл ЕГЭ",
        "category": "ege",
        "unit": "балл",
        "description": "Средний балл ЕГЭ по всем предметам",
    },
    "match_share": {
        "name_ru": "Соответствие программам",
        "category": "ege",
        "unit": "доля",
        "description": "Доля программ, для которых школа удовлетворяет набору предметов",
    },
    "threshold_share": {
        "name_ru": "Прохождение порогов ЕГЭ",
        "category": "ege",
        "unit": "доля",
        "description": "Доля предметных порогов, выполняемых школой",
    },
}

VALID_REPORT_TYPES = {"standard", "detailed"}
VALID_REPORT_FORMATS = {"json", "pdf", "xlsx"}


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Decimal):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raw = str(value).strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    try:
        return int(float(raw))
    except ValueError:
        return None


def _num_5_2(value: Any) -> float | None:
    if value is None:
        return None
    num = _to_float(value, default=0.0)
    if num > 999.99:
        num = 999.99
    if num < -999.99:
        num = -999.99
    return round(num, 2)


def _num_10_6(value: Any) -> float | None:
    if value is None:
        return None
    return round(_to_float(value, default=0.0), 6)


def _normalize_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    result: set[int] = set()
    for item in values:
        parsed = _to_int(item)
        if parsed is not None:
            result.add(parsed)
    return sorted(result)


def _guess_year_basis(filters: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    year = _to_int(filters.get("year"))
    if year is None:
        for row in rows:
            last_year = _to_int(row.get("last_year"))
            if last_year is not None:
                year = last_year
                break
    if year is None:
        year = datetime.now(timezone.utc).year
    return max(2000, min(2100, int(year)))


def _normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    raw = {
        "avg_graduates": max(0.0, _to_float(weights.get("w_graduates"), 0.25)),
        "avg_score_all": max(0.0, _to_float(weights.get("w_avg_score"), 0.45)),
        "match_share": max(0.0, _to_float(weights.get("w_match_share"), 0.20)),
        "threshold_share": max(0.0, _to_float(weights.get("w_threshold_share"), 0.10)),
    }
    total = sum(raw.values())
    if total <= 0:
        return {
            "avg_graduates": 0.25,
            "avg_score_all": 0.45,
            "match_share": 0.20,
            "threshold_share": 0.10,
        }
    return {k: v / total for k, v in raw.items()}


def _make_notes(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _ensure_metric_catalog(cur: psycopg2.extensions.cursor) -> None:
    rows = [
        (
            metric_code,
            payload["name_ru"],
            payload["category"],
            payload["unit"],
            payload["description"],
            True,
        )
        for metric_code, payload in METRIC_DEFS.items()
    ]
    execute_values(
        cur,
        """
        INSERT INTO edu.analytics_metric
            (metric_code, name_ru, category, unit, description, is_active)
        VALUES %s
        ON CONFLICT (metric_code) DO UPDATE SET
            name_ru = EXCLUDED.name_ru,
            category = EXCLUDED.category,
            unit = EXCLUDED.unit,
            description = EXCLUDED.description,
            is_active = EXCLUDED.is_active
        """,
        rows,
    )


def _append_filter_rows(
    rows: list[tuple[Any, ...]],
    request_id: int,
    filter_type: str,
    *,
    region_id: int | None = None,
    municipality_id: int | None = None,
    institute_id: int | None = None,
    profile_id: int | None = None,
    program_id: int | None = None,
    subject_id: int | None = None,
    school_id: int | None = None,
    min_score: int | None = None,
) -> None:
    rows.append(
        (
            request_id,
            filter_type,
            region_id,
            municipality_id,
            institute_id,
            profile_id,
            program_id,
            subject_id,
            school_id,
            min_score,
        )
    )


def _insert_request_filters(cur: psycopg2.extensions.cursor, request_id: int, filters: dict[str, Any]) -> None:
    rows: list[tuple[Any, ...]] = []

    region_id = _to_int(filters.get("region_id"))
    municipality_id = _to_int(filters.get("municipality_id"))
    if region_id is not None:
        _append_filter_rows(rows, request_id, "region", region_id=region_id)
    if municipality_id is not None:
        _append_filter_rows(rows, request_id, "municipality", municipality_id=municipality_id)

    for profile_id in _normalize_int_list(filters.get("profile_ids")):
        _append_filter_rows(rows, request_id, "school_profile", profile_id=profile_id)

    min_score = _to_int(filters.get("min_avg_score"))
    if min_score is not None:
        min_score = max(0, min(100, min_score))

    for subject_id in _normalize_int_list(filters.get("subject_ids")):
        _append_filter_rows(rows, request_id, "subject", subject_id=subject_id, min_score=min_score)

    for institute_id in _normalize_int_list(filters.get("institute_ids")):
        _append_filter_rows(rows, request_id, "institute", institute_id=institute_id)

    for program_id in _normalize_int_list(filters.get("program_ids")):
        _append_filter_rows(rows, request_id, "program", program_id=program_id)

    if not rows:
        return

    unique_rows: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for row in rows:
        key = (
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
            row[9],
        )
        unique_rows[key] = row

    execute_values(
        cur,
        """
        INSERT INTO edu.analytics_request_filter
            (
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
        VALUES %s
        """,
        list(unique_rows.values()),
    )


def save_search_request(
    *,
    created_by: str | None,
    filters: dict[str, Any],
    rows: list[dict[str, Any]],
    total_rows: int,
    page: int,
    per_page: int,
) -> int | None:
    if not rows and total_rows <= 0:
        return None

    cfg = _get_pg_config()
    year_basis = _guess_year_basis(filters, rows)

    notes = _make_notes(
        {
            "query": (filters.get("q") or "").strip(),
            "kind": filters.get("kind"),
            "year": filters.get("year"),
            "page": page,
            "per_page": per_page,
            "total_rows": total_rows,
        }
    )

    with psycopg2.connect(**cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO edu.analytics_request
                    (created_by, request_name, scenario, year_basis, only_active, notes)
                VALUES
                    (%s, %s, 'search', %s, FALSE, %s)
                RETURNING request_id
                """,
                (
                    (created_by or "").strip() or None,
                    "Поиск школ",
                    year_basis,
                    notes,
                ),
            )
            request_id = int(cur.fetchone()[0])

            _insert_request_filters(cur, request_id, filters)

            selection_rows: list[tuple[int, int, float | None]] = []
            for row in rows:
                school_id = _to_int(row.get("school_id"))
                if school_id is None:
                    continue
                score = _num_5_2(row.get("avg_score"))
                selection_rows.append((request_id, school_id, score))

            if selection_rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO edu.analytics_school_selection
                        (request_id, school_id, match_score)
                    VALUES %s
                    ON CONFLICT (request_id, school_id) DO NOTHING
                    """,
                    selection_rows,
                )

    return request_id


def save_rating_run(
    *,
    created_by: str | None,
    filters: dict[str, Any],
    weights: dict[str, Any],
    ranked_rows: list[dict[str, Any]],
) -> dict[str, int] | None:
    if not ranked_rows:
        return None

    cfg = _get_pg_config()
    year_basis = _guess_year_basis(filters, ranked_rows)
    min_students = _to_int(filters.get("min_graduates"))
    if min_students is not None and min_students < 0:
        min_students = 0

    normalized_weights = _normalize_weights(weights)
    primary_metric = max(METRIC_CODES, key=lambda code: normalized_weights[code])
    metric_rows = [
        (
            code,
            max(1, min(10, int(round(normalized_weights[code] * 10)))),
            code == primary_metric,
        )
        for code in METRIC_CODES
    ]

    notes = _make_notes(
        {
            "query": (filters.get("q") or "").strip(),
            "kind": filters.get("kind"),
            "year": filters.get("year"),
            "enforce_subject_threshold": bool(filters.get("enforce_subject_threshold")),
            "weights": {k: round(v, 6) for k, v in normalized_weights.items()},
            "limit": _to_int(filters.get("limit")),
        }
    )

    with psycopg2.connect(**cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO edu.analytics_request
                    (created_by, request_name, scenario, year_basis, only_active, min_students, notes)
                VALUES
                    (%s, %s, 'selection', %s, FALSE, %s, %s)
                RETURNING request_id
                """,
                (
                    (created_by or "").strip() or None,
                    "Подбор и рейтинг школ",
                    year_basis,
                    min_students,
                    notes,
                ),
            )
            request_id = int(cur.fetchone()[0])

            _insert_request_filters(cur, request_id, filters)

            cur.execute(
                """
                INSERT INTO edu.analytics_rating_run
                    (
                        request_id,
                        calculated_by,
                        run_name,
                        stage,
                        algorithm_version,
                        normalization,
                        comment
                    )
                VALUES
                    (%s, %s, %s, 'final', 'v1', 'minmax', %s)
                RETURNING run_id
                """,
                (
                    request_id,
                    (created_by or "").strip() or None,
                    "Рейтинг школ",
                    notes,
                ),
            )
            run_id = int(cur.fetchone()[0])

            _ensure_metric_catalog(cur)

            execute_values(
                cur,
                """
                INSERT INTO edu.analytics_run_metric
                    (run_id, metric_code, weight, is_primary)
                VALUES %s
                ON CONFLICT (run_id, metric_code) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    is_primary = EXCLUDED.is_primary
                """,
                [(run_id, code, weight_val, is_primary) for code, weight_val, is_primary in metric_rows],
            )

            selection_rows: list[tuple[int, int, float | None]] = []
            run_school_rows: list[tuple[int, int]] = []
            rating_rows: list[tuple[Any, ...]] = []
            metric_value_rows: list[tuple[Any, ...]] = []
            card_rows: list[tuple[Any, ...]] = []

            for row in ranked_rows:
                school_id = _to_int(row.get("school_id"))
                rank_pos = _to_int(row.get("rank_pos"))
                if school_id is None or rank_pos is None:
                    continue

                rating_score = _to_float(row.get("rating_score"), 0.0)
                avg_graduates = _to_float(row.get("avg_graduates"), 0.0)
                avg_score_all = _to_float(row.get("avg_score_all"), 0.0)
                match_share = _to_float(row.get("match_share"), 0.0)
                threshold_share = _to_float(row.get("threshold_share"), 0.0)
                norm_graduates = _to_float(row.get("norm_graduates"), 0.0)
                norm_avg_score = _to_float(row.get("norm_avg_score"), 0.0)

                metric_payload = {
                    "avg_graduates": {"raw": avg_graduates, "norm": norm_graduates},
                    "avg_score_all": {"raw": avg_score_all, "norm": norm_avg_score},
                    "match_share": {"raw": match_share, "norm": match_share},
                    "threshold_share": {"raw": threshold_share, "norm": threshold_share},
                }
                for metric_code, payload in metric_payload.items():
                    norm_value = payload["norm"]
                    weighted_value = norm_value * normalized_weights[metric_code]
                    metric_value_rows.append(
                        (
                            run_id,
                            school_id,
                            metric_code,
                            _num_5_2(payload["raw"]),
                            _num_10_6(norm_value),
                            _num_10_6(weighted_value),
                        )
                    )

                selection_rows.append((request_id, school_id, _num_5_2(rating_score * 100.0)))
                run_school_rows.append((run_id, school_id))
                rating_rows.append(
                    (
                        run_id,
                        school_id,
                        rank_pos,
                        _num_10_6(rating_score) or 0.0,
                        year_basis,
                        _to_int(round(avg_graduates)),
                        _num_5_2(avg_score_all),
                    )
                )

                card_rows.append(
                    (
                        run_id,
                        school_id,
                        Json(
                            _json_safe(
                                {
                                    "school_id": school_id,
                                    "full_name": row.get("full_name"),
                                    "region_name": row.get("region_name"),
                                    "municipality_name": row.get("municipality_name"),
                                    "profile_names": row.get("profile_names"),
                                }
                            )
                        ),
                        Json(
                            _json_safe(
                                {
                                    "avg_graduates": _num_5_2(avg_graduates),
                                    "avg_score_all": _num_5_2(avg_score_all),
                                    "ege_years": _to_int(row.get("ege_years")),
                                    "last_year": _to_int(row.get("last_year")),
                                }
                            )
                        ),
                        Json({}),
                        Json({}),
                        Json(
                            _json_safe(
                                {
                                    "rating_score": _num_10_6(rating_score),
                                    "programs_total": _to_int(row.get("programs_total")) or 0,
                                    "programs_matched": _to_int(row.get("programs_matched")) or 0,
                                    "match_share": _num_10_6(match_share),
                                    "threshold_share": _num_10_6(threshold_share),
                                    "matched_programs": row.get("matched_programs") or "",
                                }
                            )
                        ),
                    )
                )

            if selection_rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO edu.analytics_school_selection
                        (request_id, school_id, match_score)
                    VALUES %s
                    ON CONFLICT (request_id, school_id) DO UPDATE SET
                        match_score = EXCLUDED.match_score,
                        selected_at = now()
                    """,
                    selection_rows,
                )

            if run_school_rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO edu.analytics_run_school
                        (run_id, school_id)
                    VALUES %s
                    ON CONFLICT (run_id, school_id) DO NOTHING
                    """,
                    run_school_rows,
                )

            if rating_rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO edu.analytics_school_rating
                        (run_id, school_id, rank_pos, total_score, year_basis, students_cnt, ege_avg_all)
                    VALUES %s
                    ON CONFLICT (run_id, school_id) DO UPDATE SET
                        rank_pos = EXCLUDED.rank_pos,
                        total_score = EXCLUDED.total_score,
                        year_basis = EXCLUDED.year_basis,
                        students_cnt = EXCLUDED.students_cnt,
                        ege_avg_all = EXCLUDED.ege_avg_all
                    """,
                    rating_rows,
                )

            if metric_value_rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO edu.analytics_school_metric_value
                        (run_id, school_id, metric_code, raw_value, norm_value, weighted_value)
                    VALUES %s
                    ON CONFLICT (run_id, school_id, metric_code) DO UPDATE SET
                        raw_value = EXCLUDED.raw_value,
                        norm_value = EXCLUDED.norm_value,
                        weighted_value = EXCLUDED.weighted_value
                    """,
                    metric_value_rows,
                )

            if card_rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO edu.analytics_school_card
                        (
                            run_id,
                            school_id,
                            school_info,
                            ege_summary,
                            admission_summary,
                            prof_summary,
                            conclusions
                        )
                    VALUES %s
                    ON CONFLICT (run_id, school_id) DO UPDATE SET
                        generated_at = now(),
                        school_info = EXCLUDED.school_info,
                        ege_summary = EXCLUDED.ege_summary,
                        admission_summary = EXCLUDED.admission_summary,
                        prof_summary = EXCLUDED.prof_summary,
                        conclusions = EXCLUDED.conclusions
                    """,
                    card_rows,
                )

    return {"request_id": request_id, "run_id": run_id}


def fetch_calc_history(limit: int = 200) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 1000))
    cfg = _get_pg_config()
    with psycopg2.connect(**cfg) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    ar.request_id,
                    ar.created_at,
                    ar.created_by,
                    ar.request_name,
                    ar.scenario,
                    ar.year_basis,
                    ar.notes,
                    COALESCE(sel.selected_cnt, 0) AS selected_cnt,
                    fs.filter_summary,
                    rr.run_id,
                    rr.calculated_at,
                    rr.calculated_by,
                    rr.run_name,
                    rr.algorithm_version,
                    COALESCE(rs.rated_cnt, 0) AS rated_cnt
                FROM edu.analytics_request ar
                LEFT JOIN LATERAL (
                    SELECT COUNT(*)::int AS selected_cnt
                    FROM edu.analytics_school_selection ass
                    WHERE ass.request_id = ar.request_id
                ) sel ON TRUE
                LEFT JOIN LATERAL (
                    SELECT string_agg(item, ', ' ORDER BY item) AS filter_summary
                    FROM (
                        SELECT arf.filter_type || ':' || COUNT(*)::text AS item
                        FROM edu.analytics_request_filter arf
                        WHERE arf.request_id = ar.request_id
                        GROUP BY arf.filter_type
                    ) t
                ) fs ON TRUE
                LEFT JOIN edu.analytics_rating_run rr
                    ON rr.request_id = ar.request_id
                LEFT JOIN LATERAL (
                    SELECT COUNT(*)::int AS rated_cnt
                    FROM edu.analytics_school_rating asr
                    WHERE asr.run_id = rr.run_id
                ) rs ON TRUE
                ORDER BY COALESCE(rr.calculated_at, ar.created_at) DESC
                LIMIT %s
                """,
                (safe_limit,),
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_rating_runs(limit: int = 200) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 1000))
    cfg = _get_pg_config()
    with psycopg2.connect(**cfg) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    rr.run_id,
                    rr.request_id,
                    rr.calculated_at,
                    rr.calculated_by,
                    rr.run_name,
                    rr.algorithm_version,
                    rr.stage,
                    ar.request_name,
                    ar.created_by AS request_created_by,
                    ar.year_basis,
                    ar.notes,
                    COALESCE(rc.rated_cnt, 0) AS rated_cnt,
                    COALESCE(rep.reports_cnt, 0) AS reports_cnt
                FROM edu.analytics_rating_run rr
                JOIN edu.analytics_request ar
                    ON ar.request_id = rr.request_id
                LEFT JOIN LATERAL (
                    SELECT COUNT(*)::int AS rated_cnt
                    FROM edu.analytics_school_rating asr
                    WHERE asr.run_id = rr.run_id
                ) rc ON TRUE
                LEFT JOIN LATERAL (
                    SELECT COUNT(*)::int AS reports_cnt
                    FROM edu.analytics_school_report asrep
                    WHERE asrep.run_id = rr.run_id
                ) rep ON TRUE
                ORDER BY rr.calculated_at DESC
                LIMIT %s
                """,
                (safe_limit,),
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_run_schools(run_id: int, limit: int = 2000) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 5000))
    cfg = _get_pg_config()
    with psycopg2.connect(**cfg) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    rs.school_id,
                    s.full_name,
                    m.name AS municipality_name,
                    r.name AS region_name,
                    asr.rank_pos,
                    asr.total_score
                FROM edu.analytics_run_school rs
                JOIN edu.school s ON s.school_id = rs.school_id
                JOIN edu.municipality m ON m.municipality_id = s.municipality_id
                JOIN edu.region r ON r.region_id = m.region_id
                LEFT JOIN edu.analytics_school_rating asr
                    ON asr.run_id = rs.run_id
                   AND asr.school_id = rs.school_id
                WHERE rs.run_id = %s
                ORDER BY COALESCE(asr.rank_pos, 2147483647), s.full_name
                LIMIT %s
                """,
                (run_id, safe_limit),
            )
            return [dict(row) for row in cur.fetchall()]


def create_reports_for_run(
    *,
    run_id: int,
    created_by: str | None,
    report_type: str = "standard",
    report_format: str = "json",
    school_id: int | None = None,
) -> int:
    report_type_value = (report_type or "standard").strip().lower()
    if report_type_value not in VALID_REPORT_TYPES:
        report_type_value = "standard"

    report_format_value = (report_format or "json").strip().lower()
    if report_format_value not in VALID_REPORT_FORMATS:
        report_format_value = "json"

    cfg = _get_pg_config()
    with psycopg2.connect(**cfg) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    rr.run_id,
                    rr.request_id,
                    rr.calculated_at,
                    rr.algorithm_version,
                    ar.request_name,
                    ar.year_basis
                FROM edu.analytics_rating_run rr
                JOIN edu.analytics_request ar ON ar.request_id = rr.request_id
                WHERE rr.run_id = %s
                """,
                (run_id,),
            )
            run_meta = cur.fetchone()
            if run_meta is None:
                return 0

            school_clause = ""
            params: list[Any] = [run_id]
            if school_id is not None:
                school_clause = "AND rs.school_id = %s"
                params.append(school_id)

            cur.execute(
                f"""
                SELECT
                    rs.school_id,
                    s.full_name,
                    m.name AS municipality_name,
                    r.name AS region_name,
                    asr.rank_pos,
                    asr.total_score,
                    asr.students_cnt,
                    asr.ege_avg_all,
                    ascard.school_info,
                    ascard.ege_summary,
                    ascard.admission_summary,
                    ascard.prof_summary,
                    ascard.conclusions,
                    COALESCE(
                        jsonb_object_agg(
                            mv.metric_code,
                            jsonb_build_object(
                                'raw_value', mv.raw_value,
                                'norm_value', mv.norm_value,
                                'weighted_value', mv.weighted_value
                            )
                        ) FILTER (WHERE mv.metric_code IS NOT NULL),
                        '{{}}'::jsonb
                    ) AS metrics
                FROM edu.analytics_run_school rs
                JOIN edu.school s ON s.school_id = rs.school_id
                JOIN edu.municipality m ON m.municipality_id = s.municipality_id
                JOIN edu.region r ON r.region_id = m.region_id
                LEFT JOIN edu.analytics_school_rating asr
                    ON asr.run_id = rs.run_id
                   AND asr.school_id = rs.school_id
                LEFT JOIN edu.analytics_school_card ascard
                    ON ascard.run_id = rs.run_id
                   AND ascard.school_id = rs.school_id
                LEFT JOIN edu.analytics_school_metric_value mv
                    ON mv.run_id = rs.run_id
                   AND mv.school_id = rs.school_id
                WHERE rs.run_id = %s
                  {school_clause}
                GROUP BY
                    rs.school_id,
                    s.full_name,
                    m.name,
                    r.name,
                    asr.rank_pos,
                    asr.total_score,
                    asr.students_cnt,
                    asr.ege_avg_all,
                    ascard.school_info,
                    ascard.ege_summary,
                    ascard.admission_summary,
                    ascard.prof_summary,
                    ascard.conclusions
                ORDER BY COALESCE(asr.rank_pos, 2147483647), s.full_name
                """,
                params,
            )
            school_rows = [dict(row) for row in cur.fetchall()]
            if not school_rows:
                return 0

            insert_rows: list[tuple[Any, ...]] = []
            for row in school_rows:
                payload = {
                    "run": {
                        "run_id": int(run_meta["run_id"]),
                        "request_id": int(run_meta["request_id"]),
                        "calculated_at": run_meta["calculated_at"],
                        "algorithm_version": run_meta["algorithm_version"],
                        "request_name": run_meta["request_name"],
                        "year_basis": run_meta["year_basis"],
                    },
                    "school": {
                        "school_id": int(row["school_id"]),
                        "full_name": row["full_name"],
                        "region_name": row["region_name"],
                        "municipality_name": row["municipality_name"],
                    },
                    "rating": {
                        "rank_pos": row["rank_pos"],
                        "total_score": row["total_score"],
                        "students_cnt": row["students_cnt"],
                        "ege_avg_all": row["ege_avg_all"],
                    },
                    "metrics": row.get("metrics") or {},
                    "card": {
                        "school_info": row.get("school_info") or {},
                        "ege_summary": row.get("ege_summary") or {},
                        "admission_summary": row.get("admission_summary") or {},
                        "prof_summary": row.get("prof_summary") or {},
                        "conclusions": row.get("conclusions") or {},
                    },
                    "generated_by": (created_by or "").strip() or None,
                }
                insert_rows.append(
                    (
                        run_id,
                        int(row["school_id"]),
                        report_type_value,
                        report_format_value,
                        Json(_json_safe(payload)),
                    )
                )

            execute_values(
                cur,
                """
                INSERT INTO edu.analytics_school_report
                    (run_id, school_id, report_type, report_format, report_payload)
                VALUES %s
                """,
                insert_rows,
            )
            return len(insert_rows)


def fetch_report_archive(limit: int = 300) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 2000))
    cfg = _get_pg_config()
    with psycopg2.connect(**cfg) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    asrep.report_id,
                    asrep.generated_at,
                    asrep.report_type,
                    asrep.report_format,
                    asrep.run_id,
                    asrep.school_id,
                    asr.rank_pos,
                    s.full_name,
                    m.name AS municipality_name,
                    r.name AS region_name
                FROM edu.analytics_school_report asrep
                JOIN edu.school s ON s.school_id = asrep.school_id
                JOIN edu.municipality m ON m.municipality_id = s.municipality_id
                JOIN edu.region r ON r.region_id = m.region_id
                LEFT JOIN edu.analytics_school_rating asr
                    ON asr.run_id = asrep.run_id
                   AND asr.school_id = asrep.school_id
                ORDER BY asrep.generated_at DESC
                LIMIT %s
                """,
                (safe_limit,),
            )
            return [dict(row) for row in cur.fetchall()]


def get_report_payload(report_id: int) -> dict[str, Any] | None:
    cfg = _get_pg_config()
    with psycopg2.connect(**cfg) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    asrep.report_id,
                    asrep.generated_at,
                    asrep.report_type,
                    asrep.report_format,
                    asrep.run_id,
                    asrep.school_id,
                    asrep.report_payload,
                    s.full_name
                FROM edu.analytics_school_report asrep
                JOIN edu.school s ON s.school_id = asrep.school_id
                WHERE asrep.report_id = %s
                """,
                (report_id,),
            )
            row = cur.fetchone()
            return dict(row) if row is not None else None


def fetch_run_export_data(run_id: int) -> dict[str, Any] | None:
    cfg = _get_pg_config()
    with psycopg2.connect(**cfg) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    rr.run_id,
                    rr.request_id,
                    rr.calculated_at,
                    rr.calculated_by,
                    rr.run_name,
                    rr.algorithm_version,
                    rr.stage,
                    ar.request_name,
                    ar.created_at AS request_created_at,
                    ar.created_by AS request_created_by,
                    ar.year_basis,
                    ar.notes
                FROM edu.analytics_rating_run rr
                JOIN edu.analytics_request ar ON ar.request_id = rr.request_id
                WHERE rr.run_id = %s
                """,
                (run_id,),
            )
            meta = cur.fetchone()
            if meta is None:
                return None

            request_id = int(meta["request_id"])

            cur.execute(
                """
                SELECT
                    asr.rank_pos,
                    asr.school_id,
                    s.full_name,
                    r.name AS region_name,
                    m.name AS municipality_name,
                    asr.total_score,
                    asr.students_cnt,
                    asr.ege_avg_all,
                    MAX(CASE WHEN mv.metric_code = 'avg_graduates' THEN mv.raw_value END) AS avg_graduates_raw,
                    MAX(CASE WHEN mv.metric_code = 'avg_graduates' THEN mv.norm_value END) AS avg_graduates_norm,
                    MAX(CASE WHEN mv.metric_code = 'avg_graduates' THEN mv.weighted_value END) AS avg_graduates_weighted,
                    MAX(CASE WHEN mv.metric_code = 'avg_score_all' THEN mv.raw_value END) AS avg_score_raw,
                    MAX(CASE WHEN mv.metric_code = 'avg_score_all' THEN mv.norm_value END) AS avg_score_norm,
                    MAX(CASE WHEN mv.metric_code = 'avg_score_all' THEN mv.weighted_value END) AS avg_score_weighted,
                    MAX(CASE WHEN mv.metric_code = 'match_share' THEN mv.raw_value END) AS match_share_raw,
                    MAX(CASE WHEN mv.metric_code = 'match_share' THEN mv.norm_value END) AS match_share_norm,
                    MAX(CASE WHEN mv.metric_code = 'match_share' THEN mv.weighted_value END) AS match_share_weighted,
                    MAX(CASE WHEN mv.metric_code = 'threshold_share' THEN mv.raw_value END) AS threshold_share_raw,
                    MAX(CASE WHEN mv.metric_code = 'threshold_share' THEN mv.norm_value END) AS threshold_share_norm,
                    MAX(CASE WHEN mv.metric_code = 'threshold_share' THEN mv.weighted_value END) AS threshold_share_weighted,
                    COALESCE(ascard.conclusions ->> 'matched_programs', '') AS matched_programs
                FROM edu.analytics_school_rating asr
                JOIN edu.school s ON s.school_id = asr.school_id
                JOIN edu.municipality m ON m.municipality_id = s.municipality_id
                JOIN edu.region r ON r.region_id = m.region_id
                LEFT JOIN edu.analytics_school_metric_value mv
                    ON mv.run_id = asr.run_id
                   AND mv.school_id = asr.school_id
                LEFT JOIN edu.analytics_school_card ascard
                    ON ascard.run_id = asr.run_id
                   AND ascard.school_id = asr.school_id
                WHERE asr.run_id = %s
                GROUP BY
                    asr.rank_pos,
                    asr.school_id,
                    s.full_name,
                    r.name,
                    m.name,
                    asr.total_score,
                    asr.students_cnt,
                    asr.ege_avg_all,
                    ascard.conclusions
                ORDER BY asr.rank_pos, s.full_name
                """,
                (run_id,),
            )
            ranking_rows = [dict(row) for row in cur.fetchall()]

            cur.execute(
                """
                SELECT
                    arf.filter_type,
                    arf.region_id,
                    arf.municipality_id,
                    arf.institute_id,
                    arf.profile_id,
                    arf.program_id,
                    arf.subject_id,
                    arf.school_id,
                    arf.min_score
                FROM edu.analytics_request_filter arf
                WHERE arf.request_id = %s
                ORDER BY
                    arf.filter_type,
                    COALESCE(arf.region_id, -1),
                    COALESCE(arf.municipality_id, -1),
                    COALESCE(arf.institute_id, -1),
                    COALESCE(arf.profile_id, -1),
                    COALESCE(arf.program_id, -1),
                    COALESCE(arf.subject_id, -1),
                    COALESCE(arf.school_id, -1)
                """,
                (request_id,),
            )
            filter_rows = [dict(row) for row in cur.fetchall()]

            cur.execute(
                """
                SELECT
                    arm.metric_code,
                    arm.weight,
                    arm.is_primary
                FROM edu.analytics_run_metric arm
                WHERE arm.run_id = %s
                ORDER BY arm.metric_code
                """,
                (run_id,),
            )
            weight_rows = [dict(row) for row in cur.fetchall()]

    return {
        "meta": dict(meta),
        "ranking_rows": ranking_rows,
        "filter_rows": filter_rows,
        "weight_rows": weight_rows,
    }
