from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from .config import (
        DEFAULT_RATING_LIMIT,
        DEFAULT_RATING_W_AVG_SCORE,
        DEFAULT_RATING_W_GRADUATES,
        DEFAULT_RATING_W_MATCH_SHARE,
        DEFAULT_RATING_W_THRESHOLD_SHARE,
        MAX_RATING_LIMIT,
        MIN_RATING_LIMIT,
    )
    from .filter_options_repo import (
        fetch_common_filter_options,
        fetch_institutes,
        fetch_program_filter_options,
    )
    from .search_repo import get_pg_connection, normalize_search_text
except ImportError:
    from config import (
        DEFAULT_RATING_LIMIT,
        DEFAULT_RATING_W_AVG_SCORE,
        DEFAULT_RATING_W_GRADUATES,
        DEFAULT_RATING_W_MATCH_SHARE,
        DEFAULT_RATING_W_THRESHOLD_SHARE,
        MAX_RATING_LIMIT,
        MIN_RATING_LIMIT,
    )
    from filter_options_repo import (
        fetch_common_filter_options,
        fetch_institutes,
        fetch_program_filter_options,
    )
    from search_repo import get_pg_connection, normalize_search_text


_NORM_FULL_NAME_SQL = (
    "lower(replace(replace(regexp_replace(coalesce(s.full_name, ''), '\\s+', ' ', 'g'), chr(1025), chr(1045)), chr(1105), chr(1077)))"
)


@dataclass(frozen=True)
class RatingFilters:
    q: str = ""
    region_id: int | None = None
    municipality_id: int | None = None
    profile_ids: tuple[int, ...] = ()
    year: int | None = None
    kind: str | None = None
    subject_ids: tuple[int, ...] = ()
    institute_ids: tuple[int, ...] = ()
    program_ids: tuple[int, ...] = ()
    min_graduates: int | None = None
    min_avg_score: float | None = None
    enforce_subject_threshold: bool = True
    limit: int = DEFAULT_RATING_LIMIT


@dataclass(frozen=True)
class RatingWeights:
    graduates: float = DEFAULT_RATING_W_GRADUATES
    avg_score: float = DEFAULT_RATING_W_AVG_SCORE
    match_share: float = DEFAULT_RATING_W_MATCH_SHARE
    threshold_share: float = DEFAULT_RATING_W_THRESHOLD_SHARE


def _to_float(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _to_float(value)


def normalize_rating_weights(weights: RatingWeights) -> RatingWeights:
    values = [
        max(0.0, float(weights.graduates)),
        max(0.0, float(weights.avg_score)),
        max(0.0, float(weights.match_share)),
        max(0.0, float(weights.threshold_share)),
    ]
    total = sum(values)
    if total <= 0:
        return RatingWeights()
    return RatingWeights(
        graduates=values[0] / total,
        avg_score=values[1] / total,
        match_share=values[2] / total,
        threshold_share=values[3] / total,
    )


def fetch_rating_filter_options(
    *,
    region_id: int | None = None,
    institute_ids: tuple[int, ...] = (),
) -> dict[str, list[dict[str, Any]]]:
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            options = fetch_common_filter_options(
                cur,
                region_id=region_id,
                include_subject_min_score=True,
            )
            options["institutes"] = fetch_institutes(cur)
            options["programs"] = fetch_program_filter_options(cur, institute_ids=institute_ids)
            return options


def fetch_programs(institute_ids: tuple[int, ...] = ()) -> list[dict[str, Any]]:
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            rows = fetch_program_filter_options(cur, institute_ids=institute_ids)

    for row in rows:
        row.pop("is_active", None)
    return rows


def fetch_program_requirements(
    *,
    institute_ids: tuple[int, ...] = (),
    program_ids: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
    has_scope = bool(institute_ids) or bool(program_ids)
    if not has_scope:
        return []

    where_parts = ["sp.is_active IS TRUE"]
    params: list[Any] = []
    if institute_ids:
        where_parts.append("sp.institute_id = ANY(%s)")
        params.append(list(sorted(set(institute_ids))))
    if program_ids:
        where_parts.append("sp.program_id = ANY(%s)")
        params.append(list(program_ids))
    where_sql = " AND ".join(where_parts)

    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    sp.program_id,
                    i.name AS institute_name,
                    sp.code,
                    sp.name AS program_name,
                    per.role,
                    per.weight,
                    subj.subject_id,
                    subj.name AS subject_name,
                    subj.min_passing_score
                FROM edu.study_program sp
                JOIN edu.institute i ON i.institute_id = sp.institute_id
                LEFT JOIN edu.program_ege_requirement per ON per.program_id = sp.program_id
                LEFT JOIN edu.ege_subject subj ON subj.subject_id = per.subject_id
                WHERE {where_sql}
                ORDER BY
                    i.name,
                    sp.code,
                    sp.name,
                    CASE per.role WHEN 'required' THEN 1 WHEN 'choice' THEN 2 ELSE 3 END,
                    subj.name
                """,
                params,
            )
            rows = [dict(r) for r in cur.fetchall()]

    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        program_id = int(row["program_id"])
        bucket = grouped.get(program_id)
        if bucket is None:
            bucket = {
                "program_id": program_id,
                "institute_name": row["institute_name"],
                "code": row["code"],
                "name": row["program_name"],
                "required_subjects": [],
                "choice_subjects": [],
            }
            grouped[program_id] = bucket

        if row["subject_id"] is None:
            continue

        subject = {
            "subject_id": int(row["subject_id"]),
            "name": row["subject_name"],
            "weight": int(row["weight"]) if row["weight"] is not None else None,
            "min_passing_score": _to_optional_float(row["min_passing_score"]),
        }
        if (row["role"] or "").strip().lower() == "required":
            bucket["required_subjects"].append(subject)
        else:
            bucket["choice_subjects"].append(subject)

    result = list(grouped.values())
    for item in result:
        item["required_subjects"].sort(key=lambda x: x["name"])
        item["choice_subjects"].sort(key=lambda x: x["name"])
    result.sort(key=lambda x: (x["institute_name"], x["code"] or "", x["name"]))
    return result


def _apply_rating_scores_from_norms(
    rows: list[dict[str, Any]],
    *,
    normalized_weights: RatingWeights,
    should_calculate_potential: bool,
) -> None:
    for row in rows:
        potential_applicants_avg = _to_float(row.get("potential_applicants_avg"))
        threshold_share = _to_float(row.get("threshold_share"))
        if not should_calculate_potential:
            potential_applicants_avg = 0.0

        norm_graduates = _to_float(row.pop("norm_graduates_sql", 0.0))
        norm_avg_score = _to_float(row.pop("norm_avg_score_sql", 0.0))
        norm_potential_applicants = _to_float(row.pop("norm_potential_applicants_sql", 0.0))
        score_part_graduates = normalized_weights.graduates * norm_graduates
        score_part_avg_score = normalized_weights.avg_score * norm_avg_score
        score_part_potential = normalized_weights.match_share * norm_potential_applicants
        score_part_threshold = normalized_weights.threshold_share * threshold_share
        score = score_part_graduates + score_part_avg_score + score_part_potential + score_part_threshold

        row["avg_graduates"] = _to_optional_float(row.get("avg_graduates"))
        row["avg_score_all"] = _to_optional_float(row.get("avg_score_all"))
        row["admission_subject_avg"] = _to_optional_float(row.get("admission_subject_avg"))
        row["avg_score_for_rating"] = _to_optional_float(row.get("avg_score_for_rating"))
        row["potential_applicants_avg"] = round(potential_applicants_avg, 2) if should_calculate_potential else None
        row["norm_potential_applicants"] = round(norm_potential_applicants, 4)
        row["match_share"] = round(norm_potential_applicants, 4)
        row["threshold_share"] = round(threshold_share, 4)
        row["score_part_graduates"] = round(score_part_graduates, 6)
        row["score_part_avg_score"] = round(score_part_avg_score, 6)
        row["score_part_potential"] = round(score_part_potential, 6)
        row["score_part_threshold"] = round(score_part_threshold, 6)
        row["norm_graduates"] = round(norm_graduates, 4)
        row["norm_avg_score"] = round(norm_avg_score, 4)
        row["rating_score"] = round(score, 6)
        row["programs_total"] = int(row.get("programs_total") or 0)
        row["programs_matched"] = int(row.get("programs_matched") or 0)
        row["ege_years"] = int(row.get("ege_years") or 0)


def _sort_limit_and_rank_rows(rows: list[dict[str, Any]], *, safe_limit: int) -> list[dict[str, Any]]:
    rows.sort(
        key=lambda x: (
            -float(x["rating_score"]),
            -float(x["avg_score_for_rating"] or 0),
            -float(x["avg_graduates"] or 0),
            str(x["full_name"]).lower(),
        )
    )
    rows = rows[:safe_limit]
    for idx, row in enumerate(rows, start=1):
        row["rank_pos"] = idx
    return rows


def calculate_school_rating(
    filters: RatingFilters,
    weights: RatingWeights,
) -> list[dict[str, Any]]:
    base_clauses: list[str] = []
    base_params: list[Any] = []

    q_norm = normalize_search_text(filters.q)
    if q_norm:
        base_clauses.append(f"{_NORM_FULL_NAME_SQL} LIKE %s")
        base_params.append(f"%{q_norm}%")

    if filters.region_id is not None:
        base_clauses.append("r.region_id = %s")
        base_params.append(filters.region_id)

    if filters.municipality_id is not None:
        base_clauses.append("m.municipality_id = %s")
        base_params.append(filters.municipality_id)

    if filters.profile_ids:
        base_clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM edu.school_profile_link spl
                WHERE spl.school_id = s.school_id
                  AND spl.profile_id = ANY(%s)
            )
            """
        )
        base_params.append(list(filters.profile_ids))

    base_where_sql = " AND ".join(f"({c.strip()})" for c in base_clauses) if base_clauses else "TRUE"

    ege_clauses: list[str] = []
    ege_params: list[Any] = []
    if filters.year is not None:
        ege_clauses.append('y."year" = %s')
        ege_params.append(filters.year)
    if filters.kind:
        ege_clauses.append("y.kind::text = %s")
        ege_params.append(filters.kind)
    ege_where_sql = " AND ".join(ege_clauses) if ege_clauses else "TRUE"

    has_subject_filter = bool(filters.subject_ids)
    selected_subjects = sorted(set(filters.subject_ids))
    subject_filter_sql = "TRUE"
    subject_filter_params: list[Any] = []
    if has_subject_filter:
        subject_filter_sql = """
            EXISTS (
                SELECT 1
                FROM school_subject_presence ssp
                WHERE ssp.school_id = bs.school_id
                  AND ssp.subject_id = ANY(%s)
                GROUP BY ssp.school_id
                HAVING COUNT(DISTINCT ssp.subject_id) = %s
            )
            AND (
                %s = FALSE
                OR EXISTS (
                    SELECT 1
                    FROM school_subject_avg ssa
                    LEFT JOIN edu.ege_subject subj ON subj.subject_id = ssa.subject_id
                    WHERE ssa.school_id = bs.school_id
                      AND ssa.subject_id = ANY(%s)
                    GROUP BY ssa.school_id
                    HAVING COUNT(
                        DISTINCT CASE
                            WHEN ssa.avg_score >= COALESCE(subj.min_passing_score, 0)
                             AND (%s IS NULL OR ssa.avg_score >= %s)
                            THEN ssa.subject_id
                            ELSE NULL
                        END
                    ) = %s
                )
            )
        """
        subject_filter_params.extend(
            [
                selected_subjects,
                len(selected_subjects),
                filters.enforce_subject_threshold,
                selected_subjects,
                filters.min_avg_score,
                filters.min_avg_score,
                len(selected_subjects),
            ]
        )

    has_program_scope = bool(filters.institute_ids) or bool(filters.program_ids)
    should_calculate_potential = has_subject_filter or has_program_scope
    has_rating_scope_sql = "TRUE" if should_calculate_potential else "FALSE"
    program_scope_clauses = ["sp.is_active IS TRUE"]
    program_scope_params: list[Any] = []
    if filters.institute_ids:
        selected_institutes = sorted(set(filters.institute_ids))
        program_scope_clauses.append("sp.institute_id = ANY(%s)")
        program_scope_params.append(selected_institutes)
    if filters.program_ids:
        selected_programs = sorted(set(filters.program_ids))
        program_scope_clauses.append("sp.program_id = ANY(%s)")
        program_scope_params.append(selected_programs)
    program_scope_sql = " AND ".join(program_scope_clauses)
    apply_all_subject_thresholds = (
        filters.enforce_subject_threshold and not has_subject_filter and not has_program_scope
    )
    all_subject_threshold_sql = "TRUE"
    all_subject_threshold_params: list[Any] = []
    if apply_all_subject_thresholds:
        all_subject_threshold_sql = """
            EXISTS (
                SELECT 1
                FROM school_subject_avg ssa
                LEFT JOIN edu.ege_subject subj ON subj.subject_id = ssa.subject_id
                WHERE ssa.school_id = bs.school_id
                GROUP BY ssa.school_id
                HAVING COUNT(*) FILTER (
                    WHERE ssa.avg_score < COALESCE(subj.min_passing_score, 0)
                       OR (%s IS NOT NULL AND ssa.avg_score < %s)
                ) = 0
            )
        """
        all_subject_threshold_params.extend([filters.min_avg_score, filters.min_avg_score])
    use_selected_subjects_for_admission_avg = bool(selected_subjects)
    use_program_subjects_for_admission_avg = (not use_selected_subjects_for_admission_avg) and has_program_scope
    use_admission_subject_avg_for_rating = has_subject_filter or has_program_scope
    has_program_scope_sql = "TRUE" if has_program_scope else "FALSE"
    has_subject_filter_sql = "TRUE" if has_subject_filter else "FALSE"
    should_calculate_potential_sql = "TRUE" if should_calculate_potential else "FALSE"

    safe_limit = max(MIN_RATING_LIMIT, min(int(filters.limit), MAX_RATING_LIMIT))
    effective_weights = weights
    if not should_calculate_potential:
        effective_weights = RatingWeights(
            graduates=weights.graduates,
            avg_score=weights.avg_score,
            match_share=0.0,
            threshold_share=weights.threshold_share,
        )
    normalized_weights = normalize_rating_weights(effective_weights)

    sql = f"""
        WITH base_school AS (
            SELECT
                s.school_id,
                s.full_name,
                s.is_active,
                m.municipality_id,
                m.name AS municipality_name,
                r.region_id,
                r.name AS region_name
            FROM edu.school s
            JOIN edu.municipality m ON m.municipality_id = s.municipality_id
            JOIN edu.region r ON r.region_id = m.region_id
            WHERE {base_where_sql}
        ),
        ege_periods AS (
            SELECT
                y.ege_school_year_id,
                y.school_id,
                y."year",
                y.kind::text AS kind,
                y.graduates_total
            FROM edu.ege_school_year y
            JOIN base_school bs ON bs.school_id = y.school_id
            WHERE {ege_where_sql}
        ),
        ege_subject_rows AS (
            SELECT
                ep.school_id,
                ep.ege_school_year_id,
                ep."year",
                ep.kind,
                ep.graduates_total,
                ss.subject_id,
                ss.avg_score,
                ss.participants_cnt,
                ss.chosen_cnt
            FROM ege_periods ep
            LEFT JOIN edu.ege_school_subject_stat ss ON ss.ege_school_year_id = ep.ege_school_year_id
        ),
        school_subject_presence AS (
            SELECT esr.school_id, esr.subject_id
            FROM ege_subject_rows esr
            WHERE esr.subject_id IS NOT NULL
              AND COALESCE(esr.participants_cnt, esr.chosen_cnt, 0) > 0
            GROUP BY esr.school_id, esr.subject_id
        ),
        school_subject_avg AS (
            SELECT
                esr.school_id,
                esr.subject_id,
                ROUND(AVG(esr.avg_score)::numeric, 2) AS avg_score
            FROM ege_subject_rows esr
            WHERE esr.subject_id IS NOT NULL
              AND esr.avg_score IS NOT NULL
              AND COALESCE(esr.participants_cnt, esr.chosen_cnt, 0) > 0
            GROUP BY esr.school_id, esr.subject_id
        ),
        school_subject_capacity AS (
            SELECT
                esr.school_id,
                esr.subject_id,
                ROUND(AVG(COALESCE(esr.participants_cnt, esr.chosen_cnt, 0))::numeric, 2) AS participants_avg
            FROM ege_subject_rows esr
            WHERE esr.subject_id IS NOT NULL
              AND COALESCE(esr.participants_cnt, esr.chosen_cnt, 0) > 0
            GROUP BY esr.school_id, esr.subject_id
        ),
        all_subject_threshold_summary AS (
            SELECT
                ssa.school_id,
                CASE
                    WHEN COUNT(*) = 0 THEN 0::numeric
                    ELSE ROUND(
                        COUNT(*) FILTER (
                            WHERE ssa.avg_score >= COALESCE(subj.min_passing_score, 0)
                              AND (%s IS NULL OR ssa.avg_score >= %s)
                        )::numeric / COUNT(*),
                        4
                    )
                END AS threshold_share_all
            FROM school_subject_avg ssa
            LEFT JOIN edu.ege_subject subj ON subj.subject_id = ssa.subject_id
            GROUP BY ssa.school_id
        ),
        eligible_schools AS (
            SELECT bs.*
            FROM base_school bs
            WHERE (%s = FALSE OR {subject_filter_sql})
              AND (%s = FALSE OR {all_subject_threshold_sql})
        ),
        selected_programs AS (
            SELECT
                sp.program_id,
                sp.code,
                sp.name AS program_name,
                sp.institute_id
            FROM edu.study_program sp
            WHERE {has_program_scope_sql}
              AND {program_scope_sql}
            UNION ALL
            SELECT
                0::int AS program_id,
                NULL::text AS code,
                'selected_subjects'::text AS program_name,
                NULL::int AS institute_id
            WHERE NOT {has_program_scope_sql}
              AND {has_subject_filter_sql}
        ),
        program_requirements AS (
            SELECT
                sp.program_id,
                per.subject_id,
                per.role::text AS role,
                per.weight,
                subj.min_passing_score
            FROM selected_programs sp
            JOIN edu.program_ege_requirement per ON per.program_id = sp.program_id
            LEFT JOIN edu.ege_subject subj ON subj.subject_id = per.subject_id
            WHERE {has_program_scope_sql}
            UNION ALL
            SELECT
                sp.program_id,
                subj.subject_id,
                'required'::text AS role,
                NULL::int AS weight,
                subj.min_passing_score
            FROM selected_programs sp
            JOIN edu.ege_subject subj ON subj.subject_id = ANY(%s::int[])
            WHERE NOT {has_program_scope_sql}
              AND {has_subject_filter_sql}
            UNION ALL
            SELECT
                sp.program_id,
                subj.subject_id,
                'required'::text AS role,
                NULL::int AS weight,
                subj.min_passing_score
            FROM selected_programs sp
            JOIN edu.ege_subject subj ON subj.subject_id = ANY(%s::int[])
            WHERE {has_program_scope_sql}
              AND {has_subject_filter_sql}
              AND NOT EXISTS (
                  SELECT 1
                  FROM edu.program_ege_requirement per0
                  WHERE per0.program_id = sp.program_id
                    AND per0.subject_id = subj.subject_id
                    AND per0.role::text = 'required'
              )
        ),
        scoring_subjects AS (
            SELECT DISTINCT pr.subject_id
            FROM program_requirements pr
            WHERE %s = TRUE
              AND pr.subject_id IS NOT NULL
            UNION
            SELECT DISTINCT unnest(%s::int[])
            WHERE %s = TRUE
        ),
        admission_subject_score AS (
            SELECT
                es.school_id,
                ROUND(AVG(ssa.avg_score)::numeric, 2) AS admission_subject_avg
            FROM eligible_schools es
            JOIN school_subject_avg ssa ON ssa.school_id = es.school_id
            JOIN scoring_subjects ss ON ss.subject_id = ssa.subject_id
            WHERE ssa.avg_score IS NOT NULL
            GROUP BY es.school_id
        ),
        program_school_subject AS (
            SELECT
                es.school_id,
                pr.program_id,
                pr.role,
                pr.weight,
                ssc.participants_avg,
                (
                    ssc.subject_id IS NOT NULL
                    AND (%s = FALSE OR (ssa.avg_score IS NOT NULL AND ssa.avg_score >= COALESCE(pr.min_passing_score, 0)))
                    AND (%s IS NULL OR (ssa.avg_score IS NOT NULL AND ssa.avg_score >= %s))
                ) AS subject_ok
            FROM eligible_schools es
            JOIN program_requirements pr ON TRUE
            LEFT JOIN school_subject_capacity ssc
                ON ssc.school_id = es.school_id
               AND ssc.subject_id = pr.subject_id
            LEFT JOIN school_subject_avg ssa
                ON ssa.school_id = es.school_id
               AND ssa.subject_id = pr.subject_id
        ),
        program_school_eval AS (
            SELECT
                pss.school_id,
                pss.program_id,
                COUNT(*) FILTER (WHERE pss.role = 'required')::int AS required_total,
                COUNT(*) FILTER (WHERE pss.role = 'required' AND pss.subject_ok)::int AS required_ok,
                MIN(pss.participants_avg) FILTER (WHERE pss.role = 'required' AND pss.subject_ok) AS required_min_participants,
                COUNT(DISTINCT pss.weight) FILTER (WHERE pss.role = 'choice')::int AS choice_group_total,
                COUNT(DISTINCT pss.weight) FILTER (WHERE pss.role = 'choice' AND pss.subject_ok)::int AS choice_group_ok
            FROM program_school_subject pss
            GROUP BY pss.school_id, pss.program_id
        ),
        choice_group_capacity AS (
            SELECT
                pss.school_id,
                pss.program_id,
                pss.weight AS choice_group,
                MAX(pss.participants_avg) AS group_max_participants
            FROM program_school_subject pss
            WHERE pss.role = 'choice'
              AND pss.subject_ok
            GROUP BY pss.school_id, pss.program_id, pss.weight
        ),
        choice_group_eval AS (
            SELECT
                cgc.school_id,
                cgc.program_id,
                COUNT(*)::int AS choice_group_ok,
                MIN(cgc.group_max_participants) AS choice_min_participants
            FROM choice_group_capacity cgc
            GROUP BY cgc.school_id, cgc.program_id
        ),
        program_school_match AS (
            SELECT
                pse.school_id,
                pse.program_id,
                CASE
                    WHEN pse.required_total = 0 THEN 0::numeric
                    WHEN pse.required_ok <> pse.required_total THEN 0::numeric
                    WHEN pse.choice_group_ok <> pse.choice_group_total THEN 0::numeric
                    WHEN pse.choice_group_total = 0 THEN COALESCE(pse.required_min_participants, 0::numeric)
                    ELSE LEAST(
                        COALESCE(pse.required_min_participants, 0::numeric),
                        COALESCE(cge.choice_min_participants, 0::numeric)
                    )
                END AS potential_applicants,
                CASE
                    WHEN (pse.required_total + pse.choice_group_total) = 0 THEN 0::numeric
                    ELSE ROUND(
                        (pse.required_ok::numeric + pse.choice_group_ok::numeric)
                        / (pse.required_total + pse.choice_group_total),
                        4
                    )
                END AS threshold_share
            FROM program_school_eval pse
            LEFT JOIN choice_group_eval cge
              ON cge.school_id = pse.school_id
             AND cge.program_id = pse.program_id
        ),
        program_summary AS (
            SELECT
                psm.school_id,
                COUNT(*)::int AS programs_total,
                COUNT(*) FILTER (WHERE psm.potential_applicants > 0)::int AS programs_matched,
                ROUND(AVG(psm.potential_applicants)::numeric, 2) AS potential_applicants_avg,
                COALESCE(
                    ROUND(MAX(psm.threshold_share) FILTER (WHERE psm.potential_applicants > 0)::numeric, 4),
                    ROUND(MAX(psm.threshold_share)::numeric, 4),
                    0::numeric
                ) AS threshold_share
            FROM program_school_match psm
            GROUP BY psm.school_id
        ),
        matched_programs AS (
            SELECT
                psm.school_id,
                string_agg(
                    CASE
                        WHEN sp.code IS NULL OR sp.code = '' THEN sp.program_name
                        ELSE sp.code || ' ' || sp.program_name
                    END,
                    '; ' ORDER BY sp.program_name
                ) AS matched_programs
            FROM program_school_match psm
            JOIN selected_programs sp ON sp.program_id = psm.program_id
            WHERE psm.potential_applicants > 0
              AND {has_program_scope_sql}
            GROUP BY psm.school_id
        ),
        school_period_agg AS (
            SELECT
                ep.school_id,
                ROUND(AVG(ep.graduates_total)::numeric, 1) AS avg_graduates,
                MAX(ep."year") AS last_year,
                COUNT(DISTINCT ep."year")::int AS ege_years
            FROM ege_periods ep
            GROUP BY ep.school_id
        ),
        school_score_agg AS (
            SELECT
                esr.school_id,
                ROUND(AVG(esr.avg_score)::numeric, 2) AS avg_score_all
            FROM ege_subject_rows esr
            WHERE esr.avg_score IS NOT NULL
              AND COALESCE(esr.participants_cnt, esr.chosen_cnt, 0) > 0
            GROUP BY esr.school_id
        )
        SELECT
            bs.school_id,
            bs.full_name,
            bs.is_active,
            bs.region_id,
            bs.region_name,
            bs.municipality_id,
            bs.municipality_name,
            COALESCE(p.profile_names, '') AS profile_names,
            spa.avg_graduates,
            spa.last_year,
            spa.ege_years,
            ssa.avg_score_all,
            ass.admission_subject_avg,
            CASE
                WHEN %s = TRUE THEN ass.admission_subject_avg
                ELSE ssa.avg_score_all
            END AS avg_score_for_rating,
            COALESCE(ps.programs_total, 0) AS programs_total,
            COALESCE(ps.programs_matched, 0) AS programs_matched,
            COALESCE(ps.potential_applicants_avg, 0) AS potential_applicants_avg,
            CASE
                WHEN {has_rating_scope_sql}
                    THEN COALESCE(ps.threshold_share, 0)
                ELSE COALESCE(ast.threshold_share_all, 0)
            END AS threshold_share,
            COALESCE(mp.matched_programs, '') AS matched_programs,
            CASE
                WHEN MAX(COALESCE(spa.avg_graduates, 0)) OVER () > 0
                    THEN COALESCE(spa.avg_graduates, 0)::numeric
                        / MAX(COALESCE(spa.avg_graduates, 0)) OVER ()
                ELSE 0::numeric
            END AS norm_graduates_sql,
            CASE
                WHEN MAX(
                    COALESCE(
                        CASE
                            WHEN %s = TRUE THEN ass.admission_subject_avg
                            ELSE ssa.avg_score_all
                        END,
                        0
                    )
                ) OVER () > 0
                    THEN COALESCE(
                        CASE
                            WHEN %s = TRUE THEN ass.admission_subject_avg
                            ELSE ssa.avg_score_all
                        END,
                        0
                    )::numeric
                        / MAX(
                            COALESCE(
                                CASE
                                    WHEN %s = TRUE THEN ass.admission_subject_avg
                                    ELSE ssa.avg_score_all
                                END,
                                0
                            )
                        ) OVER ()
                ELSE 0::numeric
            END AS norm_avg_score_sql,
            CASE
                WHEN %s = TRUE
                  AND MAX(COALESCE(ps.potential_applicants_avg, 0)) OVER () > 0
                    THEN COALESCE(ps.potential_applicants_avg, 0)::numeric
                        / MAX(COALESCE(ps.potential_applicants_avg, 0)) OVER ()
                ELSE 0::numeric
            END AS norm_potential_applicants_sql
        FROM eligible_schools es
        JOIN base_school bs ON bs.school_id = es.school_id
        LEFT JOIN LATERAL (
            SELECT string_agg(sp.name, ', ' ORDER BY sp.name) AS profile_names
            FROM edu.school_profile_link spl
            JOIN edu.school_profile sp ON sp.profile_id = spl.profile_id
            WHERE spl.school_id = bs.school_id
        ) p ON TRUE
        LEFT JOIN school_period_agg spa ON spa.school_id = bs.school_id
        LEFT JOIN school_score_agg ssa ON ssa.school_id = bs.school_id
        LEFT JOIN admission_subject_score ass ON ass.school_id = bs.school_id
        LEFT JOIN program_summary ps ON ps.school_id = bs.school_id
        LEFT JOIN all_subject_threshold_summary ast ON ast.school_id = bs.school_id
        LEFT JOIN matched_programs mp ON mp.school_id = bs.school_id
        WHERE spa.school_id IS NOT NULL
          AND (NOT {should_calculate_potential_sql} OR COALESCE(ps.potential_applicants_avg, 0) > 0)
          AND (%s IS NULL OR COALESCE(spa.avg_graduates, 0) >= %s)
          AND (
                %s IS NULL
                OR COALESCE(
                    CASE
                        WHEN %s = TRUE THEN ass.admission_subject_avg
                        ELSE ssa.avg_score_all
                    END,
                    0
                ) >= %s
          )
        ORDER BY bs.full_name
    """

    params: list[Any] = []
    params.extend(base_params)
    params.extend(ege_params)
    params.extend([filters.min_avg_score, filters.min_avg_score])
    params.append(has_subject_filter)
    params.extend(subject_filter_params)
    params.append(apply_all_subject_thresholds)
    params.extend(all_subject_threshold_params)
    params.extend(program_scope_params)
    params.append(selected_subjects)
    params.append(selected_subjects)
    params.extend(
        [
            use_program_subjects_for_admission_avg,
            selected_subjects,
            use_selected_subjects_for_admission_avg,
        ]
    )

    params.extend(
        [
            filters.enforce_subject_threshold,
            filters.min_avg_score,
            filters.min_avg_score,
        ]
    )

    params.extend(
        [
            use_admission_subject_avg_for_rating,
            use_admission_subject_avg_for_rating,
            use_admission_subject_avg_for_rating,
            use_admission_subject_avg_for_rating,
            should_calculate_potential,
            filters.min_graduates,
            filters.min_graduates,
            filters.min_avg_score,
            use_admission_subject_avg_for_rating,
            filters.min_avg_score,
        ]
    )

    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        return []
    _apply_rating_scores_from_norms(
        rows,
        normalized_weights=normalized_weights,
        should_calculate_potential=should_calculate_potential,
    )
    return _sort_limit_and_rank_rows(rows, safe_limit=safe_limit)
