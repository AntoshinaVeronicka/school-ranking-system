from __future__ import annotations

from typing import Any


def fetch_common_filter_options(
    cur,
    *,
    region_id: int | None = None,
    include_subject_min_score: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    cur.execute("SELECT region_id, name FROM edu.region ORDER BY name")
    regions = [dict(r) for r in cur.fetchall()]

    if region_id is None:
        municipalities: list[dict[str, Any]] = []
    else:
        cur.execute(
            """
            SELECT municipality_id, name
            FROM edu.municipality
            WHERE region_id = %s
            ORDER BY name
            """,
            (region_id,),
        )
        municipalities = [dict(r) for r in cur.fetchall()]

    cur.execute("SELECT profile_id, name FROM edu.school_profile ORDER BY name")
    profiles = [dict(r) for r in cur.fetchall()]

    cur.execute('SELECT DISTINCT "year" FROM edu.ege_school_year ORDER BY "year" DESC')
    years = [dict(r) for r in cur.fetchall()]

    cur.execute("SELECT DISTINCT kind::text AS kind FROM edu.ege_school_year ORDER BY kind::text")
    kinds = [dict(r) for r in cur.fetchall()]

    if include_subject_min_score:
        cur.execute("SELECT subject_id, name, min_passing_score FROM edu.ege_subject ORDER BY name")
    else:
        cur.execute("SELECT subject_id, name FROM edu.ege_subject ORDER BY name")
    subjects = [dict(r) for r in cur.fetchall()]

    return {
        "regions": regions,
        "municipalities": municipalities,
        "profiles": profiles,
        "years": years,
        "kinds": kinds,
        "subjects": subjects,
    }


def fetch_institutes(cur) -> list[dict[str, Any]]:
    cur.execute("SELECT institute_id, name FROM edu.institute ORDER BY name")
    return [dict(r) for r in cur.fetchall()]


def fetch_program_filter_options(cur, *, institute_ids: tuple[int, ...] = ()) -> list[dict[str, Any]]:
    if not institute_ids:
        cur.execute(
            """
            SELECT
                sp.program_id,
                sp.institute_id,
                i.name AS institute_name,
                sp.code,
                sp.name,
                sp.is_active
            FROM edu.study_program sp
            JOIN edu.institute i ON i.institute_id = sp.institute_id
            WHERE sp.is_active IS TRUE
            ORDER BY i.name, sp.code, sp.name
            """
        )
    else:
        cur.execute(
            """
            SELECT
                sp.program_id,
                sp.institute_id,
                i.name AS institute_name,
                sp.code,
                sp.name,
                sp.is_active
            FROM edu.study_program sp
            JOIN edu.institute i ON i.institute_id = sp.institute_id
            WHERE sp.is_active IS TRUE
              AND sp.institute_id = ANY(%s)
            ORDER BY i.name, sp.code, sp.name
            """,
            (list(sorted(set(institute_ids))),),
        )
    return [dict(r) for r in cur.fetchall()]

