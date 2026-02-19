from __future__ import annotations

import os
import threading
from copy import deepcopy
from time import monotonic

from typing import Any


def _read_cache_ttl_seconds() -> float:
    raw = os.getenv("FILTER_OPTIONS_CACHE_TTL_SEC", "300").strip()
    try:
        ttl = float(raw)
    except ValueError:
        ttl = 300.0
    return max(0.0, ttl)


_CACHE_TTL_SECONDS = _read_cache_ttl_seconds()
_CACHE_LOCK = threading.Lock()
_COMMON_FILTER_CACHE: dict[tuple[int | None, bool], tuple[float, dict[str, list[dict[str, Any]]]]] = {}
_INSTITUTES_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_PROGRAMS_CACHE: dict[tuple[int, ...], tuple[float, list[dict[str, Any]]]] = {}


def _cache_get(cache: dict[Any, tuple[float, Any]], key: Any) -> Any | None:
    if _CACHE_TTL_SECONDS <= 0:
        return None
    now = monotonic()
    with _CACHE_LOCK:
        item = cache.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at <= now:
            cache.pop(key, None)
            return None
        return deepcopy(value)


def _cache_put(cache: dict[Any, tuple[float, Any]], key: Any, value: Any) -> None:
    if _CACHE_TTL_SECONDS <= 0:
        return
    expires_at = monotonic() + _CACHE_TTL_SECONDS
    with _CACHE_LOCK:
        cache[key] = (expires_at, deepcopy(value))


def clear_filter_options_cache() -> None:
    with _CACHE_LOCK:
        _COMMON_FILTER_CACHE.clear()
        _INSTITUTES_CACHE.clear()
        _PROGRAMS_CACHE.clear()


def fetch_common_filter_options(
    cur,
    *,
    region_id: int | None = None,
    include_subject_min_score: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    cache_key = (region_id, include_subject_min_score)
    cached = _cache_get(_COMMON_FILTER_CACHE, cache_key)
    if cached is not None:
        return cached

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

    result = {
        "regions": regions,
        "municipalities": municipalities,
        "profiles": profiles,
        "years": years,
        "kinds": kinds,
        "subjects": subjects,
    }
    _cache_put(_COMMON_FILTER_CACHE, cache_key, result)
    return result


def fetch_institutes(cur) -> list[dict[str, Any]]:
    cached = _cache_get(_INSTITUTES_CACHE, "all")
    if cached is not None:
        return cached

    cur.execute("SELECT institute_id, name FROM edu.institute ORDER BY name")
    result = [dict(r) for r in cur.fetchall()]
    _cache_put(_INSTITUTES_CACHE, "all", result)
    return result


def fetch_program_filter_options(cur, *, institute_ids: tuple[int, ...] = ()) -> list[dict[str, Any]]:
    key = tuple(sorted(set(institute_ids)))
    cached = _cache_get(_PROGRAMS_CACHE, key)
    if cached is not None:
        return cached

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
    result = [dict(r) for r in cur.fetchall()]
    _cache_put(_PROGRAMS_CACHE, key, result)
    return result

