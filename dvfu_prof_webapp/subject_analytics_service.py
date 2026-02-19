from __future__ import annotations

from typing import Any


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _heat_color(value: float | None, *, min_value: float, max_value: float, invert: bool = False) -> str:
    if value is None:
        return "#f5f7fb"

    if max_value > min_value:
        ratio = (value - min_value) / (max_value - min_value)
    else:
        ratio = 1.0

    ratio = max(0.0, min(1.0, ratio))
    if invert:
        ratio = 1.0 - ratio

    hue = int(12 + ratio * 108)  # Цветовая шкала: от красного к зелёному.
    light = int(95 - ratio * 14)
    return f"hsl({hue} 72% {light}%)"


def build_subject_analytics(ege_timeline: list[dict[str, object]]) -> dict[str, object]:
    actual_periods: list[dict[str, object]] = []
    for period in ege_timeline:
        kind = str(period.get("kind") or "").strip().lower()
        if kind != "actual":
            continue
        year = _to_int_or_none(period.get("year"))
        if year is None:
            continue
        actual_periods.append(
            {
                "year": year,
                "subjects": period.get("subjects") or [],
            }
        )

    if not actual_periods:
        return {
            "heatmap_period_label": "",
            "heatmap_rows": [],
            "histogram_rows": [],
            "histogram_latest_year": None,
            "histogram_prev_year": None,
        }

    years = sorted({int(p["year"]) for p in actual_periods})
    if len(years) > 1:
        heatmap_period_label = f"всем данным факта ({years[0]}-{years[-1]})"
    else:
        heatmap_period_label = f"всем данным факта ({years[0]})"

    by_subject: dict[int, dict[str, object]] = {}
    for period in actual_periods:
        year = int(period["year"])
        for subject in (period.get("subjects") or []):
            if not isinstance(subject, dict):
                continue
            sid = _to_int_or_none(subject.get("subject_id"))
            if sid is None:
                continue
            subject_name = str(subject.get("subject_name") or "").strip()
            if not subject_name:
                continue

            participants_cnt = _to_int_or_none(subject.get("participants_cnt"))
            chosen_cnt = _to_int_or_none(subject.get("chosen_cnt"))
            participants_base = participants_cnt if participants_cnt is not None else chosen_cnt
            participants = max(0, int(participants_base or 0))
            not_passed_cnt = max(0, int(_to_int_or_none(subject.get("not_passed_cnt")) or 0))
            high_80_99_cnt = max(0, int(_to_int_or_none(subject.get("high_80_99_cnt")) or 0))
            score_100_cnt = max(0, int(_to_int_or_none(subject.get("score_100_cnt")) or 0))
            avg_score = _to_float_or_none(subject.get("avg_score"))
            min_passing_score = _to_float_or_none(subject.get("min_passing_score"))

            bucket = by_subject.setdefault(
                sid,
                {
                    "subject_name": subject_name,
                    "min_passing_score": min_passing_score,
                    "year_stats": {},
                },
            )

            if not bucket.get("subject_name"):
                bucket["subject_name"] = subject_name
            if bucket.get("min_passing_score") is None and min_passing_score is not None:
                bucket["min_passing_score"] = min_passing_score

            year_stats = bucket["year_stats"]
            if not isinstance(year_stats, dict):
                year_stats = {}
                bucket["year_stats"] = year_stats
            yb = year_stats.setdefault(
                year,
                {
                    "participants_total": 0,
                    "not_passed_total": 0,
                    "high_80_99_total": 0,
                    "score_100_total": 0,
                    "score_weighted_sum": 0.0,
                    "score_weight": 0.0,
                    "score_plain_sum": 0.0,
                    "score_plain_cnt": 0,
                },
            )

            yb["participants_total"] = int(yb["participants_total"]) + participants
            yb["not_passed_total"] = int(yb["not_passed_total"]) + not_passed_cnt
            yb["high_80_99_total"] = int(yb["high_80_99_total"]) + high_80_99_cnt
            yb["score_100_total"] = int(yb["score_100_total"]) + score_100_cnt

            if avg_score is not None:
                if participants > 0:
                    yb["score_weighted_sum"] = float(yb["score_weighted_sum"]) + (avg_score * participants)
                    yb["score_weight"] = float(yb["score_weight"]) + participants
                else:
                    yb["score_plain_sum"] = float(yb["score_plain_sum"]) + avg_score
                    yb["score_plain_cnt"] = int(yb["score_plain_cnt"]) + 1

    heatmap_rows: list[dict[str, object]] = []
    per_subject_year_avg: dict[int, dict[int, float]] = {}

    def _year_avg(y_bucket: object) -> float | None:
        if not isinstance(y_bucket, dict):
            return None
        weight = float(y_bucket.get("score_weight") or 0.0)
        if weight > 0:
            return round(float(y_bucket.get("score_weighted_sum") or 0.0) / weight, 2)
        plain_cnt = int(y_bucket.get("score_plain_cnt") or 0)
        if plain_cnt > 0:
            return round(float(y_bucket.get("score_plain_sum") or 0.0) / plain_cnt, 2)
        return None

    def _mean(values: list[float], digits: int = 2) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), digits)

    for sid, bucket in sorted(by_subject.items(), key=lambda x: str(x[1].get("subject_name") or "")):
        year_stats = bucket.get("year_stats")
        if not isinstance(year_stats, dict):
            continue

        score_values: list[float] = []
        participants_values: list[float] = []
        not_passed_pct_values: list[float] = []
        high_80_99_pct_values: list[float] = []
        score_100_pct_values: list[float] = []
        year_avg_map: dict[int, float] = {}

        for year, yb in sorted(year_stats.items()):
            if not isinstance(yb, dict):
                continue

            participants = int(yb.get("participants_total") or 0)
            not_passed = int(yb.get("not_passed_total") or 0)
            high_80_99 = int(yb.get("high_80_99_total") or 0)
            score_100 = int(yb.get("score_100_total") or 0)

            avg_year_score = _year_avg(yb)
            if avg_year_score is not None:
                score_values.append(avg_year_score)
                year_avg_map[int(year)] = avg_year_score

            if participants > 0:
                participants_values.append(float(participants))
                not_passed_pct_values.append((not_passed * 100.0) / participants)
                high_80_99_pct_values.append((high_80_99 * 100.0) / participants)
                score_100_pct_values.append((score_100 * 100.0) / participants)

        per_subject_year_avg[sid] = year_avg_map

        avg_score = _mean(score_values, digits=2)
        participants_avg = _mean(participants_values, digits=1)
        not_passed_pct = _mean(not_passed_pct_values, digits=2)
        high_80_99_pct = _mean(high_80_99_pct_values, digits=2)
        score_100_pct = _mean(score_100_pct_values, digits=2)
        min_passing_score = _to_float_or_none(bucket.get("min_passing_score"))

        threshold_status = "na"
        if avg_score is not None and min_passing_score is not None:
            if avg_score >= min_passing_score + 5.0:
                threshold_status = "green"
            elif avg_score >= min_passing_score:
                threshold_status = "yellow"
            else:
                threshold_status = "red"

        heatmap_rows.append(
            {
                "subject_name": str(bucket.get("subject_name") or ""),
                "participants_cnt": participants_avg,
                "avg_score": avg_score,
                "not_passed_pct": not_passed_pct,
                "high_80_99_pct": high_80_99_pct,
                "score_100_pct": score_100_pct,
                "min_passing_score": min_passing_score,
                "threshold_status": threshold_status,
                "avg_score_pos": max(0.0, min(100.0, avg_score)) if avg_score is not None else None,
                "min_score_pos": max(0.0, min(100.0, min_passing_score)) if min_passing_score is not None else None,
                "threshold_delta": round(avg_score - min_passing_score, 2)
                if avg_score is not None and min_passing_score is not None
                else None,
            }
        )

    def _bounds(key: str) -> tuple[float, float]:
        values = [float(row[key]) for row in heatmap_rows if row.get(key) is not None]
        if not values:
            return (0.0, 0.0)
        return (min(values), max(values))

    avg_min, avg_max = _bounds("avg_score")
    part_min, part_max = _bounds("participants_cnt")
    np_min, np_max = _bounds("not_passed_pct")
    h80_min, h80_max = _bounds("high_80_99_pct")
    s100_min, s100_max = _bounds("score_100_pct")

    for row in heatmap_rows:
        row["avg_score_heat"] = _heat_color(
            _to_float_or_none(row.get("avg_score")),
            min_value=avg_min,
            max_value=avg_max,
        )
        row["participants_heat"] = _heat_color(
            _to_float_or_none(row.get("participants_cnt")),
            min_value=part_min,
            max_value=part_max,
        )
        row["not_passed_heat"] = _heat_color(
            _to_float_or_none(row.get("not_passed_pct")),
            min_value=np_min,
            max_value=np_max,
            invert=True,
        )
        row["high_80_99_heat"] = _heat_color(
            _to_float_or_none(row.get("high_80_99_pct")),
            min_value=h80_min,
            max_value=h80_max,
        )
        row["score_100_heat"] = _heat_color(
            _to_float_or_none(row.get("score_100_pct")),
            min_value=s100_min,
            max_value=s100_max,
        )

    latest_year = years[-1] if years else None
    prev_year = years[-2] if len(years) > 1 else None
    histogram_rows: list[dict[str, object]] = []

    for sid, bucket in by_subject.items():
        year_avg_map = per_subject_year_avg.get(sid, {})
        latest_score = year_avg_map.get(latest_year) if latest_year is not None else None
        if latest_score is None:
            continue
        prev_score = year_avg_map.get(prev_year) if prev_year is not None else None
        delta = round(latest_score - prev_score, 2) if prev_score is not None else None

        delta_status = "flat"
        if delta is None:
            delta_status = "na"
        elif delta > 0.2:
            delta_status = "up"
        elif delta < -0.2:
            delta_status = "down"

        histogram_rows.append(
            {
                "subject_name": str(bucket.get("subject_name") or ""),
                "latest_score": latest_score,
                "prev_score": prev_score,
                "delta": delta,
                "delta_status": delta_status,
                "bar_width": max(0.0, min(100.0, latest_score)),
            }
        )

    histogram_rows.sort(key=lambda x: (-float(x.get("latest_score") or 0.0), str(x.get("subject_name") or "")))

    max_abs_delta = 0.0
    for row in histogram_rows:
        delta_value = _to_float_or_none(row.get("delta"))
        if delta_value is None:
            continue
        max_abs_delta = max(max_abs_delta, abs(delta_value))

    def _hist_bar_gradient(delta: float | None, max_delta: float) -> str:
        if delta is None:
            return "linear-gradient(90deg, #dde5f2, #c6d3e8)"

        ratio = 0.0
        if max_delta > 0:
            ratio = max(0.0, min(1.0, abs(delta) / max_delta))

        start_light = 90.0 - (ratio * 18.0)
        end_light = 83.0 - (ratio * 28.0)
        if delta >= 0:
            return (
                "linear-gradient(90deg, "
                f"hsl(140 60% {start_light:.1f}%), "
                f"hsl(140 62% {end_light:.1f}%))"
            )
        return (
            "linear-gradient(90deg, "
            f"hsl(8 78% {start_light:.1f}%), "
            f"hsl(8 76% {end_light:.1f}%))"
        )

    for row in histogram_rows:
        row["bar_gradient"] = _hist_bar_gradient(
            _to_float_or_none(row.get("delta")),
            max_abs_delta,
        )

    return {
        "heatmap_period_label": heatmap_period_label,
        "heatmap_rows": heatmap_rows,
        "histogram_rows": histogram_rows,
        "histogram_latest_year": latest_year,
        "histogram_prev_year": prev_year,
    }
