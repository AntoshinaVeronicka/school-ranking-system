# -*- coding: utf-8 -*-
"""
CLI ??? ???????? SQL-???????? ??????? ?????? ??????.

????????? ???????? ???????? ????????, ????????? ?????
? ????????? ???????? ????? ???????? ?? PostgreSQL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from decimal import Decimal

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "dvfu_prof_webapp") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "dvfu_prof_webapp"))

from search_repo import SchoolSearchFilters, fetch_filter_options, fetch_school_card, search_schools


def _json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запросы для раздела 'Поиск школы'")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("options", help="Вывести значения фильтров (регионы, профили, годы, предметы)")

    p_search = sub.add_parser("search", help="Выполнить поиск школ")
    p_search.add_argument("--q", default="", help="Поиск по вхождению в название школы")
    p_search.add_argument("--region-id", type=int, default=None)
    p_search.add_argument("--municipality-id", type=int, default=None)
    p_search.add_argument("--profile-id", type=int, action="append", default=[], help="Можно указывать несколько раз")
    p_search.add_argument("--year", type=int, default=None)
    p_search.add_argument("--kind", default="", help="actual | plan")
    p_search.add_argument("--subject-id", type=int, action="append", default=[])
    p_search.add_argument("--page", type=int, default=1)
    p_search.add_argument("--per-page", type=int, default=20)
    p_search.add_argument("--no-pagination", action="store_true", help="Без LIMIT/OFFSET")

    p_card = sub.add_parser("card", help="Вывести карточку школы")
    p_card.add_argument("--school-id", type=int, required=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.cmd == "options":
        data = fetch_filter_options(region_id=None)
        print(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default))
        return 0

    if args.cmd == "search":
        filters = SchoolSearchFilters(
            q=args.q,
            region_id=args.region_id,
            municipality_id=args.municipality_id,
            profile_ids=tuple(sorted(set(args.profile_id))),
            year=args.year,
            kind=args.kind or None,
            subject_ids=tuple(sorted(set(args.subject_id))),
        )
        rows, total = search_schools(
            filters,
            page=max(1, args.page),
            per_page=max(1, args.per_page),
            apply_pagination=not args.no_pagination,
        )
        print(f"Найдено: {total}")
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=_json_default))
        return 0

    if args.cmd == "card":
        data = fetch_school_card(args.school_id)
        if data is None:
            print(f"Школа не найдена: school_id={args.school_id}")
            return 2
        print(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
