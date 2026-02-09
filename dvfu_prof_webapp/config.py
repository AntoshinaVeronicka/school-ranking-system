from __future__ import annotations

from pathlib import Path

APP_NAME = "DVFU - Профориентация и анализ школ"
SECRET_KEY = "CHANGE_ME"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATABASE_URL = f"sqlite:///{(BASE_DIR / 'app.db').as_posix()}"
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

LOAD_EGE_SCRIPT = PROJECT_ROOT / "load" / "load_ege_to_db.py"
LOAD_SCHOOLS_SCRIPT = PROJECT_ROOT / "load" / "load_schools_to_db.py"
LOAD_PROGRAMS_SCRIPT = PROJECT_ROOT / "load" / "load_programs_requirements.py"

DEFAULT_EGE_SUBJECT_SCORES = [
    {"subject_id": 1, "name": "Русский язык", "min_passing_score": 40},
    {"subject_id": 2, "name": "Математика", "min_passing_score": 40},
    {"subject_id": 3, "name": "Физика", "min_passing_score": 41},
    {"subject_id": 4, "name": "Обществознание", "min_passing_score": 45},
    {"subject_id": 5, "name": "История", "min_passing_score": 40},
    {"subject_id": 6, "name": "ИКТ", "min_passing_score": 46},
    {"subject_id": 7, "name": "Иностранный язык", "min_passing_score": 40},
    {"subject_id": 8, "name": "Литература", "min_passing_score": 40},
    {"subject_id": 9, "name": "Биология", "min_passing_score": 40},
    {"subject_id": 10, "name": "География", "min_passing_score": 40},
    {"subject_id": 11, "name": "Химия", "min_passing_score": 40},
]

DEFAULT_ADMIN_LOGIN = "admin"
DEFAULT_ADMIN_PASSWORD = "admin"
