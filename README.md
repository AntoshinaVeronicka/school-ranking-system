# School Ranking System

Репозиторий с SQL-схемой и Python-скриптами для загрузки справочников и данных в PostgreSQL (схема `edu`).

## Текущая структура проекта

```text
school-ranking-system/
├─ db/
│  ├─ schema.sql
│  ├─ edu.png
│  └─ view_data/
├─ load/
│  ├─ load_common.py
│  ├─ load_schools_to_db.py
│  ├─ load_ege_to_db.py
│  ├─ load_programs_requirements.py
│  └─ requirements.txt
├─ .env
├─ docker-compose.yml
└─ README.md
```

## Назначение скриптов

### `load/load_common.py`
Общие функции для всех загрузчиков:
- чтение `.env` и сбор DB-конфига (`get_db_config`);
- выбор Excel-файла с рабочего стола (`pick_file_dialog_desktop`);
- разрешение путей (`resolve_user_path`);
- нормализация текста и ключей;
- общие DB-хелперы (`table_is_empty`, `fetch_map`);
- общие функции для работы с листами Excel (`resolve_excel_sheet_name`).

### `load/load_schools_to_db.py`
Загружает школы и профили:
- `edu.region`
- `edu.municipality`
- `edu.school`
- `edu.school_profile`
- `edu.school_profile_link`

### `load/load_ege_to_db.py`
Загружает статистику ЕГЭ (режимы `plan`/`actual`):
- `edu.ege_school_year`
- `edu.ege_school_subject_stat`

### `load/load_programs_requirements.py`
Загружает направления и требования к ЕГЭ:
- `edu.institute` (сидирование/добавление)
- `edu.ege_subject` (сидирование/добавление)
- `edu.study_program`
- `edu.program_ege_requirement`

## Подготовка окружения

### 1. Python-зависимости

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r load/requirements.txt
```

### 2. Переменные окружения (`.env`)

Скрипты читают подключение к БД из `.env`:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=school_ranking
POSTGRES_USER=app
POSTGRES_PASSWORD=app
```

`POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` обязательны.

### 3. База данных

Можно поднять Postgres через Docker:

```bash
docker compose up -d
```

Схема будет инициализирована из `db/schema.sql`.

## Запуск загрузчиков

Все команды из корня проекта.

### Школы и профили

```bash
python load/load_schools_to_db.py --pick
python load/load_schools_to_db.py "C:\path\file.xlsx" --sheet 0 --dry-run
```

### ЕГЭ

```bash
python load/load_ege_to_db.py --file "C:\path\ege.xlsx" --sheet 2024 --kind plan --year 2024 --region "Хабаровский край" --dry-run
python load/load_ege_to_db.py --file "C:\path\ege.xlsx" --sheet 2024 --kind actual --year 2024 --region "Хабаровский край"
```

### Направления и требования ЕГЭ

```bash
python load/load_programs_requirements.py --pick --dry-run
python load/load_programs_requirements.py "C:\path\Файл.xlsx"
```

Опция:
- `--create-missing-subjects` — добавлять отсутствующие предметы в `edu.ege_subject` с `min_passing_score = NULL`.

## Примечания

- Скрипты используют `INSERT ... ON CONFLICT` и безопасны для повторного запуска.
- `load_programs_requirements.py` поддерживает блоки с `или` во второй колонке ВИ и корректно пишет их как `role='choice'`.
- Нормализация названий школ и общие функции теперь централизованы в `load/load_common.py`.
