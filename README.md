# school-ranking-system

Веб-приложение для загрузки, поиска и рейтинга школ на данных ЕГЭ и связанных справочниках.

## Что есть в проекте
- Загрузка данных из Excel (ЕГЭ, справочники, прием, профориентация).
- Поиск школ с фильтрами, карточкой школы и экспортом в Excel.
- Подбор и расчет рейтинга школ с весами, фильтрами и экспортом.
- Сохранение результатов в аналитику только по кнопкам:
  - `/search/save`
  - `/rating/profile/save`
- Раздел отчетности: генерация, архив, экспорт.

## Минимальная структура
```text
school-ranking-system/
├── dvfu_prof_webapp/               # FastAPI-приложение
│   ├── main.py                     # Точка входа ASGI
│   ├── routes.py                   # HTTP-роуты
│   ├── services.py                 # Общие сервисы/рендер/запуск скриптов
│   ├── search_repo.py              # SQL-логика поиска школ
│   ├── rating_repo.py              # SQL-логика рейтинга
│   ├── analytics_repo.py           # Сохранение/чтение аналитики и отчетов
│   ├── db.py                       # SQLite-модели (пользователи/служебные записи)
│   ├── config.py                   # Конфигурация приложения
│   ├── templates/                  # Jinja2-шаблоны
│   └── static/                     # CSS и статические файлы
├── load/                           # CLI-скрипты загрузки и обслуживания БД
│   ├── load_ege_to_db.py
│   ├── load_schools_to_db.py
│   ├── load_programs_requirements.py
│   ├── delete_region_cascade.py
│   └── setup_search_school_optimization.py
├── db/
│   ├── schema.sql                  # Базовая схема PostgreSQL
│   └── sql/                        # Дополнительные SQL-скрипты
├── docker-compose.yml              # Подъем PostgreSQL
└── README.md
```

## Основные технологии
- FastAPI + Jinja2
- PostgreSQL (основные предметные данные)
- SQLite (`dvfu_prof_webapp/app.db`) для пользователей и служебных записей веб-части
- Pandas/OpenPyXL для импорта и экспорта Excel

## Быстрый старт

### 1) Требования
- Python 3.10+
- Docker (или локальный PostgreSQL)
- Windows PowerShell (команды ниже под него)

### 2) Установка зависимостей
```powershell
cd C:\Users\Veronika\Desktop\school-ranking-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r dvfu_prof_webapp\requirements.txt
```

### 3) Настройка `.env`
Создайте/проверьте файл `.env` в корне проекта:

```env
SECRET_KEY=replace-with-strong-secret
DEFAULT_ADMIN_LOGIN=admin
DEFAULT_ADMIN_PASSWORD=admin

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=school_ranking
POSTGRES_USER=app
POSTGRES_PASSWORD=app
```

Важно:
- `SECRET_KEY` и `DEFAULT_ADMIN_PASSWORD` обязательны для веб-приложения.
- `POSTGRES_*` обязательны для поиска/рейтинга/отчетности и загрузчиков.

### 4) Поднять PostgreSQL
```powershell
docker compose up -d
```

### 5) Запустить веб-приложение
```powershell
python -m uvicorn dvfu_prof_webapp.main:app --reload --host 127.0.0.1 --port 8000
```

Открыть: `http://127.0.0.1:8000/login`

## Ключевые разделы и маршруты
- Аутентификация: `/login`, `/logout`
- Данные и загрузки: `/data/*`
- Поиск школ: `/search`, `/search/export`, `/search/school/{school_id}`
- Рейтинг: `/rating/profile`, `/rating/export`
- Отчетность: `/reports`, `/reports/generate`, `/reports/archive`, `/reports/calc-history`

## CLI-скрипты
Примеры:

```powershell
# ЕГЭ
python load\load_ege_to_db.py --file "C:\path\region.xlsx" --sheet 2024 --kind actual --year 2024 --region "Приморский край"

# Справочники школ
python load\load_schools_to_db.py "C:\path\region.xlsx" --sheet 2024

# Программы и требования
python load\load_programs_requirements.py "C:\path\programs.xlsx" --sheet 0

# Индексы для оптимизации поиска
python load\setup_search_school_optimization.py
```

## Полезные SQL-скрипты
- `db/sql/search_school_optimization.sql`
- `db/sql/municipality_case_insensitive_unique.sql`
- `db/sql/clear_analytics_data.sql`

## Примечания
- Неавторизованный доступ к защищенным страницам обрабатывается централизованно:
  - HTML-запросы -> редирект на `/login`
  - API-запросы -> `401`
- Runtime-файлы уже исключены из git через `.gitignore` (`.env`, `.venv`, `uploads`, `app.db`).
- Рекомендуется использовать один виртуальный env в корне проекта (`.venv`) и не создавать вложенные env внутри `dvfu_prof_webapp/`.
