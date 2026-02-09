# school-ranking-system

## Что это

`school-ranking-system` — учебный веб-проект для работы с данными школ:

- импорт данных из Excel (ЕГЭ, справочники, прием, профориентация);
- поиск школ с фильтрами, карточкой школы и экспортом;
- расчет рейтинга школ с фильтрацией и экспортом;
- раздел отчетности: формирование отчетов, архив выгрузок, история расчетов.

Технологический стек:

- FastAPI + Jinja2 (веб-интерфейс);
- PostgreSQL (предметные данные и аналитика);
- SQLite (`dvfu_prof_webapp/app.db`) только для веб-части (пользователи, сессии, журнал загрузок).

## Что реализовано сейчас

- Авторизация (`/login`) и главное меню.
- Раздел **Данные и загрузки**:
  - импорт ЕГЭ с выбором файла/листа/года/типа (`plan`/`actual`);
  - импорт справочников (регион/муниципалитет/школа/профиль, программы/требования, минимальные баллы ЕГЭ);
  - формы для импорта приема и профориентации.
- Раздел **Поиск школ**:
  - регистронезависимый поиск по вхождению в названии;
  - фильтры (регион, муниципалитет, профили, год, plan/fact, предметы);
  - пагинация, экспорт результатов в Excel;
  - карточка школы + экспорт карточки в Excel.
- Раздел **Подбор и рейтинг**:
  - фильтры, аналогичные поиску;
  - мультивыбор институтов/направлений/предметов;
  - расчет итогового рейтинга с весами;
  - фильтр по порогам ЕГЭ и соответствию программам;
  - экспорт рейтинга в Excel.
- Раздел **Отчетность**:
  - генерация отчетов по выбранному run;
  - типы отчета: `standard`, `detailed`;
  - архив отчетов и история расчетов;
  - экспорт run/report.

## Структура проекта

```text
school-ranking-system/
├─ dvfu_prof_webapp/                 # FastAPI приложение
│  ├─ main.py                        # Точка входа ASGI
│  ├─ routes.py                      # HTTP-роуты
│  ├─ services.py                    # Сервисные функции (импорт/рендер/auth)
│  ├─ search_repo.py                 # SQL логика поиска
│  ├─ rating_repo.py                 # SQL логика рейтинга
│  ├─ analytics_repo.py              # Сохранение/чтение расчетов и отчетов
│  ├─ db.py                          # SQLite модели для веб-части
│  ├─ config.py                      # Конфигурация приложения
│  ├─ templates/                     # HTML шаблоны
│  ├─ static/                        # CSS/статические ресурсы
│  └─ requirements.txt               # Python-зависимости веб-приложения
├─ load/                             # Скрипты загрузки и утилиты БД
│  ├─ load_ege_to_db.py
│  ├─ load_schools_to_db.py
│  ├─ load_programs_requirements.py
│  ├─ delete_region_cascade.py
│  └─ setup_search_school_optimization.py
├─ db/
│  ├─ schema.sql                     # Актуальная схема PostgreSQL
│  └─ sql/                           # Отдельные SQL-миграции/оптимизации
├─ mermaid/                          # Диаграммы переходов и сценариев
├─ docker-compose.yml                # PostgreSQL в Docker
├─ .env                              # Локальные переменные окружения
└─ README.md
```

## Локальный запуск

### 1. Требования

- Python 3.10+
- Docker Desktop (или локальный PostgreSQL 14+)
- PowerShell (ниже команды под Windows)

### 2. Установка зависимостей

```powershell
cd C:\Users\Veronika\Desktop\school-ranking-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r dvfu_prof_webapp\requirements.txt
```

### 3. Настройка `.env`

Создайте/проверьте файл `.env` в корне проекта:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=school_ranking
POSTGRES_USER=app
POSTGRES_PASSWORD=app

SECRET_KEY=replace-with-strong-secret
DEFAULT_ADMIN_LOGIN=admin
DEFAULT_ADMIN_PASSWORD=admin
```

Обязательные переменные: `SECRET_KEY`, `DEFAULT_ADMIN_PASSWORD`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`.

### 4. Поднять PostgreSQL

```powershell
docker compose up -d
```

Если база новая, схема загрузится автоматически из `db/schema.sql` при первом старте контейнера.

### 5. Запуск веб-приложения

Запускайте из корня проекта:

```powershell
python -m uvicorn dvfu_prof_webapp.main:app --reload --host 127.0.0.1 --port 8000
```

Открыть в браузере:

- `http://127.0.0.1:8000/login`

### 6. Вход

- Логин: `DEFAULT_ADMIN_LOGIN` (по умолчанию `admin`)
- Пароль: `DEFAULT_ADMIN_PASSWORD` из `.env`

## Импорт данных (CLI)

### ЕГЭ

```powershell
python load\load_ege_to_db.py --file "C:\path\region.xlsx" --sheet 2024 --kind actual --year 2024 --region "Приморский край"
```

Dry-run:

```powershell
python load\load_ege_to_db.py --file "C:\path\region.xlsx" --sheet 2024 --kind plan --year 2024 --dry-run
```

### Справочники (регион/муниципалитет/школа/профиль)

```powershell
python load\load_schools_to_db.py "C:\path\region.xlsx" --sheet 2024
```

С выбором файла через диалог:

```powershell
python load\load_schools_to_db.py --pick
```

### Направления и требования по ВИ

```powershell
python load\load_programs_requirements.py "C:\path\programs.xlsx" --sheet 0
```

### Оптимизация поиска (pg_trgm + индексы)

```powershell
python load\setup_search_school_optimization.py
```

## Очистка данных региона

Скрипт `load/delete_region_cascade.py` удаляет регион и все связанные данные (муниципалитеты, школы, ЕГЭ, аналитика и связанные сущности), с поддержкой dry-run.

Dry-run:

```powershell
python load\delete_region_cascade.py --region "Забайкальский край"
```

Применить удаление + пересчитать identity/sequence:

```powershell
python load\delete_region_cascade.py --region "Забайкальский край" --apply
```

## Экспорты

- Поиск школ: `/search/export` (текущая выдача).
- Карточка школы: `/search/school/{school_id}/export`.
- Рейтинг: `/rating/export`.
- Run отчетности: `/reports/run/{run_id}/export`.
- Отдельный отчет из архива: `/reports/report/{report_id}/export`.

## Полезные замечания

- В репозиторий не должны попадать runtime-файлы (`dvfu_prof_webapp/app.db`, `dvfu_prof_webapp/uploads/*`) — они уже в `.gitignore`.
- Если в PowerShell видите «кракозябры», переключите кодировку:

```powershell
chcp 65001
$env:PYTHONIOENCODING="utf-8"
```

- Если `uvicorn` не стартует на `8000` с `WinError 10013`, используйте другой порт, например `--port 8010`.
