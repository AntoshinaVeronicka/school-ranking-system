# School Ranking System

Веб-прототип для анализа школ, загрузки данных (ЕГЭ/справочники/прием/профориентация), поиска школ, карточки школы и рейтинга по фильтрам.

## Что внутри

- FastAPI-приложение с интерфейсом: `dvfu_prof_webapp/`
- SQL-схема PostgreSQL: `db/schema.sql`
- Скрипты загрузки данных в PostgreSQL: `load/`
- Mermaid-диаграммы сценариев: `mermaid/`

## Основные возможности

- Авторизация и главное меню.
- Раздел **Данные и загрузки**:
- Импорт ЕГЭ из Excel (plan/actual, sheet/year, dry-run, лог выполнения в UI).
- Импорт справочников:
- Регион / муниципалитет / школа / профиль.
- Направления и требования по ВИ.
- Обновление минимальных проходных баллов по предметам ЕГЭ.
- Заглушки форм для импорта приема и профориентации (с загрузкой файла и выводом диалога).
- Раздел **Поиск школ**:
- Фильтры (регион, муниципалитет, профили, год, план/факт, предметы).
- Пагинация, карточка школы, экспорт результатов в Excel.
- Карточка школы с экспортом полной информации в Excel.
- Раздел **Рейтинг**:
- Фильтры, включая институты и направления.
- Расчет итогового рейтинга школ.
- Экспорт рейтинга в Excel.

## Важная архитектурная деталь

- Аутентификация/сессии и журнал импортов хранятся в локальной SQLite базе `dvfu_prof_webapp/app.db`.
- Предметные данные (школы, ЕГЭ, направления и т.д.) берутся из PostgreSQL схемы `edu`.
- Поэтому для полноценных разделов Поиск/Рейтинг/Загрузки нужен PostgreSQL и корректный `.env`.

## Требования

- Python 3.10+
- PostgreSQL 14+ (или Docker)
- Windows PowerShell (ниже примеры для Windows)

## Быстрый старт

1. Перейдите в корень проекта:

```powershell
cd C:\Users\Veronika\Desktop\school-ranking-system
```

2. Создайте и активируйте виртуальное окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Установите зависимости (достаточно веб-приложения, они покрывают и загрузчики):

```powershell
pip install -r dvfu_prof_webapp\requirements.txt
```

4. Создайте `.env` в корне проекта:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=school_ranking
POSTGRES_USER=app
POSTGRES_PASSWORD=app
```

5. Поднимите PostgreSQL через Docker (опционально, если нет локальной БД):

```powershell
docker compose up -d
```

6. Запустите сайт:

```powershell
cd dvfu_prof_webapp
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

7. Откройте в браузере:

`http://127.0.0.1:8000/login`

## Демо-доступ

- Логин: `admin`
- Пароль: `admin`

Задаются в `dvfu_prof_webapp/config.py` (`DEFAULT_ADMIN_LOGIN`, `DEFAULT_ADMIN_PASSWORD`).

## Запуск из корня (альтернатива)

Если не хотите переходить в `dvfu_prof_webapp`:

```powershell
python -m uvicorn dvfu_prof_webapp.main:app --reload --host 127.0.0.1 --port 8000
```

## Загрузка данных через UI

После входа:

- **Данные и загрузки -> Импорт ЕГЭ**
- **Данные и загрузки -> Справочники**
- **Данные и загрузки -> Импорт приема** (заглушка)
- **Данные и загрузки -> Импорт профориентации** (заглушка)

Во всех формах поддерживается загрузка файла и отображение диалога/лога выполнения.

## CLI-скрипты загрузки

Все команды запускать из корня проекта с активным `.venv`.

### 1) Школы и профили

```powershell
python load\load_schools_to_db.py --pick
python load\load_schools_to_db.py "C:\path\region.xlsx" --sheet 2024
python load\load_schools_to_db.py "C:\path\region.xlsx" --sheet 2024 --dry-run
```

### 2) ЕГЭ

```powershell
python load\load_ege_to_db.py --file "C:\path\ege.xlsx" --sheet 2024 --kind actual --year 2024 --region "Еврейская автономная область"
python load\load_ege_to_db.py --file "C:\path\ege.xlsx" --sheet 2024 --kind plan --year 2024 --region "Еврейская автономная область" --dry-run
```

### 3) Направления и требования по ВИ

```powershell
python load\load_programs_requirements.py --pick
python load\load_programs_requirements.py "C:\path\programs.xlsx" --sheet 0
python load\load_programs_requirements.py "C:\path\programs.xlsx" --sheet 0 --create-missing-subjects
```

### 4) Каскадное удаление региона

```powershell
python load\delete_region_cascade.py --region "Еврейская автономная область"
python load\delete_region_cascade.py --region "Еврейская автономная область" --apply
```

По умолчанию скрипт работает как dry-run.

## SQL-оптимизации и защита от дублей

### Оптимизация поиска школ (pg_trgm + GIN)

```powershell
python load\setup_search_school_optimization.py
```

SQL-файл: `db/sql/search_school_optimization.sql`

### Защита от дублей муниципалитетов по регистру/формату

SQL-файл: `db/sql/municipality_case_insensitive_unique.sql`

Что делает:

- Показывает дубликаты муниципалитетов после нормализации.
- Создает уникальный индекс по нормализованному имени муниципалитета внутри региона.

## Полезные пути

- Веб-приложение: `dvfu_prof_webapp/main.py`, `dvfu_prof_webapp/routes.py`
- Репозитории запросов: `dvfu_prof_webapp/search_repo.py`, `dvfu_prof_webapp/rating_repo.py`
- Шаблоны: `dvfu_prof_webapp/templates/`
- Стили: `dvfu_prof_webapp/static/app.css`
- Загрузчики: `load/*.py`
- Схема БД: `db/schema.sql`

## Частые проблемы

### 1) `Could not import module "main"`

Причина: запуск `uvicorn` не из той папки.

Решение:

- либо `cd dvfu_prof_webapp` и `python -m uvicorn main:app --reload`
- либо из корня `python -m uvicorn dvfu_prof_webapp.main:app --reload`

### 2) `WinError 10013` при запуске Uvicorn

Порт занят или заблокирован.

Решение:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

### 3) Ошибка установки `psycopg2-binary` с `pg_config executable not found`

Используйте зависимости из `dvfu_prof_webapp/requirements.txt` (там версия `psycopg2-binary==2.9.11` с готовыми wheel для Windows/Python 3.13).

### 4) Поиск/рейтинг пустые или падают по подключению

Проверьте `.env` и доступность PostgreSQL.

## Текущее состояние модулей

- Поиск школ: реализован.
- Карточка школы + экспорт: реализован.
- Рейтинг + экспорт: реализован.
- Импорт ЕГЭ/справочников: реализован.
- Импорт приема и профориентации: форма и загрузка файла есть, предметная бизнес-обработка пока как прототип.

