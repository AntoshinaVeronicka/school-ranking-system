# Анализ школ ДФО (school-ranking-system)

Веб-приложение для загрузки, поиска и рейтинга школ на данных ЕГЭ, справочниках направлений и порогах по предметам.

## Возможности
- Импорт данных ЕГЭ и справочников из Excel.
- Поиск школ с фильтрами и карточкой школы.
- Расчет рейтинга школ с настраиваемыми весами.
- Предметная аналитика и экспорт результатов в Excel.
- Сохранение запусков расчетов и отчетность.

## Структура проекта
```text
school-ranking-system/
├── dvfu_prof_webapp/                 # FastAPI-приложение
│   ├── main.py                       # Точка входа ASGI
│   ├── routes.py                     # HTTP-маршруты и страницы
│   ├── services.py                   # Сервисы, валидация, запуск скриптов
│   ├── search_repo.py                # SQL-логика поиска школ
│   ├── rating_repo.py                # SQL-логика расчета рейтинга
│   ├── analytics_repo.py             # История расчетов и отчеты
│   ├── filter_options_repo.py        # Справочники фильтров
│   ├── subject_analytics_service.py  # Предметная аналитика
│   ├── db.py                         # SQLite-модели (пользователи/служебные записи)
│   ├── config.py                     # Конфигурация
│   ├── templates/                    # HTML-шаблоны Jinja2
│   ├── static/                       # CSS/JS/статические файлы
│   └── requirements.txt              # Совместимость: ссылается на ../requirements.txt
├── load/                             # CLI-скрипты импорта и обслуживания PostgreSQL
│   ├── load_ege_to_db.py
│   ├── load_schools_to_db.py
│   ├── load_programs_requirements.py
│   ├── merge_duplicate_schools.py
│   └── setup_search_school_optimization.py
├── db/
│   ├── schema.sql                    # Базовая схема PostgreSQL
│   └── sql/                          # Дополнительные SQL-скрипты
├── docker-compose.yml                # PostgreSQL в Docker
├── requirements.txt                  # Единый список зависимостей проекта
└── README.md
```

## Технологии
- FastAPI + Jinja2
- PostgreSQL (основные данные)
- SQLite (`dvfu_prof_webapp/app.db`) для пользователей и служебных сущностей
- Pandas + OpenPyXL для импорта/экспорта Excel

## Запуск локально

### 1. Требования
- Python 3.10+
- Docker Desktop (или локальный PostgreSQL)
- PowerShell (команды ниже)

### 2. Установка зависимостей
```powershell
cd C:\Users\Veronika\Desktop\school-ranking-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Используйте один общий `.venv` в корне проекта. Вложенный `dvfu_prof_webapp/.venv` создавать не нужно.

### 3. Настройка `.env`
Создайте файл `.env` в корне проекта:

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

### 4. Поднять PostgreSQL
```powershell
docker compose up -d
```

### 5. Запуск веб-приложения
```powershell
python -m uvicorn dvfu_prof_webapp.main:app --reload --host 127.0.0.1 --port 8000
```

Откройте: `http://127.0.0.1:8000/login`


## Полезные запуски скриптов (удаление и объединение)

Перед массовыми изменениями рекомендуется сделать резервную копию:
```powershell
pg_dump -h localhost -p 5432 -U app -d school_ranking -Fc -f backup_before_cleanup.dump
```

### Объединение дублей школ (`merge_duplicate_schools.py`)
Проверка без записи в БД (dry-run):
```powershell
python load\merge_duplicate_schools.py --sample-limit 30
```

Проверка по одному региону и ограниченному числу групп:
```powershell
python load\merge_duplicate_schools.py --region "Приморский край" --limit-groups 50 --sample-limit 30
```

Применить объединение только по региону:
```powershell
python load\merge_duplicate_schools.py --region "Приморский край" --apply
```

Применить объединение по всем регионам:
```powershell
python load\merge_duplicate_schools.py --apply
```

### Каскадное удаление региона (`delete_region_cascade.py`)
Проверка объема удаления (dry-run):
```powershell
python load\delete_region_cascade.py --region "Забайкальский край"
```

Применить удаление:
```powershell
python load\delete_region_cascade.py --region "Забайкальский край" --apply
```

Применить удаление без пересчета sequence/identity:
```powershell
python load\delete_region_cascade.py --region "Забайкальский край" --apply --no-reset-identities
```

### Очистка только аналитики (без удаления школ)
```powershell
psql -h localhost -p 5432 -U app -d school_ranking -f db\sql\clear_analytics_data.sql
```

## Основные маршруты
- Аутентификация: `/login`, `/logout`
- Данные и загрузка: `/data/*`
- Поиск школ: `/search`, `/search/export`, `/search/school/{school_id}`
- Рейтинг: `/rating/profile`, `/rating/export`
- Отчеты: `/reports`, `/reports/generate`, `/reports/archive`, `/reports/calc-history`

## Результаты
В этом разделе фиксируются итоговые экраны и краткие выводы по ним.

### Что описывать для каждого скриншота
- Контекст: дата, период данных, кто запускал расчет.
- Входные параметры: регион, муниципалитет, год, тип данных, фильтры по институтам/направлениям/предметам.
- Логика расчета: какие метрики участвовали и какие веса использовались.
- Результат: ключевые числа и 2–3 наблюдения по данным.
- Ограничения: что могло повлиять на интерпретацию (неполные данные, фильтры, нормализация).

### 1. Рейтинг школ (`/rating/profile`)
![Рейтинг школ](docs/results/rating.png)

Рекомендуется описать:
- Цель расчета: зачем запускался рейтинг в этой выборке.
- Параметры и веса: `w_graduates`, `w_avg_score`, `w_match_share`, `w_threshold_share`.
- Какие фильтры были включены: регион/муниципалитет, институт, направление, предметы ЕГЭ.
- Табличный итог: топ-10/топ-20, лидер, разрыв между 1 и 2 местом.
- График `Состав рейтинга`: какой компонент дал основной вклад у лидеров.
- Пороги: как повлияла проверка минимальных баллов на итоговые позиции.

### 2. Карточка школы (`/search/school/{school_id}`)
![Карточка школы](docs/results/school-card.png)

Рекомендуется описать:
- Идентификация школы: регион, муниципалитет, профиль, активность записи.
- Динамика выпускников и среднего балла по фактическим годам.
- Предметные зоны силы/риска: где высокий средний балл и где просадка.
- Показатели `% не преодолели`, `% 80+`, `% 100` и их интерпретация.
- Связь со сценарием набора: какие предметы усиливают потенциал абитуриентов.

### 3. Предметная аналитика (`блок аналитики ЕГЭ`)
![Предметная аналитика](docs/results/subject-analytics.png)

Рекомендуется описать:
- Период агрегации: по каким годам считалось среднее (факт).
- Таблица по предметам: `Средний балл`, `Участники`, `% не преодолели`, `% 80+`, `% 100`.
- Сравнение по предметам: рост/падение к предыдущему году (цвет и величина изменения).
- Массовость против качества: где много участников при хорошем балле, а где наоборот.
- Практический вывод: какие предметы приоритетно усиливать в работе со школами.

### Шаблон подписи к блоку результата
Можно использовать один формат подписи под каждым скриншотом:

- Период и срез: `...`
- Фильтры: `...`
- Веса формулы: `...`
- Ключевой результат: `...`
- Интерпретация: `...`
- Ограничения данных: `...`
