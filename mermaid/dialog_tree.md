# Диаграмма переходов: вход, восстановление доступа и главное меню

```mermaid
---
config:
  layout: dagre
  flowchart:
    nodeSpacing: 15
    rankSpacing: 25
---

flowchart LR
  LoginForm["0. Вход (/login)"]
  RecoveryForm["0.1 Восстановление доступа (/recovery)"]
  MainForm["1. Главное меню (/)"]
  LogoutAction["1.1 Выход (/logout)"]

  DataMgmtForm["2. Данные и загрузки (/data)"]
  SearchForm["3. Поиск школ (/search)"]
  RatingForm["4. Подбор и рейтинг школ (/rating/profile)"]
  SettingsForm["5. Настройки анализа (/settings)"]
  ReportsHome["6. Отчетность (/reports)"]
  AdminPanel["7. Администрирование (/admin, только admin)"]

  LoginForm -->|POST /login| MainForm
  LoginForm -->|Восстановление| RecoveryForm
  RecoveryForm -->|Назад| LoginForm
  MainForm -->|Выход| LogoutAction
  LogoutAction -->|Редирект 303| LoginForm

  MainForm -->|Раздел| DataMgmtForm
  MainForm -->|Раздел| SearchForm
  MainForm -->|Раздел| RatingForm
  MainForm -->|Раздел| SettingsForm
  MainForm -->|Раздел| ReportsHome
  MainForm -->|Раздел (если admin)| AdminPanel

  DataMgmtForm -.->|В меню| MainForm
  SearchForm -.->|В меню| MainForm
  RatingForm -.->|В меню| MainForm
  SettingsForm -.->|В меню| MainForm
  ReportsHome -.->|В меню| MainForm
  AdminPanel -.->|В меню| MainForm

  classDef auth fill:#f2f2f2,stroke:#333,stroke-width:1px;
  classDef main fill:#ffffff,stroke:#333,stroke-width:2px;
  classDef data fill:#eef7ff,stroke:#2b6cb0,stroke-width:1px;
  classDef search fill:#f5f3ff,stroke:#5a67d8,stroke-width:1px;
  classDef rating fill:#fff7ed,stroke:#c05621,stroke-width:1px;
  classDef set fill:#f7fafc,stroke:#4a5568,stroke-width:1px;
  classDef rep fill:#f1fff3,stroke:#2f855a,stroke-width:1px;
  classDef adm fill:#fff1f2,stroke:#b83280,stroke-width:1px;

  class LoginForm,RecoveryForm,LogoutAction auth;
  class MainForm main;
  class DataMgmtForm data;
  class SearchForm search;
  class RatingForm rating;
  class SettingsForm set;
  class ReportsHome rep;
  class AdminPanel adm;
```

---

# Диаграмма переходов в разделе «Данные и загрузки»

```mermaid
---
config:
  layout: dagre
  flowchart:
    nodeSpacing: 15
    rankSpacing: 25
---

flowchart LR
  MainForm["1. Главное меню"]

  subgraph DATA["2. Данные и загрузки"]
    direction TB
    DataMgmtForm["2. Управление данными (/data)"]
    ImportEGE["2.1 Импорт ЕГЭ (/data/ege)"]
    ImportAdmissions["2.2 Импорт приема (/data/admissions)"]
    ImportEvents["2.3 Импорт профориентации (/data/events)"]
    Directories["2.4 Справочники (/data/directories)"]
    DirLoadAction["2.4.a Загрузка справочников (POST /data/directories/load)"]
    MinScoresAction["2.4.b Обновление мин. баллов ЕГЭ (POST /data/directories/min-scores)"]

    DataMgmtForm -->|ЕГЭ| ImportEGE
    DataMgmtForm -->|Прием| ImportAdmissions
    DataMgmtForm -->|Профориентация| ImportEvents
    DataMgmtForm -->|Справочники| Directories

    ImportEGE -->|Запуск загрузки| ImportEGE
    ImportAdmissions -->|Запуск загрузки| ImportAdmissions
    ImportEvents -->|Запуск загрузки| ImportEvents

    Directories -->|Загрузка Excel| DirLoadAction
    Directories -->|Обновить баллы| MinScoresAction
    DirLoadAction -->|Рендер /data/directories| Directories
    MinScoresAction -->|Рендер /data/directories| Directories

    ImportEGE -->|Назад| DataMgmtForm
    ImportAdmissions -->|Назад| DataMgmtForm
    ImportEvents -->|Назад| DataMgmtForm
    Directories -->|Назад| DataMgmtForm
  end

  MainForm -->|Данные и загрузки| DataMgmtForm
  DATA -.->|В меню| MainForm

  classDef main fill:#ffffff,stroke:#333,stroke-width:2px;
  classDef data fill:#eef7ff,stroke:#2b6cb0,stroke-width:1px;

  class MainForm main;
  class DataMgmtForm,ImportEGE,ImportAdmissions,ImportEvents,Directories,DirLoadAction,MinScoresAction data;
```

---

# Диаграмма переходов в разделе «Поиск школ и карточка школы»

```mermaid
---
config:
  layout: dagre
  flowchart:
    nodeSpacing: 15
    rankSpacing: 25
---

flowchart LR
  MainForm["1. Главное меню"]
  ReportsHome["6. Отчетность (/reports)"]

  subgraph SEARCH["3. Поиск школ и карточка школы"]
    direction TB
    SearchForm["3. Поиск школ (/search)"]
    SaveSearchAction["3.s Сохранить результат (POST /search/save)"]
    SearchExport["3.e Экспорт списка (GET /search/export)"]
    SchoolCard["3.1 Карточка школы (/search/school/:school_id)"]
    SchoolCardExport["3.1.e Экспорт карточки (GET /search/school/:school_id/export)"]
    MunicipalitiesApi["3.api Муниципалитеты (GET /search/municipalities)"]

    SearchForm -->|Фильтры (POST /search)| SearchForm
    SearchForm -->|Сохранить результат| SaveSearchAction
    SaveSearchAction -->|Редирект 303 с save_status| SearchForm
    SearchForm -->|Экспорт Excel| SearchExport
    SearchForm -->|Открыть карточку| SchoolCard
    SearchForm -->|AJAX по региону| MunicipalitiesApi

    SchoolCard -->|Экспорт карточки| SchoolCardExport
    SchoolCard -->|Назад к поиску| SearchForm
    SchoolCard -->|Выгрузка и отчет| ReportsHome
  end

  MainForm -->|Поиск школ| SearchForm
  SEARCH -.->|В меню| MainForm

  classDef main fill:#ffffff,stroke:#333,stroke-width:2px;
  classDef search fill:#f5f3ff,stroke:#5a67d8,stroke-width:1px;
  classDef rep fill:#f1fff3,stroke:#2f855a,stroke-width:1px;

  class MainForm main;
  class SearchForm,SaveSearchAction,SearchExport,SchoolCard,SchoolCardExport,MunicipalitiesApi search;
  class ReportsHome rep;
```

---

# Диаграмма переходов в разделе «Подбор и рейтинг школ»

```mermaid
---
config:
  layout: dagre
  flowchart:
    nodeSpacing: 15
    rankSpacing: 25
---

flowchart LR
  MainForm["1. Главное меню"]
  ReportsHome["6. Отчетность (/reports)"]

  subgraph RATING["4. Подбор и рейтинг школ"]
    direction TB
    RatingProfile["4. Подбор и рейтинг (/rating/profile)"]
    RatingSaveAction["4.s Сохранить рейтинг (POST /rating/profile/save)"]
    RatingExport["4.e Экспорт рейтинга (GET /rating/export)"]
    ProgramsApi["4.api Направления по институтам (GET /rating/programs)"]
    SchoolCardFromRating["3.1 Карточка школы (/search/school/:school_id?src=rating)"]

    RatingProfile -->|Расчет по параметрам| RatingProfile
    RatingProfile -->|Сохранить рейтинг| RatingSaveAction
    RatingSaveAction -->|Редирект 303 с save_status| RatingProfile
    RatingProfile -->|Экспорт Excel| RatingExport
    RatingProfile -->|AJAX: обновить направления| ProgramsApi

    RatingProfile -->|Карточка школы| SchoolCardFromRating
    SchoolCardFromRating -->|Назад (src=rating)| RatingProfile
    SchoolCardFromRating -->|Выгрузка и отчет| ReportsHome
  end

  MainForm -->|Подбор и рейтинг| RatingProfile
  RATING -.->|В меню| MainForm

  classDef main fill:#ffffff,stroke:#333,stroke-width:2px;
  classDef rating fill:#fff7ed,stroke:#c05621,stroke-width:1px;
  classDef search fill:#f5f3ff,stroke:#5a67d8,stroke-width:1px;
  classDef rep fill:#f1fff3,stroke:#2f855a,stroke-width:1px;

  class MainForm main;
  class RatingProfile,RatingSaveAction,RatingExport,ProgramsApi rating;
  class SchoolCardFromRating search;
  class ReportsHome rep;
```

---

# Диаграмма переходов в разделе «Настройки анализа»

```mermaid
---
config:
  layout: dagre
  flowchart:
    nodeSpacing: 15
    rankSpacing: 25
---

flowchart LR
  MainForm["1. Главное меню"]

  subgraph SETTINGS["5. Настройки анализа (заглушки)"]
    direction TB
    SettingsHome["5. Настройки анализа (/settings)"]
    SettingsFilters["5.1 Конструктор фильтров (/settings/filters)"]
    SettingsWeights["5.2 Метрики и веса (/settings/weights)"]
    SettingsProfiles["5.3 Профили расчета (/settings/profiles)"]

    SettingsHome -->|Фильтры| SettingsFilters
    SettingsHome -->|Метрики и веса| SettingsWeights
    SettingsHome -->|Профили| SettingsProfiles

    SettingsFilters -->|Назад| SettingsHome
    SettingsWeights -->|Назад| SettingsHome
    SettingsProfiles -->|Назад| SettingsHome
  end

  MainForm -->|Настройки анализа| SettingsHome
  SETTINGS -.->|В меню| MainForm

  classDef main fill:#ffffff,stroke:#333,stroke-width:2px;
  classDef set fill:#f7fafc,stroke:#4a5568,stroke-width:1px;

  class MainForm main;
  class SettingsHome,SettingsFilters,SettingsWeights,SettingsProfiles set;
```

---

# Диаграмма переходов в разделе «Отчетность»

```mermaid
---
config:
  layout: dagre
  flowchart:
    nodeSpacing: 15
    rankSpacing: 25
---

flowchart LR
  MainForm["1. Главное меню"]
  SearchForm["3. Поиск школ"]
  RatingProfile["4. Подбор и рейтинг школ"]

  subgraph REPORTS["6. Отчетность"]
    direction TB
    ReportsHome["6. Отчетность (/reports)"]
    ReportGenerate["6.1 Выгрузка и отчет (/reports/generate)"]
    ReportGenerateAction["6.1.a Сформировать отчеты (POST /reports/generate)"]
    ReportArchive["6.2 Архив выгрузок (/reports/archive)"]
    CalcHistory["6.3 История расчетов (/reports/calc-history)"]
    RunExport["6.e Экспорт run (GET /reports/run/:run_id/export)"]
    ReportExport["6.e Экспорт отчета (GET /reports/report/:report_id/export)"]

    ReportsHome -->|Выгрузка и отчет| ReportGenerate
    ReportsHome -->|Архив выгрузок| ReportArchive
    ReportsHome -->|История расчетов| CalcHistory

    ReportGenerate -->|Выбрать run / показать| ReportGenerate
    ReportGenerate -->|Сформировать и сохранить| ReportGenerateAction
    ReportGenerateAction -->|Редирект 303| ReportArchive
    ReportGenerate -->|Экспорт run в Excel| RunExport
    ReportGenerate -->|К архиву| ReportArchive
    ReportGenerate -->|Назад| ReportsHome
    ReportGenerate -.->|Если нет run: перейти к сохранению| SearchForm
    ReportGenerate -.->|Если нет run: перейти к сохранению| RatingProfile

    ReportArchive -->|Скачать отчет| ReportExport
    ReportArchive -->|Выгрузка и отчет| ReportGenerate
    ReportArchive -->|История расчетов| CalcHistory
    ReportArchive -->|Назад| ReportsHome

    CalcHistory -->|Excel по run| RunExport
    CalcHistory -->|Без run: перейти к поиску| SearchForm
    CalcHistory -->|Назад| ReportsHome
  end

  MainForm -->|Отчетность| ReportsHome
  REPORTS -.->|В меню| MainForm

  classDef main fill:#ffffff,stroke:#333,stroke-width:2px;
  classDef search fill:#f5f3ff,stroke:#5a67d8,stroke-width:1px;
  classDef rating fill:#fff7ed,stroke:#c05621,stroke-width:1px;
  classDef rep fill:#f1fff3,stroke:#2f855a,stroke-width:1px;

  class MainForm main;
  class SearchForm search;
  class RatingProfile rating;
  class ReportsHome,ReportGenerate,ReportGenerateAction,ReportArchive,CalcHistory,RunExport,ReportExport rep;
```

---

# Диаграмма переходов в разделе «Администрирование»

```mermaid
---
config:
  layout: dagre
  flowchart:
    nodeSpacing: 15
    rankSpacing: 25
---

flowchart LR
  MainForm["1. Главное меню"]
  DataDirectories["2.4 Справочники (/data/directories)"]

  subgraph ADMIN["7. Администрирование (только admin)"]
    direction TB
    AdminPanel["7. Админ-панель (/admin)"]
    Roles["7.1 Роли и права (/admin/roles)"]
    Methodologies["7.2 Методики (/admin/methodologies)"]
    AdminDirectories["7.x /admin/directories (редирект)"]

    AdminPanel -->|Роли и права| Roles
    AdminPanel -->|Методики| Methodologies
    Roles -->|Назад| AdminPanel
    Methodologies -->|Назад| AdminPanel

    AdminDirectories -->|303 -> /data/directories| DataDirectories
    AdminPanel -.->|Прямой URL| AdminDirectories
  end

  MainForm -->|Администрирование| AdminPanel
  ADMIN -.->|В меню| MainForm

  classDef main fill:#ffffff,stroke:#333,stroke-width:2px;
  classDef data fill:#eef7ff,stroke:#2b6cb0,stroke-width:1px;
  classDef adm fill:#fff1f2,stroke:#b83280,stroke-width:1px;

  class MainForm main;
  class DataDirectories data;
  class AdminPanel,Roles,Methodologies,AdminDirectories adm;
```

---

# Диаграмма переходов между экранными формами (актуальная сводная)

```mermaid
---
config:
  layout: dagre
  flowchart:
    nodeSpacing: 15
    rankSpacing: 25
---

flowchart LR
  LoginForm["0. Вход"]
  RecoveryForm["0.1 Восстановление"]
  MainForm["1. Главное меню"]
  LogoutAction["1.1 Выход"]

  DataMgmtForm["2. Данные и загрузки"]
  SearchForm["3. Поиск школ"]
  SchoolCard["3.1 Карточка школы"]
  RatingProfile["4. Подбор и рейтинг"]
  SettingsHome["5. Настройки анализа"]
  ReportsHome["6. Отчетность"]
  ReportGenerate["6.1 Выгрузка и отчет"]
  ReportArchive["6.2 Архив выгрузок"]
  CalcHistory["6.3 История расчетов"]
  AdminPanel["7. Администрирование"]

  LoginForm -->|POST /login| MainForm
  LoginForm -->|Восстановление| RecoveryForm
  RecoveryForm -->|Назад| LoginForm
  MainForm -->|Выход| LogoutAction
  LogoutAction -->|Редирект| LoginForm

  MainForm --> DataMgmtForm
  MainForm --> SearchForm
  MainForm --> RatingProfile
  MainForm --> SettingsHome
  MainForm --> ReportsHome
  MainForm -->|Если admin| AdminPanel

  SearchForm -->|Открыть школу| SchoolCard
  RatingProfile -->|Открыть школу| SchoolCard
  SchoolCard -->|Назад| SearchForm
  SchoolCard -->|Назад при src=rating| RatingProfile
  SchoolCard -->|Выгрузка и отчет| ReportGenerate

  ReportsHome --> ReportGenerate
  ReportsHome --> ReportArchive
  ReportsHome --> CalcHistory
  ReportGenerate --> ReportArchive
  ReportArchive --> ReportGenerate
  ReportArchive --> CalcHistory
  CalcHistory --> SearchForm

  SearchForm -.->|После «Сохранить результат»| ReportsHome
  RatingProfile -.->|После «Сохранить рейтинг»| ReportsHome

  DataMgmtForm -.->|В меню| MainForm
  SearchForm -.->|В меню| MainForm
  RatingProfile -.->|В меню| MainForm
  SettingsHome -.->|В меню| MainForm
  ReportsHome -.->|В меню| MainForm
  AdminPanel -.->|В меню| MainForm

  classDef auth fill:#f2f2f2,stroke:#333,stroke-width:1px;
  classDef main fill:#ffffff,stroke:#333,stroke-width:2px;
  classDef data fill:#eef7ff,stroke:#2b6cb0,stroke-width:1px;
  classDef search fill:#f5f3ff,stroke:#5a67d8,stroke-width:1px;
  classDef rating fill:#fff7ed,stroke:#c05621,stroke-width:1px;
  classDef set fill:#f7fafc,stroke:#4a5568,stroke-width:1px;
  classDef rep fill:#f1fff3,stroke:#2f855a,stroke-width:1px;
  classDef adm fill:#fff1f2,stroke:#b83280,stroke-width:1px;

  class LoginForm,RecoveryForm,LogoutAction auth;
  class MainForm main;
  class DataMgmtForm data;
  class SearchForm,SchoolCard search;
  class RatingProfile rating;
  class SettingsHome set;
  class ReportsHome,ReportGenerate,ReportArchive,CalcHistory rep;
  class AdminPanel adm;
```
