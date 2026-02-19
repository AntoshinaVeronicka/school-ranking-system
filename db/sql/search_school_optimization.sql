-- Оптимизация поиска школ по подстроке (регистронезависимо, с нормализацией).
-- Выполнять под ролью с правами CREATE EXTENSION и CREATE INDEX.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Индекс под выражение из поискового запроса:
-- lower(replace(replace(regexp_replace(coalesce(full_name, ''), '\s+', ' ', 'g'), 'Ё', 'Е'), 'ё', 'е'))
CREATE INDEX IF NOT EXISTS idx_school_full_name_norm_trgm
ON edu.school
USING gin (
  lower(replace(replace(regexp_replace(coalesce(full_name, ''), '\s+', ' ', 'g'), 'Ё', 'Е'), 'ё', 'е')) gin_trgm_ops
);

-- Индексы под частые фильтры поиска.
CREATE INDEX IF NOT EXISTS idx_school_profile_link_profile_school
ON edu.school_profile_link (profile_id, school_id);

CREATE INDEX IF NOT EXISTS idx_ege_school_subject_stat_subject
ON edu.ege_school_subject_stat (subject_id);
