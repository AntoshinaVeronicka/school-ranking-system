-- Исключаем дубли названий муниципалитетов в пределах одного региона
-- после нормализации:
-- 1) обрезка и схлопывание пробелов
-- 2) нормализация префикса города: "г. ", "г " и "Город " -> "город "
-- 3) регистронезависимое сравнение + "ё" -> "е"
--
-- Шаг 1 (диагностика): если запрос вернул строки, сначала объедините дубли.
WITH normed AS (
    SELECT
        m.municipality_id,
        m.region_id,
        replace(
            lower(
                regexp_replace(
                    regexp_replace(
                        regexp_replace(btrim(m.name), '[[:space:]]+', ' ', 'g'),
                        '^[[:space:]]*г(\.|[[:space:]]+)', 'город ', 'i'
                    ),
                    '^[[:space:]]*город[[:space:]]+', 'город ', 'i'
                )
            ),
            'ё', 'е'
        ) AS name_norm
    FROM edu.municipality m
)
SELECT
    n.region_id,
    n.name_norm,
    COUNT(*) AS dup_count,
    array_agg(n.municipality_id ORDER BY n.municipality_id) AS municipality_ids
FROM normed n
GROUP BY n.region_id, n.name_norm
HAVING COUNT(*) > 1;

-- Шаг 2: включаем уникальность по нормализованному (lowercase) имени.
-- Если на шаге 1 остались дубли, этот индекс не создастся до их очистки.
CREATE UNIQUE INDEX IF NOT EXISTS uq_municipality_region_name_norm
    ON edu.municipality (
        region_id,
        replace(
            lower(
                regexp_replace(
                    regexp_replace(
                        regexp_replace(btrim(name), '[[:space:]]+', ' ', 'g'),
                        '^[[:space:]]*г(\.|[[:space:]]+)', 'город ', 'i'
                    ),
                    '^[[:space:]]*город[[:space:]]+', 'город ', 'i'
                )
            ),
            'ё', 'е'
        )
    );
