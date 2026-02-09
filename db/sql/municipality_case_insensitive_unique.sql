-- Prevent duplicates in municipality names inside one region
-- after normalization:
-- 1) trim/collapse spaces
-- 2) normalize city prefix: "г. ", "г " and "Город " -> "город "
-- 3) case-insensitive compare + "ё" -> "е"
--
-- Step 1 (diagnostic): if this query returns rows, merge duplicates first.
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

-- Step 2: enforce uniqueness by normalized (lowercase) name.
-- If Step 1 returned duplicates, this statement will fail until duplicates are cleaned.
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
