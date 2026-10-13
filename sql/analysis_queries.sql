
-- 1. Overall dataset and backorder summary
SELECT
    COUNT(*) AS total_skus,
    COUNT(DISTINCT sku) AS unique_skus,
    SUM(went_on_backorder) AS backorder_skus,
    ROUND(100.0 * AVG(went_on_backorder), 2) AS backorder_rate_pct
FROM read_csv_auto('data/raw/train.csv');


-- 2. Backorder rate by inventory status
SELECT
    CASE
        WHEN national_inv < 0 THEN 'Negative inventory'
        WHEN national_inv = 0 THEN 'Zero inventory'
        ELSE 'Positive inventory'
    END AS inventory_status,
    COUNT(*) AS sku_count,
    SUM(went_on_backorder) AS backorder_count,
    ROUND(100.0 * AVG(went_on_backorder), 2) AS backorder_rate_pct
FROM read_csv_auto('data/raw/train.csv')
GROUP BY 1
ORDER BY backorder_rate_pct DESC;


-- 3. Backorder rate by supplier performance data availability
SELECT
    CASE
        WHEN perf_6_month_avg = -99 THEN 'Sentinel value (-99)'
        ELSE 'Performance value available'
    END AS performance_status,
    COUNT(*) AS sku_count,
    ROUND(100.0 * AVG(went_on_backorder), 2) AS backorder_rate_pct
FROM read_csv_auto('data/raw/train.csv')
GROUP BY 1;


-- 4. Inventory compared with three-month forecast
-- Restrict the denominator to SKUs with positive forecast.
SELECT
    COUNT(*) AS skus_with_positive_forecast,
    SUM(
        CASE WHEN national_inv < forecast_3_month
             THEN 1 ELSE 0 END
    ) AS inventory_below_forecast_count,
    ROUND(
        100.0 * SUM(
            CASE WHEN national_inv < forecast_3_month
                 THEN 1 ELSE 0 END
        ) / COUNT(*),
        2
    ) AS inventory_below_forecast_pct
FROM read_csv_auto('data/raw/train.csv')
WHERE forecast_3_month > 0;


-- 5. Backorder rate by selected risk flags
SELECT
    'deck_risk' AS risk_flag,
    deck_risk AS flag_value,
    COUNT(*) AS sku_count,
    ROUND(100.0 * AVG(went_on_backorder), 2) AS backorder_rate_pct
FROM read_csv_auto('data/raw/train.csv')
GROUP BY deck_risk

UNION ALL

SELECT
    'ppap_risk',
    ppap_risk,
    COUNT(*),
    ROUND(100.0 * AVG(went_on_backorder), 2)
FROM read_csv_auto('data/raw/train.csv')
GROUP BY ppap_risk

UNION ALL

SELECT
    'stop_auto_buy',
    stop_auto_buy,
    COUNT(*),
    ROUND(100.0 * AVG(went_on_backorder), 2)
FROM read_csv_auto('data/raw/train.csv')
GROUP BY stop_auto_buy

ORDER BY risk_flag, flag_value;


-- 6. Overdue pieces and local backorder exposure
SELECT
    COUNT(*) AS total_skus,
    SUM(CASE WHEN pieces_past_due > 0 THEN 1 ELSE 0 END)
        AS skus_with_past_due_pieces,
    SUM(CASE WHEN local_bo_qty > 0 THEN 1 ELSE 0 END)
        AS skus_with_local_backorders,
    SUM(pieces_past_due) AS total_pieces_past_due,
    SUM(local_bo_qty) AS total_local_backorder_qty
FROM read_csv_auto('data/raw/train.csv');
