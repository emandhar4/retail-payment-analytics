-- ============================================================
-- Retail Payment Analytics — Core SQL Queries
-- Author: Mandhar Eppakayala
-- Purpose: Business intelligence queries for payment analytics
--          mirroring real-world FinTech operations workflows
-- ============================================================

-- ── 1. Total Revenue by Product Category ────────────────────────────────────
-- Identifies highest-value product lines for offer/promotion targeting
SELECT
    p.category,
    COUNT(t.transaction_id)          AS total_transactions,
    SUM(t.total_amount)              AS total_revenue,
    ROUND(AVG(t.total_amount), 2)    AS avg_transaction_value,
    ROUND(SUM(t.total_amount) * 100.0 /
          (SELECT SUM(total_amount) FROM transactions
           WHERE status = 'Completed'), 2) AS revenue_share_pct
FROM transactions t
JOIN products p ON t.product_id = p.product_id
WHERE t.status = 'Completed'
GROUP BY p.category
ORDER BY total_revenue DESC;

-- ── 2. Monthly Revenue Trend (YoY Comparison) ───────────────────────────────
-- Tracks payment volume growth over time — key metric for payment processors
SELECT
    year,
    month,
    COUNT(transaction_id)            AS total_transactions,
    ROUND(SUM(total_amount), 2)      AS monthly_revenue,
    ROUND(AVG(total_amount), 2)      AS avg_order_value
FROM transactions
WHERE status = 'Completed'
GROUP BY year, month
ORDER BY year, month;

-- ── 3. Top 20 Customers by Lifetime Value ───────────────────────────────────
-- Customer LTV segmentation used in loyalty and offer targeting programs
SELECT
    t.customer_id,
    c.country,
    c.customer_tier,
    COUNT(t.transaction_id)          AS total_orders,
    ROUND(SUM(t.total_amount), 2)    AS lifetime_value,
    ROUND(AVG(t.total_amount), 2)    AS avg_order_value,
    MIN(t.transaction_date)          AS first_purchase,
    MAX(t.transaction_date)          AS last_purchase
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
WHERE t.status = 'Completed'
GROUP BY t.customer_id, c.country, c.customer_tier
ORDER BY lifetime_value DESC
LIMIT 20;

-- ── 4. Payment Method Performance ───────────────────────────────────────────
-- Critical for prepaid card companies like InComm to track payment mix
SELECT
    payment_method,
    COUNT(transaction_id)            AS total_transactions,
    ROUND(SUM(total_amount), 2)      AS total_volume,
    ROUND(AVG(total_amount), 2)      AS avg_transaction_value,
    ROUND(COUNT(CASE WHEN status = 'Completed' THEN 1 END) * 100.0 /
          COUNT(transaction_id), 2)  AS success_rate_pct,
    COUNT(CASE WHEN status = 'Failed' THEN 1 END) AS failed_transactions
FROM transactions
GROUP BY payment_method
ORDER BY total_volume DESC;

-- ── 5. Transaction Anomaly Flags ─────────────────────────────────────────────
-- Flags high-risk transactions for fraud review
-- Criteria: total_amount > 3 standard deviations above mean
WITH stats AS (
    SELECT
        AVG(total_amount)   AS mean_amount,
        AVG(total_amount * total_amount) - AVG(total_amount) * AVG(total_amount) AS variance
    FROM transactions
    WHERE status = 'Completed'
),
flagged AS (
    SELECT
        t.transaction_id,
        t.customer_id,
        t.product_id,
        p.category,
        t.total_amount,
        t.quantity,
        t.payment_method,
        t.transaction_date,
        ROUND((t.total_amount - s.mean_amount) /
              SQRT(s.variance), 2) AS z_score
    FROM transactions t
    JOIN products p ON t.product_id = p.product_id
    CROSS JOIN stats s
    WHERE t.status = 'Completed'
)
SELECT *
FROM flagged
WHERE ABS(z_score) > 3
ORDER BY z_score DESC
LIMIT 50;

-- ── 6. Country-Level Revenue Distribution ───────────────────────────────────
-- Geographic payment flow analysis used by global payment networks
SELECT
    c.country,
    COUNT(t.transaction_id)          AS total_transactions,
    ROUND(SUM(t.total_amount), 2)    AS total_revenue,
    ROUND(AVG(t.total_amount), 2)    AS avg_order_value,
    COUNT(DISTINCT t.customer_id)    AS unique_customers
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
WHERE t.status = 'Completed'
GROUP BY c.country
ORDER BY total_revenue DESC;

-- ── 7. Merchant Performance Ranking ─────────────────────────────────────────
-- Tracks merchant-level transaction volume for partner management
SELECT
    p.merchant,
    COUNT(t.transaction_id)          AS total_transactions,
    ROUND(SUM(t.total_amount), 2)    AS total_revenue,
    ROUND(AVG(t.total_amount), 2)    AS avg_transaction_value,
    COUNT(DISTINCT t.customer_id)    AS unique_customers,
    ROUND(COUNT(CASE WHEN t.status = 'Refunded' THEN 1 END) * 100.0 /
          COUNT(t.transaction_id), 2) AS refund_rate_pct
FROM transactions t
JOIN products p ON t.product_id = p.product_id
GROUP BY p.merchant
ORDER BY total_revenue DESC;

-- ── 8. Customer Tier Revenue Breakdown ──────────────────────────────────────
-- Supports tiered loyalty and promotional offer strategies
SELECT
    c.customer_tier,
    COUNT(DISTINCT t.customer_id)    AS customer_count,
    COUNT(t.transaction_id)          AS total_transactions,
    ROUND(SUM(t.total_amount), 2)    AS total_revenue,
    ROUND(AVG(t.total_amount), 2)    AS avg_transaction_value,
    ROUND(SUM(t.total_amount) /
          COUNT(DISTINCT t.customer_id), 2) AS revenue_per_customer
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
WHERE t.status = 'Completed'
GROUP BY c.customer_tier
ORDER BY total_revenue DESC;

-- ── 9. Day-of-Week Transaction Patterns ─────────────────────────────────────
-- Identifies peak payment processing windows for capacity planning
SELECT
    day_of_week,
    COUNT(transaction_id)            AS total_transactions,
    ROUND(SUM(total_amount), 2)      AS total_revenue,
    ROUND(AVG(total_amount), 2)      AS avg_order_value
FROM transactions
WHERE status = 'Completed'
GROUP BY day_of_week
ORDER BY total_transactions DESC;

-- ── 10. Failed & Refunded Transaction Analysis ───────────────────────────────
-- Operational health metric — high failure rates indicate processing issues
SELECT
    status,
    payment_method,
    COUNT(transaction_id)            AS transaction_count,
    ROUND(SUM(total_amount), 2)      AS total_value,
    ROUND(AVG(total_amount), 2)      AS avg_value
FROM transactions
WHERE status IN ('Failed', 'Refunded', 'Pending')
GROUP BY status, payment_method
ORDER BY status, transaction_count DESC;
