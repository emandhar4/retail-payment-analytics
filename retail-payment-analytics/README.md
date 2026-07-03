# Retail Payment Analytics & Anomaly Detection Engine

**Tools:** Python · SQL · Tableau  
**Libraries:** pandas · scikit-learn · matplotlib · seaborn · SQLite  
**Models:** Isolation Forest · K-Means Clustering

---

## Overview

This project builds an end-to-end retail payment analytics system that mirrors real-world workflows used by payment processors and FinTech companies. It ingests 50,000+ synthetic retail transactions, structures them in a normalized SQL database, applies machine learning models to detect fraudulent transactions and segment customers by spending behavior, and exports clean data for Tableau visualization.

The project was built to demonstrate practical data engineering and machine learning skills relevant to analytics roles at payment companies such as Visa, Mastercard, Fiserv, InComm Payments, and financial services firms.

---

## Business Problem

Payment processors and FinTech companies face two core operational challenges:

1. **Fraud and anomaly detection** — identifying transactions that deviate from normal behavior before they result in financial loss
2. **Customer segmentation** — understanding which customers drive the most value so offers and promotions can be targeted effectively

This project addresses both using unsupervised machine learning models applied to transaction-level data.

---

## Project Structure

```
retail-payment-analytics/
├── data/
│   ├── transactions.csv       # 50,000 transaction records
│   ├── customers.csv          # 3,000 customer profiles
│   ├── products.csv           # 200 products across 10 categories
│   └── retail_payments.db     # Normalized SQLite database
├── sql/
│   └── analysis_queries.sql   # 10 business intelligence queries
├── models/
│   └── anomaly_detection.py   # Isolation Forest + K-Means models
├── exports/
│   ├── anomaly_results.csv         # Flagged transactions with scores
│   ├── customer_segments.csv       # RFM-based customer segments
│   ├── tableau_dashboard_data.csv  # Combined data for Tableau
│   ├── isolation_forest_results.png
│   └── kmeans_segments.png
└── generate_data.py           # Synthetic dataset generator
```

---

## Dataset

Synthetic dataset of 50,000 retail payment transactions generated to reflect realistic e-commerce transaction patterns including:
- Transaction ID, customer ID, product ID, date, quantity, amount, payment method, status
- 3,000 unique customers across 10 countries with Gold/Silver/Bronze tier classifications
- 200 products across 10 categories (Electronics, Apparel, Food & Beverage, etc.)
- 4 payment methods: Credit Card, Debit Card, Prepaid Card, Digital Wallet
- ~750 intentionally injected anomalies simulating fraud patterns (high-value spikes, rapid repeat transactions)

---

## SQL Analysis

10 business intelligence queries covering:

| Query | Business Purpose |
|-------|-----------------|
| Revenue by Category | Identifies highest-value product lines for promotion targeting |
| Monthly Revenue Trend | Tracks YoY payment volume growth |
| Top 20 Customers by LTV | Customer lifetime value segmentation |
| Payment Method Performance | Transaction success rates by payment type |
| Anomaly Flags (Z-Score) | Statistical outlier detection via SQL |
| Country-Level Distribution | Geographic payment flow analysis |
| Merchant Performance | Partner-level transaction volume ranking |
| Customer Tier Breakdown | Supports tiered loyalty program strategy |
| Day-of-Week Patterns | Peak payment processing window identification |
| Failed Transaction Analysis | Operational health monitoring |

---

## Machine Learning Models

### Model 1: Isolation Forest — Anomaly Detection

**What it does:** Flags transactions that deviate significantly from normal spending patterns — high purchase amounts relative to customer baseline, unusual quantities, or outlier behavior across the transaction population.

**Why it matters:** Payment processors run anomaly detection at scale to catch fraudulent transactions before they process. Isolation Forest is an industry-standard approach for this use case because it works without labeled fraud data (unsupervised).

**Results:**
- Flagged 857 anomalous transactions out of 28,563 completed transactions (3.0% rate)
- Caught 65.7% of intentionally injected anomalies as true positives
- Features used: transaction amount, quantity, customer average amount, transaction count, amount deviation from customer baseline

### Model 2: K-Means Clustering — Customer Segmentation

**What it does:** Groups customers into distinct segments based on RFM (Recency, Frequency, Monetary) features — identifying High-Value, Mid-Tier, and Occasional customers.

**Why it matters:** FinTech companies and payment processors use customer segmentation to personalize offers, target promotions to the right customers, and prioritize retention efforts for high-value accounts.

**Results (3 optimal clusters via silhouette analysis):**

| Segment | Customers | Avg Recency | Avg Frequency | Avg LTV |
|---------|-----------|-------------|---------------|---------|
| High-Value | 88 | 94 days | 10.2 orders | $9,093 |
| Mid-Tier | 1,577 | 78 days | 11.6 orders | $1,874 |
| Occasional | 1,335 | 157 days | 7.0 orders | $1,047 |

---

## Key Findings

- **Electronics** drives the highest revenue share across all categories at 14.2%
- **Prepaid Card** transactions show the highest failure rate (relevant to InComm's core product)
- **High-Value customers** (88 customers, ~3% of base) account for a disproportionate share of total revenue — typical of a 80/20 distribution in payments
- **Tuesday and Wednesday** show peak transaction volume — useful for payment processing capacity planning
- Anomaly score distribution clearly separates fraudulent and normal transactions, validating the Isolation Forest approach

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 2. Generate dataset
python generate_data.py

# 3. Run SQL queries
# Open data/retail_payments.db in DB Browser for SQLite
# Run queries from sql/analysis_queries.sql

# 4. Run ML models
python models/anomaly_detection.py

# 5. Open exports/ folder and load tableau_dashboard_data.csv into Tableau
```

---

## Tableau Dashboard

The `exports/tableau_dashboard_data.csv` file contains the full transaction dataset enriched with:
- ML anomaly flags and scores from Isolation Forest
- Customer segment labels from K-Means
- RFM metrics per customer

Recommended Tableau views:
1. Transaction volume over time (line chart)
2. Anomaly flags by merchant and category (scatter/heat map)
3. Customer segments by LTV and recency (bubble chart)
4. Revenue by category and country (bar/map)

---

## Skills Demonstrated

- Data engineering: normalized SQL schema design, complex analytical queries
- Machine learning: unsupervised anomaly detection, customer segmentation, feature engineering
- Business analytics: translating model outputs into actionable business insights
- Python: pandas, scikit-learn, matplotlib, seaborn
- Domain knowledge: payment processing workflows, fraud detection, customer lifecycle management
