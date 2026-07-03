"""
Generate synthetic retail payment transaction dataset
Mirrors real-world e-commerce transaction data structure
used by payment processors like Visa, Mastercard, InComm, Fiserv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sqlite3
import os

np.random.seed(42)
random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
N_TRANSACTIONS = 50000
N_CUSTOMERS    = 3000
N_PRODUCTS     = 200
START_DATE     = datetime(2022, 1, 1)
END_DATE       = datetime(2024, 12, 31)

# ── Reference data ───────────────────────────────────────────────────────────
CATEGORIES = [
    "Electronics", "Apparel", "Home & Garden", "Sports", "Beauty",
    "Food & Beverage", "Books", "Toys", "Automotive", "Health"
]

COUNTRIES = [
    "United States", "United Kingdom", "Canada", "Germany",
    "France", "Australia", "Japan", "Brazil", "Mexico", "Spain"
]

MERCHANTS = [
    "RetailCo", "ShopMart", "QuickBuy", "MegaStore", "ValuePlus",
    "TrendShop", "DailyDeals", "PrimeMart", "CityGoods", "UrbanSelect"
]

PAYMENT_METHODS = ["Credit Card", "Debit Card", "Prepaid Card", "Digital Wallet"]

# ── Generate customers ────────────────────────────────────────────────────────
customers = pd.DataFrame({
    "customer_id":   [f"C{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS + 1)],
    "country":       np.random.choice(COUNTRIES, N_CUSTOMERS,
                         p=[0.45,0.15,0.10,0.07,0.06,0.05,0.04,0.03,0.03,0.02]),
    "customer_tier": np.random.choice(["Gold","Silver","Bronze"],
                         N_CUSTOMERS, p=[0.15, 0.35, 0.50]),
    "join_date":     [START_DATE + timedelta(days=random.randint(0,365))
                      for _ in range(N_CUSTOMERS)],
})

# ── Generate products ─────────────────────────────────────────────────────────
products = pd.DataFrame({
    "product_id":   [f"P{str(i).zfill(4)}" for i in range(1, N_PRODUCTS + 1)],
    "category":     np.random.choice(CATEGORIES, N_PRODUCTS),
    "unit_price":   np.round(np.random.lognormal(mean=3.5, sigma=1.2, size=N_PRODUCTS), 2),
    "merchant":     np.random.choice(MERCHANTS, N_PRODUCTS),
})
products["unit_price"] = products["unit_price"].clip(1.99, 999.99)

# ── Generate transactions ─────────────────────────────────────────────────────
date_range = (END_DATE - START_DATE).days
transaction_dates = [START_DATE + timedelta(days=random.randint(0, date_range))
                     for _ in range(N_TRANSACTIONS)]

transactions = pd.DataFrame({
    "transaction_id":  [f"T{str(i).zfill(7)}" for i in range(1, N_TRANSACTIONS + 1)],
    "customer_id":     np.random.choice(customers["customer_id"], N_TRANSACTIONS),
    "product_id":      np.random.choice(products["product_id"],   N_TRANSACTIONS),
    "transaction_date": transaction_dates,
    "quantity":        np.random.choice(range(1, 11), N_TRANSACTIONS,
                           p=[0.40,0.25,0.15,0.08,0.04,0.03,0.02,0.01,0.01,0.01]),
    "payment_method":  np.random.choice(PAYMENT_METHODS, N_TRANSACTIONS,
                           p=[0.40, 0.30, 0.20, 0.10]),
    "status":          np.random.choice(
                           ["Completed","Completed","Completed","Completed",
                            "Refunded","Failed","Pending"],
                           N_TRANSACTIONS),
})

# Merge price info
transactions = transactions.merge(
    products[["product_id","unit_price","category","merchant"]],
    on="product_id", how="left"
)
transactions["total_amount"] = np.round(
    transactions["quantity"] * transactions["unit_price"], 2
)

# ── Inject anomalies (realistic fraud patterns) ───────────────────────────────
anomaly_idx = np.random.choice(N_TRANSACTIONS, size=int(N_TRANSACTIONS * 0.03), replace=False)

# High-value anomalies
hv_idx = anomaly_idx[:len(anomaly_idx)//2]
transactions.loc[hv_idx, "total_amount"] *= np.random.uniform(8, 20, len(hv_idx))
transactions.loc[hv_idx, "quantity"]     *= np.random.randint(5, 15, len(hv_idx))
transactions.loc[hv_idx, "anomaly_injected"] = 1

# Rapid repeat transactions (same customer, multiple transactions same day)
rr_idx = anomaly_idx[len(anomaly_idx)//2:]
repeat_customers = transactions.loc[rr_idx, "customer_id"].values
for cust in repeat_customers[:20]:
    mask = transactions["customer_id"] == cust
    if mask.sum() > 0:
        transactions.loc[mask, "transaction_date"] = transaction_dates[0]

transactions["anomaly_injected"] = transactions.get("anomaly_injected", pd.Series(0)).fillna(0).astype(int)
transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
transactions["year"]  = transactions["transaction_date"].dt.year
transactions["month"] = transactions["transaction_date"].dt.month
transactions["day_of_week"] = transactions["transaction_date"].dt.day_name()

# ── Save CSV exports ──────────────────────────────────────────────────────────
output_dir = os.path.dirname(os.path.abspath(__file__))
transactions.to_csv(f"{output_dir}/data/transactions.csv", index=False)
customers.to_csv(f"{output_dir}/data/customers.csv",       index=False)
products.to_csv(f"{output_dir}/data/products.csv",         index=False)

# ── Build SQLite database ─────────────────────────────────────────────────────
conn = sqlite3.connect(f"{output_dir}/data/retail_payments.db")

customers.to_sql("customers",    conn, if_exists="replace", index=False)
products.to_sql("products",      conn, if_exists="replace", index=False)
transactions.to_sql("transactions", conn, if_exists="replace", index=False)

conn.close()

print(f"✅ Dataset generated successfully")
print(f"   Transactions : {len(transactions):,}")
print(f"   Customers    : {len(customers):,}")
print(f"   Products     : {len(products):,}")
print(f"   Anomalies    : {int(transactions['anomaly_injected'].sum()):,} injected")
