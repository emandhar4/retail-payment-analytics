"""
Retail Payment Analytics — ML Models
=====================================
Models:
  1. Isolation Forest  — Unsupervised anomaly detection (fraud signals)
  2. K-Means Clustering — Customer segmentation by spending behavior

Author : Mandhar Eppakayala
Tools  : Python, scikit-learn, pandas, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3, os, warnings
from sklearn.ensemble       import IsolationForest
from sklearn.cluster        import KMeans
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import silhouette_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB     = "/tmp/retail_payments_working.db"
EXPORT = os.path.join(BASE, "exports")
os.makedirs(EXPORT, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB)
transactions = pd.read_sql("SELECT * FROM transactions", conn)
customers    = pd.read_sql("SELECT * FROM customers",    conn)
conn.close()

transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
completed = transactions[transactions["status"] == "Completed"].copy()
print(f"Loaded {len(transactions):,} transactions | {len(completed):,} completed")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — ISOLATION FOREST: Anomaly / Fraud Detection
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Running Isolation Forest ──")

# Feature engineering
iso_df = completed[["transaction_id","customer_id","total_amount",
                     "quantity","anomaly_injected"]].copy()

# Customer-level behavioral features
customer_stats = completed.groupby("customer_id").agg(
    customer_avg_amount  = ("total_amount", "mean"),
    customer_txn_count   = ("transaction_id", "count"),
    customer_total_spend = ("total_amount", "sum"),
).reset_index()

iso_df = iso_df.merge(customer_stats, on="customer_id", how="left")

# Amount deviation from customer's own baseline
iso_df["amount_vs_customer_avg"] = (
    iso_df["total_amount"] / iso_df["customer_avg_amount"]
)

features_iso = ["total_amount", "quantity",
                "customer_avg_amount", "customer_txn_count",
                "amount_vs_customer_avg"]

X_iso = iso_df[features_iso].fillna(0)

scaler_iso = StandardScaler()
X_iso_scaled = scaler_iso.fit_transform(X_iso)

# Train Isolation Forest (contamination = estimated anomaly rate)
iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.03,   # ~3% anomaly rate
    random_state=42,
    n_jobs=-1
)
iso_df["anomaly_score"]      = iso_model.fit_predict(X_iso_scaled)
iso_df["anomaly_raw_score"]  = iso_model.score_samples(X_iso_scaled)
iso_df["is_anomaly"]         = (iso_df["anomaly_score"] == -1).astype(int)

# Results
n_flagged  = iso_df["is_anomaly"].sum()
n_injected = iso_df["anomaly_injected"].sum()
detected   = iso_df[(iso_df["is_anomaly"] == 1) &
                    (iso_df["anomaly_injected"] == 1)].shape[0]

print(f"  Flagged as anomalies  : {n_flagged:,}")
print(f"  Known injected anomaly: {n_injected:,}")
print(f"  True positives caught : {detected:,} ({detected/n_injected*100:.1f}%)")

# ── Plot 1: Anomaly Score Distribution ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Isolation Forest — Fraud / Anomaly Detection", fontsize=14, fontweight="bold")

axes[0].hist(iso_df[iso_df["is_anomaly"]==0]["anomaly_raw_score"],
             bins=60, alpha=0.7, color="#4C72B0", label="Normal")
axes[0].hist(iso_df[iso_df["is_anomaly"]==1]["anomaly_raw_score"],
             bins=60, alpha=0.7, color="#DD4949", label="Anomaly")
axes[0].set_title("Anomaly Score Distribution")
axes[0].set_xlabel("Isolation Forest Score (lower = more anomalous)")
axes[0].set_ylabel("Count")
axes[0].legend()

axes[1].scatter(
    iso_df[iso_df["is_anomaly"]==0]["total_amount"],
    iso_df[iso_df["is_anomaly"]==0]["quantity"],
    alpha=0.3, s=5, color="#4C72B0", label="Normal"
)
axes[1].scatter(
    iso_df[iso_df["is_anomaly"]==1]["total_amount"],
    iso_df[iso_df["is_anomaly"]==1]["quantity"],
    alpha=0.7, s=20, color="#DD4949", label="Anomaly", zorder=5
)
axes[1].set_title("Transaction Amount vs Quantity — Anomaly Map")
axes[1].set_xlabel("Transaction Amount ($)")
axes[1].set_ylabel("Quantity")
axes[1].set_xlim(0, iso_df["total_amount"].quantile(0.99))
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{EXPORT}/isolation_forest_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: isolation_forest_results.png")

# Export flagged transactions for Tableau
anomaly_export = iso_df.merge(
    completed[["transaction_id","product_id","category","merchant",
               "payment_method","transaction_date"]],
    on="transaction_id", how="left"
)
anomaly_export.to_csv(f"{EXPORT}/anomaly_results.csv", index=False)
print("  Saved: anomaly_results.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — K-MEANS CLUSTERING: Customer Segmentation
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Running K-Means Customer Segmentation ──")

# Build customer-level feature matrix (RFM = Recency, Frequency, Monetary)
snapshot_date = completed["transaction_date"].max()

rfm = completed.groupby("customer_id").agg(
    recency   = ("transaction_date",
                 lambda x: (snapshot_date - x.max()).days),
    frequency = ("transaction_id", "count"),
    monetary  = ("total_amount",   "sum"),
    avg_order = ("total_amount",   "mean"),
    categories_purchased = ("category", "nunique"),
).reset_index()

rfm = rfm.merge(customers[["customer_id","customer_tier","country"]],
                on="customer_id", how="left")

features_kmeans = ["recency","frequency","monetary","avg_order",
                   "categories_purchased"]
X_kmeans = rfm[features_kmeans].fillna(0)

scaler_km = StandardScaler()
X_kmeans_scaled = scaler_km.fit_transform(X_kmeans)

# Find optimal K using silhouette scores
print("  Finding optimal K...")
sil_scores = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_kmeans_scaled)
    sil_scores[k] = silhouette_score(X_kmeans_scaled, labels)
    print(f"    K={k} → Silhouette Score: {sil_scores[k]:.4f}")

optimal_k = max(sil_scores, key=sil_scores.get)
print(f"  Optimal K: {optimal_k}")

# Final model
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm["cluster"] = kmeans.fit_predict(X_kmeans_scaled)

# Label clusters by monetary value ranking
cluster_summary = rfm.groupby("cluster")["monetary"].mean().sort_values(ascending=False)
label_map = {old: new for new, old in enumerate(cluster_summary.index)}
rfm["cluster"] = rfm["cluster"].map(label_map)

segment_labels = {
    0: "High-Value",
    1: "Mid-Tier",
    2: "Occasional",
    3: "At-Risk",
    4: "Dormant",
}
rfm["segment"] = rfm["cluster"].map(
    lambda x: segment_labels.get(x, f"Segment {x}")
)

# Segment stats
seg_stats = rfm.groupby("segment")[features_kmeans].mean().round(2)
seg_counts = rfm.groupby("segment")["customer_id"].count().rename("customer_count")
seg_report = pd.concat([seg_counts, seg_stats], axis=1).sort_values(
    "monetary", ascending=False
)
print("\n  Customer Segment Summary:")
print(seg_report.to_string())

# ── Plot 2: Customer Segments ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("K-Means Customer Segmentation (RFM Analysis)",
             fontsize=14, fontweight="bold")

palette = sns.color_palette("Set2", optimal_k)

# Scatter: Recency vs Monetary
for i, (seg, grp) in enumerate(rfm.groupby("segment")):
    axes[0].scatter(grp["recency"], grp["monetary"],
                    alpha=0.5, s=15, label=seg, color=palette[i % len(palette)])
axes[0].set_title("Recency vs Lifetime Spend")
axes[0].set_xlabel("Recency (days since last purchase)")
axes[0].set_ylabel("Lifetime Monetary Value ($)")
axes[0].legend(fontsize=8)

# Bar: Avg order value by segment
seg_avg = rfm.groupby("segment")["avg_order"].mean().sort_values(ascending=False)
axes[1].bar(seg_avg.index, seg_avg.values,
            color=palette[:len(seg_avg)])
axes[1].set_title("Average Order Value by Segment")
axes[1].set_xlabel("Customer Segment")
axes[1].set_ylabel("Avg Order Value ($)")
axes[1].tick_params(axis="x", rotation=20)

# Bar: Customer count by segment
seg_cnt = rfm.groupby("segment")["customer_id"].count().sort_values(ascending=False)
axes[2].bar(seg_cnt.index, seg_cnt.values,
            color=palette[:len(seg_cnt)])
axes[2].set_title("Customer Count by Segment")
axes[2].set_xlabel("Customer Segment")
axes[2].set_ylabel("Number of Customers")
axes[2].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(f"{EXPORT}/kmeans_segments.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved: kmeans_segments.png")

# Export for Tableau
rfm.to_csv(f"{EXPORT}/customer_segments.csv", index=False)
print("  Saved: customer_segments.csv")

# Export full transaction data with segment labels for Tableau dashboard
tableau_export = completed.merge(
    rfm[["customer_id","segment","recency","frequency","monetary"]],
    on="customer_id", how="left"
).merge(
    iso_df[["transaction_id","is_anomaly","anomaly_raw_score"]],
    on="transaction_id", how="left"
)
tableau_export.to_csv(f"{EXPORT}/tableau_dashboard_data.csv", index=False)
print("  Saved: tableau_dashboard_data.csv (main Tableau source)")

print("\n✅ All models complete. Export files ready in /exports/")
print(f"   → anomaly_results.csv")
print(f"   → customer_segments.csv")
print(f"   → tableau_dashboard_data.csv")
print(f"   → isolation_forest_results.png")
print(f"   → kmeans_segments.png")
