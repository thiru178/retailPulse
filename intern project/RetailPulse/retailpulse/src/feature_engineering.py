"""
RetailPulse - Feature Engineering & RFM Segmentation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


def load_and_clean(path="data/retail_transactions.csv"):
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    # Remove cancellations and bad rows
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


def compute_rfm(df, snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency = ("InvoiceNo",   "nunique"),
        Monetary  = ("Revenue",     "sum"),
    ).reset_index()

    # Score 1-5
    rfm["R_Score"] = pd.qcut(rfm["Recency"],   5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4:          return "Champions"
        elif r >= 3 and f >= 3:        return "Loyal Customers"
        elif r >= 4 and f <= 2:        return "New Customers"
        elif r >= 3 and f <= 2:        return "Potential Loyalists"
        elif r <= 2 and f >= 3:        return "At-Risk"
        elif r <= 2 and f <= 2:        return "Hibernating"
        else:                          return "Lost"

    rfm["Segment"] = rfm.apply(segment, axis=1)
    return rfm


def kmeans_segmentation(rfm, n_clusters=6):
    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["KMeans_Cluster"] = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, rfm["KMeans_Cluster"])
    print(f"  ✅ Silhouette Score: {sil:.3f}")
    return rfm, km, scaler


def add_purchase_features(df):
    cust = df.groupby("CustomerID").agg(
        TotalOrders       = ("InvoiceNo",   "nunique"),
        TotalItems        = ("Quantity",    "sum"),
        TotalRevenue      = ("Revenue",     "sum"),
        AvgOrderValue     = ("Revenue",     "mean"),
        UniqueProducts    = ("StockCode",   "nunique"),
        FavCategory       = ("Category",    lambda x: x.value_counts().index[0]),
        FirstPurchase     = ("InvoiceDate", "min"),
        LastPurchase      = ("InvoiceDate", "max"),
    ).reset_index()
    cust["CustomerLifeDays"] = (cust["LastPurchase"] - cust["FirstPurchase"]).dt.days
    cust["AvgDaysBetweenOrders"] = cust["CustomerLifeDays"] / cust["TotalOrders"].clip(lower=1)
    return cust


def churn_labels(rfm, threshold_days=90):
    """Mark customers who haven't purchased in threshold_days as churned"""
    rfm["Churned"] = (rfm["Recency"] > threshold_days).astype(int)
    return rfm


if __name__ == "__main__":
    print("🔄 Loading data...")
    df = load_and_clean()
    print(f"  {len(df):,} clean transactions | {df['CustomerID'].nunique()} customers")

    print("🔄 Computing RFM...")
    rfm = compute_rfm(df)
    print(rfm["Segment"].value_counts())

    print("🔄 K-Means segmentation...")
    rfm, km, scaler = kmeans_segmentation(rfm)

    print("🔄 Customer features...")
    cust_features = add_purchase_features(df)

    rfm = churn_labels(rfm)
    rfm.to_csv("data/rfm_segments.csv", index=False)
    cust_features.to_csv("data/customer_features.csv", index=False)
    print("✅ Saved rfm_segments.csv and customer_features.csv")
