"""
RetailPulse - Synthetic Data Generator
Simulates Online Retail II (UCI) dataset structure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
N_CUSTOMERS   = 4000
N_PRODUCTS    = 200
N_ROWS        = 100_000
START_DATE    = datetime(2022, 1, 1)
END_DATE      = datetime(2023, 12, 31)

COUNTRIES = ["United Kingdom", "Germany", "France", "Spain", "Netherlands",
             "Belgium", "Switzerland", "Australia", "Japan", "USA"]

PRODUCT_CATEGORIES = {
    "Home Decor":    (2.50, 45.00),
    "Kitchen":       (3.00, 60.00),
    "Clothing":      (5.00, 80.00),
    "Stationery":    (1.00, 25.00),
    "Toys & Games":  (4.00, 55.00),
    "Electronics":   (10.00, 150.00),
    "Beauty":        (3.50, 70.00),
    "Garden":        (5.00, 90.00),
}

def generate_products(n=N_PRODUCTS):
    categories = list(PRODUCT_CATEGORIES.keys())
    products = []
    for i in range(1, n + 1):
        cat = random.choice(categories)
        low, high = PRODUCT_CATEGORIES[cat]
        products.append({
            "StockCode":    f"P{i:04d}",
            "Description":  f"{cat} Item {i:03d}",
            "Category":     cat,
            "UnitPrice":    round(random.uniform(low, high), 2),
        })
    return pd.DataFrame(products)

def generate_customers(n=N_CUSTOMERS):
    customers = []
    for i in range(1, n + 1):
        country = random.choices(COUNTRIES, weights=[50,8,8,5,5,4,4,4,6,6])[0]
        customers.append({
            "CustomerID": 10000 + i,
            "Country":    country,
            "Segment":    random.choice(["Champions","Loyal","At-Risk","New","Hibernating","Potential"]),
        })
    return pd.DataFrame(customers)

def generate_transactions(products_df, customers_df, n=N_ROWS):
    rows = []
    date_range = (END_DATE - START_DATE).days

    for _ in range(n):
        prod   = products_df.sample(1).iloc[0]
        cust   = customers_df.sample(1).iloc[0]
        date   = START_DATE + timedelta(days=random.randint(0, date_range))

        # Seasonal multiplier (Q4 boost)
        month  = date.month
        season_mult = 1.5 if month in [11, 12] else 1.2 if month in [3, 4, 5] else 1.0

        qty    = max(1, int(np.random.poisson(4 * season_mult)))
        price  = prod["UnitPrice"] * random.uniform(0.9, 1.1)

        # ~3% cancellations (negative qty)
        if random.random() < 0.03:
            qty = -qty

        rows.append({
            "InvoiceNo":   f"INV{random.randint(100000, 999999)}",
            "StockCode":   prod["StockCode"],
            "Description": prod["Description"],
            "Category":    prod["Category"],
            "Quantity":    qty,
            "InvoiceDate": date.strftime("%Y-%m-%d"),
            "UnitPrice":   round(price, 2),
            "CustomerID":  cust["CustomerID"],
            "Country":     cust["Country"],
        })

    df = pd.DataFrame(rows)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df

def main():
    os.makedirs("data", exist_ok=True)
    print("🔄 Generating products...")
    products  = generate_products()

    print("🔄 Generating customers...")
    customers = generate_customers()

    print("🔄 Generating transactions...")
    transactions = generate_transactions(products, customers)

    # Save
    products.to_csv("data/products.csv", index=False)
    customers.to_csv("data/customers.csv", index=False)
    transactions.to_csv("data/retail_transactions.csv", index=False)

    print(f"✅ Done! {len(transactions):,} transactions | {N_CUSTOMERS} customers | {N_PRODUCTS} products")
    print(f"   Files saved to: data/")

if __name__ == "__main__":
    main()
