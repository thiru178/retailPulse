"""
RetailPulse - Inventory Optimization
Economic Order Quantity + Safety Stock using forecasted demand
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def compute_eoq(demand_per_day, ordering_cost=50, holding_cost_pct=0.25, unit_price=10):
    """Economic Order Quantity formula"""
    annual_demand = demand_per_day * 365
    holding_cost  = unit_price * holding_cost_pct
    if holding_cost <= 0 or annual_demand <= 0:
        return 0
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    return round(eoq, 0)


def compute_safety_stock(demand_std, lead_time_days=7, service_level=0.95):
    """Safety stock = Z * σ_demand * sqrt(lead_time)"""
    z_scores = {0.90: 1.28, 0.95: 1.645, 0.99: 2.326}
    z = z_scores.get(service_level, 1.645)
    safety = z * demand_std * np.sqrt(lead_time_days)
    return round(safety, 0)


def compute_reorder_point(avg_daily_demand, lead_time_days=7, safety_stock=0):
    return round(avg_daily_demand * lead_time_days + safety_stock, 0)


def build_inventory_recommendations(df, forecast_df):
    """
    df           : transaction dataframe
    forecast_df  : 30-day forecast
    """
    # Product-level daily demand
    prod_daily = (
        df.groupby(["StockCode", "Description", "Category"])
          .agg(
              TotalQty   = ("Quantity",    "sum"),
              TotalRev   = ("Revenue",     "sum"),
              AvgPrice   = ("UnitPrice",   "mean"),
              NumOrders  = ("InvoiceNo",   "nunique"),
          )
          .reset_index()
    )

    # Days span
    df_parsed = df.copy()
    df_parsed["InvoiceDate"] = pd.to_datetime(df_parsed["InvoiceDate"])
    n_days = (df_parsed["InvoiceDate"].max() - df_parsed["InvoiceDate"].min()).days or 1

    prod_daily["AvgDailyQty"]  = prod_daily["TotalQty"] / n_days
    prod_daily["DemandStd"]    = (
        df.groupby("StockCode")["Quantity"]
          .std()
          .fillna(0)
          .reset_index()["Quantity"]
    )

    # Forecasted uplift (use ensemble forecast daily average)
    if "ensemble_yhat" in forecast_df.columns:
        rev_col = "ensemble_yhat"
    else:
        rev_col = "yhat"

    avg_forecast_rev = forecast_df[rev_col].mean()
    avg_hist_rev     = df["Revenue"].sum() / n_days
    uplift           = avg_forecast_rev / avg_hist_rev if avg_hist_rev > 0 else 1.0
    uplift           = np.clip(uplift, 0.8, 1.5)

    prod_daily["ForecastedDailyQty"] = prod_daily["AvgDailyQty"] * uplift

    # Compute inventory metrics
    prod_daily["EOQ"]          = prod_daily.apply(
        lambda r: compute_eoq(r["ForecastedDailyQty"], unit_price=max(r["AvgPrice"], 1)), axis=1
    )
    prod_daily["SafetyStock"]  = prod_daily.apply(
        lambda r: compute_safety_stock(r["DemandStd"]), axis=1
    )
    prod_daily["ReorderPoint"] = prod_daily.apply(
        lambda r: compute_reorder_point(r["ForecastedDailyQty"], safety_stock=r["SafetyStock"]), axis=1
    )
    prod_daily["Stock30Days"]  = prod_daily["ForecastedDailyQty"] * 30
    prod_daily["ReorderQty"]   = (prod_daily["EOQ"] + prod_daily["SafetyStock"]).round(0)

    # Risk flag
    def risk_flag(row):
        if row["AvgDailyQty"] > row["ForecastedDailyQty"] * 1.2: return "⚠️ Overstock Risk"
        if row["ForecastedDailyQty"] > row["AvgDailyQty"] * 1.3: return "🔴 Stockout Risk"
        return "✅ Optimal"

    prod_daily["InventoryStatus"] = prod_daily.apply(risk_flag, axis=1)

    return prod_daily.sort_values("TotalRev", ascending=False)


def run_inventory_optimization():
    from feature_engineering import load_and_clean
    print("🔄 Loading data...")
    df          = load_and_clean()
    forecast_df = pd.read_csv("data/forecast_30d.csv", parse_dates=["ds"])

    print("🔄 Computing inventory recommendations...")
    inv = build_inventory_recommendations(df, forecast_df)

    inv.to_csv("data/inventory_recommendations.csv", index=False)
    print(f"✅ Inventory recommendations for {len(inv)} products saved")
    print(inv["InventoryStatus"].value_counts())
    return inv


if __name__ == "__main__":
    run_inventory_optimization()
