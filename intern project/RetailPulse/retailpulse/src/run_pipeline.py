"""
RetailPulse - Main Pipeline Runner
Run this once to generate all data files needed by the dashboard
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

def run_pipeline():
    print("=" * 60)
    print("  🛒 RetailPulse - Full Pipeline")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n📦 STEP 1: Generating synthetic retail data...")
    from data_generator import main as gen_data
    gen_data()

    # 2. Feature engineering + RFM
    print("\n📊 STEP 2: Feature engineering & RFM segmentation...")
    from feature_engineering import load_and_clean, compute_rfm, kmeans_segmentation, add_purchase_features, churn_labels
    df = load_and_clean()
    rfm = compute_rfm(df)
    rfm, km, scaler = kmeans_segmentation(rfm)
    cust_features = add_purchase_features(df)
    rfm = churn_labels(rfm)
    rfm.to_csv("data/rfm_segments.csv", index=False)
    cust_features.to_csv("data/customer_features.csv", index=False)
    df.to_csv("data/clean_transactions.csv", index=False)
    print("  ✅ RFM + Customer features saved")

    # 3. Demand forecasting
    print("\n📈 STEP 3: Demand forecasting...")
    from forecasting import run_forecasting
    ts, forecast_df, metrics = run_forecasting(df)
    print(f"  📊 Forecast MAPE: {metrics['MAPE']}%")

    # 4. Churn prediction
    print("\n🔮 STEP 4: Churn prediction model...")
    from churn_model import run_churn_pipeline
    model, churn_metrics, fi = run_churn_pipeline()

    # 5. Inventory optimization
    print("\n📦 STEP 5: Inventory optimization...")
    from inventory_optimizer import run_inventory_optimization
    inv = run_inventory_optimization()

    print("\n" + "=" * 60)
    print("  ✅ Pipeline Complete! All files in data/")
    print("=" * 60)
    print("\nFiles generated:")
    for f in sorted(os.listdir("data")):
        size = os.path.getsize(f"data/{f}") / 1024
        print(f"  📄 {f:<40} {size:>8.1f} KB")

    print("\n🚀 Now run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_pipeline()
