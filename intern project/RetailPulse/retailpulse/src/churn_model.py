"""
RetailPulse - Churn Prediction
XGBoost + SHAP explainability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, classification_report,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")


def build_churn_features(rfm_df, cust_features_df):
    """Merge RFM + customer features for churn model"""
    df = rfm_df.merge(cust_features_df, on="CustomerID", how="left")

    features = [
        "Recency", "Frequency", "Monetary",
        "R_Score", "F_Score", "M_Score", "RFM_Score",
        "TotalOrders", "TotalItems", "TotalRevenue",
        "AvgOrderValue", "UniqueProducts",
        "CustomerLifeDays", "AvgDaysBetweenOrders",
    ]

    # Encode favorite category
    le = LabelEncoder()
    df["FavCategory_enc"] = le.fit_transform(df["FavCategory"].fillna("Unknown"))
    features.append("FavCategory_enc")

    X = df[features].fillna(0)
    y = df["Churned"]
    return X, y, df, features


def train_churn_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # GradientBoosting (XGBoost-equivalent without xgboost dependency)
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42
        )
        model_name = "XGBoost"
    except ImportError:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        model_name = "GradientBoosting"

    model.fit(X_train, y_train)

    # Metrics
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    p20 = precision_at_top_k(y_test.values, y_proba, k=0.20)

    metrics = {
        "Model":          model_name,
        "AUC_ROC":        round(auc, 4),
        "Precision@20%":  round(p20, 4),
        "Precision":      round(precision_score(y_test, y_pred), 4),
        "Recall":         round(recall_score(y_test, y_pred), 4),
        "F1":             round(f1_score(y_test, y_pred), 4),
    }

    print(f"  🎯 {model_name} | AUC-ROC: {auc:.4f} | Precision@20%: {p20:.4f}")
    return model, metrics, X_test, y_test, y_proba


def precision_at_top_k(y_true, y_proba, k=0.20):
    """Precision for top k% predicted churners"""
    n = max(1, int(len(y_true) * k))
    top_idx = np.argsort(y_proba)[::-1][:n]
    return y_true[top_idx].mean()


def get_feature_importance(model, feature_names):
    try:
        imp = model.feature_importances_
    except:
        imp = np.zeros(len(feature_names))
    fi = pd.DataFrame({"Feature": feature_names, "Importance": imp})
    return fi.sort_values("Importance", ascending=False)


def score_all_customers(model, X, rfm_df):
    """Add churn probability to all customers"""
    rfm_df = rfm_df.copy()
    rfm_df["ChurnProb"]  = model.predict_proba(X)[:, 1]
    rfm_df["ChurnRisk"]  = pd.cut(
        rfm_df["ChurnProb"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"]
    )
    return rfm_df


def run_churn_pipeline():
    print("🔄 Loading features...")
    rfm_df   = pd.read_csv("data/rfm_segments.csv")
    cust_df  = pd.read_csv("data/customer_features.csv")

    print("🔄 Building churn features...")
    X, y, merged, features = build_churn_features(rfm_df, cust_df)
    print(f"  Churn rate: {y.mean():.1%} | Samples: {len(y)}")

    print("🔄 Training churn model...")
    model, metrics, X_test, y_test, y_proba = train_churn_model(X, y)

    print("🔄 Feature importance...")
    fi = get_feature_importance(model, features)
    print(fi.head(5).to_string(index=False))

    print("🔄 Scoring all customers...")
    scored = score_all_customers(model, X, rfm_df)

    # Save
    scored.to_csv("data/churn_scored.csv", index=False)
    fi.to_csv("data/feature_importance.csv", index=False)
    pd.DataFrame([metrics]).to_csv("data/churn_metrics.csv", index=False)
    print("✅ Churn pipeline complete!")
    return model, metrics, fi


if __name__ == "__main__":
    run_churn_pipeline()
