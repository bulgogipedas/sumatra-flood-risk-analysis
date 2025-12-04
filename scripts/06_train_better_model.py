"""
Train model untuk ~80% accuracy.
"""
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent

def load_data():
    path = PROJECT_ROOT / "data" / "processed" / "complete_analysis_dataset.parquet"
    return pd.read_parquet(path)

def prepare_data(df):
    """Prepare features with better signal."""
    df = df.copy()
    
    # Create derived features
    if "forest_loss_pct" in df.columns:
        df["high_deforestation"] = (df["forest_loss_pct"] > df["forest_loss_pct"].quantile(0.7)).astype(float)
    
    if "rainfall_annual_mean_mm" in df.columns:
        df["high_rainfall"] = (df["rainfall_annual_mean_mm"] > df["rainfall_annual_mean_mm"].quantile(0.6)).astype(float)
    
    if "palm_oil_pct" in df.columns:
        df["high_agriculture"] = (df["palm_oil_pct"] > df["palm_oil_pct"].quantile(0.6)).astype(float)
    
    # Interaction
    df["defor_rain"] = df.get("high_deforestation", 0) * df.get("high_rainfall", 0)
    df["agri_rain"] = df.get("high_agriculture", 0) * df.get("high_rainfall", 0)
    
    # All numeric features
    feature_cols = [
        "forest_loss_pct", "palm_oil_pct", 
        "rainfall_annual_mean_mm", "rainfall_extreme_days_avg",
        "high_deforestation", "high_rainfall", "high_agriculture",
        "defor_rain", "agri_rain"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].fillna(0)
    
    # Target: significant flood area (top 40%)
    threshold = df["flood_events_total"].quantile(0.6)
    y = (df["flood_events_total"] > threshold).astype(int)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target threshold: {threshold:.0f} events")
    print(f"Class balance: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

def train(X, y, feature_names):
    """Train with optimized parameters."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Class weights
    class_counts = np.bincount(y_train)
    scale_pos = class_counts[0] / class_counts[1] if len(class_counts) > 1 and class_counts[1] > 0 else 1
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        gamma=0.2,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
    
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    
    # CV with fresh model (no early stopping)
    cv_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        gamma=0.2,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_full = scaler.fit_transform(X)
    
    cv_acc = cross_val_score(cv_model, X_full, y, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(cv_model, X_full, y, cv=cv, scoring="f1")
    cv_roc = cross_val_score(cv_model, X_full, y, cv=cv, scoring="roc_auc")
    
    cv_results = {
        "accuracy": {"mean": cv_acc.mean(), "std": cv_acc.std(), "scores": cv_acc.tolist()},
        "f1": {"mean": cv_f1.mean(), "std": cv_f1.std(), "scores": cv_f1.tolist()},
        "roc_auc": {"mean": cv_roc.mean(), "std": cv_roc.std(), "scores": cv_roc.tolist()},
    }
    
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    return model, scaler, metrics, cv_results, importance, y_test, y_pred

def save(model, scaler, features, metrics, cv_results, importance):
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Clean old models
    for f in models_dir.glob("flood_risk_*.pkl"):
        f.unlink()
    for f in models_dir.glob("flood_risk_*.json"):
        f.unlink()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"flood_risk_xgboost_{timestamp}"
    
    with open(models_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(models_dir / f"{name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    metadata = {
        "model_type": "xgboost",
        "feature_names": features,
        "created_at": datetime.now().isoformat(),
        "training_history": {
            "metrics": metrics,
            "cv_results": cv_results,
            "feature_importance": importance.to_dict(orient="records"),
        }
    }
    
    with open(models_dir / f"{name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return name

def main():
    print("="*60)
    print("TRAINING MODEL FOR ~80% ACCURACY")
    print("="*60)
    
    df = load_data()
    print(f"Data: {len(df)} records\n")
    
    X, y, features = prepare_data(df)
    model, scaler, metrics, cv, importance, y_test, y_pred = train(X, y, features)
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.1%}")
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION (5-fold)")
    print("="*60)
    print(f"  Accuracy: {cv['accuracy']['mean']:.1%} (+/- {cv['accuracy']['std']:.1%})")
    print(f"  F1-Score: {cv['f1']['mean']:.1%} (+/- {cv['f1']['std']:.1%})")
    print(f"  ROC-AUC:  {cv['roc_auc']['mean']:.1%} (+/- {cv['roc_auc']['std']:.1%})")
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negative:  {cm[0,0]:4d}  |  False Positive: {cm[0,1]:4d}")
    print(f"  False Negative: {cm[1,0]:4d}  |  True Positive:  {cm[1,1]:4d}")
    
    print("\n" + "="*60)
    print("TOP FEATURES")
    print("="*60)
    for _, row in importance.head(5).iterrows():
        bar = "█" * int(row["importance"] * 40)
        print(f"  {row['feature']:25s} {row['importance']:.3f} {bar}")
    
    name = save(model, scaler, features, metrics, cv, importance)
    print(f"\nModel saved: {name}")

if __name__ == "__main__":
    main()
