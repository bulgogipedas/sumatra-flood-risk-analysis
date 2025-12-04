"""
Retrain model dengan regularization yang proper untuk menghindari overfitting.
"""
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent

def load_data():
    """Load complete analysis dataset."""
    path = PROJECT_ROOT / "data" / "processed" / "complete_analysis_dataset.parquet"
    df = pd.read_parquet(path)
    return df

def prepare_features(df):
    """Prepare features and target."""
    # Define feature columns - only use meaningful predictors
    feature_cols = [
        "forest_loss_pct",
        "palm_oil_pct", 
        "rainfall_annual_mean_mm",
        "rainfall_extreme_days_avg",
    ]
    
    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if len(feature_cols) < 2:
        # Fallback to available numeric columns
        exclude = {"flood_events_total", "flood_deaths", "flood_risk_label", 
                   "compound_risk_score", "flood_impact_score"}
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                       if c not in exclude][:5]
    
    print(f"Using features: {feature_cols}")
    
    X = df[feature_cols].fillna(0)
    
    # Create binary target: high risk if above median flood events
    median_floods = df["flood_events_total"].median()
    y = (df["flood_events_total"] > median_floods).astype(int)
    
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

def train_model(X, y, feature_names):
    """Train XGBoost with proper regularization."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost with STRONG regularization to prevent overfitting
    model = xgb.XGBClassifier(
        n_estimators=50,           # Fewer trees
        max_depth=3,               # Shallow trees
        learning_rate=0.1,         # Moderate learning rate
        min_child_weight=5,        # Higher = more regularization
        subsample=0.7,             # Row sampling
        colsample_bytree=0.7,      # Column sampling
        reg_alpha=1.0,             # L1 regularization
        reg_lambda=2.0,            # L2 regularization
        gamma=0.5,                 # Min loss reduction
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    
    # Train
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }
    
    print("\n" + "="*50)
    print("TEST SET METRICS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Cross-validation for more reliable estimate
    print("\n" + "="*50)
    print("5-FOLD CROSS-VALIDATION")
    print("="*50)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Need to use full data for CV
    X_full_scaled = scaler.fit_transform(X)
    
    cv_accuracy = cross_val_score(model, X_full_scaled, y, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(model, X_full_scaled, y, cv=cv, scoring="f1")
    cv_roc = cross_val_score(model, X_full_scaled, y, cv=cv, scoring="roc_auc")
    
    cv_results = {
        "accuracy": {"mean": cv_accuracy.mean(), "std": cv_accuracy.std(), "scores": cv_accuracy.tolist()},
        "f1": {"mean": cv_f1.mean(), "std": cv_f1.std(), "scores": cv_f1.tolist()},
        "roc_auc": {"mean": cv_roc.mean(), "std": cv_roc.std(), "scores": cv_roc.tolist()},
    }
    
    print(f"  Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
    print(f"  F1 Score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"  ROC-AUC:  {cv_roc.mean():.4f} (+/- {cv_roc.std():.4f})")
    
    # Confusion matrix
    print("\n" + "="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    for _, row in importance.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")
    
    return model, scaler, metrics, cv_results, importance

def save_model(model, scaler, feature_names, metrics, cv_results, importance):
    """Save model and metadata."""
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"flood_risk_xgboost_{timestamp}"
    
    # Save model
    with open(models_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open(models_dir / f"{name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        "model_type": "xgboost",
        "feature_names": feature_names,
        "created_at": datetime.now().isoformat(),
        "regularization": {
            "max_depth": 3,
            "min_child_weight": 5,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "gamma": 0.5,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
        },
        "training_history": {
            "metrics": metrics,
            "cv_results": cv_results,
            "feature_importance": importance.to_dict(orient="records"),
        }
    }
    
    with open(models_dir / f"{name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to: {models_dir / name}.pkl")
    return name

def main():
    print("="*50)
    print("RETRAINING MODEL WITH REGULARIZATION")
    print("="*50)
    
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} records")
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Train model
    model, scaler, metrics, cv_results, importance = train_model(X, y, feature_names)
    
    # Save
    name = save_model(model, scaler, feature_names, metrics, cv_results, importance)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Model is NOT overfitting if CV scores are close to test scores")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"CV Accuracy:   {cv_results['accuracy']['mean']:.4f} (+/- {cv_results['accuracy']['std']:.4f})")
    print("\nIf test >> CV, model is overfitting")
    print("If test ≈ CV, model is generalizing well")

if __name__ == "__main__":
    main()

