"""
Train flood risk classification model for SawitFlood Lab.

This module handles:
1. Model training (XGBoost, Random Forest)
2. Hyperparameter tuning
3. Feature importance analysis
4. Model persistence

Usage:
    python src/models/train_model.py
    python src/models/train_model.py --model xgboost --tune
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class FloodRiskModel:
    """Flood risk classification model."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize FloodRiskModel.

        Args:
            config_path: Path to settings.yaml
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "configs" / "settings.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.processed_dir = PROJECT_ROOT / "data" / "processed"
        self.models_dir = PROJECT_ROOT / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_history = {}

        # Setup logging
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        logger.add(log_dir / "training.log", rotation="10 MB", level="INFO")

        logger.info("Initialized FloodRiskModel")

    def load_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load analysis dataset.

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Loading dataset...")

        # Try different file formats
        parquet_path = self.processed_dir / "analysis_dataset.parquet"
        csv_path = self.processed_dir / "analysis_dataset.csv"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            logger.warning("Dataset not found. Building dataset...")
            from src.data.build_dataset import DatasetBuilder

            builder = DatasetBuilder()
            gdf = builder.build_analysis_dataset()
            df = gdf.drop(columns=["geometry"]) if "geometry" in gdf.columns else gdf

        # Get feature columns
        feature_cols = self._get_feature_columns(df)
        self.feature_names = feature_cols

        X = df[feature_cols]
        y = df["flood_risk_label"]

        logger.info(f"Loaded dataset with {len(X)} samples and {len(feature_cols)} features")
        return X, y

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Get feature columns from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            List of feature column names
        """
        # Exclude non-feature columns
        exclude_cols = {
            "geometry",
            "kabupaten_id",
            "id",
            "name",
            "province",
            "kabupaten",
            "flood_risk_label",
        }

        # Get numeric columns that aren't excluded
        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols
            and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]

        return feature_cols

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        scale: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion for test set
            scale: Whether to scale features

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training...")

        # Handle missing values
        X = X.fillna(X.median())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        else:
            X_train = X_train.values
            X_test = X_test.values

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Class distribution - Train: {np.bincount(y_train.values.astype(int))}")
        logger.info(f"Class distribution - Test: {np.bincount(y_test.values.astype(int))}")

        return X_train, X_test, y_train.values, y_test.values

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: dict | None = None,
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            params: Model parameters (uses config if None)

        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model...")

        if params is None:
            params = self.config["model"]["xgboost"]

        model = xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=params["random_state"],
            use_label_encoder=False,
            eval_metric="logloss",
        )

        model.fit(X_train, y_train)

        self.model = model
        self.model_type = "xgboost"

        logger.info("XGBoost model trained successfully")
        return model

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: dict | None = None,
    ) -> RandomForestClassifier:
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels
            params: Model parameters (uses config if None)

        Returns:
            Trained Random Forest model
        """
        logger.info("Training Random Forest model...")

        if params is None:
            params = self.config["model"]["random_forest"]

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=params["random_state"],
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        self.model = model
        self.model_type = "random_forest"

        logger.info("Random Forest model trained successfully")
        return model

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = "xgboost",
        n_iter: int = 20,
    ) -> dict[str, Any]:
        """
        Tune hyperparameters using randomized search.

        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model ('xgboost' or 'random_forest')
            n_iter: Number of iterations for random search

        Returns:
            Best parameters
        """
        logger.info(f"Tuning hyperparameters for {model_type}...")

        if model_type == "xgboost":
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            }
        else:
            model = RandomForestClassifier(n_jobs=-1)
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        search.fit(X_train, y_train)

        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV F1 score: {search.best_score_:.4f}")

        self.model = search.best_estimator_
        self.model_type = model_type

        return search.best_params_

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Labels
            cv: Number of folds

        Returns:
            Dictionary with CV scores
        """
        logger.info(f"Performing {cv}-fold cross-validation...")

        if self.model is None:
            raise ValueError("Model not trained. Call train_* method first.")

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        scores = {
            "accuracy": cross_val_score(self.model, X, y, cv=cv_strategy, scoring="accuracy"),
            "f1": cross_val_score(self.model, X, y, cv=cv_strategy, scoring="f1"),
            "roc_auc": cross_val_score(self.model, X, y, cv=cv_strategy, scoring="roc_auc"),
        }

        results = {
            metric: {
                "mean": values.mean(),
                "std": values.std(),
                "scores": values.tolist(),
            }
            for metric, values in scores.items()
        }

        logger.info("Cross-validation results:")
        for metric, values in results.items():
            logger.info(f"  {metric}: {values['mean']:.4f} (+/- {values['std']:.4f})")

        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        else:
            raise ValueError("Model doesn't have feature_importances_ attribute")

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

        return importance_df

    def explain_with_shap(
        self,
        X: np.ndarray,
        max_display: int = 10,
    ) -> Any | None:
        """
        Explain model predictions using SHAP.

        Args:
            X: Feature matrix
            max_display: Max features to display

        Returns:
            SHAP values object
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            return None

        logger.info("Calculating SHAP values...")

        if self.model_type == "xgboost":
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.TreeExplainer(self.model)

        shap_values = explainer.shap_values(X)

        logger.info("SHAP values calculated successfully")
        return shap_values

    def save_model(
        self,
        name: str | None = None,
        include_metadata: bool = True,
    ) -> Path:
        """
        Save trained model to disk.

        Args:
            name: Model name (auto-generated if None)
            include_metadata: Whether to save metadata

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save")

        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"flood_risk_{self.model_type}_{timestamp}"

        # Save model
        model_path = self.models_dir / f"{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Save scaler
        scaler_path = self.models_dir / f"{name}_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        # Save metadata
        if include_metadata:
            metadata = {
                "model_type": self.model_type,
                "feature_names": self.feature_names,
                "created_at": datetime.now().isoformat(),
                "training_history": self.training_history,
            }

            metadata_path = self.models_dir / f"{name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        return model_path

    def load_model(self, name: str) -> None:
        """
        Load a saved model.

        Args:
            name: Model name (without extension)
        """
        model_path = self.models_dir / f"{name}.pkl"
        scaler_path = self.models_dir / f"{name}_scaler.pkl"
        metadata_path = self.models_dir / f"{name}_metadata.json"

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                self.model_type = metadata.get("model_type")
                self.feature_names = metadata.get("feature_names", [])

        logger.info(f"Model loaded from {model_path}")

    def train_and_evaluate(
        self,
        model_type: str = "xgboost",
        tune: bool = False,
    ) -> dict[str, Any]:
        """
        Complete training and evaluation pipeline.

        Args:
            model_type: Type of model to train
            tune: Whether to tune hyperparameters

        Returns:
            Dictionary with training results
        """
        # Load data
        X, y = self.load_dataset()

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, scale=False)

        # Train model
        if tune:
            best_params = self.tune_hyperparameters(X_train, y_train, model_type)
        else:
            if model_type == "xgboost":
                self.train_xgboost(X_train, y_train)
            else:
                self.train_random_forest(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        # Cross-validation
        cv_results = self.cross_validate(
            np.vstack([X_train, X_test]), np.concatenate([y_train, y_test])
        )

        # Feature importance
        importance_df = self.get_feature_importance()

        # Store training history
        self.training_history = {
            "metrics": metrics,
            "cv_results": cv_results,
            "feature_importance": importance_df.to_dict(),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        # Save model
        model_path = self.save_model()

        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Model: {model_type}")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        logger.info("\nTop 10 Feature Importance:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        logger.info("=" * 60)

        return {
            "metrics": metrics,
            "cv_results": cv_results,
            "feature_importance": importance_df,
            "model_path": model_path,
        }


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(description="Train flood risk model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "random_forest"],
        default="xgboost",
        help="Model type to train",
    )
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters")
    parser.add_argument("--retrain", action="store_true", help="Retrain existing model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    args = parser.parse_args()

    trainer = FloodRiskModel(config_path=args.config)
    results = trainer.train_and_evaluate(model_type=args.model, tune=args.tune)

    print(f"\nModel saved to: {results['model_path']}")
    print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    print(f"ROC-AUC: {results['metrics']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
