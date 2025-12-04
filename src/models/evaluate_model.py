"""
Model evaluation and interpretability for SawitFlood Lab.

This module handles:
1. Model performance evaluation
2. SHAP-based interpretability
3. Risk typology analysis
4. Scenario analysis

Usage:
    python src/models/evaluate_model.py
    python src/models/evaluate_model.py --model flood_risk_xgboost_latest
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ModelEvaluator:
    """Evaluate and interpret flood risk models."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize ModelEvaluator.

        Args:
            config_path: Path to settings.yaml
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "configs" / "settings.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.processed_dir = PROJECT_ROOT / "data" / "processed"
        self.models_dir = PROJECT_ROOT / "models"
        self.output_dir = PROJECT_ROOT / "outputs"
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"

        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = None
        self.feature_names = []

        # Setup logging
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        logger.add(log_dir / "evaluate.log", rotation="10 MB", level="INFO")

        logger.info("Initialized ModelEvaluator")

    def load_model(self, model_name: str) -> None:
        """
        Load a trained model.

        Args:
            model_name: Name of the model (without extension)
        """
        model_path = self.models_dir / f"{model_name}.pkl"
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
        metadata_path = self.models_dir / f"{model_name}_metadata.json"

        if not model_path.exists():
            # Try to find the latest model (exclude scaler files)
            model_files = [
                f for f in self.models_dir.glob("flood_risk_*.pkl")
                if "_scaler" not in f.name
            ]
            if model_files:
                model_path = sorted(model_files)[-1]
                model_name = model_path.stem
                scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
                metadata_path = self.models_dir / f"{model_name}_metadata.json"
            else:
                raise FileNotFoundError(f"No model found: {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                self.feature_names = metadata.get("feature_names", [])
                self.model_type = metadata.get("model_type", "unknown")

        logger.info(f"Loaded model: {model_path}")

    def load_test_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load test data for evaluation.

        Returns:
            Tuple of (features, labels)
        """
        parquet_path = self.processed_dir / "analysis_dataset.parquet"
        csv_path = self.processed_dir / "analysis_dataset.csv"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError("Dataset not found. Run build_dataset.py first.")

        X = df[self.feature_names]
        y = df["flood_risk_label"]

        return X, y

    def evaluate_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model performance...")

        # Handle missing values
        X = X.fillna(X.median())

        # Predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_pred_proba),
        }

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)

        # Classification report
        report = classification_report(y, y_pred, output_dict=True)

        results = {
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": report,
            "y_true": y.values,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

        logger.info("Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return results

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"],
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Flood Risk Classification - Confusion Matrix")

        plt.tight_layout()

        if save_path is None:
            save_path = self.figures_dir / "confusion_matrix.png"

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")

        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, color="#1976d2", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")

        ax.fill_between(fpr, tpr, alpha=0.2, color="#1976d2")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Flood Risk Classification - ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.figures_dir / "roc_curve.png"

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved ROC curve to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        top_n: int = 15,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot feature importance.

        Args:
            top_n: Number of top features to show
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        else:
            raise ValueError("Model doesn't have feature importance")

        importance_df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": importance,
                }
            )
            .sort_values("importance", ascending=True)
            .tail(top_n)
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.RdYlGn(importance_df["importance"] / importance_df["importance"].max())

        bars = ax.barh(importance_df["feature"], importance_df["importance"], color=colors)

        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Feature Importance - Flood Risk Model")
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.figures_dir / "feature_importance.png"

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved feature importance to {save_path}")

        return fig

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        sample_size: int | None = 100,
    ) -> np.ndarray | None:
        """
        Compute SHAP values for model interpretability.

        Args:
            X: Features
            sample_size: Number of samples to use (None for all)

        Returns:
            SHAP values array
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            return None

        logger.info("Computing SHAP values...")

        # Handle missing values
        X = X.fillna(X.median())

        # Sample if needed
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X

        # Create explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        # Store for later use
        self.shap_values = shap_values
        self.shap_X = X_sample

        logger.info(f"Computed SHAP values for {len(X_sample)} samples")
        return shap_values

    def plot_shap_summary(
        self,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot SHAP summary plot.

        Args:
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not SHAP_AVAILABLE or not hasattr(self, "shap_values"):
            logger.warning("SHAP values not computed. Run compute_shap_values first.")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Handle binary classification SHAP values
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # Use class 1 (high risk)

        shap.summary_plot(
            shap_vals,
            self.shap_X,
            plot_type="bar",
            show=False,
        )

        plt.title("SHAP Feature Importance - Flood Risk Model")
        plt.tight_layout()

        if save_path is None:
            save_path = self.figures_dir / "shap_summary.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved SHAP summary to {save_path}")

        return fig

    def plot_shap_dependence(
        self,
        feature: str,
        interaction_feature: str | None = None,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot SHAP dependence plot for a feature.

        Args:
            feature: Feature name to plot
            interaction_feature: Feature to show interaction with
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not SHAP_AVAILABLE or not hasattr(self, "shap_values"):
            logger.warning("SHAP values not computed.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap.dependence_plot(
            feature,
            shap_vals,
            self.shap_X,
            interaction_index=interaction_feature,
            show=False,
            ax=ax,
        )

        ax.set_title(f"SHAP Dependence: {feature}")
        plt.tight_layout()

        if save_path is None:
            feature_clean = feature.replace(" ", "_").replace("/", "_")
            save_path = self.figures_dir / f"shap_dependence_{feature_clean}.png"

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved SHAP dependence plot to {save_path}")

        return fig

    def create_risk_typology(
        self,
        X: pd.DataFrame,
        n_clusters: int = 5,
    ) -> pd.DataFrame:
        """
        Create risk typology using clustering.

        Args:
            X: Features
            n_clusters: Number of clusters

        Returns:
            DataFrame with cluster assignments and descriptions
        """
        logger.info(f"Creating risk typology with {n_clusters} clusters...")

        # Select key features for clustering
        cluster_features = self.config["clustering"]["features"]
        available_features = [f for f in cluster_features if f in X.columns]

        if len(available_features) < 2:
            logger.warning("Not enough clustering features. Using all features.")
            available_features = X.columns.tolist()[:5]

        X_cluster = X[available_features].fillna(X[available_features].median())

        # Standardize
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Create typology descriptions
        result = X.copy()
        result["cluster"] = clusters

        # Get predictions
        X_filled = X.fillna(X.median())
        result["predicted_risk"] = self.model.predict(X_filled)
        result["risk_probability"] = self.model.predict_proba(X_filled)[:, 1]

        # Summarize clusters
        cluster_summary = (
            result.groupby("cluster").agg(dict.fromkeys(available_features, "mean")).round(2)
        )

        cluster_summary["risk_probability_mean"] = result.groupby("cluster")[
            "risk_probability"
        ].mean()
        cluster_summary["count"] = result.groupby("cluster").size()

        # Assign typology names
        typology_names = self._assign_typology_names(cluster_summary, available_features)
        result["typology"] = result["cluster"].map(typology_names)

        logger.info("Risk Typology Summary:")
        for cluster_id, name in typology_names.items():
            count = (result["cluster"] == cluster_id).sum()
            risk = result[result["cluster"] == cluster_id]["risk_probability"].mean()
            logger.info(f"  {name}: {count} zones, avg risk: {risk:.2f}")

        return result

    def _assign_typology_names(
        self,
        cluster_summary: pd.DataFrame,
        features: list[str],
    ) -> dict[int, str]:
        """
        Assign descriptive names to clusters.

        Args:
            cluster_summary: Cluster statistics
            features: Feature names used

        Returns:
            Dictionary mapping cluster ID to name
        """
        names = {}

        # Sort by risk probability
        sorted_clusters = cluster_summary.sort_values("risk_probability_mean", ascending=False)

        risk_levels = ["Sangat Tinggi", "Tinggi", "Sedang", "Rendah", "Sangat Rendah"]

        for i, (cluster_id, row) in enumerate(sorted_clusters.iterrows()):
            if i < len(risk_levels):
                risk_level = risk_levels[i]
            else:
                risk_level = f"Level {i + 1}"

            # Add description based on features
            if "forest_loss_cumulative_pct" in features:
                forest_loss = row.get("forest_loss_cumulative_pct", 0)
                if forest_loss > 30:
                    forest_desc = "Deforestasi Tinggi"
                elif forest_loss > 15:
                    forest_desc = "Deforestasi Sedang"
                else:
                    forest_desc = "Hutan Relatif Utuh"
            else:
                forest_desc = ""

            names[cluster_id] = f"Cluster {cluster_id}: {risk_level} ({forest_desc})"

        return names

    def scenario_analysis(
        self,
        X: pd.DataFrame,
        feature_to_change: str,
        change_pct: float = 20.0,
    ) -> dict[str, Any]:
        """
        Perform scenario analysis by modifying features.

        Args:
            X: Base features
            feature_to_change: Feature to modify
            change_pct: Percentage change to apply

        Returns:
            Dictionary with scenario results
        """
        logger.info(f"Running scenario analysis: {feature_to_change} +{change_pct}%")

        X_filled = X.fillna(X.median())

        # Baseline predictions
        baseline_proba = self.model.predict_proba(X_filled)[:, 1]
        baseline_high_risk = (baseline_proba >= 0.5).sum()

        # Modified scenario
        X_modified = X_filled.copy()
        X_modified[feature_to_change] = X_modified[feature_to_change] * (1 + change_pct / 100)

        scenario_proba = self.model.predict_proba(X_modified)[:, 1]
        scenario_high_risk = (scenario_proba >= 0.5).sum()

        results = {
            "feature": feature_to_change,
            "change_pct": change_pct,
            "baseline": {
                "high_risk_zones": int(baseline_high_risk),
                "mean_risk_probability": float(baseline_proba.mean()),
            },
            "scenario": {
                "high_risk_zones": int(scenario_high_risk),
                "mean_risk_probability": float(scenario_proba.mean()),
            },
            "impact": {
                "additional_high_risk_zones": int(scenario_high_risk - baseline_high_risk),
                "risk_probability_change": float(scenario_proba.mean() - baseline_proba.mean()),
            },
        }

        logger.info("Scenario Results:")
        logger.info(f"  Baseline high-risk zones: {baseline_high_risk}")
        logger.info(f"  Scenario high-risk zones: {scenario_high_risk}")
        logger.info(f"  Additional high-risk zones: {scenario_high_risk - baseline_high_risk}")

        return results

    def generate_report(
        self,
        results: dict[str, Any],
        output_path: Path | None = None,
    ) -> Path:
        """
        Generate evaluation report.

        Args:
            results: Evaluation results
            output_path: Path to save report

        Returns:
            Path to saved report
        """
        logger.info("Generating evaluation report...")

        if output_path is None:
            output_path = self.reports_dir / "evaluation_report.md"

        report_content = f"""# SawitFlood Lab - Model Evaluation Report

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | {results["metrics"]["accuracy"]:.4f} |
| Precision | {results["metrics"]["precision"]:.4f} |
| Recall | {results["metrics"]["recall"]:.4f} |
| F1-Score | {results["metrics"]["f1_score"]:.4f} |
| ROC-AUC | {results["metrics"]["roc_auc"]:.4f} |

## Confusion Matrix

```
{results["confusion_matrix"]}
```

## Key Findings

### Classification Performance
- The model achieves an F1-score of {results["metrics"]["f1_score"]:.4f}, {"meeting" if results["metrics"]["f1_score"] >= 0.75 else "below"} the target of 0.75.
- ROC-AUC of {results["metrics"]["roc_auc"]:.4f} indicates {"good" if results["metrics"]["roc_auc"] >= 0.80 else "moderate"} discrimination ability.

### Feature Importance
See `feature_importance.png` for visualization of the most important features.

### SHAP Analysis
See `shap_summary.png` for SHAP-based feature importance.

## Recommendations

1. Features with highest impact on flood risk prediction:
   - Examine top features from importance plot
   - These represent key intervention points for policy

2. Areas for model improvement:
   - Consider adding more granular spatial features
   - Incorporate temporal dynamics (seasonal patterns)

3. Deployment considerations:
   - Model can be used for preliminary risk screening
   - Human review recommended for high-stakes decisions

## Limitations

1. **Correlation vs Causation**: Model shows associations, not causal relationships.
2. **Data Quality**: Flood reporting may be inconsistent across regions.
3. **Temporal Coverage**: Model based on historical data may not capture emerging patterns.
"""

        with open(output_path, "w") as f:
            f.write(report_content)

        logger.info(f"Saved evaluation report to {output_path}")
        return output_path

    def run_full_evaluation(self, model_name: str | None = None) -> dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Args:
            model_name: Name of model to evaluate

        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Running full evaluation pipeline...")

        # Load model
        if model_name:
            self.load_model(model_name)
        else:
            # Find latest model (exclude scaler files)
            model_files = [
                f for f in self.models_dir.glob("flood_risk_*.pkl")
                if "_scaler" not in f.name
            ]
            if not model_files:
                raise FileNotFoundError("No trained models found. Run train_model.py first.")
            latest_model = sorted(model_files)[-1]
            self.load_model(latest_model.stem)

        # Load data
        X, y = self.load_test_data()

        # Evaluate performance
        results = self.evaluate_performance(X, y)

        # Generate plots
        self.plot_confusion_matrix(results["confusion_matrix"])
        self.plot_roc_curve(results["y_true"], results["y_pred_proba"])
        self.plot_feature_importance()

        # SHAP analysis
        if SHAP_AVAILABLE:
            self.compute_shap_values(X)
            self.plot_shap_summary()

        # Risk typology
        typology_results = self.create_risk_typology(X)

        # Scenario analysis - use a feature that exists in the dataset
        scenario_feature = "flood_events_total" if "flood_events_total" in X.columns else X.columns[0]
        scenario_results = self.scenario_analysis(X, scenario_feature, change_pct=20)

        # Generate report
        report_path = self.generate_report(results)

        results["typology"] = typology_results
        results["scenario"] = scenario_results
        results["report_path"] = report_path

        logger.info("Evaluation complete!")
        return results


def main():
    """Main entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate flood risk model")
    parser.add_argument("--model", type=str, default=None, help="Model name to evaluate")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    args = parser.parse_args()

    evaluator = ModelEvaluator(config_path=args.config)
    results = evaluator.run_full_evaluation(model_name=args.model)

    print("\nEvaluation complete!")
    print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    print(f"ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"Report saved to: {results['report_path']}")


if __name__ == "__main__":
    main()

