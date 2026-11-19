from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve

from app.config import CLASSIFICATION_RESULTS_DIR, REGRESSION_RESULTS_DIR
from app.services.ml_service import ModelRun

sns.set_style("whitegrid")


class PlotService:
    @staticmethod
    def _slugify(text: str) -> str:
        return (
            text.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )

    @staticmethod
    def _save_figure(fig: plt.Figure, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")

    @staticmethod
    def _empty_axis(ax: plt.Axes, title: str, message: str) -> None:
        ax.set_title(title)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=11)

    @staticmethod
    def _clean_learning_data(model_run: ModelRun, full_X: pd.DataFrame, full_y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        if model_run.task_type == "regression":
            y = pd.to_numeric(full_y, errors="coerce")
            valid_mask = y.notna()
            return full_X.loc[valid_mask].copy(), y.loc[valid_mask].copy()

        valid_mask = full_y.notna()
        return full_X.loc[valid_mask].copy(), full_y.loc[valid_mask].copy()

    @staticmethod
    def _get_cv_folds(y: pd.Series, task_type: str) -> int | None:
        if len(y) < 3:
            return None

        if task_type == "classification":
            class_counts = y.value_counts()
            if class_counts.empty or class_counts.min() < 2:
                return None
            return min(5, int(class_counts.min()))

        if len(y) >= 5:
            return 5
        return 3 if len(y) >= 3 else None

    @staticmethod
    def create_overview_plot(
        df: pd.DataFrame,
        target_name: str,
        task_type: str,
        output_path: Path,
    ) -> tuple[plt.Figure, Path]:
        if target_name not in df.columns:
            raise ValueError("Target column not found in dataset.")

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        feature_df = df.drop(columns=[target_name])
        numeric_features = feature_df.select_dtypes(include="number").columns.tolist()
        target_series = df[target_name]

        if task_type == "classification":
            sns.countplot(
                x=target_series.astype(str),
                ax=axes[0, 0],
                hue=target_series.astype(str),
                legend=False,
            )
            axes[0, 0].set_title(f"Target Class Distribution: {target_name}")
            axes[0, 0].tick_params(axis="x", rotation=30)

            value_counts = target_series.astype(str).value_counts()
            if len(value_counts) <= 8:
                axes[0, 1].pie(
                    value_counts.values,
                    labels=value_counts.index.tolist(),
                    autopct="%1.1f%%",
                    startangle=140,
                )
                axes[0, 1].set_title(f"Class Share: {target_name}")
            else:
                top_counts = value_counts.head(8)
                sns.barplot(
                    x=top_counts.index,
                    y=top_counts.values,
                    ax=axes[0, 1],
                    hue=top_counts.index,
                    legend=False,
                )
                axes[0, 1].set_title(f"Top Classes: {target_name}")
                axes[0, 1].tick_params(axis="x", rotation=30)

            if len(numeric_features) >= 2:
                sns.scatterplot(
                    data=df,
                    x=numeric_features[0],
                    y=numeric_features[1],
                    hue=target_series.astype(str),
                    ax=axes[1, 0],
                )
                axes[1, 0].set_title(f"{numeric_features[0]} vs {numeric_features[1]}")
            else:
                PlotService._empty_axis(
                    axes[1, 0],
                    "Feature Scatter Plot",
                    "At least 2 numeric feature columns are needed.",
                )

            if len(numeric_features) >= 2:
                corr_cols = numeric_features[: min(8, len(numeric_features))]
                corr = df[corr_cols].corr(numeric_only=True)
                sns.heatmap(corr, annot=True, cmap="Blues", ax=axes[1, 1])
                axes[1, 1].set_title("Numeric Feature Correlation")
            elif len(numeric_features) == 1:
                sns.histplot(df[numeric_features[0]], kde=True, ax=axes[1, 1], color="purple")
                axes[1, 1].set_title(f"Distribution: {numeric_features[0]}")
            else:
                PlotService._empty_axis(
                    axes[1, 1],
                    "Correlation / Distribution",
                    "No numeric feature columns were found.",
                )

        else:
            target_numeric = pd.to_numeric(target_series, errors="coerce")

            sns.histplot(target_numeric.dropna(), kde=True, ax=axes[0, 0], color="teal")
            axes[0, 0].set_title(f"Target Distribution: {target_name}")

            if numeric_features:
                sns.scatterplot(
                    data=df,
                    x=numeric_features[0],
                    y=target_numeric,
                    ax=axes[0, 1],
                    color="darkorange",
                )
                axes[0, 1].set_title(f"{numeric_features[0]} vs {target_name}")
            else:
                PlotService._empty_axis(
                    axes[0, 1],
                    "Feature vs Target",
                    "No numeric feature columns were found.",
                )

            corr_candidates = numeric_features + [target_name]
            numeric_df = df[corr_candidates].copy()
            numeric_df[target_name] = target_numeric

            if numeric_df.shape[1] >= 2:
                corr = numeric_df.corr(numeric_only=True)
                sns.heatmap(corr, annot=True, cmap="viridis", ax=axes[1, 0])
                axes[1, 0].set_title("Correlation Matrix")
            else:
                PlotService._empty_axis(
                    axes[1, 0],
                    "Correlation Matrix",
                    "Not enough numeric columns to build a correlation matrix.",
                )

            if numeric_features:
                sns.boxplot(y=target_numeric, ax=axes[1, 1], color="mediumpurple")
                axes[1, 1].set_title(f"Target Boxplot: {target_name}")
            else:
                PlotService._empty_axis(
                    axes[1, 1],
                    "Target Boxplot",
                    "No numeric feature columns were found.",
                )

        PlotService._save_figure(fig, output_path)
        return fig, output_path

    @staticmethod
    def create_model_plots(model_run: ModelRun, full_X: pd.DataFrame, full_y: pd.Series) -> list[Path]:
        if model_run.task_type == "classification":
            return PlotService.create_classification_plots(model_run, full_X, full_y)
        return PlotService.create_regression_plots(model_run, full_X, full_y)

    @staticmethod
    def create_classification_plots(
        model_run: ModelRun,
        full_X: pd.DataFrame,
        full_y: pd.Series,
    ) -> list[Path]:
        model_dir = CLASSIFICATION_RESULTS_DIR / PlotService._slugify(model_run.name)
        model_dir.mkdir(parents=True, exist_ok=True)

        paths = [
            model_dir / "plot1_confusion_matrix.png",
            model_dir / "plot2_prediction_counts.png",
            model_dir / "plot3_learning_curve.png",
            model_dir / "plot4_report_heatmap.png",
        ]

        y_test = pd.Series(model_run.y_test)
        y_pred = pd.Series(model_run.y_pred)

        # Plot 1: confusion matrix
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        conf_mat = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", ax=ax1)
        ax1.set_title(f"Confusion Matrix - {model_run.name}")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        PlotService._save_figure(fig1, paths[0])
        plt.close(fig1)

        # Plot 2: actual vs predicted counts
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        labels = pd.concat([y_test.astype(str), y_pred.astype(str)]).drop_duplicates().tolist()
        true_counts = y_test.astype(str).value_counts().reindex(labels, fill_value=0)
        pred_counts = y_pred.astype(str).value_counts().reindex(labels, fill_value=0)

        x = np.arange(len(labels))
        width = 0.35
        ax2.bar(x - width / 2, true_counts.values, width=width, label="Actual")
        ax2.bar(x + width / 2, pred_counts.values, width=width, label="Predicted")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=30)
        ax2.set_title(f"Actual vs Predicted Counts - {model_run.name}")
        ax2.legend()
        PlotService._save_figure(fig2, paths[1])
        plt.close(fig2)

        # Plot 3: learning curve
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        X_curve, y_curve = PlotService._clean_learning_data(model_run, full_X, full_y)
        cv = PlotService._get_cv_folds(y_curve, "classification")

        if cv is None:
            PlotService._empty_axis(
                ax3,
                "Learning Curve",
                "Not enough class samples are available to compute a learning curve.",
            )
        else:
            train_sizes, train_scores, test_scores = learning_curve(
                model_run.pipeline,
                X_curve,
                y_curve,
                train_sizes=np.linspace(0.2, 1.0, 5),
                cv=cv,
                n_jobs=1,
            )

            ax3.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Training Score")
            ax3.plot(train_sizes, test_scores.mean(axis=1), marker="o", label="Validation Score")
            ax3.set_title(f"Learning Curve - {model_run.name}")
            ax3.set_xlabel("Training Set Size")
            ax3.set_ylabel("Score")
            ax3.legend()

        PlotService._save_figure(fig3, paths[2])
        plt.close(fig3)

        # Plot 4: classification report heatmap
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).T
        report_df = report_df.drop(index=["accuracy"], errors="ignore")

        metric_cols = [col for col in ["precision", "recall", "f1-score"] if col in report_df.columns]
        sns.heatmap(report_df[metric_cols], annot=True, cmap="YlGnBu", ax=ax4)
        ax4.set_title(f"Classification Report Heatmap - {model_run.name}")
        PlotService._save_figure(fig4, paths[3])
        plt.close(fig4)

        return paths

    @staticmethod
    def create_regression_plots(
        model_run: ModelRun,
        full_X: pd.DataFrame,
        full_y: pd.Series,
    ) -> list[Path]:
        model_dir = REGRESSION_RESULTS_DIR / PlotService._slugify(model_run.name)
        model_dir.mkdir(parents=True, exist_ok=True)

        paths = [
            model_dir / "plot1_actual_vs_predicted.png",
            model_dir / "plot2_residuals.png",
            model_dir / "plot3_learning_curve.png",
            model_dir / "plot4_feature_importance.png",
        ]

        y_test = pd.Series(model_run.y_test)
        y_pred = pd.Series(model_run.y_pred)

        # Plot 1: actual vs predicted
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(y_test, y_pred, alpha=0.7, color="royalblue")
        if len(y_test) > 1:
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax1.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red")
        ax1.set_title(f"Actual vs Predicted - {model_run.name}")
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        PlotService._save_figure(fig1, paths[0])
        plt.close(fig1)

        # Plot 2: residuals
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.7, color="darkorange")
        ax2.axhline(0, linestyle="--", color="red")
        ax2.set_title(f"Residual Plot - {model_run.name}")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residuals")
        PlotService._save_figure(fig2, paths[1])
        plt.close(fig2)

        # Plot 3: learning curve
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        X_curve, y_curve = PlotService._clean_learning_data(model_run, full_X, full_y)
        cv = PlotService._get_cv_folds(y_curve, "regression")

        if cv is None:
            PlotService._empty_axis(
                ax3,
                "Learning Curve",
                "Not enough numeric samples are available to compute a learning curve.",
            )
        else:
            train_sizes, train_scores, test_scores = learning_curve(
                model_run.pipeline,
                X_curve,
                y_curve,
                train_sizes=np.linspace(0.2, 1.0, 5),
                cv=cv,
                n_jobs=1,
            )

            ax3.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Training Score")
            ax3.plot(train_sizes, test_scores.mean(axis=1), marker="o", label="Validation Score")
            ax3.set_title(f"Learning Curve - {model_run.name}")
            ax3.set_xlabel("Training Set Size")
            ax3.set_ylabel("Score")
            ax3.legend()

        PlotService._save_figure(fig3, paths[2])
        plt.close(fig3)

        # Plot 4: feature importance / coefficients
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        preprocessor = model_run.pipeline.named_steps["preprocessor"]
        estimator = model_run.pipeline.named_steps["model"]

        try:
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(model_run.feature_columns))]

        importance_values = None

        if hasattr(estimator, "feature_importances_"):
            importance_values = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = np.ravel(estimator.coef_)
            importance_values = coef

        if importance_values is None:
            PlotService._empty_axis(
                ax4,
                "Feature Importance",
                "This model does not expose feature importance coefficients.",
            )
        else:
            importance_series = pd.Series(importance_values, index=feature_names)
            top_features = importance_series.reindex(
                importance_series.abs().sort_values(ascending=False).head(15).index
            )
            top_features.sort_values().plot(kind="barh", ax=ax4, color="seagreen")
            ax4.set_title(f"Feature Importance - {model_run.name}")
            ax4.set_xlabel("Importance")

        PlotService._save_figure(fig4, paths[3])
        plt.close(fig4)

        return paths