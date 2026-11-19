from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass
class ModelRun:
    name: str
    task_type: str
    pipeline: Any
    metrics: dict[str, Any]
    y_test: pd.Series
    y_pred: np.ndarray
    X_test: pd.DataFrame
    feature_columns: list[str]


class MLService:
    """Train and evaluate supported scikit-learn models."""

    CLASSIFIERS = {
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=5, random_state=42
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "Extra Trees Classifier": ExtraTreesClassifier(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
        "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
        "K-NN Classifier": KNeighborsClassifier(),
        "SVM Classifier": SVC(kernel="rbf", probability=True, random_state=42),
        "Neural Network Classifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=600,
            early_stopping=True,
            random_state=42,
        ),
    }

    REGRESSORS = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(max_iter=5000),
        "ElasticNet Regression": ElasticNet(max_iter=5000),
        "Decision Tree Regression": DecisionTreeRegressor(
            max_depth=5, min_samples_split=5, random_state=42
        ),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "Extra Trees Regression": ExtraTreesRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regression": GradientBoostingRegressor(random_state=42),
        "AdaBoost Regression": AdaBoostRegressor(random_state=42),
        "K-NN Regression": KNeighborsRegressor(),
        "SVM Regression": SVR(),
        "Neural Network Regression": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=600,
            early_stopping=True,
            random_state=42,
        ),
    }

    @classmethod
    def get_model_names(cls, task_type: str) -> list[str]:
        if task_type == "classification":
            return list(cls.CLASSIFIERS)
        if task_type == "regression":
            return list(cls.REGRESSORS)
        raise ValueError("Task type must be 'classification' or 'regression'.")

    @staticmethod
    def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
        if X.shape[1] == 0:
            raise ValueError("The dataset must contain at least one feature column.")

        numeric_features = X.select_dtypes(include="number").columns.tolist()
        categorical_features = X.select_dtypes(exclude="number").columns.tolist()
        transformers = []

        if numeric_features:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                )
            )

        if categorical_features:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    categorical_features,
                )
            )

        if not transformers:
            raise ValueError("No valid feature columns were found.")

        return ColumnTransformer(transformers=transformers, remainder="drop")

    @classmethod
    def _prepare_data(
        cls,
        df: pd.DataFrame,
        target_name: str,
        task_type: str,
        test_size: float,
    ):
        if target_name not in df.columns:
            raise ValueError("Target column not found in dataset.")
        if task_type not in {"classification", "regression"}:
            raise ValueError("Task type must be 'classification' or 'regression'.")
        if not math.isfinite(test_size) or not 0.1 <= test_size <= 0.5:
            raise ValueError("Test size must be between 0.1 and 0.5.")

        clean_df = df.dropna(subset=[target_name]).copy()
        if clean_df.empty:
            raise ValueError("No rows with a valid target value are available.")

        X = clean_df.drop(columns=[target_name]).copy()
        y = clean_df[target_name].copy()

        if task_type == "regression":
            y = pd.to_numeric(y, errors="coerce")
            valid_rows = y.notna()
            X = X.loc[valid_rows].copy()
            y = y.loc[valid_rows].copy()
            if y.empty:
                raise ValueError(
                    "No valid numeric target values are available for regression."
                )

        if len(X) < 3:
            raise ValueError("At least 3 usable rows are needed to train a model.")

        stratify = None
        if task_type == "classification":
            if y.nunique() < 2:
                raise ValueError("Classification requires at least two target classes.")
            class_counts = y.value_counts()
            class_count = len(class_counts)
            test_rows = math.ceil(len(y) * test_size)
            train_rows = len(y) - test_rows
            if (
                class_counts.min() >= 2
                and test_rows >= class_count
                and train_rows >= class_count
            ):
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
        return X, X_train, X_test, y_train, y_test

    @classmethod
    def train_models(
        cls,
        df: pd.DataFrame,
        target_name: str,
        task_type: str,
        test_size: float = 0.2,
        model_names: list[str] | None = None,
    ) -> tuple[dict[str, ModelRun], dict[str, str]]:
        """Train requested model(s), returning successes and per-model errors."""
        X, X_train, X_test, y_train, y_test = cls._prepare_data(
            df, target_name, task_type, test_size
        )
        registry = cls.CLASSIFIERS if task_type == "classification" else cls.REGRESSORS
        requested_names = list(registry) if model_names is None else model_names
        preprocessor = cls._build_preprocessor(X)
        results: dict[str, ModelRun] = {}
        errors: dict[str, str] = {}

        for model_name in requested_names:
            estimator = registry.get(model_name)
            if estimator is None:
                errors[model_name] = "This model is not available for the selected task."
                continue
            try:
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", clone(preprocessor)),
                        ("model", clone(estimator)),
                    ]
                )
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                if task_type == "classification":
                    report_dict = classification_report(
                        y_test, y_pred, output_dict=True, zero_division=0
                    )
                    metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "f1_weighted": report_dict["weighted avg"]["f1-score"],
                        "report": classification_report(
                            y_test, y_pred, zero_division=0
                        ),
                    }
                else:
                    metrics = {
                        "mse": mean_squared_error(y_test, y_pred),
                        "mae": mean_absolute_error(y_test, y_pred),
                        "r2": r2_score(y_test, y_pred),
                        "explained_variance": explained_variance_score(y_test, y_pred),
                    }

                results[model_name] = ModelRun(
                    name=model_name,
                    task_type=task_type,
                    pipeline=pipeline,
                    metrics=metrics,
                    y_test=y_test,
                    y_pred=np.asarray(y_pred),
                    X_test=X_test,
                    feature_columns=X.columns.tolist(),
                )
            except Exception as exc:
                errors[model_name] = str(exc)

        return results, errors

    @classmethod
    def train_one(
        cls,
        df: pd.DataFrame,
        target_name: str,
        task_type: str,
        test_size: float,
        model_name: str,
    ) -> ModelRun:
        results, errors = cls.train_models(
            df, target_name, task_type, test_size, model_names=[model_name]
        )
        if model_name in results:
            return results[model_name]
        raise ValueError(errors.get(model_name, "The selected model could not be trained."))

    @classmethod
    def train_all(
        cls,
        df: pd.DataFrame,
        target_name: str,
        task_type: str,
        test_size: float = 0.2,
    ) -> dict[str, ModelRun]:
        results, errors = cls.train_models(df, target_name, task_type, test_size)
        if not results:
            details = "\n".join(f"{name}: {message}" for name, message in errors.items())
            raise ValueError(details or "No models could be trained.")
        return results

    @staticmethod
    def predict(model_run: ModelRun, input_row: dict[str, Any]) -> Any:
        input_df = pd.DataFrame([input_row], columns=model_run.feature_columns)
        return model_run.pipeline.predict(input_df)[0]

    @staticmethod
    def format_result_text(model_run: ModelRun) -> str:
        if model_run.task_type == "classification":
            return (
                f"Accuracy: {model_run.metrics['accuracy'] * 100:.2f}%\n"
                f"Weighted F1-score: {model_run.metrics['f1_weighted']:.4f}\n\n"
                f"{model_run.metrics['report']}"
            )
        return (
            f"MSE: {model_run.metrics['mse']:.6f}\n"
            f"MAE: {model_run.metrics['mae']:.6f}\n"
            f"R²: {model_run.metrics['r2']:.6f}\n"
            f"Explained Variance: {model_run.metrics['explained_variance']:.6f}\n"
        )
