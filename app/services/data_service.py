from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    Normalizer,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
)


class DataService:
    SUPPORTED_FILE_TYPES = [
        ("CSV Files", "*.csv"),
        ("Excel Files", "*.xlsx"),
        ("Text Files", "*.txt"),
        ("All Files", "*.*"),
    ]

    EXPORT_FILE_TYPES = [
        ("CSV Files", "*.csv"),
        ("Excel Files", "*.xlsx"),
        ("Text Files", "*.txt"),
    ]

    @staticmethod
    def load_dataset(file_path: str | Path) -> pd.DataFrame:
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == ".csv":
            df = pd.read_csv(path)
        elif extension == ".xlsx":
            df = pd.read_excel(path)
        elif extension == ".txt":
            df = pd.read_csv(path, sep="\t")
        else:
            raise ValueError("Unsupported file format.")

        if df.empty:
            raise ValueError("The selected dataset is empty.")

        return df

    @staticmethod
    def create_sample_template(file_path: str | Path) -> None:
        df = pd.DataFrame(
            {
                "feature_1": [1, 2, 3],
                "feature_2": ["A", "B", "C"],
                "target": [0, 1, 0],
            }
        )
        df.to_excel(file_path, index=False)

    @staticmethod
    def export_dataset(df: pd.DataFrame, file_path: str | Path) -> None:
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == ".csv":
            df.to_csv(path, index=False)
        elif extension == ".xlsx":
            df.to_excel(path, index=False)
        elif extension == ".txt":
            df.to_csv(path, index=False, sep="\t")
        else:
            raise ValueError("Unsupported export file format.")

    @staticmethod
    def dataset_info_text(df: pd.DataFrame) -> str:
        buffer = StringIO()
        df.info(buf=buffer)
        return buffer.getvalue()

    @staticmethod
    def parse_columns_text(columns_text: str) -> list[str]:
        return [item.strip() for item in columns_text.split(",") if item.strip()]

    @staticmethod
    def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Unknown columns: {', '.join(missing)}")

    @staticmethod
    def delete_columns(df: pd.DataFrame, columns_text: str) -> pd.DataFrame:
        columns = DataService.parse_columns_text(columns_text)
        if not columns:
            raise ValueError("Please enter at least one column name.")

        DataService._validate_columns(df, columns)
        return df.drop(columns=columns).copy()

    @staticmethod
    def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        duplicate_count = int(df.duplicated().sum())
        return df.drop_duplicates().copy(), duplicate_count

    @staticmethod
    def missing_values_summary(df: pd.DataFrame) -> pd.Series:
        summary = df.isna().sum()
        return summary[summary > 0]

    @staticmethod
    def handle_missing_by_method(df: pd.DataFrame, columns_text: str, method: str) -> pd.DataFrame:
        new_df = df.copy()
        columns = DataService.parse_columns_text(columns_text)

        if method != "Delete Row" and not columns:
            raise ValueError("Please enter at least one feature name.")

        if columns:
            DataService._validate_columns(new_df, columns)

        if method == "Delete Row":
            return new_df.dropna(subset=columns if columns else None).copy()

        if method == "Forward Fill":
            new_df[columns] = new_df[columns].ffill()
            return new_df

        if method == "Backward Fill":
            new_df[columns] = new_df[columns].bfill()
            return new_df

        reducer_map = {
            "Mean": np.nanmean,
            "Median": np.nanmedian,
            "Max": np.nanmax,
            "Min": np.nanmin,
        }

        if method not in reducer_map:
            raise ValueError("Unsupported missing-value handling method.")

        for col in columns:
            numeric_series = pd.to_numeric(new_df[col], errors="coerce")
            if numeric_series.notna().sum() == 0:
                raise ValueError(f"Column '{col}' is not numeric, so '{method}' cannot be applied.")

            fill_value = reducer_map[method](numeric_series.to_numpy(dtype=float))
            new_df[col] = numeric_series.fillna(fill_value)

        return new_df

    @staticmethod
    def handle_missing_by_value(df: pd.DataFrame, columns_text: str, value: str) -> pd.DataFrame:
        new_df = df.copy()
        columns = DataService.parse_columns_text(columns_text)

        if not columns:
            raise ValueError("Please enter at least one feature name.")
        if value == "":
            raise ValueError("Please enter a value.")

        DataService._validate_columns(new_df, columns)

        for col in columns:
            if pd.api.types.is_numeric_dtype(new_df[col]):
                try:
                    cast_value = float(value)
                except ValueError as exc:
                    raise ValueError(f"Column '{col}' is numeric; please enter a numeric value.") from exc
            else:
                cast_value = value

            new_df[col] = new_df[col].fillna(cast_value)

        return new_df

    @staticmethod
    def encode_replace(df: pd.DataFrame, column: str, old_value: str, new_value: str) -> pd.DataFrame:
        new_df = df.copy()

        if column not in new_df.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if old_value == "":
            raise ValueError("Please enter the current categorical value.")

        if pd.api.types.is_numeric_dtype(new_df[column]):
            try:
                new_value_cast = float(new_value)
            except ValueError:
                new_value_cast = new_value
        else:
            new_value_cast = new_value

        new_df[column] = new_df[column].replace(old_value, new_value_cast)
        return new_df

    @staticmethod
    def scale_features(df: pd.DataFrame, target_name: str, method: str) -> pd.DataFrame:
        if target_name not in df.columns:
            raise ValueError("Please choose a valid target column first.")

        scaler_map = {
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "Normalizer": Normalizer(),
            "RobustScaler": RobustScaler(),
            "PowerTransformer": PowerTransformer(),
            "QuantileTransformer": QuantileTransformer(),
        }

        scaler = scaler_map.get(method)
        if scaler is None:
            raise ValueError("Unsupported scaling method.")

        new_df = df.copy()
        feature_df = new_df.drop(columns=[target_name])

        numeric_columns = feature_df.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            raise ValueError("No numeric feature columns were found to scale.")

        new_df[numeric_columns] = scaler.fit_transform(new_df[numeric_columns])
        return new_df