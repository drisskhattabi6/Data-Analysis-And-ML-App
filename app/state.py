from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass
class AppState:
    dataset: Optional[pd.DataFrame] = None
    original_dataset: Optional[pd.DataFrame] = None
    target_name: str = ""
    target_type: str = ""  # "classification" or "regression"
    model_results: dict[str, Any] = field(default_factory=dict)
    selected_model: str = ""
    overview_plot_path: Optional[Path] = None
    model_plot_paths: dict[str, list[Path]] = field(default_factory=dict)
    source_file: Optional[Path] = None

    def has_dataset(self) -> bool:
        return self.dataset is not None and not self.dataset.empty

    def set_dataset(self, df: pd.DataFrame, source_file: Optional[str | Path] = None) -> None:
        self.dataset = df.copy()
        self.original_dataset = df.copy()
        self.source_file = Path(source_file) if source_file else None

        if self.target_name and self.target_name not in self.dataset.columns:
            self.target_name = ""
            self.target_type = ""

        self.reset_training()

    def update_dataset(self, df: pd.DataFrame) -> None:
        self.dataset = df.copy()

        if self.target_name and self.target_name not in self.dataset.columns:
            self.target_name = ""
            self.target_type = ""

        self.reset_training()

    def reset_training(self) -> None:
        self.model_results.clear()
        self.model_plot_paths.clear()
        self.selected_model = ""

    def reset_to_original(self) -> None:
        if self.original_dataset is not None:
            self.dataset = self.original_dataset.copy()

            if self.target_name and self.target_name not in self.dataset.columns:
                self.target_name = ""
                self.target_type = ""

            self.reset_training()