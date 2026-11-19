import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

from app.services.data_service import DataService
from app.state import AppState
from app.ui.utils import populate_treeview, show_error, show_success, show_warning


class ProcessPage(ctk.CTkFrame):
    def __init__(self, parent, state: AppState, controller) -> None:
        super().__init__(parent)
        self.state = state
        self.controller = controller

        if not self.state.has_dataset():
            ctk.CTkLabel(
                self,
                text="No dataset loaded.\nGo to the Import Dataset page first.",
                font=("Arial", 20, "bold"),
            ).pack(expand=True)
            return

        self._build_ui()
        self._refresh_all()

    def _build_ui(self) -> None:
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        self.tabview = ctk.CTkTabview(self, height=280)
        self.tabview.pack(fill="x", padx=10, pady=(5, 10))

        self.tabview.add("Delete / Reset")
        self.tabview.add("Missing Values")
        self.tabview.add("Data Encoding")
        self.tabview.add("Feature Scaling")

        self._build_delete_tab()
        self._build_missing_tab()
        self._build_encoding_tab()
        self._build_scaling_tab()

    def _build_delete_tab(self) -> None:
        tab = self.tabview.tab("Delete / Reset")

        left = ctk.CTkFrame(tab)
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        right = ctk.CTkFrame(tab, width=350)
        right.pack(side="right", fill="both", padx=10, pady=10)

        ctk.CTkLabel(left, text="Columns to delete (comma-separated)", font=("Arial", 14, "bold")).pack(anchor="w", pady=(10, 6), padx=15)
        self.delete_columns_entry = ctk.CTkEntry(left, placeholder_text="col1, col2, col3")
        self.delete_columns_entry.pack(fill="x", padx=15, pady=(0, 10))


        buttons_frame = ctk.CTkFrame(left, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=15, pady=6)

        # Give each button an equal share of the available width.
        buttons_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="buttons")

        ctk.CTkButton(
            buttons_frame,
            text="Delete Columns",
            command=self.delete_columns,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))

        ctk.CTkButton(
            buttons_frame,
            text="Save Current Dataset",
            fg_color="green",
            command=self.save_dataset,
        ).grid(row=0, column=1, sticky="ew", padx=5)

        ctk.CTkButton(
            buttons_frame,
            text="Reset to Original Dataset",
            fg_color="#a64b00",
            command=self.reset_dataset,
        ).grid(row=0, column=2, sticky="ew", padx=(5, 0))

        self.info_box = ctk.CTkTextbox(left, height=140)
        self.info_box.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        ctk.CTkLabel(right, text="Duplicate Rows", font=("Arial", 14, "bold")).pack(pady=(18, 8))
        self.duplicate_var = tk.StringVar(value="")
        ctk.CTkLabel(right, textvariable=self.duplicate_var, justify="center").pack(pady=8, padx=10)

        ctk.CTkButton(right, text="Drop Duplicated Rows", command=self.drop_duplicates).pack(fill="x", padx=25, pady=8)

    def _build_missing_tab(self) -> None:
        tab = self.tabview.tab("Missing Values")

        left = ctk.CTkFrame(tab)
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        right = ctk.CTkFrame(tab)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(left, text="Missing Values Summary", font=("Arial", 11, "bold")).pack(anchor="w", pady=6, padx=10)
        self.missing_box = ctk.CTkTextbox(left, height=180)
        self.missing_box.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        ctk.CTkLabel(right, text="Handle Missing Values by Method", font=("Arial", 11, "bold")).pack(pady=6)
        self.mv_columns_entry = ctk.CTkEntry(right, placeholder_text="col1, col2")
        self.mv_columns_entry.pack(fill="x", padx=10, pady=6)

        self.mv_method_var = tk.StringVar(value="Forward Fill")
        ctk.CTkOptionMenu(
            right,
            values=["Forward Fill", "Backward Fill", "Delete Row", "Mean", "Median", "Max", "Min"],
            variable=self.mv_method_var,
        ).pack(fill="x", padx=10, pady=6)

        ctk.CTkButton(right, text="Apply Method", command=self.apply_missing_method).pack(fill="x", padx=20, pady=6)

        ctk.CTkLabel(right, text="Handle Missing Values by Fixed Value", font=("Arial", 11, "bold")).pack(pady=(16, 8))
        self.mv_value_columns_entry = ctk.CTkEntry(right, placeholder_text="col1, col2")
        self.mv_value_columns_entry.pack(fill="x", padx=20, pady=6)

        self.mv_value_entry = ctk.CTkEntry(right, placeholder_text="Replacement value")
        self.mv_value_entry.pack(fill="x", padx=20, pady=6)

        ctk.CTkButton(right, text="Apply Fixed Value", command=self.apply_missing_value).pack(fill="x", padx=20, pady=6)

    def _build_encoding_tab(self) -> None:
        tab = self.tabview.tab("Data Encoding")

        ctk.CTkLabel(tab, text="Replace a categorical value in one column", font=("Arial", 12, "bold")).pack(pady=(15, 8))

        self.encoding_column_entry = ctk.CTkEntry(tab, placeholder_text="Column name", width=300)
        self.encoding_column_entry.pack(pady=6)

        self.old_value_entry = ctk.CTkEntry(tab, placeholder_text="Current value", width=300)
        self.old_value_entry.pack(pady=6)

        self.new_value_entry = ctk.CTkEntry(tab, placeholder_text="New value", width=300)
        self.new_value_entry.pack(pady=6)

        ctk.CTkButton(tab, text="Replace Value", command=self.apply_encoding).pack(pady=10)
        ctk.CTkButton(tab, text="Save Current Dataset", fg_color="green", command=self.save_dataset).pack(pady=8)

    def _build_scaling_tab(self) -> None:
        tab = self.tabview.tab("Feature Scaling")

        ctk.CTkLabel(tab, text="Scale numeric feature columns only", font=("Arial", 14, "bold")).pack(pady=(15, 12))

        self.scaling_var = tk.StringVar(value="StandardScaler")
        ctk.CTkOptionMenu(
            tab,
            values=[
                "MinMaxScaler",
                "StandardScaler",
                "Normalizer",
                "RobustScaler",
                "PowerTransformer",
                "QuantileTransformer",
            ],
            variable=self.scaling_var,
            width=250,
        ).pack(pady=8)

        ctk.CTkButton(tab, text="Apply Scaling", command=self.apply_scaling).pack(pady=15)
        ctk.CTkButton(tab, text="Save Current Dataset", fg_color="green", command=self.save_dataset).pack(pady=8)

    def _refresh_all(self) -> None:
        populate_treeview(self.table_frame, self.state.dataset)

        self.info_box.delete("1.0", "end")
        self.info_box.insert("1.0", DataService.dataset_info_text(self.state.dataset))

        missing_summary = DataService.missing_values_summary(self.state.dataset)
        self.missing_box.delete("1.0", "end")
        if missing_summary.empty:
            self.missing_box.insert("1.0", "No missing values found.")
        else:
            self.missing_box.insert("1.0", missing_summary.to_string())
            self.missing_box.insert("end", f"\n\nTotal missing values: {int(missing_summary.sum())}")

        duplicate_count = int(self.state.dataset.duplicated().sum())
        self.duplicate_var.set(f"Duplicated rows: {duplicate_count}")

    def _apply_new_dataset(self, new_df, success_message: str) -> None:
        old_target = self.state.target_name
        self.state.update_dataset(new_df)
        self._refresh_all()
        show_success("Success", success_message)

        if old_target and not self.state.target_name:
            show_warning("Warning", "The target column was removed. Please choose a new target column.")

    def delete_columns(self) -> None:
        try:
            new_df = DataService.delete_columns(self.state.dataset, self.delete_columns_entry.get())
            self.delete_columns_entry.delete(0, "end")
            self._apply_new_dataset(new_df, "Columns deleted successfully.")
        except Exception as exc:
            show_error("Delete Error", str(exc))

    def drop_duplicates(self) -> None:
        try:
            new_df, duplicate_count = DataService.drop_duplicates(self.state.dataset)
            self._apply_new_dataset(new_df, f"{duplicate_count} duplicated rows removed.")
        except Exception as exc:
            show_error("Duplicate Error", str(exc))

    def reset_dataset(self) -> None:
        if self.state.original_dataset is None:
            show_error("Error", "No original dataset is available.")
            return

        self.state.reset_to_original()
        self._refresh_all()
        show_success("Reset", "Dataset reset to its original imported version.")

    def apply_missing_method(self) -> None:
        try:
            new_df = DataService.handle_missing_by_method(
                self.state.dataset,
                self.mv_columns_entry.get(),
                self.mv_method_var.get(),
            )
            self._apply_new_dataset(new_df, "Missing values handled successfully.")
        except Exception as exc:
            show_error("Missing Values Error", str(exc))

    def apply_missing_value(self) -> None:
        try:
            new_df = DataService.handle_missing_by_value(
                self.state.dataset,
                self.mv_value_columns_entry.get(),
                self.mv_value_entry.get(),
            )
            self._apply_new_dataset(new_df, "Missing values handled successfully.")
        except Exception as exc:
            show_error("Missing Values Error", str(exc))

    def apply_encoding(self) -> None:
        try:
            new_df = DataService.encode_replace(
                self.state.dataset,
                self.encoding_column_entry.get().strip(),
                self.old_value_entry.get().strip(),
                self.new_value_entry.get().strip(),
            )
            self._apply_new_dataset(new_df, "Data encoded successfully.")
        except Exception as exc:
            show_error("Encoding Error", str(exc))

    def apply_scaling(self) -> None:
        if not self.state.target_name:
            show_error("Error", "Please configure the target column first in the Import Dataset page.")
            return

        try:
            new_df = DataService.scale_features(
                self.state.dataset,
                self.state.target_name,
                self.scaling_var.get(),
            )
            self._apply_new_dataset(new_df, "Feature scaling applied successfully.")
        except Exception as exc:
            show_error("Scaling Error", str(exc))

    def save_dataset(self) -> None:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=DataService.EXPORT_FILE_TYPES,
        )
        if not file_path:
            return

        try:
            DataService.export_dataset(self.state.dataset, file_path)
            show_success("Saved", f"Dataset saved successfully:\n{file_path}")
        except Exception as exc:
            show_error("Save Error", str(exc))