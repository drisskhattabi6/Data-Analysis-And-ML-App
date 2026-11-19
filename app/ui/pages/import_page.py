import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from app.services.data_service import DataService
from app.state import AppState
from app.ui.utils import populate_treeview, show_error, show_success


class ImportPage(ctk.CTkFrame):
    TYPE_LABEL_TO_VALUE = {
        "Qualitative values (classification)": "classification",
        "Quantitative values (regression)": "regression",
    }

    TYPE_VALUE_TO_LABEL = {
        "classification": "Qualitative values (classification)",
        "regression": "Quantitative values (regression)",
    }

    def __init__(self, parent, state: AppState, controller) -> None:
        super().__init__(parent)
        self.state = state
        self.controller = controller

        self.target_type_var = tk.StringVar(
            value=self.TYPE_VALUE_TO_LABEL.get(self.state.target_type, "Choose type")
        )

        self._build_ui()
        self._refresh_view()

    def _build_ui(self) -> None:
        self.left_panel = ctk.CTkFrame(self, width=250)
        self.left_panel.pack(side="left", fill="y", padx=10, pady=10)

        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=(0, 10), pady=10)

        ctk.CTkLabel(
            self.left_panel,
            text="Import or Create Dataset",
            font=("Arial", 20, "bold"),
        ).pack(pady=(20, 15))

        ctk.CTkButton(
            self.left_panel,
            text="Import Dataset",
            height=35,
            command=self.import_dataset,
        ).pack(fill="x", padx=15, pady=8)

        ctk.CTkButton(
            self.left_panel,
            text="Create Sample Template",
            height=35,
            fg_color="green",
            command=self.create_template,
        ).pack(fill="x", padx=15, pady=8)

        ctk.CTkLabel(
            self.left_panel,
            text="Target Column Name",
            font=("Arial", 14, "bold"),
        ).pack(padx=20, pady=(20, 6), anchor="w")

        self.target_entry = ctk.CTkEntry(self.left_panel, placeholder_text="Enter target column name")
        self.target_entry.pack(fill="x", padx=20, pady=4)

        if self.state.target_name:
            self.target_entry.insert(0, self.state.target_name)

        ctk.CTkLabel(
            self.left_panel,
            text="Target type",
            font=("Arial", 14, "bold"),
        ).pack(padx=15, pady=(15, 6), anchor="w")

        self.type_menu = ctk.CTkOptionMenu(
            self.left_panel,
            values=["Qualitative values (classification)", "Quantitative values (regression)"],
            variable=self.target_type_var,
        )
        self.type_menu.pack(fill="x", padx=15, pady=4)

        ctk.CTkButton(
            self.left_panel,
            text="Save Target Configuration",
            command=self.save_target_config,
        ).pack(fill="x", padx=15, pady=14)

        ctk.CTkLabel(
            self.left_panel,
            text="Dataset Info",
            font=("Arial", 16, "bold"),
        ).pack(padx=15, pady=(15, 8), anchor="w")

        self.info_box = ctk.CTkTextbox(self.left_panel, width=250, height=270)
        self.info_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.table_frame = ctk.CTkFrame(self.right_panel)
        self.table_frame.pack(fill="both", expand=True)

    def import_dataset(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=DataService.SUPPORTED_FILE_TYPES)
        if not file_path:
            return

        try:
            df = DataService.load_dataset(file_path)
            self.state.set_dataset(df, file_path)
            self._refresh_view()
            show_success("Success", f"Dataset loaded successfully:\n{Path(file_path).name}")
        except Exception as exc:
            show_error("Import Error", str(exc))

    def create_template(self) -> None:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
        )
        if not file_path:
            return

        try:
            DataService.create_sample_template(file_path)
            show_success("Success", f"Template created successfully:\n{Path(file_path).name}")
        except Exception as exc:
            show_error("Template Error", str(exc))

    def save_target_config(self) -> None:
        if not self.state.has_dataset():
            show_error("Error", "Please import a dataset first.")
            return

        target_name = self.target_entry.get().strip()
        target_type_label = self.target_type_var.get().strip()

        if not target_name:
            show_error("Error", "Please enter a target column name.")
            return

        if target_name not in self.state.dataset.columns:
            show_error("Error", f"Column '{target_name}' does not exist in the dataset.")
            return

        if target_type_label not in self.TYPE_LABEL_TO_VALUE:
            show_error("Error", "Please choose a valid target type.")
            return

        self.state.target_name = target_name
        self.state.target_type = self.TYPE_LABEL_TO_VALUE[target_type_label]
        self.state.reset_training()

        show_success(
            "Saved",
            f"Target column: {target_name}\nTask type: {self.state.target_type}",
        )
        self._refresh_view()

    def _refresh_view(self) -> None:
        self.info_box.delete("1.0", "end")

        if not self.state.has_dataset():
            self.info_box.insert("1.0", "No dataset loaded.")
            for widget in self.table_frame.winfo_children():
                widget.destroy()
            return

        info_text = DataService.dataset_info_text(self.state.dataset)
        if self.state.source_file:
            info_text = f"Source: {self.state.source_file.name}\n\n{info_text}"

        self.info_box.insert("1.0", info_text)
        populate_treeview(self.table_frame, self.state.dataset)