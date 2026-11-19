import threading
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk
import pandas as pd
from PIL import Image

from app.services.export_service import ExportService
from app.services.ml_service import MLService, ModelRun
from app.services.plot_service import PlotService
from app.state import AppState
from app.ui.utils import (
    clear_frame,
    show_error,
    show_success,
)


class MLPage(ctk.CTkFrame):
    def __init__(self, parent, state: AppState, controller) -> None:
        super().__init__(parent)
        self.state = state
        self.controller = controller
        self.is_training = False
        self.results_visible = True
        self.plot_images = []

        self.test_size_var = tk.StringVar(self, value="0.2")
        self.selected_model_names: set[str] = set()
        self.model_selection_text = tk.StringVar(
            self,
            value="No models selected",
        )
        self.model_var = tk.StringVar(
            self,
            value=self.state.selected_model or "Choose a trained model",
        )
        self.status_var = tk.StringVar(
            self,
            value="Train a model to display results.",
        )

        if not self.state.has_dataset():
            ctk.CTkLabel(
                self,
                text="No dataset loaded.\nGo to the Import Dataset page first.",
                font=("Arial", 20, "bold"),
            ).pack(expand=True)
            return

        if not self.state.target_name or not self.state.target_type:
            ctk.CTkLabel(
                self,
                text=(
                    "Please configure the target column and task type "
                    "in the Import Dataset page first."
                ),
                font=("Arial", 18, "bold"),
                wraplength=700,
                justify="center",
            ).pack(expand=True)
            return

        self.available_model_names = MLService.get_model_names(
            self.state.target_type
        )

        if self.available_model_names:
            self.selected_model_names = {self.available_model_names[0]}
            self._update_model_selection_text()

        self._build_ui()
        self._render_result_tabs()
        self._configure_trained_model_menu()

    def _build_ui(self) -> None:
        controls = ctk.CTkFrame(self)
        controls.pack(fill="x", padx=10, pady=10)

        training_row = ctk.CTkFrame(controls, fg_color="transparent")
        training_row.pack(fill="x", padx=5, pady=(5, 2))

        ctk.CTkLabel(
            training_row,
            text="Test Size",
            font=("Arial", 11, "bold"),
        ).pack(side="left", padx=(8, 5))

        self.test_size_menu = ctk.CTkOptionMenu(
            training_row,
            variable=self.test_size_var,
            values=[
                "0.1",
                "0.15",
                "0.2",
                "0.25",
                "0.3",
                "0.35",
                "0.4",
                "0.45",
                "0.5",
            ],
            width=75,
        )
        self.test_size_menu.pack(side="left", padx=5, pady=6)

        ctk.CTkLabel(
            training_row,
            text="Models to Train",
            font=("Arial", 11, "bold"),
        ).pack(side="left", padx=(12, 5))

        self.select_models_button = ctk.CTkButton(
            training_row,
            text="Choose Models...",
            width=145,
            command=self.open_model_selection_dialog,
        )
        self.select_models_button.pack(
            side="left",
            padx=(5, 2),
            pady=6,
        )

        ctk.CTkLabel(
            training_row,
            textvariable=self.model_selection_text,
            anchor="w",
            width=170,
        ).pack(side="left", padx=(2, 5))

        self.train_selected_button = ctk.CTkButton(
            training_row,
            text="Train Selected",
            fg_color="green",
            command=self.train_selected_model,
        )
        self.train_selected_button.pack(side="left", padx=6)

        self.train_all_button = ctk.CTkButton(
            training_row,
            text="Train All",
            fg_color="#176B87",
            command=self.train_all_models,
        )
        self.train_all_button.pack(side="left", padx=6)

        actions_row = ctk.CTkFrame(controls, fg_color="transparent")
        actions_row.pack(fill="x", padx=5, pady=(2, 5))

        ctk.CTkLabel(
            actions_row,
            text="Trained Model",
            font=("Arial", 11, "bold"),
        ).pack(side="left", padx=(8, 5))

        self.trained_model_menu = ctk.CTkOptionMenu(
            actions_row,
            variable=self.model_var,
            values=["Choose a trained model"],
            width=180,
            command=self._on_trained_model_selected,
        )
        self.trained_model_menu.pack(side="left", padx=5, pady=6)

        ctk.CTkButton(
            actions_row,
            text="Show Visuals",
            width=120,
            command=self.show_visuals,
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            actions_row,
            text="Predict Sample",
            width=120,
            command=self.open_prediction_dialog,
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            actions_row,
            text="Save Model",
            width=120,
            command=self.save_model,
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            actions_row,
            text="Export Report",
            width=120,
            command=self.export_report,
        ).pack(side="left", padx=5)

        self.toggle_results_button = ctk.CTkButton(
            actions_row,
            text="Hide Results",
            width=115,
            command=self.toggle_results,
        )
        self.toggle_results_button.pack(side="left", padx=3)

        ctk.CTkLabel(
            actions_row,
            textvariable=self.status_var,
        ).pack(side="right", padx=5)

        body = ctk.CTkFrame(self)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=2)
        self.body = body

        self.results_panel = ctk.CTkFrame(body)
        self.results_panel.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 5),
        )

        ctk.CTkLabel(
            self.results_panel,
            text="Model Results",
            font=("Arial", 17, "bold"),
        ).pack(fill="x", padx=10, pady=(10, 5))

        self.results_textbox = ctk.CTkTextbox(self.results_panel)
        self.results_textbox.pack(
            fill="both",
            expand=True,
            padx=8,
            pady=8,
        )
        self.results_textbox.configure(state="disabled")

        self.visuals_container = ctk.CTkFrame(body)
        self.visuals_container.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(5, 0),
        )

        ctk.CTkLabel(
            self.visuals_container,
            text="Choose a trained model, then click 'Show Visuals'.",
            font=("Arial", 15, "bold"),
            wraplength=450,
            justify="center",
        ).pack(expand=True, padx=20, pady=20)

    def _configure_trained_model_menu(self) -> None:
        trained_names = [
            name
            for name in self.state.model_results
            if name in self.available_model_names
        ]

        if not trained_names:
            self.trained_model_menu.configure(
                values=["Choose a trained model"]
            )
            self.model_var.set("Choose a trained model")
            self._render_result_tabs()
            return

        self.trained_model_menu.configure(values=trained_names)

        if self.state.selected_model in trained_names:
            self.model_var.set(self.state.selected_model)
        else:
            self.model_var.set(trained_names[0])
            self.state.selected_model = trained_names[0]

        self._render_result_tabs()

    def _on_trained_model_selected(self, model_name: str) -> None:
        if model_name in self.state.model_results:
            self.state.selected_model = model_name
        self._render_result_tabs()

    def _update_model_selection_text(self) -> None:
        count = len(self.selected_model_names)

        if count == 0:
            text = "No models selected"
        elif count == 1:
            text = "1 model selected"
        else:
            text = f"{count} models selected"

        self.model_selection_text.set(text)

    def open_model_selection_dialog(self) -> None:
        dialog = ctk.CTkToplevel(self)
        dialog.title("Select Models to Train")
        dialog.geometry("420x540")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()

        ctk.CTkLabel(
            dialog,
            text="Choose one or more models to train",
            font=("Arial", 17, "bold"),
        ).pack(padx=15, pady=(15, 8))

        list_frame = ctk.CTkScrollableFrame(
            dialog,
            label_text="Available models",
        )
        list_frame.pack(fill="both", expand=True, padx=15, pady=8)

        check_vars: dict[str, tk.BooleanVar] = {}

        for model_name in self.available_model_names:
            var = tk.BooleanVar(
                value=model_name in self.selected_model_names
            )
            check_vars[model_name] = var

            ctk.CTkCheckBox(
                list_frame,
                text=model_name,
                variable=var,
            ).pack(anchor="w", padx=10, pady=5)

        quick_actions = ctk.CTkFrame(
            dialog,
            fg_color="transparent",
        )
        quick_actions.pack(fill="x", padx=15, pady=(4, 0))

        def set_all(value: bool) -> None:
            for variable in check_vars.values():
                variable.set(value)

        ctk.CTkButton(
            quick_actions,
            text="Select All",
            width=110,
            command=lambda: set_all(True),
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            quick_actions,
            text="Clear",
            width=90,
            command=lambda: set_all(False),
        ).pack(side="left")

        def apply_selection() -> None:
            self.selected_model_names = {
                name
                for name, variable in check_vars.items()
                if variable.get()
            }
            self._update_model_selection_text()
            dialog.destroy()

        ctk.CTkButton(
            dialog,
            text="Apply Selection",
            command=apply_selection,
        ).pack(fill="x", padx=15, pady=15)

    def train_selected_model(self) -> None:
        model_names = [
            name
            for name in self.available_model_names
            if name in self.selected_model_names
        ]

        if not model_names:
            show_error(
                "Error",
                "Please choose at least one model to train.",
            )
            return

        self._start_training(model_names)

    def train_all_models(self) -> None:
        self._start_training(self.available_model_names.copy())

    def _start_training(self, model_names: list[str]) -> None:
        if self.is_training:
            return

        try:
            test_size = float(self.test_size_var.get())
        except ValueError:
            show_error("Error", "Invalid test size.")
            return

        dataset_snapshot = self.state.dataset.copy()
        target_name = self.state.target_name
        task_type = self.state.target_type

        self.is_training = True
        self._set_training_controls_enabled(False)
        self.status_var.set(
            f"Training {len(model_names)} model(s)..."
        )

        thread = threading.Thread(
            target=self._train_worker,
            args=(
                dataset_snapshot,
                target_name,
                task_type,
                test_size,
                model_names,
            ),
            daemon=True,
        )
        thread.start()

    def _set_training_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.train_selected_button.configure(state=state)
        self.train_all_button.configure(state=state)
        self.select_models_button.configure(state=state)
        self.test_size_menu.configure(state=state)

    def _train_worker(
        self,
        df: pd.DataFrame,
        target_name: str,
        task_type: str,
        test_size: float,
        model_names: list[str],
    ) -> None:
        try:
            results, errors = MLService.train_models(
                df=df,
                target_name=target_name,
                task_type=task_type,
                test_size=test_size,
                model_names=model_names,
            )

            self.after(
                0,
                lambda results=results, errors=errors: (
                    self._on_training_complete(results, errors)
                ),
            )
        except Exception as exc:
            message = str(exc)
            self.after(
                0,
                lambda message=message: self._on_training_error(message),
            )

    def _on_training_complete(
        self,
        results: dict[str, ModelRun],
        errors: dict[str, str],
    ) -> None:
        self.is_training = False
        self._set_training_controls_enabled(True)

        if not results:
            self.status_var.set("Training failed.")
            details = "\n".join(
                f"{name}: {message}"
                for name, message in errors.items()
            )
            show_error(
                "Training Error",
                details or "No models could be trained.",
            )
            return

        # Preserve previous runs and replace only models trained in this run.
        self.state.model_results.update(results)

        for name in results:
            self.state.model_plot_paths.pop(name, None)

        self.state.selected_model = next(iter(results))
        self._render_result_tabs()
        self._configure_trained_model_menu()
        self.status_var.set(
            f"Training completed: {len(results)} model(s)."
        )

        if errors:
            details = "\n".join(
                f"{name}: {message}"
                for name, message in errors.items()
            )
            show_error(
                "Training Finished with Warnings",
                f"Successfully trained {len(results)} model(s).\n\n"
                f"Some models could not be trained:\n{details}",
            )
        else:
            show_success(
                "Training Complete",
                f"Successfully trained {len(results)} model(s).",
            )

    def _on_training_error(self, message: str) -> None:
        self.is_training = False
        self._set_training_controls_enabled(True)
        self.status_var.set("Training failed.")
        show_error("Training Error", message)

    def _render_result_tabs(self) -> None:
        """Render the selected model's report in one scrollable text area."""
        model_name = self.model_var.get().strip()

        self.results_textbox.configure(state="normal")
        self.results_textbox.delete("1.0", "end")

        if model_name not in self.state.model_results:
            self.results_textbox.insert(
                "1.0",
                "No trained model results to display yet.",
            )
        else:
            model_run = self.state.model_results[model_name]
            self.results_textbox.insert(
                "1.0",
                MLService.format_result_text(model_run),
            )

        self.results_textbox.configure(state="disabled")

    def toggle_results(self) -> None:
        if self.results_visible:
            self.results_panel.grid_remove()
            self.visuals_container.grid_configure(
                column=0,
                columnspan=2,
            )
            self.body.grid_columnconfigure(0, weight=0)
            self.body.grid_columnconfigure(1, weight=1)
            self.toggle_results_button.configure(text="Show Results")
            self.results_visible = False
        else:
            self.results_panel.grid()
            self.visuals_container.grid_configure(
                column=1,
                columnspan=1,
            )
            self.body.grid_columnconfigure(0, weight=1)
            self.body.grid_columnconfigure(1, weight=2)
            self.toggle_results_button.configure(text="Hide Results")
            self.results_visible = True

    def _get_selected_model_name(self) -> str | None:
        model_name = self.model_var.get().strip()

        if model_name not in self.state.model_results:
            return None

        return model_name

    def _ensure_plot_paths(self, model_name: str) -> list:
        if model_name in self.state.model_plot_paths:
            return self.state.model_plot_paths[model_name]

        model_run = self.state.model_results[model_name]
        full_df = self.state.dataset.dropna(
            subset=[self.state.target_name]
        ).copy()
        X = full_df.drop(columns=[self.state.target_name])
        y = full_df[self.state.target_name]

        plot_paths = PlotService.create_model_plots(
            model_run,
            X,
            y,
        )
        self.state.model_plot_paths[model_name] = plot_paths
        return plot_paths

    def show_visuals(self) -> None:
        model_name = self._get_selected_model_name()

        if not model_name:
            show_error(
                "Error",
                "Please choose a trained model first.",
            )
            return

        try:
            plot_paths = self._ensure_plot_paths(model_name)
            self.state.selected_model = model_name
            self._display_plot_grid(plot_paths)
            self.status_var.set(
                f"Showing visuals for: {model_name}"
            )
        except Exception as exc:
            show_error("Visualization Error", str(exc))

    def _display_plot_grid(self, plot_paths: list) -> None:
        clear_frame(self.visuals_container)
        self.plot_images.clear()

        grid = ctk.CTkFrame(self.visuals_container)
        grid.pack(fill="both", expand=True, padx=5, pady=5)

        for row in range(2):
            grid.grid_rowconfigure(row, weight=1)

        for column in range(2):
            grid.grid_columnconfigure(column, weight=1)

        for index, plot_path in enumerate(plot_paths):
            with Image.open(plot_path) as image:
                pil_image = image.copy()

            ctk_image = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=(430, 300),
            )
            self.plot_images.append(ctk_image)

            ctk.CTkLabel(
                grid,
                text="",
                image=ctk_image,
            ).grid(
                row=index // 2,
                column=index % 2,
                padx=8,
                pady=8,
                sticky="nsew",
            )

    def open_prediction_dialog(self) -> None:
        model_name = self._get_selected_model_name()

        if not model_name:
            show_error(
                "Error",
                "Please choose a trained model first.",
            )
            return

        model_run = self.state.model_results[model_name]
        feature_df = self.state.dataset.drop(
            columns=[self.state.target_name]
        )

        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Predict Sample - {model_name}")
        dialog.geometry("700x650")
        dialog.grab_set()

        ctk.CTkLabel(
            dialog,
            text=f"Enter values for {model_name}",
            font=("Arial", 18, "bold"),
        ).pack(pady=(15, 10))

        scroll = ctk.CTkScrollableFrame(dialog)
        scroll.pack(fill="both", expand=True, padx=15, pady=10)

        entries: dict[str, ctk.CTkEntry] = {}

        for feature_name in feature_df.columns:
            row = ctk.CTkFrame(scroll)
            row.pack(fill="x", padx=10, pady=6)

            ctk.CTkLabel(
                row,
                text=str(feature_name),
                width=220,
                anchor="w",
            ).pack(side="left", padx=10)

            entry = ctk.CTkEntry(row)
            entry.pack(side="left", fill="x", expand=True, padx=10)
            entries[feature_name] = entry

        result_var = tk.StringVar(dialog, value="")

        ctk.CTkLabel(
            dialog,
            textvariable=result_var,
            font=("Arial", 15, "bold"),
        ).pack(pady=10)

        def predict_now() -> None:
            try:
                input_row = {}

                for column in feature_df.columns:
                    raw_value = entries[column].get().strip()

                    if raw_value == "":
                        raise ValueError(
                            f"Value required for '{column}'."
                        )

                    if pd.api.types.is_numeric_dtype(feature_df[column]):
                        input_row[column] = float(raw_value)
                    else:
                        input_row[column] = raw_value

                prediction = MLService.predict(
                    model_run,
                    input_row,
                )

                if self.state.target_type == "regression":
                    result_var.set(
                        f"Predicted value: {float(prediction):.4f}"
                    )
                else:
                    result_var.set(
                        f"Predicted class: {prediction}"
                    )

            except Exception as exc:
                show_error("Prediction Error", str(exc))

        ctk.CTkButton(
            dialog,
            text="Predict",
            command=predict_now,
        ).pack(pady=(0, 15))

    def save_model(self) -> None:
        model_name = self._get_selected_model_name()

        if not model_name:
            show_error(
                "Error",
                "Please choose a trained model first.",
            )
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle Files", "*.pkl")],
        )

        if not file_path:
            return

        try:
            ExportService.save_model(
                self.state.model_results[model_name],
                file_path,
            )
            show_success(
                "Saved",
                f"Model saved successfully:\n{file_path}",
            )
        except Exception as exc:
            show_error("Save Error", str(exc))

    def export_report(self) -> None:
        model_name = self._get_selected_model_name()

        if not model_name:
            show_error(
                "Error",
                "Please choose a trained model first.",
            )
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
        )

        if not file_path:
            return

        try:
            self._ensure_plot_paths(model_name)
            ExportService.export_report(
                self.state,
                model_name,
                self.state.model_results[model_name],
                file_path,
            )
            show_success(
                "Saved",
                f"PDF report exported successfully:\n{file_path}",
            )
        except Exception as exc:
            show_error("Export Error", str(exc))
