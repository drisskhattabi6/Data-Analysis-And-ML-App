from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk

from app.config import OVERVIEW_PLOT_PATH
from app.services.plot_service import PlotService
from app.state import AppState
from app.ui.utils import show_error


class VisualizePage(ctk.CTkFrame):
    def __init__(self, parent, state: AppState, controller) -> None:
        super().__init__(parent)
        self.state = state
        self.controller = controller
        self.figure_canvas = None

        self._build_ui()

    def _build_ui(self) -> None:
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
                text="Please configure the target column and task type in the Import Dataset page first.",
                font=("Arial", 18, "bold"),
                wraplength=700,
                justify="center",
            ).pack(expand=True)
            return

        top_bar = ctk.CTkFrame(self)
        top_bar.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            top_bar,
            text=f"Dataset Overview • Target: {self.state.target_name} • Task: {self.state.target_type}",
            font=("Arial", 18, "bold"),
        ).pack(side="left", padx=12, pady=12)

        ctk.CTkButton(
            top_bar,
            text="Regenerate",
            command=self._render_plot,
        ).pack(side="right", padx=12, pady=12)

        self.plot_holder = ctk.CTkScrollableFrame(self)
        self.plot_holder.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._render_plot()

    def _render_plot(self) -> None:
        try:
            fig, plot_path = PlotService.create_overview_plot(
                self.state.dataset,
                self.state.target_name,
                self.state.target_type,
                OVERVIEW_PLOT_PATH,
            )
            self.state.overview_plot_path = plot_path

            for widget in self.plot_holder.winfo_children():
                widget.destroy()

            self.figure_canvas = FigureCanvasTkAgg(fig, master=self.plot_holder)
            self.figure_canvas.draw()
            self.figure_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        except Exception as exc:
            show_error("Visualization Error", str(exc))