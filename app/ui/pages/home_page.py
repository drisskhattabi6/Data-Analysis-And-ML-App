import customtkinter as ctk
from PIL import Image

from app.config import COLORS, HOME_BG_IMAGE
from app.state import AppState


class HomePage(ctk.CTkFrame):

    def __init__(self, parent, state: AppState, controller) -> None:
        super().__init__(parent, fg_color="transparent")
        self.state = state
        self.controller = controller

        self.bg_image = None

        self._build_ui()

    def _build_ui(self) -> None:
        if HOME_BG_IMAGE.exists():
            bg_pil = Image.open(HOME_BG_IMAGE)
            self.bg_image = ctk.CTkImage(
                light_image=bg_pil,
                dark_image=bg_pil,
                size=(1300, 780),
            )
            ctk.CTkLabel(self, text="", image=self.bg_image).pack(fill="both", expand=True)
        else:
            self.configure(fg_color="#1b1b1b")

        overlay = ctk.CTkFrame(self, fg_color="#000000", corner_radius=18)
        overlay.place(relx=0.5, rely=0.5, anchor="center")

        subtitle = ctk.CTkLabel(
            overlay,
            text="Desktop application for data analysis, preprocessing, visualization, and machine learning.",
            font=("Arial", 14),
            wraplength=700,
            justify="center",
            text_color=COLORS["white"],
        )
        subtitle.pack(padx=30, pady=(0, 20))

        if self.state.has_dataset():
            dataset_text = (
                f"Loaded dataset: {self.state.dataset.shape[0]} rows × {self.state.dataset.shape[1]} columns\n"
                f"Target: {self.state.target_name or 'Not selected'}\n"
                f"Task type: {self.state.target_type or 'Not selected'}"
            )
        else:
            dataset_text = "No dataset loaded yet. Start with the 'Import Dataset' page."

        info = ctk.CTkLabel(
            overlay,
            text=dataset_text,
            font=("Arial", 14),
            justify="center",
        )
        info.pack(padx=20, pady=(0, 25))