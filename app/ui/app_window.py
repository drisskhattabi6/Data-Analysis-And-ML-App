import customtkinter as ctk
from PIL import Image

from app.config import (
    APP_ICON,
    APP_NAME,
    COLORS,
    ASSETS_DIR,
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
    WINDOW_SIZE,
)
from app.state import AppState
from app.ui.pages.home_page import HomePage
from app.ui.pages.import_page import ImportPage
from app.ui.pages.ml_page import MLPage
from app.ui.pages.process_page import ProcessPage
from app.ui.pages.visualize_page import VisualizePage
from app.ui.utils import clear_frame


class AppWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.app_state = AppState()
        self.current_page = None
        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        self.nav_icons: list[ctk.CTkImage] = []

        self.title(APP_NAME)
        self.geometry(WINDOW_SIZE)
        self.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        try:
            if APP_ICON.exists():
                self.iconbitmap(str(APP_ICON))
        except Exception:
            pass

        self.page_map = {
            "home": HomePage,
            "import": ImportPage,
            "visualize": VisualizePage,
            "process": ProcessPage,
            "ml": MLPage,
        }

        self._build_layout()
        self.show_page("home")

    def _build_layout(self) -> None:

        self.body_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.body_frame.pack(fill="both", expand=True)

        self.sidebar_frame = ctk.CTkFrame(
            self.body_frame,
            width=220,
            fg_color=COLORS["sidebar"],
            corner_radius=0,
        )
        self.sidebar_frame.pack(side="left", fill="y")

        self.content_frame = ctk.CTkFrame(self.body_frame, fg_color="transparent")
        self.content_frame.pack(side="right", fill="both", expand=True)

        ctk.CTkLabel(
            self.sidebar_frame,
            text="Dashboard",
            font=("Arial", 28, "bold"),
            text_color=COLORS["white"],
        ).pack(pady=(30, 10))

        ctk.CTkLabel(
            self.sidebar_frame,
            text="",
            fg_color=COLORS["sidebar_accent"],
            width=150,
            height=3,
            corner_radius=10,
        ).pack(pady=(0, 20))

        buttons = [
            ("home", "Home", None, COLORS["primary"]),
            ("import", "Import Dataset", ASSETS_DIR / "importicon.png", COLORS["primary"]),
            ("visualize", "Data Visualization", ASSETS_DIR / "vise-icon.png", COLORS["primary"]),
            ("process", "Data Processing", ASSETS_DIR / "processing-icon.png", COLORS["primary"]),
            ("ml", "Machine Learning", ASSETS_DIR / "ml-icon.png", COLORS["primary"]),
        ]

        for page_key, label, icon_path, text_color in buttons:
            icon = self._load_icon(icon_path) if icon_path else None
            if icon is not None:
                self.nav_icons.append(icon)

            button = ctk.CTkButton(
                self.sidebar_frame,
                text=label,
                image=icon,
                compound="left",
                anchor="w",
                width=210,
                height=48,
                fg_color="transparent",
                border_width=2,
                border_color=text_color,
                text_color=text_color,
                hover_color=COLORS["light_surface"],
                command=lambda name=page_key: self.show_page(name),
            )
            button.pack(pady=10, padx=18)
            self.nav_buttons[page_key] = button

    def _load_icon(self, path, size=(34, 34)):
        if path is None or not path.exists():
            return None

        pil_img = Image.open(path)
        return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=size)

    def show_page(self, page_name: str) -> None:
        if page_name not in self.page_map:
            return

        for key, button in self.nav_buttons.items():
            default_color = COLORS["secondary"] if key in {"tutorial", "documentation"} else COLORS["primary"]
            if key == page_name:
                button.configure(fg_color=COLORS["sidebar_accent"], text_color=COLORS["white"])
            else:
                button.configure(fg_color="transparent", text_color=default_color)

        clear_frame(self.content_frame)
        page_class = self.page_map[page_name]
        self.current_page = page_class(self.content_frame, self.app_state, self)
        self.current_page.pack(fill="both", expand=True)