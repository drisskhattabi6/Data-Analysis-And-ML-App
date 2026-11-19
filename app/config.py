from pathlib import Path

APP_NAME = "ML Toolkit"
WINDOW_SIZE = "1100x700"
MIN_WINDOW_WIDTH = 1000
MIN_WINDOW_HEIGHT = 700

BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR / "assets"
# SCREENSHOTS_DIR = BASE_DIR / "screenshots"

APP_ICON = ASSETS_DIR / "icon.ico"
HOME_BG_IMAGE = ASSETS_DIR / "bag-img.png"
PDF_HEADER_IMAGE = ASSETS_DIR / "pdf-header.png"

RESULTS_DIR = BASE_DIR / "results"
CLASSIFICATION_RESULTS_DIR = RESULTS_DIR / "classification"
REGRESSION_RESULTS_DIR = RESULTS_DIR / "regression"
OVERVIEW_PLOT_PATH = RESULTS_DIR / "visualization.png"

for directory in (RESULTS_DIR, CLASSIFICATION_RESULTS_DIR, REGRESSION_RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

COLORS = {
    "sidebar": "#4D1230",
    "sidebar_accent": "#e06d14",
    "header": "#5e2c59",
    "primary": "#e06d14",
    "secondary": "#3887BE",
    "success": "#2E8B57",
    "surface": "#1f1f1f",
    "light_surface": "#2b2b2b",
    "white": "#FFFFFF",
}