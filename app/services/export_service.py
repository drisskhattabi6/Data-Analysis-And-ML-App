from datetime import datetime
from pathlib import Path

import joblib
from fpdf import FPDF

from app.config import PDF_HEADER_IMAGE
from app.services.ml_service import ModelRun
from app.state import AppState


class ExportService:
    @staticmethod
    def save_model(model_run: ModelRun, file_path: str | Path) -> None:
        joblib.dump(model_run.pipeline, file_path)

    @staticmethod
    def _epw(pdf: FPDF) -> float:
        return pdf.w - pdf.l_margin - pdf.r_margin

    @staticmethod
    def _add_plot_page(pdf: FPDF, title: str, image_path: Path) -> None:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        if image_path.exists():
            pdf.image(str(image_path), x=pdf.l_margin, w=ExportService._epw(pdf))

    @staticmethod
    def export_report(
        state: AppState,
        model_name: str,
        model_run: ModelRun,
        output_path: str | Path,
    ) -> None:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.add_page()

        if PDF_HEADER_IMAGE.exists():
            pdf.image(str(PDF_HEADER_IMAGE), x=pdf.l_margin, w=ExportService._epw(pdf) * 0.7)
            pdf.ln(28)

        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 10, "ModelMaster Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(2)

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Model: {model_name}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Target column: {state.target_name}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Task type: {state.target_type}", new_x="LMARGIN", new_y="NEXT")

        if state.source_file:
            pdf.cell(0, 8, f"Source file: {state.source_file.name}", new_x="LMARGIN", new_y="NEXT")

        if state.has_dataset():
            pdf.cell(
                0,
                8,
                f"Dataset shape: {state.dataset.shape[0]} rows x {state.dataset.shape[1]} columns",
                new_x="LMARGIN",
                new_y="NEXT",
            )

        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Performance Summary", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 11)
        if state.target_type == "classification":
            pdf.cell(0, 8, f"Accuracy: {model_run.metrics['accuracy']:.4f}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 8, f"Weighted F1-score: {model_run.metrics['f1_weighted']:.4f}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

            pdf.set_font("Courier", "", 9)
            pdf.multi_cell(0, 5, model_run.metrics["report"])
        else:
            pdf.cell(0, 8, f"MSE: {model_run.metrics['mse']:.6f}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 8, f"MAE: {model_run.metrics['mae']:.6f}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 8, f"R²: {model_run.metrics['r2']:.6f}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(
                0,
                8,
                f"Explained Variance: {model_run.metrics['explained_variance']:.6f}",
                new_x="LMARGIN",
                new_y="NEXT",
            )

        if state.overview_plot_path and state.overview_plot_path.exists():
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Dataset Overview", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(4)
            pdf.image(str(state.overview_plot_path), x=pdf.l_margin, w=ExportService._epw(pdf))

        plot_paths = state.model_plot_paths.get(model_name, [])
        for index, plot_path in enumerate(plot_paths, start=1):
            ExportService._add_plot_page(pdf, f"Model Plot {index}", plot_path)

        pdf.output(str(output_path))