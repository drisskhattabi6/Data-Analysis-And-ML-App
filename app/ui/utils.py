import tkinter as tk
from tkinter import ttk

import customtkinter as ctk
import pandas as pd
from CTkMessagebox import CTkMessagebox


def clear_frame(frame: tk.Widget) -> None:
    for widget in frame.winfo_children():
        widget.destroy()


def show_success(title: str, message: str) -> None:
    CTkMessagebox(title=title, message=message, icon="check", option_1="Ok")


def show_error(title: str, message: str) -> None:
    CTkMessagebox(title=title, message=message, icon="cancel", option_1="Ok")


def show_warning(title: str, message: str) -> None:
    CTkMessagebox(title=title, message=message, icon="warning", option_1="Ok")


def populate_treeview(parent: tk.Widget, df: pd.DataFrame, max_rows: int = 500) -> ttk.Treeview:
    clear_frame(parent)

    wrapper = ctk.CTkFrame(parent)
    wrapper.pack(fill="both", expand=True)

    preview_df = df.head(max_rows)

    info_label = ctk.CTkLabel(
        wrapper,
        text=f"Previewing {len(preview_df)} of {len(df)} rows × {len(df.columns)} columns",
        anchor="w",
    )
    info_label.pack(fill="x", padx=10, pady=(10, 5))

    table_container = ctk.CTkFrame(wrapper)
    table_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview", rowheight=28, font=("Arial", 10))
    style.configure("Treeview.Heading", font=("Arial", 10, "bold"))

    tree = ttk.Treeview(
        table_container,
        columns=list(preview_df.columns),
        show="headings",
        selectmode="extended",
    )

    for col in preview_df.columns:
        tree.heading(col, text=str(col), anchor=tk.CENTER)
        tree.column(col, width=130, anchor=tk.CENTER, stretch=True)

    for _, row in preview_df.iterrows():
        values = ["" if pd.isna(value) else value for value in row.tolist()]
        tree.insert("", "end", values=values)

    vsb = ttk.Scrollbar(table_container, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(table_container, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")

    table_container.grid_rowconfigure(0, weight=1)
    table_container.grid_columnconfigure(0, weight=1)

    return tree