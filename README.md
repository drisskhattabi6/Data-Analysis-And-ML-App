# ML ToolKit

**ML ToolKit** is a Python desktop application for exploring datasets, preparing data, visualizing patterns, and training introductory machine-learning models for classification and regression.

The GUI is built with **CustomTkinter**. Data handling uses **pandas**, visualizations use **Matplotlib** and **Seaborn**, and machine-learning workflows use **scikit-learn**.

> This is an educational analysis tool. Evaluate models on appropriate data and do not treat their outputs as professional, medical, financial, or other high-stakes advice.

## Features

### Dataset import and inspection
- Import CSV (`.csv`), Excel (`.xlsx`), and tab-delimited text (`.txt`) files.
- Preview dataset rows and inspect columns and data types.
- Create a sample Excel template.
- Select the target column and indicate whether the task is classification or regression.

### Data preparation
- Delete selected columns.
- Identify and remove duplicate rows.
- Review missing-value counts.
- Handle missing values by forward fill, backward fill, row deletion, mean, median, minimum, maximum, or a specified replacement value.
- Replace a categorical value in a selected column.
- Scale numeric feature columns using a selected scaler.
- Reset the working dataset to the imported version.
- Export the current dataset as CSV, Excel, or tab-delimited text.

### Visualization
- Generate an overview based on the configured target and task type.
- For classification, inspect class distributions and available numeric-feature relationships.
- For regression, inspect the target distribution, feature relationships, and correlations where applicable.
- Generate model-specific plots, including confusion matrices or actual-versus-predicted and residual plots, plus learning curves where the data permits.

### Machine learning
- Train a group of classification or regression estimators and compare their results.
- Choose the test-set proportion.
- Review classification metrics and reports, or regression metrics such as MSE, MAE, R², and explained variance.
- Enter a sample to obtain a prediction from a trained model.
- Save the trained scikit-learn pipeline, including its preprocessing steps.
- Export a PDF report with model metrics and available plots.

The app's ML preprocessing pipeline imputes missing feature values, scales numeric features, and one-hot encodes categorical features. The exact supported models and behavior are defined in `ML ToolKit/services/ml_service.py`.

## Requirements

- Windows, macOS, or Linux with a working Tkinter installation. The setup commands below are for Windows PowerShell.
- **Python 3.12** is recommended for the pinned dependencies in this repository. Python 3.13 or 3.14 may not have compatible wheels for every pinned package.
- Dependencies listed in `requirements.txt`.

## Project layout

```text
Data-Analysis-And-ML-App/
├── ML ToolKit/
│   ├── main.py
│   ├── config.py
│   ├── state.py
│   ├── services/
│   │   ├── data_service.py
│   │   ├── export_service.py
│   │   ├── ml_service.py
│   │   └── plot_service.py
│   └── ui/
│       ├── app_window.py
│       ├── utils.py
│       └── pages/
├── icons/
├── images/
├── screenshots/
├── results/                 # Created automatically for generated plots
├── documentation.pdf        # Optional in-app documentation
├── ML ToolKit.mp4          # Optional in-app tutorial
├── app.py
├── run.py
├── requirements.txt
└── README.md
```

Keep the `icons/` and `images/` folders in the project root. The app uses image assets from those folders. `documentation.pdf` and `ML ToolKit.mp4` are optional, but their corresponding pages need those files to display the PDF or video.

## Setup on Windows PowerShell

Open PowerShell in the project root—the folder containing `requirements.txt`—and create a virtual environment with Python 3.12:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python --version
```

Confirm the version shown is **Python 3.12.x**. Upgrade pip and install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

If your organization uses a private package index, pip may use that index automatically. If installation tries to compile Matplotlib or NumPy from source, check that the index has compatible prebuilt Windows wheels for your Python version, or contact your package-repository administrator. Do not install a compiler as a first workaround unless your organization specifically requires building packages from source.

### Activate the environment later

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation due to its execution policy, follow your organization's approved guidance for enabling virtual-environment activation. Alternatively, run the environment's Python directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\venv\Scripts\python.exe .\app.py
```

## Run the application

With the virtual environment activated and the current directory set to the project root:

```powershell
python .\app.py
```

Alternatively:

```powershell
python .\run.py
```

## Basic workflow

1. Open **Import Dataset** and load a CSV, XLSX, or tab-delimited TXT file.
2. Enter the exact name of the target column and select the task type.
3. Optionally visit **Data Processing** to inspect or prepare the data. Export a copy if you want to keep a checkpoint.
4. Open **Data Visualization** to view an overview of the configured dataset.
5. Open **Machine Learning**, choose a test size, and train the models.
6. Review model metrics, select a trained model, and use **Show Visuals** or **Predict Sample**.
7. Save the selected model or export a PDF report if needed.

## Generated files

The application creates output directories under `results/` for dataset-overview and model plots. The `results/` folder can be regenerated and does not need to be committed to source control unless you want to keep particular outputs.

Models are saved in joblib/pickle format. **Only load model files from sources you trust**; serialized Python objects can execute unsafe code when loaded.

## Troubleshooting

### `py -3.12` cannot find Python
Check installed Python versions with:

```powershell
py -0p
```

Install an approved Python 3.12 distribution if it is not listed, then recreate the virtual environment using `py -3.12 -m venv venv`.

### Pip downloads a Matplotlib or NumPy source archive and fails to build
Make sure the environment is actually using Python 3.12:

```powershell
python --version
```

If it is, the configured package index may not provide a compatible wheel. Contact the administrator of that index and provide the pip error. Avoid using a different package index if your organization requires its approved repository.

### An image, PDF, or tutorial does not appear
Check the file and path: image assets should be inside the root `images/` or `icons/` folder; the documentation PDF and tutorial video should be in the project root with the expected filenames.

### Excel support is unavailable
Confirm that `openpyxl` is installed in the active virtual environment:

```powershell
python -m pip show openpyxl
```
