# this app has developed by Idriss khattabi - Mohammed Amine Sabbahi - Aymen Boufarhi

import customtkinter as ctk
from PIL import Image
import tkinter.filedialog as fd
import tkinter as tk
from tkinter import ttk
from CTkMessagebox import CTkMessagebox
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import io
import contextlib

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler, PowerTransformer, QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

# *********************************************************************************

ctk.set_appearance_mode("dark")

app = ctk.CTk()
app.geometry('1200x1000')
app.iconbitmap('images/icon.ico')
app.title('ModelMaster')
dataset = None
creat_file_path = None


# Fonction permet de cree un jeu de donnees
def creat_dataset():
    global creat_file_path
    # Create a sample DataFrame (you can replace this with your dataset creation logic)
    creat_file_path = fd.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])

    # Check if the user canceled file selection
    if not creat_file_path:
        return

    # Create a sample DataFrame (you can replace this with your dataset creation logic)
    example_data = {
        'Column1': [1, 2, 3],
        'Column2': ['A', 'B', 'C']
    }
    df = pd.DataFrame(example_data)

    # Write the DataFrame to the selected Excel file
    df.to_excel(creat_file_path, index=False)

    # Check if the file was created successfully
    if os.path.exists(creat_file_path):
        try:
            os.startfile(creat_file_path)
        except Exception as e:
            CTkMessagebox(title="Warning", message=f"Failed to open Excel: {e}", icon="warning", option_1="Ok")
    else:
        CTkMessagebox(title="Warning", message=f"Failed to creat dataset", icon="cancel", option_1="Ok")


# Fonction permet de Import un jeu de donnees
def import_dataset():
    global dataset
    filetypes = [
        ("CSV Files", "*.csv"),
        ("Excel Files", "*.xlsx"),
        ("Text Files", "*.txt"),
        ("All Files", "*.*")
    ]
    file_path = ctk.filedialog.askopenfilename(filetypes=filetypes)
    if file_path:
        try:
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == ".csv":
                dataset = pd.read_csv(file_path, dtype=object)
            elif file_extension == ".xlsx":
                dataset = pd.read_excel(file_path, dtype=object)
            elif file_extension == ".txt":
                dataset = pd.read_csv(file_path, dtype=object, delimiter='\t')  # Assuming it's tab-separated
            else:
                raise ValueError("Unsupported file format")

            if dataset is not None :
                show_table(dataset, table_frame)

        except Exception as e:
            CTkMessagebox(title="Warning", message=f"Failed to load dataset: {e}", icon="warning", option_1="Ok")


table_frame = None
columns_list = None
# Fonction permet de afficher la table de jeu de donnees danse la page Import
def show_table(data, frame):
    global columns_list
    if data is not None:
        # Clear any existing data in the table_frame
        for widget in frame.winfo_children():
            widget.destroy()

        columns_list = data.columns.tolist()
        style = ttk.Style()
        style.configure('Alternate.Treeview', background='lightblue', fieldbackground='white', font=('Courier', 15))

        # Configure the separator style for columns
        style.layout('Treeview', [('Treeview.treearea', {'sticky': 'nswe'})])  # Set separator style

        # Create a Treeview widget
        tree = ttk.Treeview(frame, columns=columns_list, show="headings", selectmode="extended")
        tree.config(style='Alternate.Treeview')

        # Set up columns
        for col in columns_list:
            tree.heading(col, text=col, anchor=tk.CENTER)  # Center-align headings
            tree.column(col, width=100, anchor=tk.CENTER)  # Adjust width as needed

        # Insert data into the table
        for index, row in data.iterrows():
            tree.insert("", "end", values=row.tolist())

        # Add a vertical scrollbar
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        vsb.pack(side="right", fill="y")
        tree.configure(yscrollcommand=vsb.set)

        # Add a horizontal scrollbar
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        hsb.pack(side="bottom", fill="x")
        tree.configure(xscrollcommand=hsb.set)

        tree.pack(fill='both', expand=True)
    else:
        CTkMessagebox(title="Error", message="No Data to display!", icon="cancel", option_1="Ok")


target_name = ""
target_type = ""
def submit_values():
    global target_name, target_type, dataset, columns_list, table_frame, creat_file_path
    if creat_file_path is not None:
        dataset = pd.read_excel(creat_file_path)
        creat_file_path = None

    columns_list = dataset.columns.tolist()
    target_name = target_column.get()
    target_type = label_type_var.get()

    if dataset is not None:
        if target_name != "" and target_type != "":
            if target_name in columns_list:
                show_table(dataset, table_frame)
                CTkMessagebox(title="Done", message=f"The target name = '{target_name}', and target_type = '{target_type}' have saved seccussfully", icon="check", option_1="Ok")
            else:
                CTkMessagebox(title="Error", message=f"There is no column named {target_name} in the imported dataset", icon="cancel", option_1="Ok")
        else:
            CTkMessagebox(title="Error", message="Please specify the target column name and type first.", icon="cancel", option_1="Ok")
    else:
        CTkMessagebox(title="Error", message="No Data, Please import a dataset first.", icon="cancel", option_1="Ok")


# Les fonctiones de traitement de dataset
def update_info_list():
    global dataset, dataset_info_var
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        dataset.info()
        info_output = buf.getvalue()
    # Displaying the formatted output excluding the first three lines
    info_lines = info_output.split('\n')
    info_to_display = '\n'.join(info_lines[5:])
    dataset_info_var.set(info_to_display)


def delete_columns():
    global dataset, column_entry
    cols = column_entry.get()
    columns_to_drop = cols.split(',')
    columns_to_drop = [s.strip() for s in columns_to_drop]

    cnt = 0
    for col in columns_to_drop:
        if col in dataset.columns.tolist():
            cnt += 1
    if cnt == len(columns_to_drop):
        dataset.drop(columns=columns_to_drop, axis=1, inplace=True)
        update_table()
        update_info_list()
        CTkMessagebox(title="Success",  message=f"Columns {" ".join(columns_to_drop)} deleted successfully.", icon="check", option_1="Ok")
        column_entry.delete(0, 'end')
    else:
        CTkMessagebox(title="Error", message="Please enter columns name correctly.", icon="cancel", option_1="Ok")
        column_entry.delete(0, 'end')


def update_table():
    global dataset, process_frame
    show_table(dataset, process_frame)


def features_scaling():
    global dataset, fs_method_option, target_name

    print(f"target_name : {target_name}")

    if not target_name in dataset.columns or target_name == "":
        print("you must choose the target column name in import Dataset Page to do feature scaling!!")
        CTkMessagebox(title="Error", message="target_name is not define, PLZ choose the target column name in import Dataset Page to do feature scaling!!", icon="cancel", option_1="Ok")
        return
    
    if dataset is not None:
        fs_method = fs_method_option.get()
        scaler = None
        if fs_method == "MinMaxScaler" :
            print("MinMaxScaler")
            scaler = MinMaxScaler()
        elif fs_method == "StandardScaler" :
            print("StandardScaler")
            scaler = StandardScaler()
        elif fs_method == "Normalizer" :
            print("Normalizer")
            scaler = Normalizer()
        elif fs_method == "RobustScaler" :
            print("RobustScaler")
            scaler = RobustScaler()
        elif fs_method == "PowerTransformer" :
            print("PowerTransformer")
            scaler = PowerTransformer()
        else :
            print("QuantileTransformer")
            scaler = QuantileTransformer()

        columns = dataset.columns
        columns_without_target = columns.drop(target_name)
        #print(pd.DataFrame(dataset[columns_without_target]))
        dataset[columns_without_target] = scaler.fit_transform(dataset[columns_without_target])

        update_table()
        CTkMessagebox(title="Success", message=f"Features scaled successfully.", icon="check", option_1="Ok")
        
    else:
        CTkMessagebox(title="Error",  message="No Data, Please import a dataset first.", icon="cancel", option_1="Ok")


def check_nan():
    global dataset, check_label_var
    columns_with_missing_values = dataset.isna().sum()
    columns_with_missing_values = columns_with_missing_values[columns_with_missing_values > 0]
    if dataset is not None:
        check_label_var.set(f"{columns_with_missing_values}\nTotal: {columns_with_missing_values.sum()} missig values")
    else:
        CTkMessagebox(title="Error", message="No Data, Please import a dataset first.", icon="cancel", option_1="Ok")


def handle_data_by_method():
    global dataset, method_option, mv_features
    if dataset is not None:
        method = method_option.get()
        columns_to_handle = mv_features.get().split(',')
        columns_to_handle = [s.strip() for s in columns_to_handle]

        if method != "" and columns_to_handle != "":

            if method == "Mean":
                for col in columns_to_handle:
                    dataset[col].fillna(value=pd.to_numeric(dataset[col], errors='coerce').mean(), inplace=True)
            elif method == "Max":
                for col in columns_to_handle:
                    dataset[col].fillna(value=pd.to_numeric(dataset[col], errors='coerce').max(), inplace=True)
            elif method == "Min":
                for col in columns_to_handle:
                    dataset[col].fillna(value=pd.to_numeric(dataset[col], errors='coerce').min(), inplace=True)
            elif method == "Previous value":
                for col in columns_to_handle:
                    dataset[col].fillna(method='ffill', inplace=True)
            elif method == "Next value":
                for col in columns_to_handle:
                    dataset[col].fillna(method='bfill', inplace=True)
            elif method == "Delete row":
                dataset.dropna(inplace=True)

            mv_features.delete(0, 'end')
            update_table()
            update_info_list()
            CTkMessagebox(title="Success", message=f"Missing Values Handled successfully.", icon="check", option_1="Ok")

        else:
            CTkMessagebox(title="Error", message="Please specify method and features.", icon="cancel", option_1="Ok")

        mv_features.delete(0, 'end')
        check_nan()

    else:
        CTkMessagebox(title="Error", message="Please specify method and features.", icon="cancel", option_1="Ok")


def handle_data_by_value():
    global value_entry, na_by_value_entry
    if dataset is not None:
        value = value_entry.get()
        features = na_by_value_entry.get().split(',')
        features = [s.strip() for s in features]

        if value != "" and features != "":
            for col in features:
                dataset[col].fillna(value=value, inplace=True)
            mv_features.delete(0, 'end')
            update_table()
            update_info_list()
            CTkMessagebox(title="Success", message=f"Missing Values Handled successfully.", icon="check", option_1="Ok")

        else:
            CTkMessagebox(title="Error", message="Please specify value and features correctly.", icon="cancel", option_1="Ok")
        na_by_value_entry.delete(0, 'end')
        value_entry.delete(0, 'end')
        check_nan()

    else:
        CTkMessagebox(title="Error", message="No Data, Please import a dataset first.", icon="cancel", option_1="Ok")

        na_by_value_entry.delete(0, 'end')
        value_entry.delete(0, 'end')


def data_encoding():
    global dataset, encoding_entry, encodin_value_entry, encodin_cat_value_entry
    feature = encoding_entry.get()
    value = encodin_value_entry.get()
    cat_value = encodin_cat_value_entry.get()

    if dataset is not None:
        if feature in dataset.columns.tolist():
            dataset[feature] = dataset[feature].replace(cat_value, value)
            update_table()
            CTkMessagebox(title="Success", message=f"The data encoded successfully.", icon="check", option_1="Ok")

        else:
            CTkMessagebox(title="Error", message="Please enter a feature from the dataset.", icon="cancel", option_1="Ok")
        encodin_value_entry.delete(0, 'end')
        encodin_cat_value_entry.delete(0, 'end')

    else:
        CTkMessagebox(title="Error", message="No Data, Please import a dataset first.", icon="cancel", option_1="Ok")
        encodin_value_entry.delete(0, 'end')
        encodin_cat_value_entry.delete(0, 'end')


def export_dataset():
    global dataset
    if dataset is not None:
        filetypes = [
            ("CSV Files", "*.csv"),
            ("Excel Files", "*.xlsx"),
            ("Text Files", "*.txt")
        ]
        file_path = ctk.filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=filetypes[0])

        if file_path:
            try:
                file_extension = os.path.splitext(file_path)[1].lower()

                if file_extension == ".csv":
                    dataset.to_csv(file_path, index=False)
                elif file_extension == ".xlsx":
                    dataset.to_excel(file_path, index=False)
                elif file_extension == ".txt":
                    dataset.to_csv(file_path, index=False, sep='\t')
                else:
                    raise ValueError("Unsupported file format")

                CTkMessagebox(title="Success", message=f"Dataset saved successfully to:\n{file_path}", icon="check", option_1="Ok")

            except Exception as e:
                CTkMessagebox(title="Error", message=f"Failed to save dataset: {e}", icon="cancel", option_1="Ok")
                
    else:
        CTkMessagebox(title="Error", message="No Data to export!", icon="cancel", option_1="Ok")


# *** Les Fonctiones des algorithmes ML *** :
model = None
repport1 = None
accuracy1 = None
classifier1 = None
y_pred1 = None
y_test1 = None
def native_bayes(ts):
    global dataset, target_name, repport1, accuracy1, classifier1, y_pred1, y_test1
    if dataset is not None:
        categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
        # Transform categorical columns using LabelEncoder
        encoder = LabelEncoder()
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                dataset[col] = encoder.fit_transform(dataset[col])
        X = dataset.drop(target_name, axis=1)
        y = dataset[target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
        # Initialize and fit the Naive Bayes model
        nb_classifier = GaussianNB()
        nb_classifier.fit(X_train, y_train)
        # Predict on the test set
        y_pred = nb_classifier.predict(X_test)
        # Evaluate the model
        accuracy1 = accuracy_score(y_test, y_pred)
        # Display classification report
        repport1 = classification_report(y_test, y_pred)
        classifier1 = nb_classifier
        y_pred1 = y_pred
        y_test1 = y_test


repport2 = None
accuracy2 = None
classifier2 = None
y_pred2 = None
y_test2 = None
def random_forest(ts):
    global dataset, target_name, repport2, accuracy2, classifier2, y_test2, y_pred2
    if dataset is not None:
        categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
        # Transform categorical columns using LabelEncoder
        encoder = LabelEncoder()
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                dataset[col] = encoder.fit_transform(dataset[col])
        X = dataset.drop(target_name, axis=1)
        y = dataset[target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
        # Initialize and fit the Random Forest model
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)
        # Predict on the test set
        y_pred = rf_classifier.predict(X_test)
        # Evaluate the model
        accuracy2 = accuracy_score(y_test, y_pred)
        # Display classification report
        repport2 = classification_report(y_test, y_pred)

        classifier2 = rf_classifier
        y_pred2 = y_pred
        y_test2 = y_test


repport3 = None
accuracy3 = None
classifier3 = None
y_pred3 = None
y_test3 = None
def decision_tree(ts):
    global dataset, target_name, repport3, accuracy3, classifier3, y_pred3, y_test3
    if dataset is not None:
        categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
        # Transform categorical columns using LabelEncoder
        encoder = LabelEncoder()
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                dataset[col] = encoder.fit_transform(dataset[col])
        X = dataset.drop(target_name, axis=1)
        y = dataset[target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
        # Initialize and fit the Naive Bayes model
        dt_classifier = DecisionTreeClassifier(max_depth=3, min_samples_split=5)
        dt_classifier.fit(X_train, y_train)
        # Predict on the test set
        y_pred = dt_classifier.predict(X_test)
        # Evaluate the model
        accuracy3 = accuracy_score(y_test, y_pred)
        # Display classification report
        repport3 = classification_report(y_test, y_pred)

        classifier3 = dt_classifier
        y_pred3 = y_pred
        y_test3 = y_test


repport4 = None
accuracy4 = None
classifier4 = None
y_pred4 = None
y_test4 = None
def knn(ts):
    global dataset, target_name, repport4, accuracy4, classifier4, y_pred4, y_test4
    if dataset is not None:
        categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
        # Transform categorical columns using LabelEncoder
        encoder = LabelEncoder()
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                dataset[col] = encoder.fit_transform(dataset[col])
        X = dataset.drop(target_name, axis=1)
        y = dataset[target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
        # Initialize and fit the Naive Bayes model
        knn_classifier = KNeighborsClassifier()
        knn_classifier.fit(X_train, y_train)
        classifier4 = knn_classifier
        # Predict on the test set
        y_pred = knn_classifier.predict(X_test)
        y_pred4 = y_pred
        y_test4 = y_test
        # Evaluate the model
        accuracy4 = accuracy_score(y_test, y_pred)
        # Display classification report
        repport4 = classification_report(y_test, y_pred)


repport5 = None
accuracy5 = None
classifier5 = None
y_pred5 = None
y_test5 = None
def sv_classifier(ts):
    global dataset, target_name, repport5, accuracy5, classifier5, y_pred5, y_test5
    if dataset is not None:
        categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
        # Transform categorical columns using LabelEncoder
        encoder = LabelEncoder()
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                dataset[col] = encoder.fit_transform(dataset[col])
        X = dataset.drop(target_name, axis=1)
        y = dataset[target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
        # Initialize and fit the Naive Bayes model
        svc = SVC(kernel='linear')
        svc.fit(X_train, y_train)
        # Predict on the test set
        y_pred = svc.predict(X_test)
        # Evaluate the model
        accuracy5 = accuracy_score(y_test, y_pred)
        # Display classification report
        repport5 = classification_report(y_test, y_pred)

        classifier5 = svc
        y_pred5 = y_pred
        y_test5 = y_test


repport6 = None
accuracy6 = None
classifier6 = None
y_pred6 = None
y_test6 = None
def nn_classifier(ts):
    global dataset, target_name, repport6, accuracy6, classifier6, y_pred6, y_test6
    if dataset is not None:
        categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
        # Transform categorical columns using LabelEncoder
        encoder = LabelEncoder()
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                dataset[col] = encoder.fit_transform(dataset[col])
        X = dataset.drop(target_name, axis=1)
        y = dataset[target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
        # Initialize and fit the Naive Bayes model
        nnc = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
        nnc.fit(X_train, y_train)
        # Predict on the test set
        y_pred = nnc.predict(X_test)
        # Evaluate the model
        accuracy6 = accuracy_score(y_test, y_pred)
        # Display classification report
        repport6 = classification_report(y_test, y_pred)

        classifier6 = nnc
        y_pred6 = y_pred
        y_test6 = y_test


def apply_ml_classifiers():
    ts = float(test_size.get())
    print(ts)
    native_bayes(ts)
    random_forest(ts)
    decision_tree(ts)
    knn(ts)
    sv_classifier(ts)
    nn_classifier(ts)
    if dataset is not None:
        # Text area 1 modification
        text_area1.config(state=tk.NORMAL)  # Set state to normal to modify
        text_area1.delete(1.0, tk.END)  # Clear previous content
        text_area1.insert(tk.END, f"\t\t\t\tAccuracy: {accuracy1 * 100:.2f}%\n\n")
        text_area1.insert(tk.END, f"\n\n {repport1}")  # Insert new classification report
        text_area1.config(state=tk.DISABLED)

        # Text area 2 modification
        text_area2.config(state=tk.NORMAL)  # Set state to normal to modify
        text_area2.delete(1.0, tk.END)  # Clear previous content
        text_area2.insert(tk.END, f"\t\t\t\tAccuracy: {accuracy2 * 100:.2f}\n\n")
        text_area2.insert(tk.END, f"\n\n {repport2}")  # Insert new classification report
        text_area2.config(state=tk.DISABLED)

        # Text area 3 modification
        text_area3.config(state=tk.NORMAL)  # Set state to normal to modify
        text_area3.delete(1.0, tk.END)  # Clear previous content
        text_area3.insert(tk.END, f"\t\t\t\tAccuracy: {accuracy3 * 100:.2f}\n\n")
        text_area3.insert(tk.END, f"\n\n {repport3}")  # Insert new classification report
        text_area3.config(state=tk.DISABLED)

        # Text area 4 modification
        text_area4.config(state=tk.NORMAL)  # Set state to normal to modify
        text_area4.delete(1.0, tk.END)  # Clear previous content
        text_area4.insert(tk.END, f"\t\t\t\tAccuracy: {accuracy4 * 100:.2f}\n\n")
        text_area4.insert(tk.END, f"\n\n {repport4}")  # Insert new classification report
        text_area4.config(state=tk.DISABLED)

        # Text area 5 modification
        text_area5.config(state=tk.NORMAL)  # Set state to normal to modify
        text_area5.delete(1.0, tk.END)  # Clear previous content
        text_area5.insert(tk.END, f"\t\t\t\tAccuracy: {accuracy5 * 100:.2f}\n\n")
        text_area5.insert(tk.END, f"\n\n {repport5}")  # Insert new classification report
        text_area5.config(state=tk.DISABLED)

        # Text area 5 modification
        text_area6.config(state=tk.NORMAL)  # Set state to normal to modify
        text_area6.delete(1.0, tk.END)  # Clear previous content
        text_area6.insert(tk.END, f"\t\t\t\tAccuracy: {accuracy6 * 100:.2f}\n\n")
        text_area6.insert(tk.END, f"\n\n {repport6}")  # Insert new classification report
        text_area6.config(state=tk.DISABLED)

    else:
        CTkMessagebox(title="Error",
                      message="No Data, Please import a dataset first.",
                      icon="cancel",
                      option_1="Ok")


mse1 = None
mae1 = None
r21 = None
explained_var1 = None
y_pred21 = None
y_test21 = None
predictor1 = None
def Linear_Regression(ts):
    global dataset, mse1, mae1, r21, explained_var1, target_name, y_pred21, y_test21, predictor1

    # Check for categorical columns
    categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()

    # Transform categorical columns using LabelEncoder
    encoder = LabelEncoder()
    for col in categorical_cols:
        dataset[col] = encoder.fit_transform(dataset[col])

    # Split data into features and target
    X = dataset.drop(target_name, axis=1)
    y = dataset[target_name]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # Create and train the Linear Regression model
    l_regressor = LinearRegression()
    l_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = l_regressor.predict(X_test)

    # Evaluate the model
    mse1 = mean_squared_error(y_test, y_pred)
    mae1 = mean_absolute_error(y_test, y_pred)
    r21 = r2_score(y_test, y_pred)
    explained_var1 = explained_variance_score(y_test, y_pred)

    y_pred21 = y_pred
    y_test21 = y_test
    predictor1 = l_regressor


mse2 = None
mae2 = None
r22 = None
explained_var2 = None
y_pred22 = None
y_test22 = None
predictor2 = None
def SV_Regression(ts):
    global dataset, mse2, mae2, r22, explained_var2, target_name, y_pred22, y_test22, predictor2

    # Check for categorical columns
    categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()

    # Transform categorical columns using LabelEncoder
    encoder = LabelEncoder()
    for col in categorical_cols:
        dataset[col] = encoder.fit_transform(dataset[col])

    # Split data into features and target
    X = dataset.drop(target_name, axis=1)
    y = dataset[target_name]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # Create and train the SVR model
    sv_regressor = SVR()
    sv_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = sv_regressor.predict(X_test)

    # Evaluate the model
    mse2 = mean_squared_error(y_test, y_pred)
    mae2 = mean_absolute_error(y_test, y_pred)
    r22 = r2_score(y_test, y_pred)
    explained_var2 = explained_variance_score(y_test, y_pred)

    y_pred22 = y_pred
    y_test22 = y_test
    predictor2 = sv_regressor


mse3 = None
mae3 = None
r23 = None
explained_var3 = None
y_pred23 = None
y_test23 = None
predictor3 = None
def DT_Regression(ts):
    global dataset, mse3, mae3, r23, explained_var3, target_name, y_test23, y_pred23, predictor3

    # Check for categorical columns
    categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
    # Transform categorical columns using LabelEncoder
    encoder = LabelEncoder()
    for col in categorical_cols:
        dataset[col] = encoder.fit_transform(dataset[col])
    # Split data into features and target
    X = dataset.drop(target_name, axis=1)
    y = dataset[target_name]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # Create and train the Decision Tree Regression model
    dt_regressor = DecisionTreeRegressor(max_depth=3, min_samples_split=5)
    dt_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = dt_regressor.predict(X_test)

    # Evaluate the model
    mse3 = mean_squared_error(y_test, y_pred)
    mae3 = mean_absolute_error(y_test, y_pred)
    r23 = r2_score(y_test, y_pred)
    explained_var3 = explained_variance_score(y_test, y_pred)

    y_pred23 = y_pred
    y_test23 = y_test
    predictor3 = dt_regressor


mse4 = None
mae4 = None
r24 = None
explained_var4 = None
y_pred24 = None
y_test24 = None
predictor4 = None
def NN_Regression(ts):
    global dataset, mse4, mae4, r24, explained_var4, target_name, y_test24, y_pred24, predictor4

    # Check for categorical columns
    categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()

    # Transform categorical columns using LabelEncoder
    encoder = LabelEncoder()
    for col in categorical_cols:
        dataset[col] = encoder.fit_transform(dataset[col])

    # Split data into features and target
    X = dataset.drop(target_name, axis=1)
    y = dataset[target_name]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # Create and train the Neural Network Regression model
    nn_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
    nn_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = nn_regressor.predict(X_test)

    # Evaluate the model
    mse4 = mean_squared_error(y_test, y_pred)
    mae4 = mean_absolute_error(y_test, y_pred)
    r24 = r2_score(y_test, y_pred)
    explained_var4 = explained_variance_score(y_test, y_pred)

    y_pred24 = y_pred
    y_test24 = y_test
    predictor4 = nn_regressor


mse5 = None
mae5 = None
r25 = None
explained_var5 = None
y_pred25 = None
y_test25 = None
predictor5 = None
def RF_Regression(ts):
    global mse5, mae5, r25, explained_var5, target_name, y_test25, y_pred25, predictor5

    # Check for categorical columns
    categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()

    # Transform categorical columns using LabelEncoder
    encoder = LabelEncoder()
    for col in categorical_cols:
        dataset[col] = encoder.fit_transform(dataset[col])

    # Split data into features and target
    X = dataset.drop(target_name, axis=1)
    y = dataset[target_name]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # Create and train the Neural Network Regression model
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_regressor.predict(X_test)

    # Evaluate the model
    mse5 = mean_squared_error(y_test, y_pred)
    mae5 = mean_absolute_error(y_test, y_pred)
    r25 = r2_score(y_test, y_pred)
    explained_var5 = explained_variance_score(y_test, y_pred)

    y_pred25 = y_pred
    y_test25 = y_test
    predictor5 = rf_regressor


mse6 = None
mae6 = None
r26 = None
explained_var6 = None
y_pred26 = None
y_test26 = None
predictor6 = None
def KNN_Regression(ts):
    global mse6, mae6, r26, explained_var6, y_test26, y_pred26, predictor6

    # Check for categorical columns
    categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()

    # Transform categorical columns using LabelEncoder
    encoder = LabelEncoder()
    for col in categorical_cols:
        dataset[col] = encoder.fit_transform(dataset[col])

    # Split data into features and target
    X = dataset.drop(target_name, axis=1)
    y = dataset[target_name]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # Create and train the Neural Network Regression model
    rf_regressor = KNeighborsRegressor(n_neighbors=5)
    rf_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_regressor.predict(X_test)

    # Evaluate the model
    mse6 = mean_squared_error(y_test, y_pred)
    mae6 = mean_absolute_error(y_test, y_pred)
    r26 = r2_score(y_test, y_pred)
    explained_var6 = explained_variance_score(y_test, y_pred)

    y_pred26 = y_pred
    y_test26 = y_test
    predictor6 = rf_regressor


def apply_ml_predictions():
    ts = float(test_size.get())
    print(ts)
    # Appliquer les algorithmes de machine learning pour les problemes de regression
    Linear_Regression(ts)
    SV_Regression(ts)
    DT_Regression(ts)
    NN_Regression(ts)
    RF_Regression(ts)
    KNN_Regression(ts)
    if dataset is not None:
        # Text area 1 modification
        text_area1.config(state=tk.NORMAL)  # Set state to normal to modify
        text_area1.delete(1.0, tk.END)  # Clear previous content
        text_area1.insert(tk.END, f"\n\n\t\tMean Squared Error (MSE): {mse1}\n\n")
        text_area1.insert(tk.END, f"\n\n\t\tMean Absolute Error (MAE): {mae1}\n\n")
        text_area1.insert(tk.END, f"\n\n\t\tR-squared (R²): {r21}\n\n")
        text_area1.insert(tk.END, f"\n\n\t\tExplained Variance Score: {explained_var1}\n\n")
        text_area1.config(state=tk.DISABLED)

        # Text area 2 modification
        text_area2.config(state=tk.NORMAL)  # Set state to normal to modify
        text_area2.delete(1.0, tk.END)  # Clear previous content
        text_area2.insert(tk.END, f"\n\n\t\tMean Squared Error (MSE): {mse2}\n\n")
        text_area2.insert(tk.END, f"\n\n\t\tMean Absolute Error (MAE): {mae2}\n\n")
        text_area2.insert(tk.END, f"\n\n\t\tR-squared (R²): {r22}\n\n")
        text_area2.insert(tk.END, f"\n\n\t\tExplained Variance Score: {explained_var2}\n\n")
        text_area2.config(state=tk.DISABLED)

        # Text area 3 modification
        text_area3.config(state=tk.NORMAL)  # Set state to normal to modify
        text_area3.delete(1.0, tk.END)  # Clear previous content
        text_area3.insert(tk.END, f"\n\n\t\tMean Squared Error (MSE): {mse3}\n\n")
        text_area3.insert(tk.END, f"\n\n\t\tMean Absolute Error (MAE): {mae3}\n\n")
        text_area3.insert(tk.END, f"\n\n\t\tR-squared (R²): {r23}\n\n")
        text_area3.insert(tk.END, f"\n\n\t\tExplained Variance Score: {explained_var3}\n\n")
        text_area3.config(state=tk.DISABLED)

        # Text area 4 modification
        text_area4.config(state=tk.NORMAL)  # Set state to normal to modify
        text_area4.delete(1.0, tk.END)  # Clear previous content
        text_area4.insert(tk.END, f"\n\n\t\tMean Squared Error (MSE): {mse4}\n\n")
        text_area4.insert(tk.END, f"\n\n\t\tMean Absolute Error (MAE): {mae4}\n\n")
        text_area4.insert(tk.END, f"\n\n\t\tR-squared (R²): {r24}\n\n")
        text_area4.insert(tk.END, f"\n\n\t\tExplained Variance Score: {explained_var4}\n\n")
        text_area4.config(state=tk.DISABLED)

        # Text area 5 modification
        text_area5.config(state=tk.NORMAL)  # Set state to normal to modify
        text_area5.delete(1.0, tk.END)  # Clear previous content
        text_area5.insert(tk.END, f"\n\n\t\tMean Squared Error (MSE): {mse5}\n\n")
        text_area5.insert(tk.END, f"\n\n\t\tMean Absolute Error (MAE): {mae5}\n\n")
        text_area5.insert(tk.END, f"\n\n\t\tR-squared (R²): {r25}\n\n")
        text_area5.insert(tk.END, f"\n\n\t\tExplained Variance Score: {explained_var5}\n\n")
        text_area5.config(state=tk.DISABLED)

        # Text area 5 modification
        text_area6.config(state=tk.NORMAL)  # Set state to normal to modify
        text_area6.delete(1.0, tk.END)  # Clear previous content
        text_area6.insert(tk.END, f"\n\n\t\tMean Squared Error (MSE): {mse6}\n\n")
        text_area6.insert(tk.END, f"\n\n\t\tMean Absolute Error (MAE): {mae6}\n\n")
        text_area6.insert(tk.END, f"\n\n\t\tR-squared (R²): {r26}\n\n")
        text_area6.insert(tk.END, f"\n\n\t\tExplained Variance Score: {explained_var6}\n\n")
        text_area6.config(state=tk.DISABLED)

    else:
        CTkMessagebox(title="Error",
                      message="No Data, Please import a dataset first.",
                      icon="cancel",
                      option_1="Ok")


classifier = None
def visualise_classification_result():
    global  classifier
    model = chosed_model.get()
    if model == 'Native Bayes':
        y_pred = y_pred1
        y_test = y_test1
        classifier = classifier1
    elif model == 'Decision Tree':
        y_pred = y_pred3
        y_test = y_test3
        classifier = classifier3
    elif model == 'Random Forest Classifier':
        y_pred = y_pred2
        y_test = y_test2
        classifier = classifier2
    elif model == 'K-NN Classifier':
        y_pred = y_pred4
        y_test = y_test4
        classifier = classifier4
    elif model == 'SVM Classifier':
        y_pred = y_pred5
        y_test = y_test5
        classifier = classifier5
    elif model == 'Neural Network Classifier':
        y_pred = y_pred6
        y_test = y_test6
        classifier = classifier6
    X = dataset.drop(target_name, axis=1)
    y = dataset[target_name]

    conf_matrix = confusion_matrix(y_test, y_pred)
    # Displaying the confusion matrix using Seaborn heatmap
    plt.figure()  # Create a new figure instance with a unique number
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix of {model} classifier')
    # Save the figure as PNG
    plt.savefig('results/classification/plot1.png', bbox_inches='tight')
    plt.close()  # Close the figure without displaying
    accuracy = accuracy_score(y_test, y_pred)
    incorrect = 1 - accuracy
    # Labels for the pie chart
    labels = ['Correct Predictions', 'Incorrect Predictions']
    sizes = [accuracy, incorrect]
    colors = ['purple', 'orange']
    # Create a pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(f'Accuracy of Predictions for {model} classifier')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('results/classification/plot2.png', bbox_inches='tight')
    plt.close()


    train_sizes, train_scores, test_scores = learning_curve(
        classifier, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
    )

    # Calculate mean and standard deviation of training and test scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plotting the learning curve
    plt.figure(figsize=(8, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='orange')

    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='Validation Score')

    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(f'Learning Curve for {model} Classifier')
    plt.legend()
    plt.grid()
    plt.savefig('results/classification/plot3.png', bbox_inches='tight')
    plt.close()
    if model == 'Decision Tree':
        from sklearn.tree import plot_tree
        X_df = pd.DataFrame(X, columns=dataset.drop(target_name, axis=1).columns)

        # Plot the decision tree
        plt.figure(figsize=(10, 7))  # Adjust the figure size if needed
        plot_tree(classifier3, filled=True, feature_names=X_df.columns.tolist())
        plt.title("Decision Tree Visualization")
        plt.savefig('results/classification/plot4.png', bbox_inches='tight')
        plt.close()
    else:
        # Plotting ROC curve
        if len(np.unique(y_pred)) == 2:
            # Compute ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.savefig('results/classification/plot4.png', bbox_inches='tight')
            plt.close()
        else:
            differences = np.array(y_test) - np.array(y_pred)
            # Plotting the differences as a bar chart
            plt.figure()
            plt.bar(range(len(differences)), differences, alpha=0.7)
            plt.xlabel('Data Points')
            plt.ylabel('Difference (Actual - Predicted)')
            plt.title('Differences between Actual and Predicted Values')
            plt.savefig('results/classification/plot4.png', bbox_inches='tight')
            plt.close()


predictor = None
def visualise_regression_result():
    global predictor
    model = chosed_model.get()
    if model == 'Linear Regression':
        y_pred = y_pred21
        y_test = y_test21
        predictor = predictor1
    elif model == 'Decision Tree Regression':
        y_pred = y_pred23
        y_test = y_test23
        predictor = predictor3
    elif model == 'SVM Regression':
        y_pred = y_pred22
        y_test = y_test22
        predictor = predictor2
    elif model == 'Neural Network Regression':
        y_pred = y_pred24
        y_test = y_test24
        predictor = predictor4
    elif model == 'Random Forest Regression':
        y_pred = y_pred25
        y_test = y_test25
        predictor = predictor5
    elif model == 'K-NN Regression':
        y_pred = y_pred26
        y_test = y_test26
        predictor = predictor6
    X = dataset.drop(target_name, axis=1)
    y = dataset[target_name]

    # Create a scatter plot of actual vs predicted values
    plt.scatter(y_test, y_pred, alpha=0.5, label='Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    # Plotting the regression line
    fit = np.polyfit(y_test, y_pred, deg=1)
    plt.plot(y_test, fit[0] * y_test + fit[1], color='red', label='Fitted Line')
    plt.legend()
    plt.title(f'Fitting of {model}')
    plt.savefig('results/regression/plot1.png', bbox_inches='tight')
    plt.close()

    # Calculate learning curves for the Naive Bayes classifier
    train_sizes, train_scores, test_scores = learning_curve(
        predictor, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
    )

    # Calculate mean and standard deviation of training and test scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plotting the learning curve
    plt.figure(figsize=(8, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='orange')

    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='Validation Score')

    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(f'Learning Curve for {model} predictor')
    plt.legend()
    plt.grid()
    plt.savefig('results/regression/plot2.png', bbox_inches='tight')
    plt.close()

    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='--')  # Adding a horizontal line at y=0
    plt.title(f'Residual Plot of {model} model')
    plt.savefig('results/regression/plot3.png', bbox_inches='tight')
    plt.close()

    if model == 'Decision Tree Regression':
        from sklearn.tree import plot_tree
        X_df = pd.DataFrame(X, columns=dataset.drop(target_name, axis=1).columns)

        # Plot the decision tree
        plt.figure(figsize=(10, 7))  # Adjust the figure size if needed
        plot_tree(predictor3, filled=True, feature_names=X_df.columns.tolist())
        plt.title("Decision Tree Visualization")
        plt.savefig('results/regression/plot4.png', bbox_inches='tight')
        plt.close()
    else:
        importances = predictor.coef_
        feature_names = X.columns

        plt.barh(feature_names, importances)
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title(f"Feature Importance for {model}")
        plt.savefig('results/regression/plot4.png')
        plt.close()


def process_chosen_model1():
    global models_frame, chosed_model
    model = chosed_model.get()
    if model != "Choose an algorithm":
        visualise_classification_result()
        for frame in models_frame.winfo_children():
            frame.destroy()
        models_frame.configure(fg_color="white")
        top_frame = ctk.CTkFrame(models_frame, fg_color="white")
        top_frame.pack(side="top", fill='both', expand=True)
        bottom_frame = ctk.CTkFrame(models_frame, fg_color="white")
        bottom_frame.pack(side="bottom", fill='both', expand=True)

        conf_mat = ctk.CTkImage(light_image=Image.open("results/classification/plot1.png"), dark_image=Image.open("results/classification/plot1.png"),
                              size=(470, 350))

        conf_mat_label = ctk.CTkLabel(top_frame, text="", image=conf_mat)
        conf_mat_label.pack(side='left', expand=True)

        pie_chart = ctk.CTkImage(light_image=Image.open("results/classification/plot2.png"),
                                dark_image=Image.open("results/classification/plot2.png"),
                                size=(470, 350))

        pie_chart_label = ctk.CTkLabel(top_frame, text="", image=pie_chart)
        pie_chart_label.pack(side='right', expand=True)

        learn_curve = ctk.CTkImage(light_image=Image.open("results/classification/plot3.png"),
                                 dark_image=Image.open("results/classification/plot3.png"),
                                 size=(450, 310))

        learn_curve_label = ctk.CTkLabel(bottom_frame, text="", image=learn_curve)
        learn_curve_label.pack(side='left', expand=True)

        roc_curve = ctk.CTkImage(light_image=Image.open("results/classification/plot4.png"),
                                   dark_image=Image.open("results/classification/plot4.png"),
                                   size=(450, 310))

        roc_curve_label = ctk.CTkLabel(bottom_frame, text="", image=roc_curve)
        roc_curve_label.pack(side='right', expand=True)


    else:
        CTkMessagebox(title="Error",
                      message="Please chose a model.",
                      icon="cancel",
                      option_1="Ok")


def process_chosen_model2():
    global models_frame, chosed_model
    model = chosed_model.get()
    if model != "Choose an algorithm":
        visualise_regression_result()
        for frame in models_frame.winfo_children():
            frame.destroy()
        top_frame = ctk.CTkFrame(models_frame, fg_color="white")
        top_frame.pack(side="top", fill='both', expand=True)
        bottom_frame = ctk.CTkFrame(models_frame, fg_color="white")
        bottom_frame.pack(side="bottom", fill='both', expand=True)

        plot1 = ctk.CTkImage(light_image=Image.open("results/regression/plot1.png"),
                             dark_image=Image.open("results/regression/plot1.png"),
                              size=(470, 350))

        plot1_label = ctk.CTkLabel(top_frame, text="", image=plot1)
        plot1_label.pack(side='left', expand=True)

        plot2 = ctk.CTkImage(light_image=Image.open("results/regression/plot2.png"),
                                dark_image=Image.open("results/regression/plot2.png"),
                                size=(470, 350))

        plot2_label = ctk.CTkLabel(top_frame, text="", image=plot2)
        plot2_label.pack(side='right', expand=True)

        plot3 = ctk.CTkImage(light_image=Image.open("results/regression/plot3.png"),
                                 dark_image=Image.open("results/regression/plot3.png"),
                                 size=(450, 310))

        plot3_label = ctk.CTkLabel(bottom_frame, text="", image=plot3)
        plot3_label.pack(side='left', expand=True)

        plot4 = ctk.CTkImage(light_image=Image.open("results/regression/plot4.png"),
                                   dark_image=Image.open("results/regression/plot4.png"),
                                   size=(450, 310))

        plot4_label = ctk.CTkLabel(bottom_frame, text="", image=plot4)
        plot4_label.pack(side='right', expand=True)
    else:
        CTkMessagebox(title="Error",
                      message="Please chose a model.",
                      icon="cancel",
                      option_1="Ok")


def save_chosed_model():
    global target_type, classifier, predictor, chosed_model
    if chosed_model.get() != "Choose an algorithm":
        if target_type == "Quantitative values":
            import joblib
            model = predictor
            filename = ctk.filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])

            if filename:
                # Save the model to the chosen file path
                joblib.dump(model, filename)
                print(f"Model saved successfully as '{filename}'")
                CTkMessagebox(title="Success",
                              message=f"Model saved successfully as '{filename}'",
                              icon="check",
                              option_1="Ok")
        elif target_type == "Qualitative values":
            import joblib
            model = classifier
            filename = ctk.filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])

            if filename:
                # Save the model to the chosen file path
                joblib.dump(model, filename)
                CTkMessagebox(title="Success",
                              message=f"Model saved successfully as '{filename}'",
                              icon="check",
                              option_1="Ok")
    else:
        CTkMessagebox(title="Error",
                      message="Please chose a model first.",
                      icon="cancel",
                      option_1="Ok")


prediction_entry = None
result_label = None
entry_widgets = []
def make_prediction():
    global dataset, target_name, target_type, models_frame, prediction_entry, result_label, entry_widgets
    if chosed_model.get() != "Choose an algorithm":
        for frame in models_frame.winfo_children():
            frame.destroy()
        new_frame = ctk.CTkFrame(models_frame)
        new_frame.pack(fill="both", expand=True)
        predition_frame = ctk.CTkFrame(new_frame, height=500, width=900)
        predition_frame.place(relx=0.5, rely=0.5, anchor='center')
        feature = []
        feature = dataset.drop(target_name, axis=1)
        feature_names = feature.columns.tolist()  # Replace with your feature names
        num_features = len(feature_names)

        for i, col in enumerate(feature_names):
            label = ctk.CTkLabel(predition_frame, text=col)
            label.grid(row=0, column=i, padx=5, pady=5)

        for i in range(num_features):
            prediction_entry = ctk.CTkEntry(predition_frame)
            entry_widgets.append(prediction_entry)
            prediction_entry.grid(row=1, column=i, padx=5, pady=5)

        predict_button = ctk.CTkButton(predition_frame, text="Predict", command=predict_result)
        predict_button.grid(row=2, column=0, columnspan=num_features, pady=10)

        # Label to display the prediction result
        result_label = ctk.CTkLabel(predition_frame, text="", font=("bold", 15))
        result_label.grid(row=3, column=0, columnspan=num_features, pady=10)


    else:
        CTkMessagebox(title="Error",
                      message="Please chose a model first.",
                      icon="cancel",
                      option_1="Ok")


def predict_result():
    if target_type == "Qualitative values":
        model = classifier
        input_values = []
        input_values = [int(round(float(prediction_entry.get()), 1)) for entry in entry_widgets]

        predictions = model.predict([input_values])
        result_label.configure(text=f"Predicted class: {predictions[0]}")
    elif target_type == "Quantitative values":
        model = predictor
        input_values = []
        input_values = [int(round(float(prediction_entry.get()), 1)) for entry in entry_widgets]

        predictions = model.predict([input_values])
        result_label.configure(text=f"Predicted value: {round(predictions[0], 2)}")


# delete duplicated rows
def delete_duplicated_data():
    global dataset, check_dd_var

    nbr_duplic_data = dataset.duplicated().sum()
    if dataset is not None:
        if nbr_duplic_data == 0:
            check_dd_var.set("there's no duplicated data")
        else:
            dataset.drop_duplicates(inplace=True)
            check_dd_var.set(f'Number of duplicated rows: {nbr_duplic_data} has benn deleted.')
            CTkMessagebox(title="Success",
                          message="Duplicated rows deleted successfully..",
                          icon="check",
                          option_1="Ok")
    else:
        CTkMessagebox(title="Error",
                      message="No Data, Please import a dataset first.",
                      icon="cancel",
                      option_1="Ok")


def export_pdf_report():
    model = chosed_model.get()
    if model != "Choose an algorithm":
        filename = ctk.filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if filename:
            if target_type == "Qualitative values":
                if model == 'Native Bayes':
                    c_report = repport1
                elif model == 'Decision Tree':
                    c_report = repport3
                elif model == 'Random Forest Classifier':
                    c_report = repport2
                elif model == 'K-NN Classifier':
                    c_report = repport4
                elif model == 'SVM Classifier':
                    c_report = repport5
                elif model == 'Neural Network Classifier':
                    c_report = repport6

                from fpdf import FPDF
                from io import StringIO
                import sys

                # 1. Set up the PDF doc basics
                pdf = FPDF()
                pdf.add_page()

                # Set font family, style, and size for the text
                pdf.set_font('Arial', 'B', 16)

                # 2. Layout the PDF doc contents
                pdf.image('images/pdf-header.png', x=(pdf.w - 150) / 2, y=None, w=150, h=30)
                ## Title
                pdf.cell(0, 10, 'Classification Analysis report', ln=True, align='C')
                from datetime import datetime
                # Get the current time
                current_time = datetime.now()
                # Format the time as a string (optional, adjust the format as needed)
                time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, 'ModelMaster', align='L')
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, time_string, align='R')
                ## Line breaks
                pdf.ln(10)
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, f'Model chosen:      {model}', ln=True)
                pdf.ln(10)

                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, f'Dataset visualization:', ln=True, align='C')
                pdf.ln(10)

                pdf.image('visualization.png', x=(pdf.w - 200) / 2, y=None, w=200)

                pdf.ln(10)
                pdf.cell(0, 10, 'Model metrics results:', ln=True, align='C')
                pdf.ln(5)

                # Add the images with explanatory text
                cr = c_report
                original_stdout = sys.stdout
                sys.stdout = StringIO()
                print(cr)
                info_output = sys.stdout.getvalue()
                sys.stdout = original_stdout
                pdf.set_font('Arial', '', 14)
                pdf.multi_cell(0, 5, info_output, align='C')

                pdf.add_page()
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, 'Confusion Matrix:', ln=True, align='C')
                pdf.ln(5)  # Add some space before each image
                pdf.image('results/classification/plot1.png', x=(pdf.w - 150) / 2, y=None, w=150, h=110)
                pdf.ln(5)

                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, 'Correct vs Incorrect predictions', ln=True, align='C')
                pdf.ln(5)
                pdf.image('results/classification/plot2.png', x=(pdf.w - 150) / 2, y=None,
                          w=150)  # Adjust the width 'w' as needed
                pdf.ln(5)

                pdf.add_page()
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, 'Learning Curve Analysis:', ln=True, align='C')
                pdf.ln(5)
                pdf.image('results/classification/plot3.png', x=(pdf.w - 150) / 2, y=None,
                          w=150)  # Adjust the width 'w' as needed
                pdf.ln(2)

                if model == 'Decision Tree':
                    pdf.cell(0, 10, 'Tree Visualization:', ln=True, align='C')
                else:
                    pdf.cell(0, 10, '(ROC) Curve visualization:', ln=True, align='C')

                pdf.ln(2)
                pdf.image('results/classification/plot4.png', x=(pdf.w - 150) / 2, y=None, w=150,
                          h=110)  # Adjust the width 'w' as needed

                # 3. Output the PDF file
                pdf.output(filename)
                CTkMessagebox(title="Success", message=f"PDF saved successfully as '{filename}'", icon="check",
                              option_1="Ok")

            elif target_type == "Quantitative values":
                if model == 'Linear Regression':
                    mse = mse1
                    mae = mae1
                    r2 = r21
                    explained_var = explained_var1
                elif model == 'Decision Tree Regression':
                    mse = mse3
                    mae = mae3
                    r2 = r23
                    explained_var = explained_var3
                elif model == 'SVM Regression':
                    mse = mse2
                    mae = mae2
                    r2 = r22
                    explained_var = explained_var2
                elif model == 'Neural Network Regression':
                    mse = mse4
                    mae = mae4
                    r2 = r24
                    explained_var = explained_var4
                elif model == 'Random Forest Regression':
                    mse = mse5
                    mae = mae5
                    r2 = r25
                    explained_var = explained_var5
                elif model == 'K-NN Regression':
                    mse = mse6
                    mae = mae6
                    r2 = r26
                    explained_var = explained_var6

                from fpdf import FPDF
                from io import StringIO
                import sys

                # 1. Set up the PDF doc basics
                pdf = FPDF()
                pdf.add_page()
                # Set font family, style, and size for the text
                pdf.image('images/pdf-header.png', x=(pdf.w - 150) / 2, y=None, w=150, h=30)
                pdf.set_font('Arial', 'B', 16)
                # 2. Layout the PDF doc contents
                ## Title
                pdf.cell(0, 10, 'Prediction Analysis report', ln=True, align='C')
                from datetime import datetime
                # Get the current time
                current_time = datetime.now()
                # Format the time as a string (optional, adjust the format as needed)
                time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, 'ModelMaster', align='L')
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, time_string, align='R')
                ## Line breaks
                pdf.ln(10)
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, f'Model chosen:      {model}', ln=True)
                pdf.ln(10)

                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, f'Dataset visualization:', ln=True, align='C')
                pdf.ln(10)

                pdf.image('visualization.png', x=(pdf.w - 200) / 2, y=None, w=200)

                pdf.ln(10)
                pdf.cell(0, 10, 'Model metrics results:', ln=True, align='C')
                pdf.ln(5)

                # Add the images with explanatory text
                pdf.set_font('Arial', '', 14)
                pdf.cell(0, 5, f"Mean Squared Error (MSE): {round(mse, 2)}", ln=True, align='C')
                pdf.ln(3)
                pdf.cell(0, 5, f"Mean Absolute Error (MAE): {round(mae, 2)}", ln=True, align='C')
                pdf.ln(3)
                pdf.cell(0, 5, f"Explained Variance Score: {round(explained_var, 2)}", ln=True, align='C')
                pdf.ln(3)
                pdf.cell(0, 5, f"R-squared (R²): {round(r2, 2)}", ln=True, align='C')

                pdf.add_page()
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, 'Model fitting graph:', ln=True, align='C')
                pdf.ln(5)  # Add some space before each image
                pdf.image('results/regression/plot1.png', x=(pdf.w - 150) / 2, y=None, w=150, h=110)
                pdf.ln(5)

                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, 'Learning Curve Analysis:', ln=True, align='C')
                pdf.ln(5)
                pdf.image('results/regression/plot2.png', x=(pdf.w - 150) / 2, y=None, w=150)  # Adjust the width 'w' as needed
                pdf.ln(5)

                pdf.add_page()
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, 'Residuals vs Predicted values:', ln=True, align='C')
                pdf.ln(5)
                pdf.image('results/regression/plot3.png', x=(pdf.w - 150) / 2, y=None,
                          w=150)  # Adjust the width 'w' as needed
                pdf.ln(2)

                if model == 'Decision Tree Regression':
                    pdf.cell(0, 10, 'Tree Visualization:', ln=True, align='C')
                else:
                    pdf.cell(0, 10, 'Feature Importance:', ln=True, align='C')

                pdf.ln(2)
                pdf.image('results/regression/plot4.png', x=(pdf.w - 150) / 2, y=None, w=150,
                          h=110)  # Adjust the width 'w' as needed

                # 3. Output the PDF file
                pdf.output(filename)
                CTkMessagebox(title="Success", message=f"PDF saved successfully as '{filename}'", icon="check",
                              option_1="Ok")
    else:
        CTkMessagebox(title="Error",
                      message="Please chose a model first.",
                      icon="cancel",
                      option_1="Ok")



# *** Les fonctions des pages ***
label_type_var = ""
target_column = ""
def import_page():
    global label_type_var, target_column, table_frame
    data_frame1 = ctk.CTkFrame(master=content_page)
    data_frame1.pack(fill='both', expand=True)

    import_frame = ctk.CTkFrame(master=data_frame1, width=200, fg_color='#525050', bg_color='#525050')
    import_frame.pack(side='left', fill='y')

    import_label = ctk.CTkLabel(import_frame, text="Import or Create\n  DataSet:", font=('bold', 20))
    import_label.place(relx=0.5, rely=0.11, anchor='center')
    # Un bouton pour charger l'ensemble de données
    btn_import = ctk.CTkButton(master=import_frame, font=("bold", 18), text="Click to import", fg_color="green",
                               width=160, height=50, command=import_dataset)
    btn_import.place(relx=0.5, rely=0.2, anchor='center')


    creat_btn = ctk.CTkButton(master=import_frame, font=("bold", 18), text="Click to Create", fg_color="green",
                              width=160, height=50, command=creat_dataset)
    creat_btn.place(relx=0.5, rely=0.3, anchor='center')

    entry_label1 = ctk.CTkLabel(master=import_frame, font=("bold", 15), text="Target column name:", padx=0, pady=0)
    entry_label1.place(relx=0.5, rely=0.45, anchor='center')

    target_column = ctk.CTkEntry(import_frame, placeholder_text="Enter target column name", width=170)
    target_column.place(relx=0.5, rely=0.5, anchor='center')

    entry_label2 = ctk.CTkLabel(master=import_frame, font=("bold", 15), text="Target column data type:")
    entry_label2.place(relx=0.5, rely=0.55, anchor='center')
    label_type_var = ctk.StringVar(value="Choose type")
    label_type = ctk.CTkOptionMenu(import_frame, values=["Quantitative values", "Qualitative values"],
                                   variable=label_type_var)
    label_type.place(relx=0.5, rely=0.6, anchor='center')

    # Button to submit the content of target column name and type of data
    submit_button = ctk.CTkButton(import_frame, text="Submit", command=submit_values)
    submit_button.place(relx=0.5, rely=0.65, anchor='center')

    table_frame = ctk.CTkFrame(master=data_frame1)
    table_frame.pack(side='right', fill='both', expand=True)


def vise_page():
    global dataset, target_name, target_type
    # vise_frame = ctk.CTkFrame(master=content_page, fg_color='red')
    # vise_frame.pack(fill='both', expand=True)

    if dataset is not None:
        if target_type == 'Qualitative values':
            label = dataset[target_name]
            features = dataset.drop(target_name, axis=1)
            # Create a separate frame for the visualizations
            vis_frame = ctk.CTkFrame(master=content_page)
            vis_frame.pack(fill='both', expand=True)

            # Create a canvas
            canvas = ctk.CTkCanvas(master=vis_frame)
            canvas.pack(side='left', fill='both', expand=True)

            # Attach a vertical scrollbar to the canvas
            scrollbar = ttk.Scrollbar(master=vis_frame, orient='vertical', command=canvas.yview)
            scrollbar.pack(side='right', fill='y')
            canvas.configure(yscrollcommand=scrollbar.set)

            # Create another frame to contain the plots within the canvas
            plot_frame = ctk.CTkFrame(master=canvas)
            canvas.create_window((0, 0), window=plot_frame, anchor='nw')

            # Creating 6 subplots
            value_counts = label.value_counts()
            explode = [0.1 if i == value_counts.idxmax() else 0 for i in value_counts.index]
            fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

            dataset[target_name].value_counts().plot.pie(explode=explode, autopct='%1.1f%%', ax=axes[0, 0], shadow=True)

            # Réglages du premier sous-graphique (diagramme circulaire)
            axes[0, 0].set_title(f'diagramme circulaire {target_name}')
            axes[0, 0].set_xlabel(target_name)

            sns.countplot(x=target_name, data=dataset, ax=axes[0, 1], hue=dataset[target_name])
            # Réglages du deuxième sous-graphique ()
            axes[0, 1].set_title(f'diagramme de comptage {target_name}')

            features_name = features.columns.tolist()

            sns.scatterplot(x=features.iloc[:, 0], y=features.iloc[:, 1], data=dataset, ax=axes[0, 2],
                            hue=dataset[target_name])
            axes[0, 2].set_title(f'Scatter Plot: {features_name[0]} vs {features_name[1]}')

            sns.barplot(x=features_name[0], y=target_name, data=dataset, ax=axes[1, 0], hue=dataset[features_name[0]])
            axes[1, 0].set_title(f'{target_name} by {features_name[0]}')

            sns.barplot(x=features_name[1], y=target_name, data=dataset, ax=axes[1, 1], hue=dataset[features_name[1]])
            axes[1, 1].set_title(f'{target_name} by {features_name[1]}')

            sns.scatterplot(x=features.iloc[:, 2], y=features.iloc[:, 3], data=dataset, ax=axes[1, 2],
                            hue=dataset[target_name])
            axes[1, 2].set_title(f'Scatter Plot: {features_name[2]} vs {features_name[3]}')

            # Adjust layout
            plt.tight_layout()
            plt.savefig('visualization.png')  # You can change 'output_image.png' to the desired file name and format
            plt.close()

            # Embedding the subplot in the Tkinter canvas
            canvas_hist = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas_hist.draw()
            canvas_hist.get_tk_widget().pack()

            # Update the canvas to adapt to the frame size
            plot_frame.update_idletasks()
            canvas.config(scrollregion=canvas.bbox('all'))
        else:
            label = dataset[target_name]
            features = dataset.drop(target_name, axis=1)
            # Create a separate frame for the visualizations
            vis_frame = ctk.CTkFrame(master=content_page)
            vis_frame.pack(fill='both', expand=True)

            # Create a canvas
            canvas = ctk.CTkCanvas(master=vis_frame)
            canvas.pack(side='left', fill='both', expand=True)

            # Attach a vertical scrollbar to the canvas
            scrollbar = ttk.Scrollbar(master=vis_frame, orient='vertical', command=canvas.yview)
            scrollbar.pack(side='right', fill='y')
            canvas.configure(yscrollcommand=scrollbar.set)

            # Create another frame to contain the plots within the canvas
            plot_frame = ctk.CTkFrame(master=canvas)
            canvas.create_window((0, 0), window=plot_frame, anchor='nw')

            # Creating 6 subplots
            value_counts = label.value_counts()
            features_name = features.columns.tolist()
            fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

            # Réglages du premier sous-graphique (diagramme circulaire)
            sns.lineplot(x=features_name[0], y=target_name, data=dataset, ax=axes[0, 0])
            axes[0, 0].set_title(f'Linear graph of {target_name} by {features_name[0]}')

            sns.lineplot(x=features_name[1], y=target_name, data=dataset, ax=axes[0, 1])
            axes[0, 1].set_title(f'Linear graph of {target_name} by {features_name[1]}')

            sns.lineplot(x=features_name[2], y=target_name, data=dataset, ax=axes[0, 2])
            axes[0, 2].set_title(f'Linear graph of {target_name} by {features_name[2]}')

            sns.barplot(x=features_name[2], y=target_name, data=dataset, ax=axes[1, 0], hue=features_name[2])
            axes[1, 0].set_title(f'{target_name} by {features_name[2]}')

            sns.barplot(x=features_name[1], y=target_name, data=dataset, ax=axes[1, 1], hue=features_name[1])
            axes[1, 1].set_title(f'{target_name} by {features_name[1]}')

            sns.barplot(x=features_name[4], y=target_name, data=dataset, ax=axes[1, 2], hue=features_name[4])
            axes[1, 2].set_title(f'Scatter Plot: {features_name[4]} vs {target_name}')

            # Adjust layout
            plt.tight_layout()
            plt.savefig('visualization.png')  # You can change 'output_image.png' to the desired file name and format
            plt.close()
            # Embedding the subplot in the Tkinter canvas
            canvas_hist = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas_hist.draw()
            canvas_hist.get_tk_widget().pack()

            # Update the canvas to adapt to the frame size
            plot_frame.update_idletasks()
            canvas.config(scrollregion=canvas.bbox('all'))
    else:
        CTkMessagebox(title="Error",
                      message="No Data, Please import a dataset first.",
                      icon="cancel",
                      option_1="Ok")


column_entry = ""
process_frame = None
fs_method_option = ""
check_label_var = tk.StringVar()
mv_features = ""
method_option = ""
value_entry = ""
na_by_value_entry = ""
encodin_value_entry = None
encodin_cat_value_entry = None
encoding_entry = None
dataset_info_var = None
check_dd_var = tk.StringVar()
def process_page():
    global dataset, dataset_info_var, process_frame, column_entry, check_label_var, process_frame, mv_features, method_option, value_entry, na_by_value_entry, encodin_value_entry, encodin_cat_value_entry, encoding_entry, check_dd_var, fs_method_option
    if dataset is not None:
        process_frame = ctk.CTkFrame(master=content_page, fg_color='yellow')
        process_frame.pack(side='top', fill='both', expand=True)
        show_table(dataset, process_frame)

        tabview = ctk.CTkTabview(master=content_page, height=200)
        tabview.add("Delete Features/rows")
        tabview.add("Missing Values")
        tabview.add("Data Encoding")
        tabview.add("Feature Scaling")
        tabview.pack(side='bottom', fill='both')

        # Delete Features
        dl_columns = ctk.CTkFrame(tabview.tab("Delete Features/rows"))
        dl_columns.pack(fill="both")

        dl_columns_left = ctk.CTkFrame(dl_columns)
        dl_columns_left.pack(side='left', fill='both', expand=True)
        dl_columns_right = ctk.CTkFrame(dl_columns, width=450)
        dl_columns_right.pack(side='right', fill='both')

        col_label = ctk.CTkLabel(master=dl_columns_left, text="Enter columns to delete:", font=('bold', 15))
        col_label.place(relx=0.6, rely=0.08)

        column_entry = ctk.CTkEntry(master=dl_columns_left, placeholder_text="Enter list of columns separated with ','", width=250)
        column_entry.place(relx=0.6, rely=0.25)

        dl_btn = ctk.CTkButton(master=dl_columns_left, text="Delete", font=('bold', 14), width=200, command=delete_columns)
        dl_btn.place(relx=0.65, rely=0.5)

        save_dl_btn = ctk.CTkButton(master=dl_columns_left, text="Save Current Dataset", font=('bold', 14), width=200, fg_color='green', command=export_dataset)
        save_dl_btn.place(relx=0.65, rely=0.7)


        dataset_info_var = tk.StringVar()
        dataset_info_label = ctk.CTkLabel(dl_columns_left, width=150, height=190, font=('Arial', 11),
                                          textvariable=dataset_info_var, justify='center')
        dataset_info_label.place(relx=0.20, rely=0.02)

        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            dataset.info()
            info_output = buf.getvalue()

        # Displaying the formatted output excluding the first three lines
        info_lines = info_output.split('\n')
        info_to_display = '\n'.join(info_lines[5:])
        dataset_info_var.set(info_to_display)

        result_dd = ctk.CTkButton(master=dl_columns_right, text="Drop Duplicated Rows", width=200, font=('Arial', 14), command=delete_duplicated_data)
        result_dd.place(relx=0.5, rely=0.09, anchor='center')

        check_dd = ctk.CTkLabel(dl_columns_right, width=250, textvariable=check_dd_var, height=150, text="",
                                justify="left", font=('Arial', 16))
        check_dd.place(relx=0.5, rely=0.6, anchor="center")


        # Missing Values
        ms_values = ctk.CTkFrame(tabview.tab("Missing Values"))
        ms_values.pack(fill="both", expand=True)

        check_ms = ctk.CTkButton(master=ms_values, text="Check Missing Values", width=250, command=check_nan)
        check_ms.place(relx=0.2, rely=0.1, anchor="center")

        check_label = ctk.CTkLabel(ms_values, width=250, textvariable=check_label_var, height=150, text="",
                                   justify="left")
        check_label.place(relx=0.2, rely=0.6, anchor="center")

        na_method = ctk.CTkLabel(ms_values, text="Handle NaN by Methods:", font=("bold", 15))
        na_method.place(relx=0.45, rely=0.08, anchor="center")

        na_methode_label = ctk.CTkLabel(ms_values, text="Enter features to handle:", font=("bold", 12))
        na_methode_label.place(relx=0.45, rely=0.2, anchor="center")

        mv_features = ctk.CTkEntry(ms_values, placeholder_text="Enter features separated with ',' ", width=250)
        mv_features.place(relx=0.45, rely=0.35, anchor="center")

        option_label = ctk.CTkLabel(ms_values, text="Choose methode to hande each column", font=("bold", 12), padx=0, pady=0)
        option_label.place(relx=0.45, rely=0.5, anchor="center")

        method_option = ctk.CTkOptionMenu(ms_values, values=["Previous value", "Next value", "Delete row", "Mean", "Max", "Min"])
        method_option.place(relx=0.45, rely=0.62, anchor="center")

        mv_submit = ctk.CTkButton(ms_values, text="Submit", width=240, command=handle_data_by_method)
        mv_submit.place(relx=0.45, rely=0.77, anchor="center")

        mv_save_data = ctk.CTkButton(ms_values, text="Save Current Dataset", font=('bold', 13), width=250,
                                     fg_color='green', command=export_dataset)
        mv_save_data.place(relx=0.45, rely=0.92, anchor="center")

        na_by_value = ctk.CTkLabel(ms_values, text="Handle NaN by Value:", font=('bold', 15))
        na_by_value.place(relx=0.7, rely=0.08, anchor="center")

        na_by_value2 = ctk.CTkLabel(ms_values, text="Enter features to handle:", font=('bold', 12))
        na_by_value2.place(relx=0.7, rely=0.2, anchor="center")

        na_by_value_entry = ctk.CTkEntry(master=ms_values, placeholder_text="Enter features separated with ',' ",
                                         width=250)
        na_by_value_entry.place(relx=0.7, rely=0.35, anchor="center")

        na_by_value3 = ctk.CTkLabel(ms_values, text="Enter Value", font=('bold', 12))
        na_by_value3.place(relx=0.7, rely=0.5, anchor="center")

        value_entry = ctk.CTkEntry(ms_values, placeholder_text="Enter Numerical Value", width=250)
        value_entry.place(relx=0.7, rely=0.62, anchor="center")

        submit_value_btn = ctk.CTkButton(ms_values, text="Submit", width=240, command=handle_data_by_value)
        submit_value_btn.place(relx=0.7, rely=0.77, anchor="center")

        export_data_btn = ctk.CTkButton(ms_values, text="Save Current Dataset", font=('bold', 13), width=250,
                                        fg_color='green', command=export_dataset)
        export_data_btn.place(relx=0.7, rely=0.92, anchor="center")


        # Features Scaling
        fs_data = ctk.CTkFrame(tabview.tab("Feature Scaling"))
        fs_data.pack(side='left', fill="both", expand=True)

        fs_title = ctk.CTkLabel(fs_data, text="Feature Scaling:", font=('bold', 16))
        fs_title.place(relx=0.5, rely=0.09, anchor='center')

        fs_title = ctk.CTkLabel(fs_data, text="Choose a method for Feature Scaling:", font=('bold', 14))
        fs_title.place(relx=0.5, rely=0.22, anchor='center')

        fs_method_option = ctk.CTkOptionMenu(fs_data, values=["MinMaxScaler", "StandardScaler", "Normalizer", "RobustScaler", "PowerTransformer", "QuantileTransformer"], width=200)
        fs_method_option.place(relx=0.5, rely=0.42, anchor="center")

        fs_btn = ctk.CTkButton(master=fs_data, text="Normalize", font=('bold', 16), width=250,
                               command=features_scaling)
        fs_btn.place(relx=0.385, rely=0.6)

        save_sc_btn = ctk.CTkButton(master=fs_data, text="Save Current Dataset", font=('bold', 16), width=250,
                                    fg_color='green', command=export_dataset)
        save_sc_btn.place(relx=0.5, rely=0.85, anchor='center')


        # Data encoding
        data_encoding_fram = ctk.CTkFrame(tabview.tab("Data Encoding"))
        data_encoding_fram.pack(fill="both")

        encoding_label = ctk.CTkLabel(data_encoding_fram, text="Data Encoding:", font=('bold', 15))
        encoding_label.place(relx=0.5, rely=0.05, anchor='center')

        encoding_label2 = ctk.CTkLabel(data_encoding_fram,
                                       text="Replace categorical values within a specific column with other values",
                                       padx=0, pady=0, text_color="light grey")
        encoding_label2.place(relx=0.5, rely=0.17, anchor='center')

        encoding_label3 = ctk.CTkLabel(data_encoding_fram,
                                       text="Enter desired feature to perform the encoding:", padx=0, pady=0,
                                       font=('bold', 11))
        encoding_label3.place(relx=0.5, rely=0.27, anchor='center')

        encoding_entry = ctk.CTkEntry(data_encoding_fram, placeholder_text="Enter desired feature.", width=250,
                                      height=9)
        encoding_entry.place(relx=0.5, rely=0.39, anchor='center')

        encoding_label4 = ctk.CTkLabel(data_encoding_fram,
                                       text="Enter the categorical value and the value to be replaced with", padx=0,
                                       pady=0,
                                       font=('bold', 11))
        encoding_label4.place(relx=0.5, rely=0.52, anchor='center')

        encodin_cat_value_entry = ctk.CTkEntry(data_encoding_fram, placeholder_text='Categorical value', width=120)
        encodin_cat_value_entry.place(relx=0.45, rely=0.64, anchor="center")

        encodin_value_entry = ctk.CTkEntry(data_encoding_fram, placeholder_text='New value', width=120)
        encodin_value_entry.place(relx=0.55, rely=0.64, anchor="center")

        encoding_submit_btn = ctk.CTkButton(data_encoding_fram, text="Submit", height=8, width=240,
                                            command=data_encoding)
        encoding_submit_btn.place(relx=0.5, rely=0.79, anchor="center")

        encoding_save_btn = ctk.CTkButton(data_encoding_fram, text="Save current Dataset", height=10, width=250,
                                          fg_color='green', command=export_dataset)
        encoding_save_btn.place(relx=0.5, rely=0.92, anchor="center")

    else:
        CTkMessagebox(title="Error",
                      message="No Data, Please import a dataset first.",
                      icon="cancel",
                      option_1="Ok")


models_frame = None
text_area1 = None
text_area2 = None
text_area3 = None
text_area4 = None
text_area5 = None
text_area6 = None
chosed_model = None
test_size = tk.StringVar(value='0.2')
def ml_page():
    global dataset, target_type, models_frame, text_area1, text_area2, text_area3, text_area4, text_area5, text_area6,chosed_model, test_value_Entry, test_size
    if dataset is not None:
        if target_type == "Qualitative values":
            # Appliquer les algorithmes de machine learning pour les problemes de classification

            models_frame = ctk.CTkFrame(content_page)
            models_frame.pack(side='right', fill='both', expand=True)


            model_process_frame = ctk.CTkFrame(content_page, width=270)
            model_process_frame.pack(side='left', fill='both')

            top_frame = ctk.CTkFrame(models_frame)
            top_frame.pack(side='top', fill='both', expand=True)

            bottom_frame = ctk.CTkFrame(models_frame)
            bottom_frame.pack(side='bottom', fill='both', expand=True)

            bottom_frame2 = ctk.CTkFrame(models_frame)
            bottom_frame2.pack(side='bottom', fill='both', expand=True)


            model_algo1 = ctk.CTkFrame(top_frame)
            model_algo1.pack(side='left', fill='both', expand=True)

            model_algo1_title = ctk.CTkFrame(model_algo1, height=20)
            model_algo1_title.pack(side='top', fill='x', expand=True, pady=0)
            model_algo1_repport = ctk.CTkFrame(model_algo1)
            model_algo1_repport.pack(side='bottom', fill='both', expand=True)
            model_name1 = ctk.CTkLabel(model_algo1_title, text="Native Bayes algorithm:", padx=0, pady=0)
            model_name1.place(relx=0.5, rely=0.5, anchor="center")
            text_area1 = tk.Text(model_algo1_repport, height=10, width=20)
            text_area1.config(state=tk.DISABLED)
            text_area1.pack(fill="both", expand=True, padx=10, pady=0)



            model_algo2 = ctk.CTkFrame(top_frame)
            model_algo2.pack(side='right', fill='both', expand=True)
            model_algo2_title = ctk.CTkFrame(model_algo2, height=20)
            model_algo2_title.pack(side='top', fill='x', expand=True)
            model_algo2_repport = ctk.CTkFrame(model_algo2)
            model_algo2_repport.pack(side='bottom', fill='both', expand=True)
            model_name2 = ctk.CTkLabel(model_algo2_title, text="Random Forest algorithm:")
            model_name2.place(relx=0.5, rely=0.5, anchor="center")
            text_area2 = tk.Text(model_algo2_repport, height=10, width=20)
            text_area2.config(state=tk.DISABLED)
            text_area2.pack(fill="both", expand=True, padx=10, pady=0)

            model_algo3 = ctk.CTkFrame(bottom_frame)
            model_algo3.pack(side='left', fill='both', expand=True)
            model_algo3_title = ctk.CTkFrame(model_algo3, height=20)
            model_algo3_title.pack(side='top', fill='x', expand=True)
            model_algo3_repport = ctk.CTkFrame(model_algo3)
            model_algo3_repport.pack(side='bottom', fill='both', expand=True)
            model_name3 = ctk.CTkLabel(model_algo3_title, text="Decision Tree algorithm:")
            model_name3.place(relx=0.5, rely=0.5, anchor="center")
            text_area3 = tk.Text(model_algo3_repport, height=10, width=20)
            text_area3.config(state=tk.DISABLED)
            text_area3.pack(fill="both", expand=True, padx=10, pady=0)


            model_algo4 = ctk.CTkFrame(bottom_frame)
            model_algo4.pack(side='right', fill='both', expand=True)

            model_algo4_title = ctk.CTkFrame(model_algo4, height=20)
            model_algo4_title.pack(side='top', fill='x', expand=True)

            model_algo4_repport = ctk.CTkFrame(model_algo4)
            model_algo4_repport.pack(side='bottom', fill='both', expand=True)

            model_name4 = ctk.CTkLabel(model_algo4_title, text="K-Nearest Neighbors algorithm:")
            model_name4.place(relx=0.5, rely=0.5, anchor="center")
            text_area4 = tk.Text(model_algo4_repport, height=10, width=20)
            text_area4.config(state=tk.DISABLED)
            text_area4.pack(fill="both", expand=True, padx=10, pady=0)

            model_algo5 = ctk.CTkFrame(bottom_frame2)
            model_algo5.pack(side='left', fill='both', expand=True)
            model_algo5_title = ctk.CTkFrame(model_algo5, height=20)
            model_algo5_title.pack(side='top', fill='x', expand=True)
            model_algo5_repport = ctk.CTkFrame(model_algo5)
            model_algo5_repport.pack(side='bottom', fill='both', expand=True)
            model_name5 = ctk.CTkLabel(model_algo5_title, text="Support Vector Classifier:")
            model_name5.place(relx=0.5, rely=0.5, anchor="center")
            text_area5 = tk.Text(model_algo5_repport, height=10, width=20)
            text_area5.config(state=tk.DISABLED)
            text_area5.pack(fill="both", expand=True, padx=10, pady=0)

            model_algo6 = ctk.CTkFrame(bottom_frame2)
            model_algo6.pack(side='right', fill='both', expand=True)
            model_algo6_title = ctk.CTkFrame(model_algo6, height=20)
            model_algo6_title.pack(side='top', fill='x', expand=True)
            model_algo6_repport = ctk.CTkFrame(model_algo6)
            model_algo6_repport.pack(side='bottom', fill='both', expand=True)
            model_name6 = ctk.CTkLabel(model_algo6_title, text="Neural Network Classifier:")
            model_name6.place(relx=0.5, rely=0.5, anchor="center")
            text_area6 = tk.Text(model_algo6_repport, height=10, width=20)
            text_area6.config(state=tk.DISABLED)
            text_area6.pack(fill="both", expand=True, padx=10, pady=0)


            process_frame1 = ctk.CTkFrame(model_process_frame, height=150)
            process_frame1.pack(side='top', fill='both', expand=True)

            test_size_label = ctk.CTkLabel(process_frame1, text="Chose test size:\nDefault 20%")
            test_size_label.place(relx=0.5, rely=0.1, anchor="center")

            test_value_Entry = ctk.CTkOptionMenu(process_frame1, variable=test_size, values=["0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45","0.5"], corner_radius=3)
            test_value_Entry.place(relx=0.5, rely=0.3, anchor='center')

            apply_bottun = ctk.CTkButton(process_frame1, text="Apply\nML algorithms", font=("bold", 25), height=80,
                                         width=95, fg_color='green', command=apply_ml_classifiers)
            apply_bottun.place(relx=0.5, rely=0.7, anchor="center")

            process_frame2 = ctk.CTkFrame(model_process_frame)
            process_frame2.pack(side='top', fill='both', expand=True)

            chose_label = ctk.CTkLabel(process_frame2, text="Chose desired Model:", font=('bold', 17))
            chose_label.place(relx=0.5, rely=0.1, anchor='center')

            chosed_model = ctk.StringVar(value="Choose an algorithm")
            algorithms_list = ctk.CTkOptionMenu(process_frame2, values=['Native Bayes', 'Decision Tree','SVM Classifier' ,'Neural Network Classifier' , 'Random Forest Classifier', 'K-NN Classifier'], width=150,
                                           variable=chosed_model)
            algorithms_list.place(relx=0.5, rely=0.3, anchor='center')

            submit_chosen_model = ctk.CTkButton(process_frame2, text="Submit", width=150, command=process_chosen_model1)
            submit_chosen_model.place(relx=0.5, rely=0.45, anchor='center')

            pridection_label = ctk.CTkLabel(process_frame2, text="Make predictions\n on Model:", font=('bold', 17))
            pridection_label.place(relx=0.5, rely=0.7, anchor='center')

            predection_btn = ctk.CTkButton(process_frame2, text="Classify a sample", width=150, fg_color="#e06d14", command=make_prediction)
            predection_btn.place(relx=0.5, rely=0.9, anchor='center')


            process_frame3 = ctk.CTkFrame(model_process_frame)
            process_frame3.pack(side='top', fill='both', expand=True)

            label_frame3 = ctk.CTkLabel(process_frame3, text="Save Results:", font=('bold', 17))
            label_frame3.place(relx=0.5, rely=0.1, anchor='center')

            save_model = ctk.CTkButton(process_frame3, text="Save Trained Model", width=150, fg_color="#e06d14", command=save_chosed_model)
            save_model.place(relx=0.5, rely=0.3, anchor='center')

            export_report = ctk.CTkButton(process_frame3, text="Export Analysis report", width=150, fg_color="#e06d14",  command=export_pdf_report)
            export_report.place(relx=0.5, rely=0.5, anchor='center')

            save_dataset = ctk.CTkButton(process_frame3, text="Save current dataset", width=150, fg_color="green", command=export_dataset)
            save_dataset.place(relx=0.5, rely=0.7, anchor='center')

        elif target_type == "Quantitative values":


            models_frame = ctk.CTkFrame(content_page)
            models_frame.pack(side='right', fill='both', expand=True)

            model_process_frame = ctk.CTkFrame(content_page, width=270)
            model_process_frame.pack(side='left', fill='both')


            top_frame = ctk.CTkFrame(models_frame)
            top_frame.pack(side='top', fill='both', expand=True)

            bottom_frame = ctk.CTkFrame(models_frame)
            bottom_frame.pack(side='bottom', fill='both', expand=True)

            bottom_frame2 = ctk.CTkFrame(models_frame)
            bottom_frame2.pack(side='bottom', fill='both', expand=True)

            model_algo1 = ctk.CTkFrame(top_frame)
            model_algo1.pack(side='left', fill='both', expand=True)
            model_algo1_title = ctk.CTkFrame(model_algo1, height=20)
            model_algo1_title.pack(side='top', fill='x', expand=True, pady=0)
            model_algo1_repport = ctk.CTkFrame(model_algo1)
            model_algo1_repport.pack(side='bottom', fill='both', expand=True)
            model_name1 = ctk.CTkLabel(model_algo1_title, text="Linear Regression:", padx=0, pady=0)
            model_name1.place(relx=0.5, rely=0.5, anchor="center")
            text_area1 = tk.Text(model_algo1_repport, height=10, width=20)
            text_area1.config(state=tk.DISABLED)
            text_area1.pack(fill="both", expand=True, padx=10, pady=0)

            model_algo2 = ctk.CTkFrame(top_frame)
            model_algo2.pack(side='right', fill='both', expand=True)
            model_algo2_title = ctk.CTkFrame(model_algo2, height=20)
            model_algo2_title.pack(side='top', fill='x', expand=True)
            model_algo2_repport = ctk.CTkFrame(model_algo2)
            model_algo2_repport.pack(side='bottom', fill='both', expand=True)
            model_name2 = ctk.CTkLabel(model_algo2_title, text="Support Vector Regression:")
            model_name2.place(relx=0.5, rely=0.5, anchor="center")
            text_area2 = tk.Text(model_algo2_repport, height=10, width=20)
            text_area2.config(state=tk.DISABLED)
            text_area2.pack(fill="both", expand=True, padx=10, pady=0)

            model_algo3 = ctk.CTkFrame(bottom_frame)
            model_algo3.pack(side='left', fill='both', expand=True)
            model_algo3_title = ctk.CTkFrame(model_algo3, height=20)
            model_algo3_title.pack(side='top', fill='x', expand=True)
            model_algo3_repport = ctk.CTkFrame(model_algo3)
            model_algo3_repport.pack(side='bottom', fill='both', expand=True)
            model_name3 = ctk.CTkLabel(model_algo3_title, text="Decision Tree Regression:")
            model_name3.place(relx=0.5, rely=0.5, anchor="center")
            text_area3 = tk.Text(model_algo3_repport, height=10, width=20)
            text_area3.config(state=tk.DISABLED)
            text_area3.pack(fill="both", expand=True, padx=10, pady=0)

            model_algo4 = ctk.CTkFrame(bottom_frame)
            model_algo4.pack(side='right', fill='both', expand=True)
            model_algo4_title = ctk.CTkFrame(model_algo4, height=20)
            model_algo4_title.pack(side='top', fill='x', expand=True)
            model_algo4_repport = ctk.CTkFrame(model_algo4)
            model_algo4_repport.pack(side='bottom', fill='both', expand=True)
            model_name4 = ctk.CTkLabel(model_algo4_title, text="Neural Network Regression:")
            model_name4.place(relx=0.5, rely=0.5, anchor="center")
            text_area4 = tk.Text(model_algo4_repport, height=10, width=20)
            text_area4.config(state=tk.DISABLED)
            text_area4.pack(fill="both", expand=True, padx=10, pady=0)

            model_algo5 = ctk.CTkFrame(bottom_frame2)
            model_algo5.pack(side='right', fill='both', expand=True)
            model_algo5_title = ctk.CTkFrame(model_algo5, height=20)
            model_algo5_title.pack(side='top', fill='x', expand=True)
            model_algo5_repport = ctk.CTkFrame(model_algo5)
            model_algo5_repport.pack(side='bottom', fill='both', expand=True)
            model_name5 = ctk.CTkLabel(model_algo5_title, text="Random Forest Regression:")
            model_name5.place(relx=0.5, rely=0.5, anchor="center")
            text_area5 = tk.Text(model_algo5_repport, height=10, width=20)
            text_area5.config(state=tk.DISABLED)
            text_area5.pack(fill="both", expand=True, padx=10, pady=0)

            model_algo6 = ctk.CTkFrame(bottom_frame2)
            model_algo6.pack(side='left', fill='both', expand=True)
            model_algo6_title = ctk.CTkFrame(model_algo6, height=20)
            model_algo6_title.pack(side='top', fill='x', expand=True)
            model_algo6_repport = ctk.CTkFrame(model_algo6)
            model_algo6_repport.pack(side='bottom', fill='both', expand=True)
            model_name6 = ctk.CTkLabel(model_algo6_title, text="K-NN Regression:")
            model_name6.place(relx=0.5, rely=0.5, anchor="center")
            text_area6 = tk.Text(model_algo6_repport, height=10, width=20)
            text_area6.config(state=tk.DISABLED)
            text_area6.pack(fill="both", expand=True, padx=10, pady=0)

            process_frame1 = ctk.CTkFrame(model_process_frame)
            process_frame1.pack(side='top', fill='both', expand=True)

            test_size_label = ctk.CTkLabel(process_frame1, text="Chose test size:\nDefault 20%")
            test_size_label.place(relx=0.5, rely=0.1, anchor="center")

            test_value_Entry = ctk.CTkOptionMenu(process_frame1, variable=test_size,
                                                 values=["0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45",
                                                         "0.5"], corner_radius=3)
            test_value_Entry.place(relx=0.5, rely=0.3, anchor='center')

            apply_bottun = ctk.CTkButton(process_frame1, text="Apply\nML algorithms", font=("bold", 25), height=80,
                                         width=95, fg_color='green', command=apply_ml_predictions)
            apply_bottun.place(relx=0.5, rely=0.7, anchor="center")

            process_frame2 = ctk.CTkFrame(model_process_frame)
            process_frame2.pack(side='top', fill='both', expand=True)

            chose_label = ctk.CTkLabel(process_frame2, text="Chose desired Model:", font=('bold', 17))
            chose_label.place(relx=0.5, rely=0.1, anchor='center')

            chosed_model = ctk.StringVar(value="Choose an algorithm")
            algorithms_list = ctk.CTkOptionMenu(process_frame2,
                                                values=['Linear Regression', 'Decision Tree Regression', 'SVM Regression', 'Neural Network Regression', 'Random Forest Regression', 'K-NN Regression'],
                                                width=150,
                                                variable=chosed_model)
            algorithms_list.place(relx=0.5, rely=0.3, anchor='center')

            submit_chosen_model = ctk.CTkButton(process_frame2, text="Submit", width=150, command=process_chosen_model2)
            submit_chosen_model.place(relx=0.5, rely=0.45, anchor='center')

            pridection_label = ctk.CTkLabel(process_frame2, text="Make predections\n on Model:", font=('bold', 17))
            pridection_label.place(relx=0.5, rely=0.7, anchor='center')

            predection_btn = ctk.CTkButton(process_frame2, text="Predict a sample", width=150, fg_color="#e06d14",
                                           command=make_prediction)
            predection_btn.place(relx=0.5, rely=0.9, anchor='center')

            process_frame3 = ctk.CTkFrame(model_process_frame)
            process_frame3.pack(side='top', fill='both', expand=True)

            label_frame3 = ctk.CTkLabel(process_frame3, text="Save Results:", font=('bold', 17))
            label_frame3.place(relx=0.5, rely=0.1, anchor='center')

            save_model = ctk.CTkButton(process_frame3, text="Save Trained Model",
                                       width=150,
                                       fg_color="#e06d14",
                                       command=save_chosed_model)
            save_model.place(relx=0.5, rely=0.3, anchor='center')

            export_report = ctk.CTkButton(process_frame3, text="Export Analysis report", width=150, fg_color="#e06d14", command=export_pdf_report)
            export_report.place(relx=0.5, rely=0.5, anchor='center')

            save_dataset = ctk.CTkButton(process_frame3, text="Save current dataset", width=150, fg_color="green", command=export_dataset)
            save_dataset.place(relx=0.5, rely=0.7, anchor='center')
    else:
        CTkMessagebox(title="Error",
                      message="No Data, Please import a dataset first.",
                      icon="cancel",
                      option_1="Ok")


def tut_page():
    from tkvideo import tkvideo
    label_video = tk.Label(content_page)
    label_video.place(relx=0.5, rely=0.5, anchor="center")

    player = tkvideo("ModelMaster.mp4", label_video, loop=1, size=(1300, 700))
    player.play()


def doc_page():
    from tkPDFViewer import tkPDFViewer as pdfv

    pdf_path = "documentation.pdf"  # Replace with the actual path to your PDF file
    v1 = pdfv.ShowPdf()
    v2 = v1.pdf_view(content_page, pdf_location=open(pdf_path, "r"), width=500,height=600)
    v2.pack(fill='both', expand=True, padx=300)


def delet_frame():
    for frame in content_page.winfo_children():
        frame.destroy()


# indicate function
def hide_indicator():
    import_indicate.configure(bg_color='#4D1230')
    ml_indicate.configure(bg_color='#4D1230')
    process_indicate.configure(bg_color='#4D1230')
    vise_indicate.configure(bg_color='#4D1230')
    doc_indicate.configure(bg_color='#4D1230')
    tut_indicate.configure(bg_color='#4D1230')


def indicate(lb, page):
    hide_indicator()
    lb.configure(bg_color='#e06d14')
    delet_frame()
    page()


#*****************************************************************************

# header Frame
head_frame = ctk.CTkFrame(master=app, height=100, fg_color='#5e2c59', bg_color='#5e2c59')
head_frame.pack(side='top', fill='x')

bg_head_img = ctk.CTkImage(light_image=Image.open("images/head-bg.png"), dark_image=Image.open("images/head-bg.png"), size=(1700, 100))

image_head_label = ctk.CTkLabel(master=head_frame, image=bg_head_img, text="")
image_head_label.place(relx=0.5, rely=0.46, anchor='center')

head_separator = ctk.CTkLabel(head_frame, text="", height=1, width=1700, fg_color='#5e2c59')
head_separator.place(relx=0.5, rely=1, anchor='center')

# Dashboard farme
# old color (grey) #23272D
dashboard_frame = ctk.CTkFrame(master=app, width=250, fg_color='#4D1230', bg_color='#4D1230')
dashboard_frame.pack(side='left', fill='y')

dash_label = ctk.CTkLabel(master=dashboard_frame, text="Dashboard", font=('bold', 30), text_color='white')
dash_label.place(relx=0.5, rely=0.05, anchor='center')

dash_label2 = ctk.CTkLabel(master=dashboard_frame, text="", height=1, width=150, fg_color='#e06d14', corner_radius=30)
dash_label2.place(relx=0.5, rely=0.09, anchor='center')

# Dashboard Buttons
image_1 = ctk.CTkImage(Image.open("icons/importicon.png"), size=(40, 40))
import_btn = ctk.CTkButton(master=dashboard_frame, font=('bold', 18), text='Import DataSet', compound='left' ,text_color='#e06d14', fg_color='transparent', border_color='#e06d14', border_width=4, width=200, height=50, image=image_1, command=lambda: indicate(import_indicate, import_page))
import_btn.place(relx=0.5, rely=0.2, anchor='center', )

import_indicate = ctk.CTkLabel(master=dashboard_frame, text='', bg_color='#4D1230', width=5, height=50)
import_indicate.place(relx=0.04, rely=0.2, anchor='center')

image_2 = ctk.CTkImage(Image.open("icons/vise-icon.png"), size=(40, 40))
vise_btn = ctk.CTkButton(master=dashboard_frame, font=('bold', 16), text='Data Visualisation', text_color='#e06d14',
            fg_color='transparent', border_color='#e06d14', border_width=4, image=image_2, compound='left',width=200, height=50, command=lambda: indicate(vise_indicate, vise_page))
vise_btn.place(relx=0.5, rely=0.35, anchor='center')

vise_indicate = ctk.CTkLabel(master=dashboard_frame, text='', bg_color='#4D1230', width=5, height=50)
vise_indicate.place(relx=0.04, rely=0.35, anchor='center')

image_3 = ctk.CTkImage(Image.open("icons/processing-icon.png"), size=(40, 40))
process_btn = ctk.CTkButton(master=dashboard_frame, font=('bold', 18), text='Data Processing', text_color='#e06d14',
            fg_color='transparent', border_color='#e06d14', border_width=4, image=image_3, compound='left', width=200, height=50, command=lambda: indicate(process_indicate, process_page))
process_btn.place(relx=0.5, rely=0.5, anchor='center')

process_indicate = ctk.CTkLabel(master=dashboard_frame, text='', bg_color='#4D1230', width=5, height=50)
process_indicate.place(relx=0.04, rely=0.5, anchor='center')

image_4 = ctk.CTkImage(Image.open("icons/ml-icon.png"), size=(40, 40))
ml_btn = ctk.CTkButton(master=dashboard_frame, font=('bold', 18), text='Machine Learning', text_color='#e06d14',
            fg_color='transparent', border_color='#e06d14', border_width=4, image=image_4, compound='left', width=200, height=50, command=lambda: indicate(ml_indicate, ml_page))
ml_btn.place(relx=0.5, rely=0.65, anchor='center')

ml_indicate = ctk.CTkLabel(master=dashboard_frame, text='', bg_color='#4D1230', width=5, height=50)
ml_indicate.place(relx=0.04, rely=0.65, anchor='center')


image_5 = ctk.CTkImage(Image.open("icons/watch-icon.png"), size=(35, 35))
tut_btn = ctk.CTkButton(master=dashboard_frame, font=('bold', 18), text='  Watch Tutorial', text_color='#3887BE',
            fg_color='transparent', border_color='#3887BE', border_width=4,image=image_5, compound='left', width=200, height=50, command=lambda: indicate(tut_indicate, tut_page))
tut_btn.place(relx=0.5, rely=0.80, anchor='center')

tut_indicate = ctk.CTkLabel(master=dashboard_frame, text='', bg_color='#4D1230', width=5, height=50)
tut_indicate.place(relx=0.04, rely=0.8, anchor='center')


image_6 = ctk.CTkImage(Image.open("icons/doc-icon.png"), size=(40, 40))
doc_btn = ctk.CTkButton(master=dashboard_frame, font=('bold', 18), text='Documentation', text_color='#3887BE',
            fg_color='transparent', border_color='#3887BE', border_width=4, image=image_6, compound='left',width=200, height=50, command=lambda: indicate(doc_indicate, doc_page))
doc_btn.place(relx=0.5, rely=0.95, anchor='center')

doc_indicate = ctk.CTkLabel(master=dashboard_frame, text='', bg_color='#4D1230', width=5, height=50)
doc_indicate.place(relx=0.04, rely=0.95, anchor='center')

content_page = ctk.CTkFrame(master=app)
content_page.pack(side='right', fill="both", expand=True)

team_frame = ctk.CTkFrame(master=content_page)
team_frame.pack(fill='both', expand=True)

bg_img = ctk.CTkImage(light_image=Image.open("images/bag-img.png"), dark_image=Image.open("images/bag-img.png"), size=(1300, 750))
image_label = ctk.CTkLabel(master=team_frame, image=bg_img, text="")
image_label.pack(fill="both", expand=True)


team_img = ctk.CTkImage(light_image=Image.open("images/team.png"), dark_image=Image.open("images/team.png"), size=(400, 400))
team_img_label = ctk.CTkLabel(master=team_frame, image=team_img, text="", corner_radius=10, width=420, height=420, fg_color='#4D1230')
team_img_label.place(relx=0.5, rely=0.5, anchor='center')


app.mainloop()
