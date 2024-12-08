import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
import joblib
import shap
import io

# Dark Mode Styling (Custom CSS)
st.markdown(
    """
    <style>
        body {
            background-color: #2e2e2e;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #333;
            color: white;
        }
        .header {
            font-size: 36px;
            color: #4CAF50;
            text-align: center;
            font-weight: bold;
            padding: 10px;
        }
        .subheader {
            font-size: 20px;
            color: #aaa;
            text-align: center;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stSelectbox>div>div>input {
            background-color: #444;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="header">DataPilot</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Your Interactive Machine Learning Dashboard</div>', unsafe_allow_html=True)

# Sidebar for uploading the dataset
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load and preview data
    data = pd.read_csv(uploaded_file)
    st.write("Preview of the uploaded data:")
    st.dataframe(data.head())

    # Select the target column
    target_column = st.sidebar.selectbox("Select Target Column", options=data.columns)

    # Data Preprocessing Section
    st.sidebar.header("Preprocessing")
    st.sidebar.markdown("Choose preprocessing options:")
    standardize = st.sidebar.checkbox("Standardize Numerical Data")
    handle_missing = st.sidebar.checkbox("Handle Missing Data")

    # Model selection
    st.sidebar.header("Model Selection")
    models_selected = st.sidebar.multiselect(
        "Select Models to Compare",
        ["Random Forest", "Logistic Regression", "Decision Tree"]
    )

    # Hyperparameter optimization for selected models
    hyperparameters = {}
    if "Random Forest" in models_selected:
        hyperparameters["Random Forest"] = {
            "n_estimators": st.sidebar.slider("RF: Number of Estimators", 10, 200, 100),
            "max_depth": st.sidebar.slider("RF: Maximum Depth", 1, 20, 5)
        }
    if "Logistic Regression" in models_selected:
        hyperparameters["Logistic Regression"] = {
            "regularization_strength": st.sidebar.slider("LR: Regularization Strength", 0.01, 1.0, 0.1)
        }
    if "Decision Tree" in models_selected:
        hyperparameters["Decision Tree"] = {
            "max_depth": st.sidebar.slider("DT: Maximum Depth", 1, 20, 5)
        }

    # Button to trigger model training and comparison
    if st.button("Train and Compare Models"):
        # Split data into features (X) and target (y)
        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # Preprocessing of the data
        num_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(exclude=np.number).columns

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean' if handle_missing else 'most_frequent')),
                    ('scaler', StandardScaler() if standardize else 'passthrough')]), num_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
            ])

        X_processed = preprocessor.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        # Train and evaluate each model
        model_results = {}
        for model_name in models_selected:
            if model_name == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=hyperparameters["Random Forest"]["n_estimators"],
                    max_depth=hyperparameters["Random Forest"]["max_depth"]
                )
            elif model_name == "Logistic Regression":
                model = LogisticRegression(
                    C=hyperparameters["Logistic Regression"]["regularization_strength"]
                )
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(
                    max_depth=hyperparameters["Decision Tree"]["max_depth"]
                )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Collect metrics
            model_results[model_name] = {
                "model": model,
                "accuracy": accuracy_score(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            }

        # Display results
        st.write("### Model Comparison")
        comparison_df = pd.DataFrame({
            model_name: {
                "Accuracy": result["accuracy"],
                "Precision": np.mean([
                    result["classification_report"][str(label)]["precision"]
                    for label in np.unique(y_test)
                ]),
                "Recall": np.mean([
                    result["classification_report"][str(label)]["recall"]
                    for label in np.unique(y_test)
                ]),
                "F1-Score": np.mean([
                    result["classification_report"][str(label)]["f1-score"]
                    for label in np.unique(y_test)
                ])
            }
            for model_name, result in model_results.items()
        })
        st.dataframe(comparison_df)

        # SHAP explanations for Random Forest (if selected)
        if "Random Forest" in models_selected:
            st.write("### SHAP Explanations for Random Forest")
            explainer = shap.TreeExplainer(model_results["Random Forest"]["model"])
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, plot_type="bar")
            st.pyplot(bbox_inches="tight")

        # Allow users to download models
        st.write("### Download Trained Models")
        for model_name, result in model_results.items():
            model_filename = f"{model_name}_model.pkl"
            joblib.dump(result["model"], model_filename)
            with open(model_filename, "rb") as model_file:
                st.download_button(
                    label=f"Download {model_name} Model",
                    data=model_file,
                    file_name=model_filename,
                    mime="application/octet-stream"
                )
