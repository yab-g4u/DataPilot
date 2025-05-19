import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap
import joblib

# Dark Mode Styling
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
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="header">DataPilot</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Your Interactive Machine Learning Dashboard</div>', unsafe_allow_html=True)

# Sidebar for uploading dataset
with st.sidebar:
    st.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if data.empty or data.shape[1] < 2:
            st.error("Uploaded file is invalid. Ensure it contains at least two columns and some data.")
        else:
            st.write("Preview of the uploaded data:")
            st.dataframe(data.head())

            # Exploratory Data Analysis
            if st.checkbox("Show EDA"):
                st.subheader("Exploratory Data Analysis")
                st.write("### Pairplot of Numerical Features")
                numeric_cols = data.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 1:
                    sns.pairplot(data[numeric_cols])
                    st.pyplot(bbox_inches="tight")
                else:
                    st.write("Not enough numerical features for pairplot.")
                st.write("### Correlation Heatmap")
                if len(numeric_cols) > 1:
                    corr = data[numeric_cols].corr()
                    sns.heatmap(corr, annot=True, cmap="coolwarm")
                    st.pyplot(bbox_inches="tight")
                else:
                    st.write("Not enough numerical features for heatmap.")

            # Select target column
            target_column = st.sidebar.selectbox("Select Target Column", options=data.columns)

            # Data Preprocessing
            st.sidebar.header("Preprocessing")
            standardize = st.sidebar.checkbox("Standardize Numerical Data")
            handle_missing = st.sidebar.checkbox("Handle Missing Data")

            # Model selection and hyperparameters
            model_choices = ["Random Forest", "Logistic Regression", "Decision Tree"]
            models_selected = st.sidebar.multiselect("Select Models to Train", model_choices, default="Random Forest")
            hyperparameters = {}

            if "Random Forest" in models_selected:
                hyperparameters["Random Forest"] = {
                    "n_estimators": st.sidebar.slider("RF: Number of Estimators", 10, 200, 100),
                    "max_depth": st.sidebar.slider("RF: Maximum Depth", 1, 20, 5),
                }
            if "Logistic Regression" in models_selected:
                hyperparameters["Logistic Regression"] = {
                    "regularization_strength": st.sidebar.slider("LR: Regularization Strength", 0.01, 1.0, 0.1),
                }
            if "Decision Tree" in models_selected:
                hyperparameters["Decision Tree"] = {
                    "max_depth": st.sidebar.slider("DT: Maximum Depth", 1, 20, 5),
                }

            # Train models
            if st.button("Train Models"):
                X = data.drop(target_column, axis=1)
                y = data[target_column]

                # Data Preprocessing
                num_cols = X.select_dtypes(include=np.number).columns
                cat_cols = X.select_dtypes(exclude=np.number).columns
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", Pipeline([
                            ("imputer", SimpleImputer(strategy="mean" if handle_missing else "most_frequent")),
                            ("scaler", StandardScaler() if standardize else "passthrough")
                        ]), num_cols),
                        ("cat", Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))
                        ]), cat_cols)
                    ]
                )

                X_processed = preprocessor.fit_transform(X)

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

                model_results = {}
                for model_name in models_selected:
                    if model_name == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=hyperparameters["Random Forest"]["n_estimators"],
                            max_depth=hyperparameters["Random Forest"]["max_depth"]
                        )
                    elif model_name == "Logistic Regression":
                        model = LogisticRegression(C=hyperparameters["Logistic Regression"]["regularization_strength"])
                    elif model_name == "Decision Tree":
                        model = DecisionTreeClassifier(max_depth=hyperparameters["Decision Tree"]["max_depth"])

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    model_results[model_name] = {
                        "model": model,
                        "accuracy": accuracy_score(y_test, y_pred),
                        "confusion_matrix": confusion_matrix(y_test, y_pred),
                        "classification_report": classification_report(y_test, y_pred, output_dict=True),
                    }

                # Model Comparison Dashboard
                st.subheader("Model Comparison Dashboard")
                comparison_df = pd.DataFrame({
                    model_name: {
                        "Accuracy": model_results[model_name]["accuracy"],
                    }
                    for model_name in models_selected
                }).T
                st.write(comparison_df)

                # Download Trained Models
                st.subheader("Download Trained Models")
                for model_name, result in model_results.items():
                    st.write(f"Download {model_name} Model")
                    model_filename = f"{model_name}_model.pkl"
                    joblib.dump(result["model"], model_filename)
                    with open(model_filename, "rb") as file:
                        st.download_button(
                            label=f"Download {model_name} Model",
                            data=file,
                            file_name=model_filename,
                            mime="application/octet-stream"
                        )

                # SHAP Explainability
                st.subheader("Model Explainability with SHAP")
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    X_test_df = pd.DataFrame(X_test, columns=feature_names)
                    for model_name, result in model_results.items():
                        st.write(f"SHAP Explanation for {model_name}")
                        explainer = shap.Explainer(result["model"], X_test_df)
                        shap_values = explainer(X_test_df)
                        shap.summary_plot(shap_values, X_test_df, plot_type="bar")
                        st.pyplot(bbox_inches="tight")
                except Exception as e:
                    st.error(f"SHAP visualization failed: {e}")

            # Real-Time Predictions
            if st.checkbox("Make Predictions"):
                st.subheader("Real-Time Prediction")
                sample_input = {col: st.text_input(f"Enter value for {col}") for col in X.columns}
                if st.button("Predict"):
                    try:
                        input_df = pd.DataFrame([sample_input])
                        input_transformed = preprocessor.transform(input_df)
                        predictions = {model_name: model.predict(input_transformed)[0] for model_name, model in model_results.items()}
                        st.write(predictions)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
    except Exception as e:
        st.error(f"Failed to load the dataset: {e}")
else:
    st.info("Please upload a valid CSV file.")
