import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from fpdf import FPDF  


st.set_page_config(page_title="DataPilot for Finance", layout="wide", page_icon="💰")

st.title("DataPilot for Finance")
st.markdown("Loan Default Prediction Tool for Microfinance & Fintech")

# Sidebar - Upload + Use-Case Selection
st.sidebar.title("Step 1: Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File (e.g., loan_data.csv)", type="csv")

st.sidebar.title("Step 2: Select Use-Case")
use_case = st.sidebar.selectbox("Use-Case Template", ["Loan Default Prediction"], index=0)

role = st.sidebar.radio("Role", ["Data Analyst", "Compliance Officer"])

language = st.sidebar.selectbox("Language", ["English", "Amharic"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())

    # Step 3: Select Target Column
    st.sidebar.title("Step 3: Setup")
    target_column = st.sidebar.selectbox("Select Target Column", data.columns)

    # Step 4: Model Choice
    model_type = st.sidebar.radio("Choose Model", ["Random Forest", "Logistic Regression"])

    # Step 5: Preprocessing Pipeline
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = RandomForestClassifier() if model_type == "Random Forest" else LogisticRegression()

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.success(f"Model Trained with Accuracy: {acc:.2f}")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).T)

        # SHAP explainability
        st.subheader("Global Feature Importance (SHAP)")
        try:
            X_transformed = pipeline.named_steps['preprocessing'].transform(X_test)
            explainer = shap.Explainer(pipeline.named_steps['classifier'], X_transformed)
            shap_values = explainer(X_transformed)
            shap.summary_plot(shap_values, X_transformed, plot_type="bar", show=False)
            st.pyplot(bbox_inches="tight")

            # Local XAI for single-row explanation
            st.subheader("Explain This Row (Local Explanation)")
            row_index = st.number_input("Select Row Index to Explain (from test set)", min_value=0, max_value=len(X_test)-1, value=0)
            st.write("Row Input:", X_test.iloc[row_index])
            shap.plots.waterfall(shap_values[row_index], show=False)
            st.pyplot(bbox_inches="tight")
        except Exception as e:
            st.warning(f"SHAP explainability failed: {e}")

        # Export model
        st.subheader("Export Model")
        joblib.dump(pipeline, "trained_model.pkl")
        with open("trained_model.pkl", "rb") as file:
            st.download_button("Download Model (.pkl)", file, file_name="model.pkl")

        # PDF Audit Report (Compliance only)
        if role == "Compliance Officer":
            st.subheader("Generate PDF Report")
            if st.button("Generate Audit Report"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Model Audit Report - Loan Default Prediction", ln=True)
                pdf.cell(200, 10, txt=f"Accuracy: {acc:.2f}", ln=True)
                pdf.cell(200, 10, txt="Confusion Matrix:", ln=True)
                for row in cm:
                    pdf.cell(200, 10, txt=str(row), ln=True)
                pdf.output("audit_report.pdf")
                with open("audit_report.pdf", "rb") as file:
                    st.download_button("Download PDF Report", file, file_name="audit_report.pdf")

else:
    st.info("Please upload a dataset to begin.")
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

# Page configuration
st.set_page_config(page_title="DataPilot", layout="wide", page_icon="🧠")

# Dark Mode Styling
st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown('<div class="header">DataPilot</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Your Interactive Machine Learning Dashboard</div>', unsafe_allow_html=True)

# Sidebar for onboarding steps
with st.sidebar:
    st.title("Upload Dataset")
    st.markdown("### Steps to Use:")
    st.markdown("1. Upload your dataset (CSV)")
    st.markdown("2. Select the target column")
    st.markdown("3. Choose models and tune hyperparameters")
    st.markdown("4. Train, evaluate, and download models")
    st.markdown("5. Make predictions using the trained models")

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if data.empty or data.shape[1] < 2:
            st.error("Uploaded file is invalid. Ensure it contains at least two columns and some data.")
        else:
            st.write("### Preview of the Uploaded Data")
            st.dataframe(data.head())

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
                    fig, ax = plt.subplots()
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.write("Not enough numerical features for heatmap.")

            target_column = st.sidebar.selectbox("Select Target Column", options=data.columns)

            st.sidebar.header("Preprocessing")
            standardize = st.sidebar.checkbox("Standardize Numerical Data")
            handle_missing = st.sidebar.checkbox("Handle Missing Data")

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

            if st.button("Train Models"):
                X = data.drop(target_column, axis=1)
                y = data[target_column]

                num_cols = X.select_dtypes(include=np.number).columns
                cat_cols = X.select_dtypes(exclude=np.number).columns
                preprocessor = ColumnTransformer([
                    ("num", Pipeline([
                        ("imputer", SimpleImputer(strategy="mean" if handle_missing else "most_frequent")),
                        ("scaler", StandardScaler() if standardize else "passthrough")
                    ]), num_cols),
                    ("cat", Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ]), cat_cols)
                ])

                X_processed = preprocessor.fit_transform(X)
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

                st.subheader("Model Comparison Dashboard")
                comparison_df = pd.DataFrame({
                    model_name: {
                        "Accuracy": model_results[model_name]["accuracy"],
                    }
                    for model_name in models_selected
                }).T
                st.write(comparison_df)

                for model_name, result in model_results.items():
                    st.write(f"\n### Confusion Matrix: {model_name}")
                    fig, ax = plt.subplots()
                    sns.heatmap(result["confusion_matrix"], annot=True, fmt='d', cmap='Blues', ax=ax)
                    st.pyplot(fig)

                    st.write("Classification Report:")
                    report_df = pd.DataFrame(result["classification_report"]).T
                    st.dataframe(report_df)

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

                st.subheader("Model Explainability with SHAP")
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    X_test_df = pd.DataFrame(X_test, columns=feature_names)
                    for model_name, result in model_results.items():
                        st.write(f"SHAP Explanation for {model_name}")
                        explainer = shap.Explainer(result["model"], X_test_df)
                        shap_values = explainer(X_test_df)
                        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
                        st.pyplot(bbox_inches="tight")
                except Exception as e:
                    st.warning(f"SHAP not supported for some models: {e}")

            if st.checkbox("Make Predictions"):
                st.subheader("Real-Time Prediction")
                sample_input = {}
                for col in data.drop(target_column, axis=1).columns:
                    dtype = data[col].dtype
                    if np.issubdtype(dtype, np.number):
                        sample_input[col] = st.number_input(f"{col}:", value=0.0)
                    else:
                        sample_input[col] = st.text_input(f"{col}:")

                if st.button("Predict"):
                    try:
                        input_df = pd.DataFrame([sample_input])
                        input_transformed = preprocessor.transform(input_df)
                        predictions = {model_name: model.predict(input_transformed)[0] for model_name, model in model_results.items()}
                        st.write("Predictions:", predictions)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
    except Exception as e:
        st.error(f"Failed to load the dataset: {e}")
else:
    st.info("Please upload a valid CSV file.")
