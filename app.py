import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Title and header
st.markdown(
    """
    <style>
        .header {
            font-size: 36px;
            color: #4CAF50;
            text-align: center;
            font-weight: bold;
            padding: 10px;
        }
        .subheader {
            font-size: 20px;
            color: #555555;
            text-align: center;
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
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression", "Decision Tree"])

    # Button to trigger model training
    if st.button("Train Model"):
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

        # Choose model based on user selection
        if model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
        elif model_choice == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display evaluation metrics
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)

        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
