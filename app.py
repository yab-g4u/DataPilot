import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Title for the app
st.title("DataPilot - Interactive Machine Learning Dashboard & Data Visualizer")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load the data and display the first few rows
        data = pd.read_csv(uploaded_file)

        # Show the column names and the first few rows of the data
        st.write("Columns in the dataset:", data.columns)
        st.write("Data Preview:")
        st.write(data.head())

        # Reset index if there's an 'Index' column causing issues
        data.reset_index(drop=True, inplace=True)

        # --- Data Visualization Section ---
        st.subheader("Data Visualization")

        # Select target column
        target_column = st.selectbox("Select Target Column", data.columns)

        # Display histograms for numerical columns
        if st.checkbox("Show Histograms"):
            num_cols = data.select_dtypes(include=np.number).columns
            if target_column in num_cols:
                num_cols = num_cols.drop(target_column)
            if len(num_cols) > 0:
                for col in num_cols:
                    plt.figure(figsize=(8, 6))
                    plt.hist(data[col], bins=20)
                    plt.xlabel(col)
                    plt.ylabel("Frequency")
                    plt.title(f"Histogram of {col}")
                    st.pyplot(plt)
            else:
                st.write("No numerical columns available for histogram.")

        # Display boxplots for numerical columns
        if st.checkbox("Show Boxplots"):
            num_cols = data.select_dtypes(include=np.number).columns
            if target_column in num_cols:
                num_cols = num_cols.drop(target_column)
            if len(num_cols) > 0:
                for col in num_cols:
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(x=data[col])
                    plt.title(f"Boxplot of {col}")
                    st.pyplot(plt)
            else:
                st.write("No numerical columns available for boxplot.")

        # Display correlation matrix
        if st.checkbox("Show Correlation Matrix"):
            num_cols = data.select_dtypes(include=np.number).columns
            if target_column in num_cols:
                num_cols = num_cols.drop(target_column)
            if len(num_cols) > 0:
                corr_matrix = data[num_cols].corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
                plt.title("Correlation Matrix of Numerical Features")
                st.pyplot(plt)

        # --- Machine Learning Section ---
        st.subheader("Machine Learning")

        # Model selection
        model_options = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }
        selected_model = st.selectbox("Select a model", list(model_options.keys()))
        model = model_options[selected_model]

        # Hyperparameter tuning based on selected model
        if selected_model == "Logistic Regression":
            C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
            model.C = C
        elif selected_model == "Decision Tree":
            max_depth = st.slider("Max Depth", 1, 20, 5)
            model.max_depth = max_depth
        elif selected_model == "Random Forest":
            n_estimators = st.slider("Number of Estimators", 10, 200, 100)
            max_depth = st.slider("Max Depth", 1, 20, 5)
            model.n_estimators = n_estimators
            model.max_depth = max_depth

        # Train the selected model
        if st.button("Train Model"):
            try:
                # Split data into features (X) and target (y)
                X = data.drop(target_column, axis=1)
                y = data[target_column]

                # Re-identify numerical and categorical columns
                num_cols = X.select_dtypes(include=np.number).columns
                cat_cols = X.select_dtypes(exclude=np.number).columns

                # Define preprocessor
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', Pipeline([
                            ('imputer', SimpleImputer(strategy='mean')),
                            ('scaler', StandardScaler())
                        ]), num_cols),
                        ('cat', Pipeline([
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))
                        ]), cat_cols)
                    ])

                # Preprocess features
                X_processed = preprocessor.fit_transform(X)

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

                # Train model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Model evaluation
                cm = confusion_matrix(y_test, y_pred)
                st.write("Confusion Matrix:")
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(plt)

                st.write("Classification Report:")
                st.write(classification_report(y_test, y_pred))
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    except Exception as e:
        st.error(f"Error loading file: {e}")
