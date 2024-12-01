import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Interactive Machine Learning Dashboard")

# 1. Dataset Upload and Preprocessing
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())

        # Identify numerical and categorical columns
        numerical_cols = data.select_dtypes(include=np.number).columns
        categorical_cols = data.select_dtypes(exclude=np.number).columns

        #Target column selection
        target_column = st.selectbox("Select Target Column", data.columns)

        # 2. Model Selection
        model_options = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }
        selected_model = st.selectbox("Select a model", list(model_options.keys()))
        model = model_options[selected_model]

        # 3. Preprocessing Pipeline (Handles missing values, scaling, and encoding)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),  #Handle missing numerical values
                    ('scaler', StandardScaler()) #Scale numerical features
                ]), numerical_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')), #Handle missing categorical values
                    ('onehot', OneHotEncoder(handle_unknown='ignore')) #One-hot encode categorical features
                ]), categorical_cols)
            ])

        # 4. Train/Test Split and Model Training
        if st.button("Train Model"):
            try:
                X = data.drop(target_column, axis=1)
                y = data[target_column]
                X_processed = preprocessor.fit_transform(X)  # Apply preprocessing
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

                #Hyperparameter Tuning (Example for RandomForest - Add more as needed)
                if selected_model == "Random Forest":
                    n_estimators = st.slider("Number of Estimators", 10, 200, 100)
                    model.n_estimators = n_estimators
                    

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # 5. Evaluation and Visualization
                cm = confusion_matrix(y_test, y_pred)
                st.write("Confusion Matrix:")
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(plt)

                st.write("Classification Report:")
                st.write(classification_report(y_test, y_pred))
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

            except KeyError as e:
                st.error(f"Error: Column '{e.args[0]}' not found in the dataset. Check your target column selection.")
            except ValueError as e:
                st.error(f"Error: {e}.  Check your dataset for issues like inconsistent data types or missing values that cannot be handled by the preprocessing pipeline.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")