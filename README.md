# DataPilot - Interactive Machine Learning Platform

Welcome to the **ML Dashboard**! This is a web-based, interactive platform built with **Streamlit** that allows users to explore, visualize, and experiment with machine learning models in real-time. Whether you are a beginner or an expert in machine learning, this tool makes it easy to upload your dataset, select from a variety of models, fine-tune hyperparameters, and evaluate the performance of your model with just a few clicks.

---

## Features

### **1. Dataset Upload**
- Easily upload your own **CSV files** for analysis
- The app handles missing data with intelligent preprocessing:
  - Numerical data is imputed with **mean** or **median** values
  - Categorical data is imputed using the **most frequent** value.
  
### **2. Model Selection**
- Choose from several popular machine learning algorithms, including:
  - **Logistic Regression**
  - **Decision Tree Classifier**
  - **Random Forest Classifier**
  - (More models may be added in future updates.)

### **3. Hyperparameter Tuning**
- Adjust hyperparameters for each model to optimize performance. Fine-tune key parameters like:
  - Maximum depth for decision trees
  - Number of estimators in random forests
  - Regularization strength in logistic regression

### **4. Data Preprocessing**
- Automatically handles preprocessing tasks:
  - **Imputes missing values** for both numerical and categorical features.
  - **Standardizes numerical features** using `StandardScaler`.
  - **One-hot encodes** categorical features.

### **5. Model Training & Evaluation**
- Train your selected model and view detailed evaluation metrics:
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-score)
  - **Accuracy**
- Visualize results instantly with easy-to-understand charts.

### **6. Real-time Visualizations**
- Get immediate feedback on model performance with clear, interactive visualizations like:
  - **Confusion matrix plots**
  - **Accuracy score**
  - **Precision, Recall, and F1-Score** metrics
  
### **7. User-Friendly Interface**
- Designed for users of all levels, even those without a background in machine learning.
- Intuitive navigation and clear instructions to guide you through the entire process.

---

## Getting Started

Follow these steps to set up and run the ML Dashboard locally:

### 1. Clone the repository

Clone the project to your local machine by running:

```bash
git clone https://github.com/yab-g4u/ml-dashboard.git
```

### 2. Install dependencies

Navigate to the project folder and install the required Python dependencies:

```bash
cd ml-dashboard
pip install -r requirements.txt
```

### 3. Run the app

Start the app with the following command:

```bash
streamlit run app.py
```

This will open the dashboard in your default web browser.

---

## Technologies Used

- **Python**: Programming language used for backend development.
- **Streamlit**: Framework for building interactive web applications.
- **Pandas**: Library for data manipulation and analysis.
- **Scikit-learn**: Machine learning library for building and evaluating models.
- **Matplotlib & Seaborn**: Libraries for visualizing data and model performance.

---

## Usage Instructions

### 1. **Upload Your Dataset**
- Click the "Choose a CSV file" button and select your dataset. A preview of the first few rows will be displayed. Ensure that your dataset includes a **target variable** (the variable you want to predict).

### 2. **Select the Target Column**
- Choose the column that represents the target variable (the label you want to predict).

### 3. **Select a Model**
- From the dropdown menu, select a machine learning model. Available models include Logistic Regression, Decision Tree, and Random Forest.

### 4. **Adjust Hyperparameters**
- Some models allow you to adjust hyperparameters, such as:
  - Regularization strength for Logistic Regression
  - Tree depth for Decision Trees
  - Number of estimators for Random Forest
- Use the provided sliders or input fields to adjust these settings

### 5. **Train the Model**
- Click "Train Model" to start the training process. The app will preprocess your data, train the selected model, and provide evaluation metrics and visualizations in real-time.

---

## Try the Interactive Dashboard!

You can also explore the interactive version of the dashboard live by clicking the link below:

[Try the ML Dashboard Live!](https://ml-dashboard-egwwhfpax5cvufphgsahgd.streamlit.app/)

---

## Contributing

We welcome contributions! If you find any issues or want to add new features, feel free to open an issue or submit a pull request. 

### How to contribute:
- Fork the repository
- Make your changes
- Submit a pull request with a detailed explanation of what you’ve done.

---

## Contact

For questions or suggestions, please reach out via email:  
**g4uforlife@gmail.com**

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
