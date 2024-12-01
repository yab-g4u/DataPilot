# ml-dashboard
# Interactive Machine Learning Dashboard
A web-based application built with Streamlit for exploring and visualizing machine learning models.  Upload your own datasets, select from various algorithms, and experiment with hyperparameters to build and evaluate models in real-time.
## Features
• **Dataset Upload:** Upload CSV files for analysis.  The application handles missing data and performs basic data preprocessing.
• **Model Selection:** Choose from several popular machine learning algorithms including:
    * Logistic Regression
    * Decision Tree Classifier
    * Random Forest Classifier
    *(More models may be added in future updates)*
• **Hyperparameter Tuning:** Adjust key hyperparameters for each model to optimize performance.
• **Data Preprocessing:**  The app automatically handles missing values (numerical imputation with mean/median, categorical imputation with most frequent) and scales numerical features using StandardScaler.  Categorical features are one-hot encoded.
• **Model Training & Evaluation:** Train the selected model on your data and get a detailed evaluation, including:
    * Confusion Matrix
    * Classification Report (Precision, Recall, F1-score)
    * Accuracy
• **Real-time Visualization:**  View the results immediately using clear and informative visualizations.
• **User-Friendly Interface:** Designed for ease of use, even for users with limited machine learning experience.

## Getting Started

1. **Clone the repository:**
   
bash
   git clone <repository_url>
2. **Install dependencies:**
   
bash
   pip install -r requirements.txt
3. **Run the app:**
   
bash
   streamlit run app.py
This will launch the application in your web browser.

## Technologies Used

• **Python:**  Programming language.
• **Streamlit:**  Framework for building interactive web applications.
• **Pandas:**  Data manipulation and analysis library.
• **Scikit-learn:**  Machine learning library.
• **Matplotlib & Seaborn:** Data visualization libraries.


## Usage Instructions

1. **Upload your data:** Click the "Choose a CSV file" button and select your dataset. The first few rows of your data will be displayed for preview.  Ensure your data includes a target variable column.
2. **Select Target Column:** Choose the column representing your target variable (the variable you want to predict).
3. **Select a model:** Choose a machine learning model from the dropdown menu.
4. **Adjust Hyperparameters:**  For some models, you can adjust hyperparameters using the provided sliders or input fields.
5. **Train the model:** Click the "Train Model" button.  This will preprocess your data, train the model, and display the evaluation metrics and visualizations.


## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## Contact

g4uforlife@gmail.com

