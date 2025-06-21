🏦 DataPilot for Finance
Empowering Microfinance & Fintech with Explainable AI

DataPilot for Finance is a web-based ML-powered dashboard designed to help financial institutions like microfinance orgs and fintech startups make auditable, explainable, and data-driven loan decisions.

Built with Streamlit and scikit-learn, it enables teams to train, evaluate, and deploy models for Loan Default Prediction, with built-in regulatory features like SHAP explanations and audit-ready PDF reports.

🌟 Key Features
📂 Dataset Upload
Upload loan, credit, or risk datasets (CSV)

Automatic validation & preview

📊 Use-Case Templates
Currently Supported: Loan Default Prediction

Future: Fraud Detection, Credit Scoring, Customer Churn

⚙️ Auto Preprocessing
Handles missing values, scaling, and encoding

Supports numerical & categorical columns

No manual configuration required

🧠 Train ML Models
Choose between:

Random Forest

Logistic Regression

One-click training and validation

📈 Model Evaluation
Accuracy, Confusion Matrix, Classification Report

Interactive visual insights

🔍 Explainability (XAI)
Global SHAP Summary Plot

Local SHAP Row-wise Explanation ("Explain this row")

Enhances transparency & trust in model decisions

📄 Audit Report Generation
PDF report with:

Model metadata

Accuracy & confusion matrix

Summary for compliance officers

💾 Export & Integration
Download trained model as .pkl

Deployable via FastAPI API for production inference

🌐 Language Support (Planned)
English ✅

Amharic 🇪🇹 (Coming Soon)

🔐 Role-Based Access (Beta)
Data Analyst View: Train, test, visualize

Compliance Officer View: Generate audit reports

🧪 Getting Started
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/datapilot-finance.git
cd datapilot-finance
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run Locally
bash
Copy
Edit
streamlit run app.py
Access at http://localhost:8501

🎯 Target Users
Microfinance Institutions (e.g., Metemamen, Omo MFI)

Fintech companies (e.g., ArifPay, EthioPay)

Risk analysts & compliance teams in regulated financial sectors

🛠 Tech Stack
Layer	Tool
UI	Streamlit
Backend	Python
ML	scikit-learn
XAI	SHAP
Export	Joblib, FPDF
API Deployment	FastAPI (optional)
DB (planned)	SQLite / Firebase
Hosting	Streamlit Cloud / Railway

📌 Example Use Case: Loan Default Prediction
Input:
Loan dataset with borrower attributes

Output:
Probability of default

SHAP values for each prediction

Compliance-ready report

📤 FastAPI Integration (Optional)
You can deploy your trained .pkl model via a REST API:

bash
Copy
Edit
uvicorn api:app --reload
Send POST requests to /predict with borrower features.

🤝 Contributing
We welcome feedback and contributions—especially from:

Financial domain experts

Regulators and compliance analysts

Open-source ML enthusiasts
