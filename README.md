# 🏦 DataPilot for Finance

> **Empowering Microfinance & Fintech with Explainable AI**

**DataPilot for Finance** is a web-based interactive machine learning dashboard tailored for financial institutions like microfinance organizations and fintech startups. It helps teams build auditable, explainable, and data-driven solutions for **Loan Default Prediction**, with built-in SHAP-based interpretability and compliance-friendly reporting.

---

## 🌟 Key Features

### 📂 Dataset Upload
- Upload financial datasets in CSV format
- Automatic preview and validation

### 🧭 Use-Case Templates
- ✅ `Loan Default Prediction` (currently supported)
- 🚧 `Fraud Detection`, `Credit Scoring` (coming soon)

### ⚙️ Automated Preprocessing
- Handles missing values
- One-hot encoding for categorical data
- Scaling for numerical features

### 🧠 Model Training
- Choose from:
  - Logistic Regression
  - Random Forest Classifier
- Config-free, one-click training

### 📈 Model Evaluation
- Accuracy Score
- Confusion Matrix
- Classification Report

### 🔍 Model Explainability
- Global SHAP Summary Plot
- Local SHAP “Explain This Row” view
- Improves transparency and trust in model decisions

### 📄 PDF Audit Report
- Generate a professional PDF report for:
  - Model metadata
  - Accuracy & confusion matrix
  - Compliance purposes

### 💾 Export Trained Models
- Download trained model as `.pkl`
- Deploy to production via FastAPI

### 🌐 Language Support (Planned)
- English (default)
- Amharic 🇪🇹 *(coming soon)*

### 🔐 Role-Based Access (Beta)
- **Data Analyst View**: Train, test, visualize
- **Compliance Officer View**: Generate audit reports

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yab-g4u/DataPilot.git
cd DataPilot
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🎯 Target Audience

* Microfinance Institutions (e.g., Metemamen MFI, Omo MFI)
* Fintechs (e.g., ArifPay, EthioPay)
* Credit Analysts, Risk Officers, Compliance Teams

---

## 🧰 Tech Stack

| Layer        | Tool                      |
| ------------ | ------------------------- |
| Frontend     | Streamlit                 |
| Backend      | Python                    |
| ML           | Scikit-learn              |
| XAI          | SHAP                      |
| Export       | Joblib, FPDF              |
| Deployment   | FastAPI                   |
| Hosting      | Streamlit Cloud / Railway |
| DB (Planned) | SQLite / Firebase         |

---

## 📤 Optional API Deployment

You can expose your trained model using FastAPI.

Example:

```bash
uvicorn api:app --reload
```

Send `POST` requests to `/predict` with borrower data in JSON format.

---

## 📌 Sample Use Case

### Input:

CSV dataset with borrower history & loan details

### Output:

* Default prediction (yes/no)
* Explanation of key contributing factors (via SHAP)
* Downloadable audit PDF

---

## 📈 Roadmap

* [ ] Add full Amharic UI toggle
* [ ] Add support for additional use cases: Credit Scoring, Fraud Detection
* [ ] Add login and user-role authentication
* [ ] One-click deployment to Railway/Render
* [ ] Integration with Firebase for secure storage

---

## 🤝 Contributing

We welcome contributions and feedback from:

* Financial domain experts
* Regulators and compliance officers
* Open-source contributors

Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for more information.

---

## 🙌 Acknowledgments

Inspired by the urgent need for local, explainable, and auditable ML tools in Ethiopia's financial ecosystem.

```


