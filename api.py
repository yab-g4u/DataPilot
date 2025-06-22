from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="DataPilot Finance API")

model = joblib.load("trained_model.pkl")

class InputData(BaseModel):
    loan_amount: float
    annual_income: float
    credit_score: float
    employment_length: int
    age: int
    loan_term: int
    interest_rate: float
    purpose: str
    home_ownership: str
    state: str

@app.post("/predict")
def predict(data: InputData):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))