from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="DataPilot Finance API")

# Load the trained pipeline model once on startup
model = joblib.load("trained_model.pkl")

# Define the expected input data schema (adjust columns to your dataset)
class InputData(BaseModel):
    # Example fields (replace with your actual features)
    # For loan data example, include your columns here:
    feature1: float
    feature2: float
    feature3: str
    # add more as needed...

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to DataFrame for pipeline
        input_df = pd.DataFrame([data.dict()])
        
        # Predict with your pipeline
        prediction = model.predict(input_df)
        
        # Return prediction (convert to int if needed)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
