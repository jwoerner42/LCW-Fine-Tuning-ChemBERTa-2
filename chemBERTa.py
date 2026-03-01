from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from solubility_predictor import SolubilityPredictor

MODEL_DIR = "./output"
predictor = SolubilityPredictor(MODEL_DIR)

app = FastAPI(title="ChemBERTa Solubility Regression Predictor")

class PredictRequest(BaseModel):
    smiles: List[str]

@app.post("/predict")
def predict(req: PredictRequest):
    preds = predictor.predict_batch(req.smiles)
    formatted = [f"{p:.9f} log10(WS in mol/L)" for p in preds]
    return {"predictions": formatted}