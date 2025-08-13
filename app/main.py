from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import json
from typing import Dict, Any

app = FastAPI(
    title="Subscription Prediction API",
    description="API para predecir suscripciones usando modelo entrenado con DVC + MLflow",
    version="1.0.0"
)

# Modelo para validar entrada
class PredictionInput(BaseModel):
    age: int
    income: int
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "income": 50000
            }
        }

class PredictionOutput(BaseModel):
    prediction: str  # "yes" o "no"
    prediction_proba: Dict[str, float]
    model_info: Dict[str, Any]

# Cargar modelo y metadata al iniciar la aplicación
MODEL_PATH = "models/model.pkl"
METADATA_PATH = "models/model_metadata.json"

try:
    # Cargar modelo
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    # Cargar metadata del modelo
    with open(METADATA_PATH, "r") as f:
        model_metadata = json.load(f)
    
    print(f"✓ Modelo cargado exitosamente")
    print(f"✓ MLflow Run ID: {model_metadata.get('mlflow_run_id', 'N/A')}")
    
except FileNotFoundError as e:
    print(f"❌ Error: No se pudo cargar el modelo - {e}")
    model = None
    model_metadata = {}

@app.get("/")
def root():
    return {
        "message": "Subscription Prediction API", 
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/model/info")
def model_info():
    """Información sobre el modelo cargado"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    return {
        "model_type": str(type(model).__name__),
        "mlflow_run_id": model_metadata.get("mlflow_run_id"),
        "experiment_name": model_metadata.get("experiment_name"),
        "model_name": model_metadata.get("model_name"),
        "features": ["age", "income"]
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    """Realizar predicción de suscripción"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Convertir datos de entrada a DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Realizar predicción
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]
        
        # Obtener las clases del modelo
        classes = model.classes_
        proba_dict = {str(classes[i]): float(prediction_proba[i]) for i in range(len(classes))}
        
        return PredictionOutput(
            prediction=str(prediction),
            prediction_proba=proba_dict,
            model_info={
                "mlflow_run_id": model_metadata.get("mlflow_run_id"),
                "model_name": model_metadata.get("model_name")
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

@app.get("/health")
def health_check():
    """Endpoint de salud para monitoreo"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_info": model_metadata if model else {}
    }