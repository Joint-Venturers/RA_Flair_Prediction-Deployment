from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
from datetime import datetime
import uvicorn

app = FastAPI(
    title="RA Inflammation Prediction API - SIMPLE",
    description="Raw features only - Combined vs Separate joint counts",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model_type = joblib.load('ra_model_type.pkl')
    model = joblib.load(f'ra_model_{model_type}.pkl')
    scaler = joblib.load(f'ra_scaler_{model_type}.pkl')
    features = joblib.load(f'ra_features_{model_type}.pkl')
    print(f"✅ Loaded {model_type.upper()} model with {len(features)} features")
except Exception as e:
    print(f"❌ Error: {e}")
    model = None
    scaler = None
    features = None
    model_type = None

class PatientData(BaseModel):
    """Raw patient data - ONLY specified features"""
    age: float = Field(..., ge=25, le=85)
    gender: int = Field(..., ge=0, le=1, description="0=Male, 1=Female")
    disease_duration: float = Field(..., ge=0, le=40)
    bmi: float = Field(..., ge=15, le=50)
    min_temperature: float
    max_temperature: float
    humidity: float = Field(..., ge=0, le=100)
    barometric_pressure: float
    precipitation: float = Field(..., ge=0)
    wind_speed: float = Field(..., ge=0)
    
    # Joint counts - provide based on model
    tender_joint_count: Optional[float] = Field(None, ge=0, le=28)
    swollen_joint_count: Optional[float] = Field(None, ge=0, le=28)
    combined_joint_count: Optional[float] = Field(None, ge=0, le=56)
    
    class Config:
        schema_extra = {
            "example": {
                "age": 55.0,
                "gender": 1,
                "disease_duration": 10.0,
                "bmi": 26.5,
                "min_temperature": 15.0,
                "max_temperature": 20.0,
                "humidity": 65.0,
                "barometric_pressure": 1015.0,
                "precipitation": 0.5,
                "wind_speed": 5.0,
                "tender_joint_count": 1.0,
                "swollen_joint_count": 1.0
            }
        }

class PredictionResponse(BaseModel):
    inflammation_probability: float
    inflammation_prediction: int
    risk_level: str
    confidence: float
    prediction_date: str
    model_used: str
    joint_count_type: str

class BatchPredictionRequest(BaseModel):
    patients: List[PatientData]

class BatchPredictionResponse(BaseModel):
    total_predictions: int
    predictions: List[PredictionResponse]
    processing_time_ms: float

def prepare_features(data: dict) -> dict:
    """Prepare features based on model type"""
    if model_type == 'combined':
        if 'combined_joint_count' not in data or data['combined_joint_count'] is None:
            if data.get('tender_joint_count') is not None and data.get('swollen_joint_count') is not None:
                data['combined_joint_count'] = data['tender_joint_count'] + data['swollen_joint_count']
    elif model_type == 'separate':
        if data.get('tender_joint_count') is None or data.get('swollen_joint_count') is None:
            if data.get('combined_joint_count') is not None:
                data['tender_joint_count'] = data['combined_joint_count'] / 2
                data['swollen_joint_count'] = data['combined_joint_count'] / 2
    return data

def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MODERATE"
    else:
        return "HIGH"

@app.get("/")
async def root():
    return {
        "message": "RA Inflammation Prediction API - SIMPLE",
        "version": "3.0.0",
        "model_type": model_type if model_type else "not loaded",
        "features": features if features else [],
        "endpoints": {
            "predict": "/predict",
            "batch": "/predict/batch",
            "health": "/health",
            "info": "/model/info"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def info():
    if not model:
        raise HTTPException(503, "Model not loaded")
    return {
        "model_type": type(model).__name__,
        "joint_count_type": model_type,
        "features": features,
        "feature_count": len(features)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    if not model:
        raise HTTPException(503, "Model not loaded")
    
    try:
        data = patient.dict()
        data = prepare_features(data)
        
        X = np.array([[data.get(f, 0) for f in features]])
        X_scaled = scaler.transform(X)
        
        probability = model.predict_proba(X_scaled)[0][1]
        prediction = int(probability >= 0.5)
        confidence = max(probability, 1 - probability)
        
        return PredictionResponse(
            inflammation_probability=float(probability),
            inflammation_prediction=prediction,
            risk_level=get_risk_level(probability),
            confidence=float(confidence),
            prediction_date=datetime.now().isoformat(),
            model_used=type(model).__name__,
            joint_count_type=model_type
        )
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if not model:
        raise HTTPException(503, "Model not loaded")
    
    start = datetime.now()
    predictions = []
    
    try:
        for patient in request.patients:
            data = patient.dict()
            data = prepare_features(data)
            
            X = np.array([[data.get(f, 0) for f in features]])
            X_scaled = scaler.transform(X)
            
            probability = model.predict_proba(X_scaled)[0][1]
            prediction = int(probability >= 0.5)
            confidence = max(probability, 1 - probability)
            
            predictions.append(PredictionResponse(
                inflammation_probability=float(probability),
                inflammation_prediction=prediction,
                risk_level=get_risk_level(probability),
                confidence=float(confidence),
                prediction_date=datetime.now().isoformat(),
                model_used=type(model).__name__,
                joint_count_type=model_type
            ))
        
        processing_time = (datetime.now() - start).total_seconds() * 1000
        
        return BatchPredictionResponse(
            total_predictions=len(predictions),
            predictions=predictions,
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting RA Inflammation API - SIMPLE")
    print("=" * 60)
    print(f"🎯 Model: {model_type if model_type else 'not loaded'}")
    print(f"📊 Features: {len(features) if features else 0} (raw only)")
    print("🌐 Server: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")