# FastAPI Server for Render Deployment
# Production-ready API with CORS and health checks

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
from datetime import datetime
import os

app = FastAPI(
    title="RA Inflammation Predictor API",
    description="Gradient Boosting model for RA inflammation prediction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
print("🔧 Loading model...")
try:
    model_type = joblib.load('ra_model_type.pkl')
    model = joblib.load(f'ra_model_{model_type}.pkl')
    scaler = joblib.load(f'ra_scaler_{model_type}.pkl')
    features = joblib.load(f'ra_features_{model_type}.pkl')
    print(f"✅ Model loaded: {model_type}")
    print(f"📊 Features: {len(features)}")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    model = None
    scaler = None
    features = None
    model_type = None

# Request model
class PredictionInput(BaseModel):
    age: float = Field(..., ge=25, le=85, description="Patient age")
    gender: int = Field(..., ge=0, le=1, description="0=Male, 1=Female")
    disease_duration: float = Field(..., ge=0, le=40, description="Years with RA")
    bmi: float = Field(..., ge=15, le=50, description="Body Mass Index")
    min_temperature: float = Field(..., description="Min temperature (°C)")
    max_temperature: float = Field(..., description="Max temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    barometric_pressure: float = Field(..., description="Pressure (mb)")
    precipitation: float = Field(..., ge=0, description="Precipitation (mm)")
    wind_speed: float = Field(..., ge=0, description="Wind speed (m/s)")
    tender_joint_count: float = Field(..., ge=0, le=28, description="Tender joints")
    swollen_joint_count: float = Field(..., ge=0, le=28, description="Swollen joints")
    
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
                "tender_joint_count": 2.0,
                "swollen_joint_count": 1.5
            }
        }

# Response model
class PredictionResponse(BaseModel):
    inflammation_probability: float
    inflammation_prediction: int
    risk_level: str
    confidence: float
    prediction_date: str
    model_type: str

def get_risk_level(probability: float) -> str:
    """Determine risk level from probability"""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MODERATE"
    else:
        return "HIGH"

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RA Inflammation Predictor API",
        "version": "1.0.0",
        "model": model_type if model_type else "not loaded",
        "status": "healthy" if model else "model not loaded",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "info": "GET /info"
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_type": model_type if model_type else None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/info")
async def model_info():
    """Get model information"""
    if not model:
        raise HTTPException(503, "Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "algorithm": model_type,
        "features": features,
        "feature_count": len(features),
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """
    Make inflammation prediction
    
    Returns probability, prediction, risk level, and confidence
    """
    if not model:
        raise HTTPException(503, "Model not loaded")
    
    try:
        # Prepare features in correct order
        X = np.array([[
            input_data.age,
            input_data.gender,
            input_data.disease_duration,
            input_data.bmi,
            input_data.min_temperature,
            input_data.max_temperature,
            input_data.humidity,
            input_data.barometric_pressure,
            input_data.precipitation,
            input_data.wind_speed,
            input_data.tender_joint_count,
            input_data.swollen_joint_count
        ]])
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        probability = float(model.predict_proba(X_scaled)[0][1])
        prediction = int(probability >= 0.5)
        confidence = float(max(probability, 1 - probability))
        risk_level = get_risk_level(probability)
        
        return PredictionResponse(
            inflammation_probability=probability,
            inflammation_prediction=prediction,
            risk_level=risk_level,
            confidence=confidence,
            prediction_date=datetime.now().isoformat(),
            model_type=model_type
        )
        
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)