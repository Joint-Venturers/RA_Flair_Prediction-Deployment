# FastAPI Server for Render Deployment
# Production-ready API with CORS and health checks

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib
import numpy as np
from datetime import datetime
import os

app = FastAPI(
    title="RA Inflammation Predictor API",
    description="Gradient Boosting model for RA inflammation prediction with batch processing",
    version="2.0.0"
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

# Request models
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
    
    @validator('max_temperature')
    def validate_temperature(cls, v, values):
        if 'min_temperature' in values and v < values['min_temperature']:
            raise ValueError('max_temperature must be >= min_temperature')
        return v
    
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

class BatchPredictionInput(BaseModel):
    patients: List[PredictionInput] = Field(..., min_items=1, max_items=100, 
                                            description="List of patient data (max 100)")
    
    class Config:
        schema_extra = {
            "example": {
                "patients": [
                    {
                        "age": 55.0, "gender": 1, "disease_duration": 10.0,
                        "bmi": 26.5, "min_temperature": 15.0, "max_temperature": 20.0,
                        "humidity": 65.0, "barometric_pressure": 1015.0,
                        "precipitation": 0.5, "wind_speed": 5.0,
                        "tender_joint_count": 2.0, "swollen_joint_count": 1.5
                    },
                    {
                        "age": 45.0, "gender": 1, "disease_duration": 5.0,
                        "bmi": 24.0, "min_temperature": 18.0, "max_temperature": 24.0,
                        "humidity": 55.0, "barometric_pressure": 1018.0,
                        "precipitation": 0.0, "wind_speed": 4.0,
                        "tender_joint_count": 0.5, "swollen_joint_count": 0.5
                    }
                ]
            }
        }

# Response models
class PredictionResponse(BaseModel):
    inflammation_probability: float
    inflammation_prediction: int
    risk_level: str
    confidence: float
    prediction_date: str
    model_type: str

class BatchPredictionResponse(BaseModel):
    total_patients: int
    predictions: List[PredictionResponse]
    summary: dict

def get_risk_level(probability: float) -> str:
    """Determine risk level from probability"""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MODERATE"
    else:
        return "HIGH"

def make_single_prediction(input_data: PredictionInput) -> dict:
    """Make prediction for single patient"""
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
    
    return {
        "inflammation_probability": probability,
        "inflammation_prediction": prediction,
        "risk_level": risk_level,
        "confidence": confidence,
        "prediction_date": datetime.now().isoformat(),
        "model_type": model_type
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RA Inflammation Predictor API",
        "version": "2.0.0",
        "model": model_type if model_type else "not loaded",
        "status": "healthy" if model else "model not loaded",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
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
        "version": "2.0.0",
        "batch_support": True,
        "max_batch_size": 100
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """
    Make single inflammation prediction
    
    Returns probability, prediction, risk level, and confidence
    """
    if not model:
        raise HTTPException(503, "Model not loaded")
    
    try:
        result = make_single_prediction(input_data)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Make batch predictions for multiple patients
    
    Accepts up to 100 patients at once.
    Returns predictions for all patients plus summary statistics.
    """
    if not model:
        raise HTTPException(503, "Model not loaded")
    
    try:
        predictions = []
        
        # Process each patient
        for patient in batch_input.patients:
            result = make_single_prediction(patient)
            predictions.append(result)
        
        # Calculate summary statistics
        total = len(predictions)
        inflammation_count = sum(1 for p in predictions if p['inflammation_prediction'] == 1)
        remission_count = total - inflammation_count
        
        risk_counts = {
            'LOW': sum(1 for p in predictions if p['risk_level'] == 'LOW'),
            'MODERATE': sum(1 for p in predictions if p['risk_level'] == 'MODERATE'),
            'HIGH': sum(1 for p in predictions if p['risk_level'] == 'HIGH')
        }
        
        avg_probability = sum(p['inflammation_probability'] for p in predictions) / total
        avg_confidence = sum(p['confidence'] for p in predictions) / total
        
        summary = {
            "inflammation_count": inflammation_count,
            "remission_count": remission_count,
            "inflammation_rate": round(inflammation_count / total * 100, 2),
            "risk_distribution": risk_counts,
            "average_probability": round(avg_probability, 4),
            "average_confidence": round(avg_confidence, 4)
        }
        
        return BatchPredictionResponse(
            total_patients=total,
            predictions=[PredictionResponse(**p) for p in predictions],
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(500, f"Batch prediction error: {str(e)}")

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)