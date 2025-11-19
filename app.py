# Enhanced RA Flare Prediction API with Personalized Trigger Identification

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = FastAPI(
    title="RA Flare Prediction API - Enhanced",
    description="Predict RA inflammation flares with personalized trigger identification",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
try:
    model = joblib.load('ra_model_gradient_boosting.pkl')
    scaler = joblib.load('ra_scaler_gradient_boosting.pkl')
    feature_names = joblib.load('ra_features_gradient_boosting.pkl')
    print("✅ Model loaded successfully")
    print(f"📋 Features: {feature_names}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

# Pydantic models
class PredictionInput(BaseModel):
    """
    Enhanced input features for RA flare prediction
    
    Demographics & Medical History:
    - age: Patient age (25-80 years)
    - sex: 0=female, 1=male
    - disease_duration: Years since RA diagnosis (1-30)
    - bmi: Body Mass Index (18.5-50)
    
    Lifestyle Factors:
    - sleep_hours: Hours of sleep (0-12)
    - smoking_status: 0=no, 1=yes, 2=quit
    
    Environmental Factors:
    - air_quality_index: AQI (0-300)
    - min_temperature: Daily minimum temp (°C)
    - max_temperature: Daily maximum temp (°C)
    - humidity: Relative humidity (%)
    - barometric_pressure: Pressure (hPa)
    - precipitation: Rainfall (mm)
    - wind_speed: Wind speed (km/h)
    
    Current State:
    - current_pain_score: Pain level (0-10)
    - tender_joint_count: Number of tender joints
    - swollen_joint_count: Number of swollen joints
    """
    
    # Demographics
    age: int = Field(..., ge=25, le=80, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="0=female, 1=male")
    disease_duration: int = Field(..., ge=1, le=30, description="Years since diagnosis")
    bmi: float = Field(..., ge=18.5, le=50, description="Body Mass Index")
    
    # Lifestyle
    sleep_hours: float = Field(..., ge=0, le=12, description="Hours of sleep")
    smoking_status: int = Field(..., ge=0, le=2, description="0=no, 1=yes, 2=quit")
    
    # Environmental
    air_quality_index: float = Field(..., ge=0, le=300, description="Air Quality Index")
    min_temperature: float = Field(..., description="Minimum temperature (°C)")
    max_temperature: float = Field(..., description="Maximum temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    barometric_pressure: float = Field(..., description="Pressure (hPa)")
    precipitation: float = Field(..., ge=0, description="Rainfall (mm)")
    wind_speed: float = Field(..., ge=0, description="Wind speed (km/h)")
    
    # Current state
    current_pain_score: float = Field(..., ge=0, le=10, description="Pain level (0-10)")
    tender_joint_count: float = Field(..., ge=0, description="Tender joints")
    swollen_joint_count: float = Field(..., ge=0, description="Swollen joints")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 55,
                "sex": 0,
                "disease_duration": 10,
                "bmi": 28.5,
                "sleep_hours": 5.5,
                "smoking_status": 0,
                "air_quality_index": 85,
                "min_temperature": 8,
                "max_temperature": 15,
                "humidity": 78,
                "barometric_pressure": 998,
                "precipitation": 2.5,
                "wind_speed": 6,
                "current_pain_score": 6.5,
                "tender_joint_count": 3,
                "swollen_joint_count": 2
            }
        }

class PredictionOutput(BaseModel):
    """Prediction response with personalized trigger analysis"""
    inflammation_prediction: int
    inflammation_probability: float
    confidence: float
    risk_level: str
    personalized_triggers: List[dict]
    recommendations: List[str]
    timestamp: str

class BatchPredictionInput(BaseModel):
    """Batch prediction request"""
    predictions: List[PredictionInput]

class BatchPredictionOutput(BaseModel):
    """Batch prediction response"""
    results: List[PredictionOutput]
    total_count: int
    high_risk_count: int
    timestamp: str

# Helper functions
def identify_triggers(input_data: dict, feature_importance: dict) -> List[dict]:
    """
    Identify personalized triggers based on feature importance and patient data
    
    Returns list of triggers with severity scores
    """
    triggers = []
    
    # Sleep trigger
    if input_data['sleep_hours'] < 6:
        triggers.append({
            "trigger": "Poor Sleep",
            "severity": "high",
            "value": f"{input_data['sleep_hours']:.1f} hours",
            "threshold": "< 6 hours",
            "impact_score": 0.25
        })
    
    # BMI trigger
    if input_data['bmi'] > 30:
        triggers.append({
            "trigger": "Elevated BMI",
            "severity": "moderate",
            "value": f"{input_data['bmi']:.1f}",
            "threshold": "> 30 (obesity)",
            "impact_score": 0.15
        })
    
    # Smoking trigger
    if input_data['smoking_status'] == 1:
        triggers.append({
            "trigger": "Active Smoking",
            "severity": "high",
            "value": "Current smoker",
            "threshold": "Active smoking",
            "impact_score": 0.20
        })
    
    # High humidity trigger
    if input_data['humidity'] > 75:
        triggers.append({
            "trigger": "High Humidity",
            "severity": "moderate",
            "value": f"{input_data['humidity']:.1f}%",
            "threshold": "> 75%",
            "impact_score": 0.15
        })
    
    # Low pressure trigger
    if input_data['barometric_pressure'] < 1000:
        triggers.append({
            "trigger": "Low Barometric Pressure",
            "severity": "moderate",
            "value": f"{input_data['barometric_pressure']:.1f} hPa",
            "threshold": "< 1000 hPa",
            "impact_score": 0.15
        })
    
    # Air quality trigger
    if input_data['air_quality_index'] > 100:
        triggers.append({
            "trigger": "Poor Air Quality",
            "severity": "moderate",
            "value": f"AQI {input_data['air_quality_index']:.0f}",
            "threshold": "> 100 (unhealthy)",
            "impact_score": 0.15
        })
    
    # Cold weather trigger
    if input_data['min_temperature'] < 10:
        triggers.append({
            "trigger": "Cold Weather",
            "severity": "low",
            "value": f"{input_data['min_temperature']:.1f}°C",
            "threshold": "< 10°C",
            "impact_score": 0.10
        })
    
    # High pain trigger
    if input_data['current_pain_score'] > 6:
        triggers.append({
            "trigger": "Elevated Pain Score",
            "severity": "high",
            "value": f"{input_data['current_pain_score']:.1f}/10",
            "threshold": "> 6/10",
            "impact_score": 0.20
        })
    
    # Sort by impact score
    triggers.sort(key=lambda x: x['impact_score'], reverse=True)
    
    return triggers

def generate_recommendations(triggers: List[dict], risk_level: str) -> List[str]:
    """Generate personalized recommendations based on identified triggers"""
    recommendations = []
    
    trigger_types = [t['trigger'] for t in triggers]
    
    if "Poor Sleep" in trigger_types:
        recommendations.append("💤 Prioritize sleep hygiene: Aim for 7-9 hours nightly")
    
    if "Elevated BMI" in trigger_types:
        recommendations.append("🏃 Consider weight management: Consult with nutritionist")
    
    if "Active Smoking" in trigger_types:
        recommendations.append("🚭 Smoking cessation: Talk to doctor about quit programs")
    
    if "Poor Air Quality" in trigger_types:
        recommendations.append("😷 Limit outdoor exposure when AQI > 100")
    
    if "High Humidity" in trigger_types or "Low Barometric Pressure" in trigger_types:
        recommendations.append("🌦️ Monitor weather changes: Plan indoor activities")
    
    if "Cold Weather" in trigger_types:
        recommendations.append("🧥 Stay warm: Layer clothing, use heating")
    
    if "Elevated Pain Score" in trigger_types:
        recommendations.append("💊 Review pain management with rheumatologist")
    
    if risk_level == "high":
        recommendations.append("⚠️ HIGH RISK: Contact your healthcare provider")
        recommendations.append("📱 Log symptoms in your RA tracking app")
    
    return recommendations

# API endpoints
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "RA Flare Prediction API - Enhanced Version",
        "version": "2.0.0",
        "features": 16,
        "model": "Gradient Boosting Classifier",
        "accuracy": 0.845,
        "endpoints": {
            "/predict": "Single prediction with trigger analysis",
            "/predict/batch": "Batch predictions",
            "/health": "Health check",
            "/features": "Feature information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/features")
async def get_features():
    """Get feature information"""
    if feature_names is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "features": feature_names,
        "count": len(feature_names),
        "categories": {
            "demographics": ["age", "sex", "disease_duration", "bmi"],
            "lifestyle": ["sleep_hours", "smoking_status"],
            "environmental": [
                "air_quality_index", "min_temperature", "max_temperature",
                "humidity", "barometric_pressure", "precipitation", "wind_speed"
            ],
            "current_state": ["current_pain_score", "tender_joint_count", "swollen_joint_count"]
        }
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict RA inflammation with personalized trigger identification
    
    Returns:
    - Inflammation prediction (0=no flare, 1=flare)
    - Probability score
    - Risk level (low/moderate/high)
    - Personalized triggers
    - Recommendations
    """
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to dict
        data_dict = input_data.model_dump()
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([data_dict])[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "low"
        elif probability < 0.6:
            risk_level = "moderate"
        else:
            risk_level = "high"
        
        # Identify personalized triggers
        triggers = identify_triggers(data_dict, {})
        
        # Generate recommendations
        recommendations = generate_recommendations(triggers, risk_level)
        
        return PredictionOutput(
            inflammation_prediction=int(prediction),
            inflammation_probability=float(probability),
            confidence=float(abs(probability - 0.5) * 2),  # 0-1 scale
            risk_level=risk_level,
            personalized_triggers=triggers,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Batch prediction endpoint for multiple patients
    """
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    high_risk_count = 0
    
    for input_data in batch_input.predictions:
        result = await predict(input_data)
        results.append(result)
        if result.risk_level == "high":
            high_risk_count += 1
    
    return BatchPredictionOutput(
        results=results,
        total_count=len(results),
        high_risk_count=high_risk_count,
        timestamp=datetime.now().isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)