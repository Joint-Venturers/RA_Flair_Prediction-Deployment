from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
try:
    model_data = joblib.load("app/ra_flare_model.joblib")
    models = model_data['models']
    scalers = model_data['scalers']
    feature_columns = model_data['feature_columns']
    logger.info("✅ RA Flare Prediction model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    models = None
    scalers = None
    feature_columns = None

app = FastAPI(
    title="RA Flare Prediction API",
    description="AI-powered 24-hour rheumatoid arthritis flare prediction system",
    version="1.0.5"
)

class PredictionRequest(BaseModel):
    weather_data: Dict = {
        "temperature": 15.0,
        "humidity": 60.0,
        "pressure": 1013.0,
        "weather_condition": "cloudy",
        "temp_change_24h": 0.0,
        "pressure_change_24h": 0.0,
        "humidity_change_24h": 0.0
    }
    pain_history: Dict = {
        "1_day_avg": 3.0,
        "3_day_avg": 3.0,
        "7_day_avg": 3.0
    }
    patient_data: Dict = {
        "age": 55,
        "disease_duration": 5,
        "medication_adherence": 0.8,
        "sleep_quality": 6,
        "stress_level": 4
    }

class PredictionResponse(BaseModel):
    flare_probability: float
    risk_level: str
    confidence_score: float
    risk_factors: List[str]
    recommendations: List[str]
    model_predictions: Dict
    timestamp: str

def extract_flare_probability(prob_array):
    """
    WORKING: Extract flare probability from scikit-learn predict_proba output
    predict_proba returns [[no_flare_prob, flare_prob]] - we want the flare_prob (index 1)
    """
    try:
        logger.info(f"Extracting from array shape: {prob_array.shape}, dtype: {prob_array.dtype}")
        logger.info(f"Array content: {prob_array}")
        
        # For predict_proba output: [[prob_class_0, prob_class_1]]
        # We want prob_class_1 (flare probability)
        if prob_array.shape == (1, 2):
            flare_prob = prob_array[0, 1]  # Get the flare probability (class 1)
            result = float(flare_prob)
            logger.info(f"Extracted flare probability: {result}")
            return result
        else:
            logger.error(f"Unexpected array shape: {prob_array.shape}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error extracting probability: {e}")
        return 0.0

def safe_float(value) -> float:
    try:
        return float(value) if value is not None else 0.0
    except:
        return 0.0

def encode_weather_condition(condition: str) -> int:
    condition_map = {'sunny': 0, 'cloudy': 1, 'rainy': 2, 'stormy': 3}
    return condition_map.get(str(condition or '').lower().strip(), 1)

def prepare_features(weather_data: Dict, pain_history: Dict, patient_data: Dict) -> np.ndarray:
    if not feature_columns:
        raise ValueError("Feature columns not loaded")
    
    features = {}
    
    # Weather features
    features['temperature'] = safe_float(weather_data.get('temperature', 15))
    features['humidity'] = safe_float(weather_data.get('humidity', 60))
    features['pressure'] = safe_float(weather_data.get('pressure', 1013))
    features['temp_change_24h'] = safe_float(weather_data.get('temp_change_24h', 0))
    features['pressure_change_24h'] = safe_float(weather_data.get('pressure_change_24h', 0))
    features['humidity_change_24h'] = safe_float(weather_data.get('humidity_change_24h', 0))
    features['weather_condition_encoded'] = encode_weather_condition(weather_data.get('weather_condition', 'cloudy'))
    
    # Patient features
    features['age'] = safe_float(patient_data.get('age', 55))
    features['disease_duration'] = safe_float(patient_data.get('disease_duration', 5))
    features['medication_adherence'] = safe_float(patient_data.get('medication_adherence', 0.8))
    features['sleep_quality'] = safe_float(patient_data.get('sleep_quality', 6))
    features['stress_level'] = safe_float(patient_data.get('stress_level', 4))
    
    # Pain history
    features['pain_history_1d'] = safe_float(pain_history.get('1_day_avg', 3))
    features['pain_history_3d'] = safe_float(pain_history.get('3_day_avg', 3))
    features['pain_history_7d'] = safe_float(pain_history.get('7_day_avg', 3))
    
    # Time features
    now = datetime.now()
    features['hour_of_day'] = now.hour
    features['day_of_week'] = now.weekday()
    features['month'] = now.month
    
    # Build feature array
    feature_values = []
    for col in feature_columns:
        feature_values.append(safe_float(features.get(col, 0.0)))
    
    return np.array([feature_values], dtype=np.float64)

def identify_risk_factors(weather_data: Dict, pain_history: Dict, patient_data: Dict) -> List[str]:
    risk_factors = []
    
    try:
        humidity = safe_float(weather_data.get('humidity', 50))
        temperature = safe_float(weather_data.get('temperature', 15))
        pressure = safe_float(weather_data.get('pressure', 1013))
        
        if humidity > 70:
            risk_factors.append(f'High humidity ({humidity:.0f}%)')
        if temperature < 10:
            risk_factors.append(f'Cold temperature ({temperature:.1f}°C)')
        if pressure < 1000:
            risk_factors.append(f'Low barometric pressure ({pressure:.0f} hPa)')
        
        temp_change = abs(safe_float(weather_data.get('temp_change_24h', 0)))
        pressure_change = abs(safe_float(weather_data.get('pressure_change_24h', 0)))
        
        if temp_change > 10:
            risk_factors.append(f'Large temperature change ({temp_change:.0f}°C)')
        if pressure_change > 15:
            risk_factors.append(f'Large pressure change ({pressure_change:.0f} hPa)')
        
        recent_pain = safe_float(pain_history.get('1_day_avg', 0))
        if recent_pain > 6:
            risk_factors.append(f'High recent pain levels ({recent_pain:.1f}/10)')
        
        stress = safe_float(patient_data.get('stress_level', 0))
        if stress > 7:
            risk_factors.append(f'High stress level ({stress:.1f}/10)')
        
        sleep = safe_float(patient_data.get('sleep_quality', 10))
        if sleep < 4:
            risk_factors.append(f'Poor sleep quality ({sleep:.1f}/10)')
        
        adherence = safe_float(patient_data.get('medication_adherence', 1))
        if adherence < 0.7:
            risk_factors.append(f'Low medication adherence ({adherence:.0%})')
            
    except Exception as e:
        logger.error(f"Error identifying risk factors: {e}")
    
    return risk_factors

def generate_recommendations(risk_level: str, risk_factors: List[str]) -> List[str]:
    recommendations = []
    
    if risk_level == 'HIGH':
        recommendations.extend([
            'Consider taking prescribed preventive medication',
            'Apply heat therapy to affected joints',
            'Avoid strenuous activities today',
            'Monitor symptoms closely'
        ])
    elif risk_level == 'MODERATE':
        recommendations.extend([
            'Monitor symptoms more closely than usual',
            'Prepare pain management tools',
            'Consider gentle exercises'
        ])
    elif risk_level == 'LOW':
        recommendations.extend([
            'Continue regular medication routine',
            'Maintain normal activity level'
        ])
    else:
        recommendations.extend([
            'Great conditions for normal activities',
            'Continue current health routine'
        ])
    
    # Add specific recommendations
    risk_text = ' '.join(risk_factors).lower()
    if 'humidity' in risk_text:
        recommendations.append('Use a dehumidifier indoors if possible')
    if 'temperature' in risk_text:
        recommendations.append('Keep joints warm with layers')
    if 'stress' in risk_text:
        recommendations.append('Practice stress reduction techniques')
    
    return recommendations[:6]

@app.get("/")
async def root():
    return {
        "message": "RA Flare Prediction API - Working Version",
        "version": "1.0.5",
        "status": "Model loaded" if models else "Model not loaded"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": models is not None,
        "feature_count": len(feature_columns) if feature_columns else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_flare_risk(request: PredictionRequest):
    if not models:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        logger.info("Starting prediction request")
        
        # Prepare features
        features = prepare_features(
            request.weather_data,
            request.pain_history,
            request.patient_data
        )
        logger.info(f"Features prepared: {features.shape}")
        
        # Random Forest prediction
        logger.info("Making Random Forest prediction...")
        rf_probabilities = models['random_forest'].predict_proba(features)
        rf_prob = extract_flare_probability(rf_probabilities)
        logger.info(f"Random Forest flare probability: {rf_prob}")
        
        # Gradient Boosting prediction
        logger.info("Making Gradient Boosting prediction...")
        features_scaled = scalers['standard'].transform(features)
        gb_probabilities = models['gradient_boost'].predict_proba(features_scaled)
        gb_prob = extract_flare_probability(gb_probabilities)
        logger.info(f"Gradient Boosting flare probability: {gb_prob}")
        
        # Ensemble prediction
        ensemble_prob = (rf_prob * 0.6 + gb_prob * 0.4)
        logger.info(f"Ensemble probability: {ensemble_prob}")
        
        # Determine risk level
        if ensemble_prob >= 0.7:
            risk_level = 'HIGH'
        elif ensemble_prob >= 0.4:
            risk_level = 'MODERATE'
        elif ensemble_prob >= 0.2:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        # Calculate confidence
        confidence_score = max(0.0, 1.0 - abs(rf_prob - gb_prob))
        
        # Get risk factors and recommendations
        risk_factors = identify_risk_factors(
            request.weather_data,
            request.pain_history,
            request.patient_data
        )
        recommendations = generate_recommendations(risk_level, risk_factors)
        
        logger.info(f"Prediction completed: {risk_level} risk ({ensemble_prob:.1%})")
        
        return PredictionResponse(
            flare_probability=ensemble_prob,
            risk_level=risk_level,
            confidence_score=confidence_score,
            risk_factors=risk_factors,
            recommendations=recommendations,
            model_predictions={
                'random_forest': rf_prob,
                'gradient_boost': gb_prob
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
