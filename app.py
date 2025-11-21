# Enhanced RA Flare Prediction API with Analytics
# Includes prediction endpoints + comprehensive analytics endpoints

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import joblib
import numpy as np
from datetime import datetime, timedelta
import os
from supabase import create_client, Client
from analytics_helper import (
    get_feature_importance,
    load_training_metadata,
    calculate_trigger_statistics,
    analyze_trigger_combinations,
    calculate_trigger_impact,
    get_model_insights
)

# Initialize FastAPI app
app = FastAPI(
    title="RA Flare Prediction API - Enhanced",
    description="Machine Learning API for predicting rheumatoid arthritis flares with personalized trigger identification and comprehensive analytics",
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

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("[OK] Supabase client initialized successfully")
    except Exception as e:
        print(f"[WARN] Failed to initialize Supabase client: {e}")
else:
    print("[WARN] Supabase credentials not found - analytics will be limited")

# Load ML model and components
try:
    model = joblib.load('ra_model_gradient_boosting.pkl')
    scaler = joblib.load('ra_scaler_gradient_boosting.pkl')
    feature_names = joblib.load('ra_features_gradient_boosting.pkl')
    model_type = joblib.load('ra_model_type.pkl')
    print(f"[OK] Loaded {model_type} model with {len(feature_names)} features")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None
    scaler = None
    feature_names = None
    model_type = None

# ============================================================
# DATA MODELS
# ============================================================

class PatientData(BaseModel):
    age: float
    sex: int
    disease_duration: float
    bmi: float
    sleep_hours: float
    smoking_status: int
    air_quality_index: float
    min_temperature: float
    max_temperature: float
    humidity: float
    barometric_pressure: float
    precipitation: float
    wind_speed: float
    current_pain_score: float
    tender_joint_count: int
    swollen_joint_count: int


class BatchPredictionRequest(BaseModel):
    predictions: List[PatientData]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

async def log_prediction_to_supabase(result: dict, input_data: dict):
    """Log prediction to Supabase for analytics"""
    if not supabase:
        return False
    
    try:
        prediction_log = {
            'prediction': int(result['inflammation_prediction']),
            'probability': float(result['inflammation_probability']),
            'confidence': float(result.get('confidence', 0.0)),
            'risk_level': result['risk_level'],
            'triggers': result.get('personalized_triggers', []),
            'features': input_data,
            'timestamp': datetime.now().isoformat()
        }
        
        response = supabase.table('predictions').insert(prediction_log).execute()
        return True
    except Exception as e:
        print(f"[WARN] Failed to log prediction: {e}")
        return False


def identify_triggers(data: dict, feature_importance: dict = None) -> List[dict]:
    """
    Identify personalized triggers based on patient data
    Returns list of triggers with severity and impact scores
    """
    triggers = []
    
    # Sleep trigger
    if data['sleep_hours'] < 6:
        severity = 'high' if data['sleep_hours'] < 5 else 'moderate'
        triggers.append({
            'trigger': 'Poor Sleep',
            'severity': severity,
            'value': f"{data['sleep_hours']} hours",
            'threshold': '< 6 hours',
            'impact_score': 0.25
        })
    
    # Pain score trigger
    if data['current_pain_score'] > 6:
        severity = 'high' if data['current_pain_score'] >= 8 else 'moderate'
        triggers.append({
            'trigger': 'Elevated Pain Score',
            'severity': severity,
            'value': f"{data['current_pain_score']}/10",
            'threshold': '> 6/10',
            'impact_score': 0.20
        })
    
    # BMI trigger
    if data['bmi'] > 30:
        triggers.append({
            'trigger': 'Elevated BMI',
            'severity': 'moderate',
            'value': f"{data['bmi']}",
            'threshold': '> 30 (obesity)',
            'impact_score': 0.15
        })
    
    # Humidity trigger
    if data['humidity'] > 75:
        triggers.append({
            'trigger': 'High Humidity',
            'severity': 'moderate',
            'value': f"{data['humidity']}%",
            'threshold': '> 75%',
            'impact_score': 0.15
        })
    
    # Barometric pressure trigger
    if data['barometric_pressure'] < 1000:
        triggers.append({
            'trigger': 'Low Barometric Pressure',
            'severity': 'moderate',
            'value': f"{data['barometric_pressure']} hPa",
            'threshold': '< 1000 hPa',
            'impact_score': 0.15
        })
    
    # Air quality trigger
    if data['air_quality_index'] > 100:
        severity = 'high' if data['air_quality_index'] > 150 else 'moderate'
        triggers.append({
            'trigger': 'Poor Air Quality',
            'severity': severity,
            'value': f"AQI {data['air_quality_index']}",
            'threshold': '> 100 (unhealthy)',
            'impact_score': 0.15
        })
    
    # Cold weather trigger
    if data['min_temperature'] < 10:
        triggers.append({
            'trigger': 'Cold Weather',
            'severity': 'low',
            'value': f"{data['min_temperature']}°C",
            'threshold': '< 10°C',
            'impact_score': 0.10
        })
    
    return triggers


def generate_recommendations(triggers: List[dict], risk_level: str) -> List[str]:
    """
    Generate actionable recommendations based on triggers
    """
    recommendations = []
    
    # Trigger-specific recommendations
    trigger_names = [t['trigger'] for t in triggers]
    
    if 'Poor Sleep' in trigger_names:
        recommendations.append("💤 Prioritize sleep hygiene: Aim for 7-9 hours nightly")
    
    if 'Elevated BMI' in trigger_names:
        recommendations.append("🏃 Consider weight management: Consult with nutritionist")
    
    if 'Poor Air Quality' in trigger_names:
        recommendations.append("😷 Limit outdoor exposure when AQI > 100")
    
    if any(t in trigger_names for t in ['High Humidity', 'Low Barometric Pressure', 'Cold Weather']):
        recommendations.append("🌦️ Monitor weather changes: Plan indoor activities")
        recommendations.append("🧥 Stay warm: Layer clothing, use heating")
    
    if 'Elevated Pain Score' in trigger_names:
        recommendations.append("💊 Review pain management with rheumatologist")
    
    # Risk-based recommendations
    if risk_level == 'high':
        recommendations.append("⚠️ HIGH RISK: Contact your healthcare provider")
    
    # General recommendation
    recommendations.append("📱 Log symptoms in your RA tracking app")
    
    return recommendations


def assess_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability >= 0.7:
        return 'high'
    elif probability >= 0.4:
        return 'moderate'
    else:
        return 'low'

# ============================================================
# PREDICTION ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "RA Flare Prediction API - Enhanced v2.0",
        "model_type": model_type if model_type else "Model not loaded",
        "features": len(feature_names) if feature_names else 0,
        "endpoints": {
            "predictions": ["/predict", "/predict/batch"],
            "info": ["/health", "/features"],
            "analytics": [
                "/analytics/health",
                "/analytics/model-performance",
                "/analytics/feature-importance",
                "/analytics/trigger-frequency",
                "/analytics/trigger-combinations",
                "/analytics/trigger-impact",
                "/analytics/predictions-summary",
                "/analytics/training-history",
                "/analytics/model-insights"
            ]
        },
        "version": "2.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_type": model_type if model_type else None,
        "features": len(feature_names) if feature_names else 0,
        "supabase_connected": supabase is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/features")
async def get_features():
    """Get list of features used by the model"""
    if not feature_names:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "features": feature_names,
        "count": len(feature_names),
        "model_type": model_type
    }


@app.post("/predict")
async def predict(data: PatientData):
    """
    Single prediction endpoint with personalized trigger identification
    """
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = np.array([[
            data.age,
            data.sex,
            data.disease_duration,
            data.bmi,
            data.sleep_hours,
            data.smoking_status,
            data.air_quality_index,
            data.min_temperature,
            data.max_temperature,
            data.humidity,
            data.barometric_pressure,
            data.precipitation,
            data.wind_speed,
            data.current_pain_score,
            data.tender_joint_count,
            data.swollen_joint_count
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        
        # Calculate confidence
        confidence = abs(probability - 0.5) * 2
        
        # Assess risk level
        risk_level = assess_risk_level(probability)
        
        # Identify triggers
        data_dict = data.dict()
        triggers = identify_triggers(data_dict)
        
        # Generate recommendations
        recommendations = generate_recommendations(triggers, risk_level)
        
        # Prepare result
        result = {
            "inflammation_prediction": prediction,
            "inflammation_probability": probability,
            "confidence": confidence,
            "risk_level": risk_level,
            "personalized_triggers": triggers,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log prediction to Supabase
        await log_prediction_to_supabase(result, data_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint
    """
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        high_risk_count = 0
        
        for patient_data in request.predictions:
            # Prepare features
            features = np.array([[
                patient_data.age,
                patient_data.sex,
                patient_data.disease_duration,
                patient_data.bmi,
                patient_data.sleep_hours,
                patient_data.smoking_status,
                patient_data.air_quality_index,
                patient_data.min_temperature,
                patient_data.max_temperature,
                patient_data.humidity,
                patient_data.barometric_pressure,
                patient_data.precipitation,
                patient_data.wind_speed,
                patient_data.current_pain_score,
                patient_data.tender_joint_count,
                patient_data.swollen_joint_count
            ]])
            
            # Scale and predict
            features_scaled = scaler.transform(features)
            prediction = int(model.predict(features_scaled)[0])
            probability = float(model.predict_proba(features_scaled)[0][1])
            
            # Calculate confidence
            confidence = abs(probability - 0.5) * 2
            
            # Assess risk level
            risk_level = assess_risk_level(probability)
            
            if risk_level == 'high':
                high_risk_count += 1
            
            # Identify triggers
            data_dict = patient_data.dict()
            triggers = identify_triggers(data_dict)
            
            # Generate recommendations
            recommendations = generate_recommendations(triggers, risk_level)
            
            # Prepare result
            result = {
                "inflammation_prediction": prediction,
                "inflammation_probability": probability,
                "confidence": confidence,
                "risk_level": risk_level,
                "personalized_triggers": triggers,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log prediction to Supabase
            await log_prediction_to_supabase(result, data_dict)
            
            results.append(result)
        
        return {
            "results": results,
            "total_count": len(results),
            "high_risk_count": high_risk_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ANALYTICS ENDPOINTS
# ============================================================

@app.get("/analytics/health")
async def analytics_health():
    """Check analytics system health"""
    return {
        "status": "healthy",
        "supabase_connected": supabase is not None,
        "model_loaded": os.path.exists('ra_model_gradient_boosting.pkl'),
        "metadata_available": os.path.exists('training_metadata.json'),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/analytics/model-performance")
async def get_model_performance():
    """
    Get current model performance metrics
    
    Returns:
    - accuracy, f1_score, auc
    - training date
    - model type
    - feature count
    """
    try:
        metadata = load_training_metadata()
        
        if not metadata:
            return {
                "error": "Model metadata not found",
                "message": "Train the model first"
            }
        
        return {
            "model_performance": {
                "accuracy": metadata.get('accuracy', 0),
                "f1_score": metadata.get('f1', 0),
                "auc": metadata.get('auc', 0),
                "training_date": metadata.get('timestamp', ''),
                "model_type": metadata.get('model_type', 'unknown'),
                "total_features": metadata.get('n_features', 0),
                "training_samples": metadata.get('training_samples', 0),
                "test_samples": metadata.get('test_samples', 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/analytics/feature-importance")
async def get_feature_importance_endpoint():
    """
    Get feature importance rankings with categories
    
    Returns:
    - All features ranked by importance
    - Top 5 features
    - Features grouped by category
    - Clinical descriptions
    """
    try:
        feature_data = get_feature_importance()
        
        if 'error' in feature_data:
            return {
                "error": "Failed to load feature importance",
                "details": feature_data.get('error', '')
            }
        
        return feature_data
    except Exception as e:
        return {"error": str(e)}


@app.get("/analytics/trigger-frequency")
async def get_trigger_frequency(days: Optional[int] = 30):
    """
    Get trigger frequency statistics for the last N days
    
    Query Parameters:
    - days: Number of days to analyze (default: 30)
    
    Returns:
    - Trigger counts and percentages
    - Severity distributions
    - Average impact scores
    """
    try:
        if not supabase:
            return {
                "error": "Supabase not configured",
                "message": "Analytics database unavailable"
            }
        
        # Query predictions from last N days
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        response = supabase.table('predictions')\
            .select('*')\
            .gte('timestamp', cutoff_date)\
            .execute()
        
        predictions = response.data if response.data else []
        
        if not predictions:
            return {
                "message": f"No predictions found in the last {days} days",
                "total_predictions": 0,
                "time_period": f"last_{days}_days"
            }
        
        # Calculate trigger statistics
        stats = calculate_trigger_statistics(predictions)
        stats['time_period'] = f"last_{days}_days"
        
        return stats
    except Exception as e:
        return {"error": str(e)}


@app.get("/analytics/trigger-combinations")
async def get_trigger_combinations(days: Optional[int] = 30):
    """
    Analyze common trigger combinations
    
    Query Parameters:
    - days: Number of days to analyze (default: 30)
    
    Returns:
    - Most common trigger combinations
    - Flare rates for each combination
    - Average probabilities
    """
    try:
        if not supabase:
            return {
                "error": "Supabase not configured",
                "message": "Analytics database unavailable"
            }
        
        # Query predictions from last N days
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        response = supabase.table('predictions')\
            .select('*')\
            .gte('timestamp', cutoff_date)\
            .execute()
        
        predictions = response.data if response.data else []
        
        if not predictions:
            return {
                "message": f"No predictions found in the last {days} days",
                "total_predictions": 0
            }
        
        # Analyze combinations
        combinations = analyze_trigger_combinations(predictions)
        combinations['time_period'] = f"last_{days}_days"
        
        return combinations
    except Exception as e:
        return {"error": str(e)}


@app.get("/analytics/trigger-impact")
async def get_trigger_impact(days: Optional[int] = 30):
    """
    Calculate individual trigger impact on flare predictions
    
    Query Parameters:
    - days: Number of days to analyze (default: 30)
    
    Returns:
    - Relative risk for each trigger
    - Flare rates when trigger present vs absent
    - Impact categories (Major/Moderate/Minor/Low)
    """
    try:
        if not supabase:
            return {
                "error": "Supabase not configured",
                "message": "Analytics database unavailable"
            }
        
        # Query predictions from last N days
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        response = supabase.table('predictions')\
            .select('*')\
            .gte('timestamp', cutoff_date)\
            .execute()
        
        predictions = response.data if response.data else []
        
        if not predictions:
            return {
                "message": f"No predictions found in the last {days} days",
                "total_predictions": 0
            }
        
        # Calculate impact
        impact_data = calculate_trigger_impact(predictions)
        impact_data['time_period'] = f"last_{days}_days"
        
        return impact_data
    except Exception as e:
        return {"error": str(e)}


@app.get("/analytics/predictions-summary")
async def get_predictions_summary(days: Optional[int] = 30):
    """
    Get summary statistics for predictions
    
    Query Parameters:
    - days: Number of days to analyze (default: 30)
    
    Returns:
    - Total predictions
    - Flare rate
    - Risk level distribution
    - Average probability
    """
    try:
        if not supabase:
            return {
                "error": "Supabase not configured",
                "message": "Analytics database unavailable"
            }
        
        # Query predictions from last N days
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        response = supabase.table('predictions')\
            .select('*')\
            .gte('timestamp', cutoff_date)\
            .execute()
        
        predictions = response.data if response.data else []
        
        if not predictions:
            return {
                "message": f"No predictions found in the last {days} days",
                "total_predictions": 0,
                "time_period": f"last_{days}_days"
            }
        
        # Calculate summary statistics
        total = len(predictions)
        flares = sum(1 for p in predictions if p.get('prediction') == 1)
        flare_rate = round((flares / total) * 100, 1) if total > 0 else 0
        
        # Risk level distribution
        risk_levels = {}
        for p in predictions:
            level = p.get('risk_level', 'unknown')
            risk_levels[level] = risk_levels.get(level, 0) + 1
        
        # Average probability
        avg_prob = round(
            sum(p.get('probability', 0) for p in predictions) / total,
            3
        ) if total > 0 else 0
        
        return {
            "time_period": f"last_{days}_days",
            "total_predictions": total,
            "flare_count": flares,
            "flare_rate_percentage": flare_rate,
            "risk_distribution": risk_levels,
            "average_probability": avg_prob,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/analytics/training-history")
async def get_training_history(limit: Optional[int] = 10):
    """
    Get training history from Supabase
    
    Query Parameters:
    - limit: Number of recent training runs to retrieve (default: 10)
    
    Returns:
    - List of training runs with performance metrics
    - Feature importances for each run
    """
    try:
        if not supabase:
            return {
                "error": "Supabase not configured",
                "message": "Analytics database unavailable"
            }
        
        response = supabase.table('training_history')\
            .select('*')\
            .order('timestamp', desc=True)\
            .limit(limit)\
            .execute()
        
        training_runs = response.data if response.data else []
        
        if not training_runs:
            return {
                "message": "No training history found",
                "training_runs": []
            }
        
        return {
            "total_runs": len(training_runs),
            "training_runs": training_runs,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/analytics/model-insights")
async def get_model_insights_endpoint():
    """
    Get combined model performance + feature importance insights
    
    Returns:
    - Model performance metrics
    - Top predictive features with clinical significance
    - Trigger mapping for each feature
    """
    try:
        insights = get_model_insights()
        
        if 'error' in insights:
            return {
                "error": "Failed to load model insights",
                "details": insights.get('error', '')
            }
        
        return insights
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
