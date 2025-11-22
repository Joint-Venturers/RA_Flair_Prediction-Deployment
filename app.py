# app.py
# Enhanced RA Flare Prediction API with Analytics & 14 Features
# Version 3.0.0 - Episode history tracking with comprehensive analytics

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import joblib
import numpy as np
import json
from datetime import datetime, timedelta
import os
from supabase import create_client, Client

# Try to import analytics helper - gracefully handle if not present
try:
    from analytics_helper import (
        get_feature_importance,
        load_training_metadata,
        calculate_trigger_statistics,
        analyze_trigger_combinations,
        calculate_trigger_impact,
        get_model_insights
    )
    ANALYTICS_HELPER_AVAILABLE = True
    print("[INFO] analytics_helper.py loaded successfully")
except ImportError as e:
    print(f"[WARN] analytics_helper.py not found: {e}")
    ANALYTICS_HELPER_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="RA Flare Prediction API - Enhanced with Episode History",
    description="ML API for predicting RA flares with 14 features including episode history tracking, comprehensive analytics, and user profile integration",
    version="3.0.0"
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
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
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
    model = joblib.load('ra_flare_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Load metadata
    with open('model_metadata.json', 'r') as f:
        MODEL_METADATA = json.load(f)
    
    feature_names = MODEL_METADATA.get('feature_names', [])
    model_type = MODEL_METADATA.get('model_type', 'gradient_boosting')
    
    print(f"[OK] Loaded {model_type} model with {len(feature_names)} features")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None
    scaler = None
    feature_names = []
    model_type = None
    MODEL_METADATA = {}

# Define the 14 essential features (MUST MATCH TRAINING ORDER)
FEATURE_COLUMNS = [
    'age',
    'sex',
    'disease_duration',
    'bmi',
    'sleep_hours',
    'smoking_status',
    'air_quality_index',
    'min_temperature',
    'max_temperature',
    'humidity',
    'change_in_barometric_pressure',
    'current_pain_score',
    'last_episode_duration',
    'days_since_last_episode'
]

# Pydantic model for prediction input (14 features)
class PredictionInput(BaseModel):
    # Demographics
    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0: Female, 1: Male)")
    disease_duration: float = Field(..., ge=0, le=50, description="Years since RA diagnosis")
    
    # Body metrics
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index")
    
    # Lifestyle
    sleep_hours: float = Field(..., ge=0, le=24, description="Sleep hours per night")
    smoking_status: int = Field(..., ge=0, le=2, description="Smoking status (0: No, 1: Yes, 2: Quit)")
    
    # Environmental
    air_quality_index: float = Field(..., ge=0, le=500, description="Air Quality Index")
    min_temperature: float = Field(..., ge=-30, le=50, description="Minimum temperature (°C)")
    max_temperature: float = Field(..., ge=-20, le=55, description="Maximum temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    change_in_barometric_pressure: float = Field(..., ge=-30, le=30, description="24-hour pressure change (hPa)")
    
    # Clinical
    current_pain_score: int = Field(..., ge=0, le=10, description="Current pain level (0-10)")
    last_episode_duration: int = Field(..., ge=0, le=30, description="Duration of last flare (days)")
    days_since_last_episode: int = Field(..., ge=0, le=180, description="Days since last flare")
    
    # Optional
    user_id: Optional[str] = Field(None, description="User UUID for prediction tracking")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 45,
                "sex": 0,
                "disease_duration": 8,
                "bmi": 28.5,
                "sleep_hours": 5.5,
                "smoking_status": 0,
                "air_quality_index": 120,
                "min_temperature": 3,
                "max_temperature": 10,
                "humidity": 85,
                "change_in_barometric_pressure": -12,
                "current_pain_score": 7,
                "last_episode_duration": 5,
                "days_since_last_episode": 14,
                "user_id": "c9bbde37-fbb2-4dbb-bdd2-db7d91c3a721"
            }
        }

# Trigger thresholds for personalized trigger identification (10 triggers)
TRIGGER_THRESHOLDS = {
    'Poor Sleep': {
        'feature': 'sleep_hours',
        'condition': 'lt',
        'value': 6.0,
        'severity_high': 5.0,
        'severity_moderate': 6.0
    },
    'Smoking': {
        'feature': 'smoking_status',
        'condition': 'eq',
        'value': 1,
        'severity_high': 1,
        'severity_moderate': 1
    },
    'Elevated BMI': {
        'feature': 'bmi',
        'condition': 'gt',
        'value': 27.0,
        'severity_high': 30.0,
        'severity_moderate': 27.0
    },
    'Poor Air Quality': {
        'feature': 'air_quality_index',
        'condition': 'gt',
        'value': 100,
        'severity_high': 150,
        'severity_moderate': 100
    },
    'High Humidity': {
        'feature': 'humidity',
        'condition': 'gt',
        'value': 70,
        'severity_high': 80,
        'severity_moderate': 70
    },
    'Cold Weather': {
        'feature': 'min_temperature',
        'condition': 'lt',
        'value': 10,
        'severity_high': 5,
        'severity_moderate': 10
    },
    'Pressure Drop': {
        'feature': 'change_in_barometric_pressure',
        'condition': 'lt',
        'value': -5,
        'severity_high': -10,
        'severity_moderate': -5
    },
    'Elevated Pain Score': {
        'feature': 'current_pain_score',
        'condition': 'gte',
        'value': 5,
        'severity_high': 7,
        'severity_moderate': 5
    },
    'Recent Episode': {
        'feature': 'days_since_last_episode',
        'condition': 'lt',
        'value': 30,
        'severity_high': 14,
        'severity_moderate': 30
    },
    'Long Previous Episode': {
        'feature': 'last_episode_duration',
        'condition': 'gt',
        'value': 7,
        'severity_high': 14,
        'severity_moderate': 7
    }
}

def identify_triggers(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify triggered conditions based on feature values"""
    triggers = []
    feature_importance = {
        item['feature']: item['importance'] 
        for item in MODEL_METADATA.get('metrics', {}).get('feature_importance', [])
    }
    
    for trigger_name, config in TRIGGER_THRESHOLDS.items():
        feature = config['feature']
        value = features.get(feature)
        
        if value is None:
            continue
        
        triggered = False
        severity = 'low'
        
        if config['condition'] == 'lt':
            if value < config.get('severity_high', config['value']):
                triggered = True
                severity = 'high'
            elif value < config.get('severity_moderate', config['value']):
                triggered = True
                severity = 'moderate'
            elif value < config['value']:
                triggered = True
                severity = 'low'
        elif config['condition'] == 'gt':
            if value > config.get('severity_high', config['value']):
                triggered = True
                severity = 'high'
            elif value > config.get('severity_moderate', config['value']):
                triggered = True
                severity = 'moderate'
            elif value > config['value']:
                triggered = True
                severity = 'low'
        elif config['condition'] == 'gte':
            if value >= config.get('severity_high', config['value']):
                triggered = True
                severity = 'high'
            elif value >= config.get('severity_moderate', config['value']):
                triggered = True
                severity = 'moderate'
            elif value >= config['value']:
                triggered = True
                severity = 'low'
        elif config['condition'] == 'eq':
            if value == config['value']:
                triggered = True
                severity = 'high'
        
        if triggered:
            impact_score = feature_importance.get(feature, 0.0)
            triggers.append({
                'trigger': trigger_name,
                'severity': severity,
                'impact_score': float(impact_score)
            })
    
    return sorted(triggers, key=lambda x: x['impact_score'], reverse=True)

def calculate_confidence(probability: float, triggers: List[Dict]) -> float:
    """Calculate prediction confidence based on probability and triggers"""
    trigger_count = len(triggers)
    high_severity_count = sum(1 for t in triggers if t['severity'] == 'high')
    
    # Base confidence from probability
    if probability > 0.7 or probability < 0.3:
        base_confidence = 0.85
    else:
        base_confidence = 0.65
    
    # Increase confidence with more triggers
    if trigger_count > 0:
        trigger_confidence = min(0.15, trigger_count * 0.03 + high_severity_count * 0.02)
        confidence = min(0.95, base_confidence + trigger_confidence)
    else:
        confidence = base_confidence
    
    return float(confidence)

def determine_risk_level(probability: float) -> str:
    """Determine risk level category"""
    if probability >= 0.6:
        return 'high'
    elif probability >= 0.4:
        return 'moderate'
    else:
        return 'low'

# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "RA Flare Prediction API - Enhanced with 14 Features & Episode History",
        "version": "3.0.0",
        "model_type": MODEL_METADATA.get('model_type', 'unknown'),
        "n_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "training_date": MODEL_METADATA.get('training_date', 'unknown'),
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information",
            "/analytics/overview": "GET - System overview",
            "/analytics/model-performance": "GET - Model metrics",
            "/analytics/triggers": "GET - Trigger analysis",
            "/analytics/user-insights": "POST - User-specific insights",
            "/analytics/feature-importance": "GET - Feature importance",
            "/analytics/trigger-combinations": "GET - Trigger combination analysis"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "supabase_connected": supabase is not None,
        "n_features": len(FEATURE_COLUMNS),
        "analytics_helper": ANALYTICS_HELPER_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
def model_info():
    """Get model information and metadata"""
    return {
        "model_metadata": MODEL_METADATA,
        "feature_columns": FEATURE_COLUMNS,
        "required_features": len(FEATURE_COLUMNS),
        "trigger_thresholds": TRIGGER_THRESHOLDS,
        "feature_descriptions": {
            'age': 'Patient age in years',
            'sex': 'Biological sex (0: Female, 1: Male)',
            'disease_duration': 'Years since RA diagnosis',
            'bmi': 'Body Mass Index (kg/m²)',
            'sleep_hours': 'Average sleep duration per night',
            'smoking_status': 'Smoking status (0: No, 1: Yes, 2: Quit)',
            'air_quality_index': 'Air Quality Index (0-500)',
            'min_temperature': 'Minimum temperature (°C)',
            'max_temperature': 'Maximum temperature (°C)',
            'humidity': 'Relative humidity (%)',
            'change_in_barometric_pressure': '24-hour pressure change (hPa)',
            'current_pain_score': 'Pain level (0-10)',
            'last_episode_duration': 'Duration of last flare (days)',
            'days_since_last_episode': 'Days since last flare'
        }
    }

# ============================================================================
# PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict")
def predict(input_data: PredictionInput):
    """Make a flare prediction based on 14 essential features"""
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Extract features in correct order
        features_dict = {
            'age': input_data.age,
            'sex': input_data.sex,
            'disease_duration': input_data.disease_duration,
            'bmi': input_data.bmi,
            'sleep_hours': input_data.sleep_hours,
            'smoking_status': input_data.smoking_status,
            'air_quality_index': input_data.air_quality_index,
            'min_temperature': input_data.min_temperature,
            'max_temperature': input_data.max_temperature,
            'humidity': input_data.humidity,
            'change_in_barometric_pressure': input_data.change_in_barometric_pressure,
            'current_pain_score': input_data.current_pain_score,
            'last_episode_duration': input_data.last_episode_duration,
            'days_since_last_episode': input_data.days_since_last_episode
        }
        
        # Create feature array in correct order
        features = np.array([[features_dict[col] for col in FEATURE_COLUMNS]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        
        # Identify triggers
        triggers = identify_triggers(features_dict)
        
        # Calculate confidence and risk level
        confidence = calculate_confidence(probability, triggers)
        risk_level = determine_risk_level(probability)
        
        # Prepare response
        response = {
            "prediction": prediction,
            "probability": round(probability, 4),
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "triggers": triggers,
            "timestamp": datetime.now().isoformat(),
            "model_version": MODEL_METADATA.get('model_type', 'unknown'),
            "n_features": len(FEATURE_COLUMNS)
        }
        
        # Save to Supabase if available
        if supabase and input_data.user_id:
            try:
                prediction_record = {
                    'timestamp': response['timestamp'],
                    'prediction': prediction,
                    'probability': probability,
                    'confidence': confidence,
                    'risk_level': risk_level,
                    'triggers': json.dumps(triggers),
                    'features': json.dumps(features_dict),
                    'user_id': input_data.user_id
                }
                supabase.table('predictions').insert(prediction_record).execute()
            except Exception as e:
                print(f"[WARN] Could not save to Supabase: {e}")
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/analytics/overview")
def analytics_overview():
    """Get system-wide analytics overview"""
    
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Get predictions from last 30 days
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        
        predictions = supabase.table('predictions')\
            .select('*')\
            .gte('timestamp', thirty_days_ago)\
            .execute()
        
        if not predictions.data:
            return {
                "total_predictions": 0,
                "flare_rate": 0,
                "high_risk_count": 0,
                "avg_confidence": 0,
                "period_days": 30
            }
        
        total = len(predictions.data)
        flares = sum(1 for p in predictions.data if p['prediction'] == 1)
        high_risk = sum(1 for p in predictions.data if p['risk_level'] == 'high')
        avg_confidence = sum(p['confidence'] for p in predictions.data) / total if total > 0 else 0
        
        return {
            "total_predictions": total,
            "flare_rate": round(flares / total if total > 0 else 0, 4),
            "high_risk_count": high_risk,
            "avg_confidence": round(avg_confidence, 4),
            "period_days": 30
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")

@app.get("/analytics/model-performance")
def model_performance():
    """Get model performance metrics"""
    
    try:
        # Try analytics helper first
        if ANALYTICS_HELPER_AVAILABLE:
            try:
                training_metadata = load_training_metadata()
                insights = get_model_insights()
                return {
                    "model_type": MODEL_METADATA.get('model_type', 'unknown'),
                    "metrics": MODEL_METADATA.get('metrics', {}),
                    "n_features": MODEL_METADATA.get('n_features', 14),
                    "training_date": MODEL_METADATA.get('training_date', 'unknown'),
                    "insights": insights,
                    "training_history": training_metadata
                }
            except Exception as e:
                print(f"[WARN] Analytics helper error: {e}")
        
        # Fallback to basic metadata
        return {
            "model_type": MODEL_METADATA.get('model_type', 'unknown'),
            "metrics": MODEL_METADATA.get('metrics', {}),
            "n_features": MODEL_METADATA.get('n_features', 14),
            "training_date": MODEL_METADATA.get('training_date', 'unknown')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model performance error: {str(e)}")

@app.get("/analytics/triggers")
def trigger_analysis():
    """Get trigger frequency analysis"""
    
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        
        predictions = supabase.table('predictions')\
            .select('triggers, prediction')\
            .gte('timestamp', thirty_days_ago)\
            .execute()
        
        if not predictions.data:
            return {"triggers": []}
        
        # Count trigger occurrences
        trigger_counts = {}
        for pred in predictions.data:
            if pred.get('triggers'):
                triggers_list = json.loads(pred['triggers']) if isinstance(pred['triggers'], str) else pred['triggers']
                for trigger in triggers_list:
                    trigger_name = trigger['trigger']
                    if trigger_name not in trigger_counts:
                        trigger_counts[trigger_name] = {
                            'count': 0,
                            'severities': {'high': 0, 'moderate': 0, 'low': 0},
                            'flare_correlation': 0
                        }
                    trigger_counts[trigger_name]['count'] += 1
                    trigger_counts[trigger_name]['severities'][trigger['severity']] += 1
                    if pred['prediction'] == 1:
                        trigger_counts[trigger_name]['flare_correlation'] += 1
        
        # Format response
        triggers = [
            {
                'trigger': name,
                'count': data['count'],
                'frequency': round(data['count'] / len(predictions.data), 4),
                'severities': data['severities'],
                'flare_correlation': round(data['flare_correlation'] / data['count'] if data['count'] > 0 else 0, 4)
            }
            for name, data in trigger_counts.items()
        ]
        
        return {"triggers": sorted(triggers, key=lambda x: x['count'], reverse=True)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trigger analysis error: {str(e)}")

@app.get("/analytics/feature-importance")
def feature_importance():
    """Get feature importance rankings"""
    
    # Always use metadata - don't call analytics_helper
    # This avoids file lookup errors for old model filenames
    feature_importance_data = MODEL_METADATA.get('metrics', {}).get('feature_importance', [])
    
    if not feature_importance_data:
        # Generate basic importance from feature names if not available
        feature_importance_data = [
            {"feature": feature, "importance": 0.0, "category": "unknown"}
            for feature in FEATURE_COLUMNS
        ]
    
    return {"feature_importance": feature_importance_data}


@app.get("/analytics/trigger-combinations")
def trigger_combinations():
    """Analyze common trigger combinations"""
    
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        
        predictions = supabase.table('predictions')\
            .select('triggers, prediction')\
            .gte('timestamp', thirty_days_ago)\
            .execute()
        
        if not predictions.data:
            return {"combinations": []}
        
        # Try analytics helper if available
        if ANALYTICS_HELPER_AVAILABLE:
            try:
                combinations = analyze_trigger_combinations(predictions.data)
                return {"combinations": combinations}
            except Exception as e:
                print(f"[WARN] Analytics helper error: {e}")
        
        # Fallback: Basic combination analysis
        combo_counts = {}
        for pred in predictions.data:
            if pred.get('triggers'):
                triggers_list = json.loads(pred['triggers']) if isinstance(pred['triggers'], str) else pred['triggers']
                trigger_names = sorted([t['trigger'] for t in triggers_list])
                
                if len(trigger_names) >= 2:
                    combo = ' + '.join(trigger_names)
                    if combo not in combo_counts:
                        combo_counts[combo] = {'count': 0, 'flares': 0}
                    combo_counts[combo]['count'] += 1
                    if pred['prediction'] == 1:
                        combo_counts[combo]['flares'] += 1
        
        combinations = [
            {
                'combination': combo,
                'count': data['count'],
                'flare_rate': round(data['flares'] / data['count'] if data['count'] > 0 else 0, 4)
            }
            for combo, data in combo_counts.items()
        ]
        
        return {"combinations": sorted(combinations, key=lambda x: x['count'], reverse=True)[:10]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combination analysis error: {str(e)}")

# User insights endpoint
class UserInsightsRequest(BaseModel):
    user_id: str
    days: Optional[int] = 30

@app.post("/analytics/user-insights")
def user_insights(request: UserInsightsRequest):
    """Get comprehensive insights for a specific user"""
    
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Get user predictions
        cutoff_date = (datetime.now() - timedelta(days=request.days)).isoformat()
        
        predictions = supabase.table('predictions')\
            .select('*')\
            .eq('user_id', request.user_id)\
            .gte('timestamp', cutoff_date)\
            .order('timestamp', desc=True)\
            .execute()
        
        if not predictions.data:
            raise HTTPException(status_code=404, detail="No predictions found for user")
        
        total = len(predictions.data)
        flares = sum(1 for p in predictions.data if p['prediction'] == 1)
        
        # Get recent predictions (last 10)
        recent = predictions.data[:10]
        
        # Count user-specific triggers
        trigger_counts = {}
        for pred in predictions.data:
            if pred.get('triggers'):
                triggers_list = json.loads(pred['triggers']) if isinstance(pred['triggers'], str) else pred['triggers']
                for trigger in triggers_list:
                    trigger_name = trigger['trigger']
                    if trigger_name not in trigger_counts:
                        trigger_counts[trigger_name] = {'count': 0, 'with_flare': 0}
                    trigger_counts[trigger_name]['count'] += 1
                    if pred['prediction'] == 1:
                        trigger_counts[trigger_name]['with_flare'] += 1
        
        top_triggers = sorted(
            [
                {
                    'trigger': k,
                    'count': v['count'],
                    'flare_correlation': round(v['with_flare'] / v['count'] if v['count'] > 0 else 0, 4)
                }
                for k, v in trigger_counts.items()
            ],
            key=lambda x: x['count'],
            reverse=True
        )[:5]
        
        return {
            "user_id": request.user_id,
            "period_days": request.days,
            "total_predictions": total,
            "flare_count": flares,
            "flare_rate": round(flares / total if total > 0 else 0, 4),
            "top_triggers": top_triggers,
            "recent_predictions": recent
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User insights error: {str(e)}")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)