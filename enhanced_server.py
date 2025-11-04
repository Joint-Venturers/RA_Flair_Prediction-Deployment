# enhanced_server.py - Enhanced RA Flare Prediction API with Advanced Analytics


from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import asyncio
import uvicorn

# Import our custom modules
from ultimate_analytics_fix import generate_dashboard_analytics, UltimateAnalyticsService
from feature_engineering import AdvancedFeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RA Flare Prediction API - Enhanced",
    description="Advanced AI-powered rheumatoid arthritis flare prediction with comprehensive analytics",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and analyzers
models = {}
scalers = {}
feature_engineer = None
analytics_engines = {}

# Pydantic models for request/response validation
class WeatherData(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    pressure: float = Field(..., description="Barometric pressure in hPa")
    weather_condition: str = Field(..., description="Weather condition: sunny/cloudy/rainy/stormy")
    temp_change_24h: float = Field(0, description="Temperature change in last 24h")
    pressure_change_24h: float = Field(0, description="Pressure change in last 24h")
    humidity_change_24h: float = Field(0, description="Humidity change in last 24h")

class PainHistory(BaseModel):
    day_1_avg: float = Field(..., ge=1, le=10, description="Average pain last 24h (1-10 scale)")
    day_3_avg: float = Field(..., ge=1, le=10, description="Average pain last 3 days (1-10 scale)")
    day_7_avg: float = Field(..., ge=1, le=10, description="Average pain last 7 days (1-10 scale)")

class PatientData(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Patient age")
    disease_duration: int = Field(..., ge=0, description="Years since RA diagnosis")
    medication_adherence: float = Field(..., ge=0, le=1, description="Medication adherence rate (0-1)")
    sleep_quality: float = Field(..., ge=1, le=10, description="Sleep quality (1-10 scale)")
    stress_level: float = Field(..., ge=1, le=10, description="Stress level (1-10 scale)")

class PredictionRequest(BaseModel):
    weather_data: WeatherData
    pain_history: PainHistory
    patient_data: PatientData
    include_analytics: bool = Field(True, description="Include detailed analytics in response")
    user_id: Optional[str] = Field(None, description="User ID for personalized analytics")

class HistoricalDataPoint(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    weather_data: WeatherData
    pain_history: PainHistory
    patient_data: PatientData
    flare_occurred: Optional[bool] = Field(None, description="Whether a flare occurred")

class AnalyticsRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    historical_data: List[HistoricalDataPoint] = Field(..., description="Historical data for analysis")
    analytics_type: str = Field("comprehensive", description="Type of analytics: comprehensive/risk/correlation")

class PredictionResponse(BaseModel):
    flare_probability: float = Field(..., description="Probability of flare in next 24h")
    risk_level: str = Field(..., description="Risk level: MINIMAL/LOW/MODERATE/HIGH")
    confidence_score: float = Field(..., description="Model confidence (0-1)")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    recommendations: List[str] = Field(..., description="Personalized recommendations")
    model_predictions: Dict[str, float] = Field(..., description="Individual model predictions")
    timestamp: str = Field(..., description="Prediction timestamp")
    analytics: Optional[Dict] = Field(None, description="Additional analytics if requested")

# Startup event to load models
@app.on_event("startup")
async def startup_event():
    """Load models and initialize analytics engines on startup"""
    global models, scalers, feature_engineer, analytics_engines
    
    logger.info("Starting RA Flare Prediction API...")
    
    try:
        # Load enhanced models
        model_path = Path("models/advanced_ra_model_ensemble.joblib")
        if model_path.exists():
            model_data = joblib.load(model_path)
            models = model_data.get('models', {})
            scalers = model_data.get('scalers', {})
            logger.info(f"Loaded {len(models)} advanced models")
        else:
            # Fallback to basic model
            basic_model_path = Path("models/ra_flare_model.joblib")
            if basic_model_path.exists():
                model_data = joblib.load(basic_model_path)
                models = model_data.get('models', {})
                scalers = model_data.get('scalers', {})
                logger.info(f"Loaded {len(models)} basic models")
            else:
                logger.error("No model files found!")
                models = {}
                scalers = {}
        
        # Initialize feature engineer
        feature_engineer = AdvancedFeatureEngineer()
        
        # Initialize analytics engines
        analytics_engines = {
            'risk_analyzer': UltimateAnalyticsService(),
            'correlation_analyzer': UltimateAnalyticsService()
        }
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        models = {}
        scalers = {}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if models else "not_loaded"
    
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "model_status": model_status,
        "feature_engineer": "loaded" if feature_engineer else "not_loaded",
        "analytics_engines": len(analytics_engines),
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RA Flare Prediction API - Enhanced Version",
        "version": "2.0.0",
        "features": [
            "Advanced ML ensemble predictions",
            "Comprehensive risk analytics",
            "Correlation analysis",
            "Personalized recommendations",
            "Real-time feature engineering"
        ],
        "endpoints": {
            "prediction": "/predict",
            "analytics": "/analytics",
            "batch_prediction": "/predict/batch",
            "correlation_analysis": "/analytics/correlations",
            "health": "/health",
            "docs": "/docs"
        }
    }

def prepare_features_enhanced(weather_data: WeatherData, pain_history: PainHistory, 
                            patient_data: PatientData) -> pd.DataFrame:
    """Prepare features using enhanced feature engineering"""
    
    # Create base feature dictionary
    features = {
        # Weather features
        'temperature': weather_data.temperature,
        'humidity': weather_data.humidity,
        'pressure': weather_data.pressure,
        'temp_change_24h': weather_data.temp_change_24h,
        'pressure_change_24h': weather_data.pressure_change_24h,
        'humidity_change_24h': weather_data.humidity_change_24h,
        
        # Pain history
        'pain_history_1d': pain_history.day_1_avg,
        'pain_history_3d': pain_history.day_3_avg,
        'pain_history_7d': pain_history.day_7_avg,
        
        # Patient data
        'age': patient_data.age,
        'disease_duration': patient_data.disease_duration,
        'medication_adherence': patient_data.medication_adherence,
        'sleep_quality': patient_data.sleep_quality,
        'stress_level': patient_data.stress_level,
        
        # Time features
        'hour_of_day': datetime.now().hour,
        'day_of_week': datetime.now().weekday(),
        'month': datetime.now().month,
        
        # Weather condition encoding
        'weather_condition_encoded': {
            'sunny': 0, 'cloudy': 1, 'rainy': 2, 'stormy': 3
        }.get(weather_data.weather_condition.lower(), 1)
    }
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Apply enhanced feature engineering if available
    if feature_engineer:
        try:
            df = feature_engineer.transform_dataset(df)
        except Exception as e:
            logger.warning(f"Feature engineering failed, using basic features: {e}")
    
    return df

def make_enhanced_prediction(features_df: pd.DataFrame) -> Dict:
    """Make prediction using enhanced ensemble models - FIXED VERSION"""
    
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    predictions = {}
    probabilities = {}
    
    try:
        # Get feature columns used during training (if available)
        model_data_path = Path("models/advanced_ra_model_ensemble.joblib")
        if model_data_path.exists():
            model_data = joblib.load(model_data_path)
            expected_features = getattr(model_data, 'feature_columns', None)
        else:
            expected_features = None
        
        # Align features with training data
        if expected_features:
            # Create DataFrame with all expected features
            aligned_features = pd.DataFrame(columns=expected_features, index=features_df.index)
            
            # Fill available features
            for col in expected_features:
                if col in features_df.columns:
                    aligned_features[col] = features_df[col]
                else:
                    aligned_features[col] = 0  # Default for missing features
            
            features_df = aligned_features.fillna(0)
        
        # Get predictions from all available models
        for model_name, model in models.items():
            try:
                if model_name in ['neural_network'] and 'standard' in scalers:
                    # Use scaled features for neural network
                    features_scaled = scalers['standard'].transform(features_df)
                    prob = model.predict_proba(features_scaled)[0][1]
                elif model_name == 'stacking_ensemble':
                    # Stacking ensemble uses original features
                    prob = model.predict_proba(features_df)[0][1]
                else:
                    # Tree-based models use original features
                    prob = model.predict_proba(features_df)[0][1]
                
                probabilities[model_name] = float(prob)
                predictions[model_name] = int(prob > 0.5)
                
            except Exception as e:
                logger.warning(f"Model {model_name} prediction failed: {e}")
                continue
        
        # Calculate ensemble probability (weighted by performance)
        if probabilities:
            # Simple average for now (could be weighted by model performance)
            ensemble_prob = np.mean(list(probabilities.values()))
        else:
            ensemble_prob = 0.5
        
        return {
            'ensemble_probability': float(ensemble_prob),
            'individual_probabilities': probabilities,
            'individual_predictions': predictions
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def determine_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MODERATE"
    elif probability >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"

def generate_risk_factors(features_df: pd.DataFrame, prediction_result: Dict) -> List[str]:
    """Generate list of identified risk factors"""
    risk_factors = []
    
    # Get the first (and only) row of features
    features = features_df.iloc[0]
    
    # Weather-based risk factors
    if features.get('temperature', 20) < 5:
        risk_factors.append(f"Very cold temperature ({features['temperature']:.1f}°C)")
    elif features.get('temperature', 20) < 10:
        risk_factors.append(f"Cold temperature ({features['temperature']:.1f}°C)")
    
    if features.get('humidity', 50) > 80:
        risk_factors.append(f"High humidity ({features['humidity']:.0f}%)")
    elif features.get('humidity', 50) > 70:
        risk_factors.append(f"Elevated humidity ({features['humidity']:.0f}%)")
    
    if features.get('pressure', 1013) < 995:
        risk_factors.append(f"Low barometric pressure ({features['pressure']:.0f} hPa)")
    elif features.get('pressure', 1013) < 1005:
        risk_factors.append(f"Reduced barometric pressure ({features['pressure']:.0f} hPa)")
    
    # Weather change risk factors
    if abs(features.get('temp_change_24h', 0)) > 10:
        risk_factors.append(f"Large temperature change ({features['temp_change_24h']:+.1f}°C in 24h)")
    
    if features.get('pressure_change_24h', 0) < -15:
        risk_factors.append(f"Rapid pressure drop ({features['pressure_change_24h']:+.0f} hPa in 24h)")
    
    # Pain and lifestyle risk factors
    if features.get('pain_history_1d', 3) > 6:
        risk_factors.append(f"High recent pain levels ({features['pain_history_1d']:.1f}/10)")
    
    if features.get('stress_level', 5) > 7:
        risk_factors.append(f"High stress level ({features['stress_level']:.1f}/10)")
    
    if features.get('sleep_quality', 6) < 4:
        risk_factors.append(f"Poor sleep quality ({features['sleep_quality']:.1f}/10)")
    
    if features.get('medication_adherence', 0.8) < 0.7:
        risk_factors.append(f"Low medication adherence ({features['medication_adherence']:.0%})")
    
    # Advanced risk factors (if available from feature engineering)
    if features.get('weather_severity', 0) > 3:
        risk_factors.append("Multiple adverse weather conditions")
    
    if features.get('triple_risk', 0) > 2:
        risk_factors.append("Multiple risk factors present simultaneously")
    
    return risk_factors

def generate_recommendations(risk_level: str, risk_factors: List[str], 
                           features_df: pd.DataFrame) -> List[str]:
    """Generate personalized recommendations based on risk assessment"""
    recommendations = []
    features = features_df.iloc[0]
    
    # Risk level specific recommendations
    if risk_level == "HIGH":
        recommendations.extend([
            "Consider taking prescribed preventive medication if available",
            "Contact your healthcare provider if symptoms worsen",
            "Monitor symptoms closely throughout the day",
            "Avoid strenuous activities and get adequate rest"
        ])
    elif risk_level == "MODERATE":
        recommendations.extend([
            "Be mindful of symptom changes today",
            "Ensure medication adherence",
            "Consider gentle movement and stretching"
        ])
    elif risk_level == "LOW":
        recommendations.extend([
            "Continue current management routine",
            "Maintain good sleep and stress management"
        ])
    else:  # MINIMAL
        recommendations.append("Low risk detected - continue current healthy habits")
    
    # Specific recommendations based on risk factors
    if any("temperature" in rf.lower() for rf in risk_factors):
        recommendations.extend([
            "Keep joints warm with layers or heating pads",
            "Consider warm baths or showers to ease stiffness"
        ])
    
    if any("humidity" in rf.lower() for rf in risk_factors):
        recommendations.extend([
            "Use a dehumidifier indoors if possible",
            "Stay in air-conditioned environments when available"
        ])
    
    if any("pressure" in rf.lower() for rf in risk_factors):
        recommendations.extend([
            "Be prepared for potential weather-related symptoms",
            "Consider gentle joint exercises to maintain mobility"
        ])
    
    if any("pain" in rf.lower() for rf in risk_factors):
        recommendations.extend([
            "Apply heat or cold therapy as preferred",
            "Practice relaxation techniques",
            "Consider adjusting daily activities"
        ])
    
    if any("stress" in rf.lower() for rf in risk_factors):
        recommendations.extend([
            "Practice stress reduction techniques (meditation, deep breathing)",
            "Prioritize self-care activities today"
        ])
    
    if any("sleep" in rf.lower() for rf in risk_factors):
        recommendations.extend([
            "Focus on getting quality sleep tonight",
            "Consider establishing a consistent bedtime routine"
        ])
    
    if any("medication" in rf.lower() for rf in risk_factors):
        recommendations.extend([
            "Take prescribed medications as directed",
            "Set reminders to improve medication consistency"
        ])
    
    return recommendations

# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_flare_risk(request: PredictionRequest):
    """Enhanced flare risk prediction with comprehensive analytics"""
    
    logger.info("Starting enhanced prediction request")
    
    try:
        # Prepare features using enhanced engineering
        features_df = prepare_features_enhanced(
            request.weather_data, 
            request.pain_history, 
            request.patient_data
        )
        
        logger.info(f"Features prepared: {features_df.shape}")
        
        # Make prediction
        prediction_result = make_enhanced_prediction(features_df)
        ensemble_probability = prediction_result['ensemble_probability']
        
        # Determine risk level
        risk_level = determine_risk_level(ensemble_probability)
        
        # Generate risk factors and recommendations
        risk_factors = generate_risk_factors(features_df, prediction_result)
        recommendations = generate_recommendations(risk_level, risk_factors, features_df)
        
        # Calculate confidence score (based on model agreement)
        individual_probs = list(prediction_result['individual_probabilities'].values())
        if len(individual_probs) > 1:
            confidence_score = 1.0 - (np.std(individual_probs) / np.mean(individual_probs))
            confidence_score = max(0.1, min(1.0, confidence_score))  # Clamp between 0.1 and 1.0
        else:
            confidence_score = 0.8  # Default confidence for single model
        
        # Prepare response
        response = PredictionResponse(
            flare_probability=ensemble_probability,
            risk_level=risk_level,
            confidence_score=confidence_score,
            risk_factors=risk_factors,
            recommendations=recommendations,
            model_predictions=prediction_result['individual_probabilities'],
            timestamp=datetime.now().isoformat()
        )
        
        # Add analytics if requested
        if request.include_analytics and request.user_id:
            try:
                # Convert current request to historical data format for analytics
                current_data = pd.DataFrame([{
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'pain_history_1d': request.pain_history.day_1_avg,
                    'temperature': request.weather_data.temperature,
                    'humidity': request.weather_data.humidity,
                    'pressure': request.weather_data.pressure,
                    'stress_level': request.patient_data.stress_level,
                    'sleep_quality': request.patient_data.sleep_quality,
                    'medication_adherence': request.patient_data.medication_adherence,
                    'month': datetime.now().month,
                    'predicted_flare_probability': ensemble_probability
                }])
                
                analytics_result = generate_dashboard_analytics(request.user_id, current_data)
                response.analytics = analytics_result
                
            except Exception as e:
                logger.warning(f"Analytics generation failed: {e}")
                response.analytics = {"error": "Analytics unavailable"}
        
        logger.info(f"Prediction completed: {risk_level} ({ensemble_probability:.1%})")
        return response
        
    except Exception as e:
        logger.error(f"Prediction request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict/batch")
async def batch_predict(requests: List[PredictionRequest]):
    """Batch prediction for multiple requests"""
    
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 requests per batch")
    
    results = []
    for i, request in enumerate(requests):
        try:
            result = await predict_flare_risk(request)
            results.append({"index": i, "success": True, "result": result})
        except Exception as e:
            results.append({"index": i, "success": False, "error": str(e)})
    
    return {
        "batch_size": len(requests),
        "successful_predictions": sum(1 for r in results if r["success"]),
        "failed_predictions": sum(1 for r in results if not r["success"]),
        "results": results
    }

# Analytics endpoint
@app.post("/analytics")
async def generate_analytics(request: AnalyticsRequest):
    """Generate comprehensive analytics - BULLETPROOF VERSION"""
    
    if len(request.historical_data) < 7:
        raise HTTPException(status_code=400, detail="Minimum 7 days of historical data required")
    
    try:
        # Convert to completely safe DataFrame
        records = []
        for point in request.historical_data:
            record = {
                'date': str(point.date),
                'pain_history_1d': float(point.pain_history.day_1_avg),
                'pain_history_3d': float(point.pain_history.day_3_avg), 
                'pain_history_7d': float(point.pain_history.day_7_avg),
                'temperature': float(point.weather_data.temperature),
                'humidity': float(point.weather_data.humidity),
                'pressure': float(point.weather_data.pressure),
                'temp_change_24h': float(point.weather_data.temp_change_24h),
                'pressure_change_24h': float(point.weather_data.pressure_change_24h),
                'humidity_change_24h': float(point.weather_data.humidity_change_24h),
                'age': int(point.patient_data.age),
                'disease_duration': int(point.patient_data.disease_duration),
                'medication_adherence': float(point.patient_data.medication_adherence),
                'sleep_quality': float(point.patient_data.sleep_quality),
                'stress_level': float(point.patient_data.stress_level),
                'month': int(pd.to_datetime(point.date).month)
            }
            
            # Handle flare_occurred safely
            if point.flare_occurred is not None:
                record['flare_occurred'] = int(bool(point.flare_occurred))
            else:
                record['flare_occurred'] = 0
                
            records.append(record)
        
        # Create DataFrame
        historical_df = pd.DataFrame(records)
        
        # Generate analytics with ultimate safety
        analytics_result = generate_dashboard_analytics(request.user_id, historical_df)
        
        # Ensure complete JSON safety
        from ultimate_analytics_fix import convert_to_json_safe
        safe_analytics = convert_to_json_safe(analytics_result)
        
        response = {
            "user_id": str(request.user_id),
            "analytics_type": str(request.analytics_type),
            "data_period": {
                "start_date": str(min(point.date for point in request.historical_data)),
                "end_date": str(max(point.date for point in request.historical_data)),
                "total_records": int(len(request.historical_data))
            },
            "analytics": safe_analytics,
            "generated_at": str(datetime.now().isoformat())
        }
        
        return convert_to_json_safe(response)
        
    except Exception as e:
        logger.error(f"Analytics generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")


@app.post("/analytics/correlations")
async def analyze_correlations(request: AnalyticsRequest):
    """Analyze correlations - FINAL BULLETPROOF VERSION"""
    
    try:
        # Convert to DataFrame with explicit type safety
        records = []
        for point in request.historical_data:
            record = {
                'pain_history_1d': float(point.pain_history.day_1_avg),
                'temperature': float(point.weather_data.temperature),
                'humidity': float(point.weather_data.humidity),
                'pressure': float(point.weather_data.pressure),
                'stress_level': float(point.patient_data.stress_level),
                'sleep_quality': float(point.patient_data.sleep_quality),
                'medication_adherence': float(point.patient_data.medication_adherence),
                'flare_occurred': 1 if (point.flare_occurred and bool(point.flare_occurred)) else 0
            }
            records.append(record)
        
        # Create safe DataFrame
        historical_df = pd.DataFrame(records)
        
        # Use ultimate analytics service
        from ultimate_analytics_fix import UltimateAnalyticsService, convert_to_json_safe
        ultimate_service = UltimateAnalyticsService()
        correlation_results = ultimate_service.calculate_simple_correlations(historical_df)
        
        # Build completely safe response
        response_data = {
            "user_id": str(request.user_id),
            "correlation_analysis": correlation_results,  # Already JSON-safe from ultimate service
            "data_period": {
                "total_records": len(request.historical_data),
                "date_range": f"{min(str(point.date) for point in request.historical_data)} to {max(str(point.date) for point in request.historical_data)}"
            },
            "generated_at": str(datetime.now().isoformat())
        }
        
        # Final safety conversion
        final_response = convert_to_json_safe(response_data)
        
        return final_response
        
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return safe error response
        error_response = {
            "user_id": str(request.user_id),
            "error": str(e),
            "status": "error",
            "generated_at": str(datetime.now().isoformat())
        }
        
        return convert_to_json_safe(error_response)


# Model information endpoint
@app.get("/models/info")
async def get_model_info():
    """Get model information - JSON SAFE"""
    
    from ultimate_analytics_fix import convert_to_json_safe
    
    try:
        model_info = {}
        
        for name, model in models.items():
            info = {
                "model_type": str(type(model).__name__),
                "has_feature_importance": bool(hasattr(model, 'feature_importances_')),
                "has_predict_proba": bool(hasattr(model, 'predict_proba')),
                "model_loaded": True
            }
            
            # Get basic parameter info safely
            try:
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    safe_params = {}
                    for k, v in list(params.items())[:5]:  # Limit to first 5 params
                        if isinstance(v, (int, float, str, bool, type(None))):
                            safe_params[str(k)] = v
                        else:
                            safe_params[str(k)] = str(type(v).__name__)
                    info["sample_parameters"] = safe_params
            except:
                info["sample_parameters"] = {}
                
            model_info[name] = info
        
        response = {
            "total_models": int(len(models)),
            "model_details": model_info,
            "scalers_loaded": [str(k) for k in scalers.keys()] if scalers else [],
            "feature_engineer_loaded": bool(feature_engineer is not None),
            "status": "healthy" if models else "no_models_loaded",
            "api_version": "2.0.0"
        }
        
        return convert_to_json_safe(response)
        
    except Exception as e:
        logger.error(f"Model info failed: {e}")
        return convert_to_json_safe({
            "error": str(e),
            "total_models": 0,
            "status": "error"
        })

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "enhanced_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )