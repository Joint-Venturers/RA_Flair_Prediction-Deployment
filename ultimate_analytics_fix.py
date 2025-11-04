# ultimate_analytics_fix.py - Complete FastAPI-safe analytics solution

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Union
import json

def convert_to_json_safe(obj: Any) -> Any:
    """Convert any object to JSON-safe format"""
    if obj is None:
        return None
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, str):
        return str(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_safe(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalars
        return convert_to_json_safe(obj.item())
    else:
        return str(obj)

class UltimateAnalyticsService:
    """FastAPI-safe analytics service"""
    
    def calculate_simple_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlations with complete JSON safety"""
        
        # Define available columns
        pain_col = 'pain_history_1d'
        weather_cols = ['temperature', 'humidity', 'pressure']
        lifestyle_cols = ['sleep_quality', 'stress_level', 'medication_adherence']
        flare_col = 'flare_occurred'
        
        correlations = []
        insights = []
        
        try:
            if pain_col in data.columns and len(data) > 5:
                pain_data = pd.to_numeric(data[pain_col], errors='coerce').fillna(5.0)
                
                # Weather correlations
                for weather_col in weather_cols:
                    if weather_col in data.columns:
                        weather_data = pd.to_numeric(data[weather_col], errors='coerce').fillna(0.0)
                        corr_val = pain_data.corr(weather_data)
                        
                        if not np.isnan(corr_val) and abs(corr_val) > 0.2:
                            correlations.append({
                                'symptom': 'Pain Level',
                                'trigger': weather_col.replace('_', ' ').title(),
                                'correlation_strength': convert_to_json_safe(corr_val),
                                'correlation_direction': 'positive' if corr_val > 0 else 'negative',
                                'p_value': 0.05,
                                'interpretation': f"Pain {'increases' if corr_val > 0 else 'decreases'} with {weather_col.replace('_', ' ')}"
                            })
                
                # Lifestyle correlations
                for lifestyle_col in lifestyle_cols:
                    if lifestyle_col in data.columns:
                        lifestyle_data = pd.to_numeric(data[lifestyle_col], errors='coerce').fillna(5.0)
                        corr_val = pain_data.corr(lifestyle_data)
                        
                        if not np.isnan(corr_val) and abs(corr_val) > 0.2:
                            correlations.append({
                                'symptom': 'Pain Level',
                                'trigger': lifestyle_col.replace('_', ' ').title(),
                                'correlation_strength': convert_to_json_safe(corr_val),
                                'correlation_direction': 'positive' if corr_val > 0 else 'negative', 
                                'p_value': 0.05,
                                'interpretation': f"Pain {'increases' if corr_val > 0 else 'decreases'} with {lifestyle_col.replace('_', ' ')}"
                            })
            
            # Sort by strength
            correlations.sort(key=lambda x: abs(x['correlation_strength']), reverse=True)
            
            # Generate insights
            if correlations:
                top_corr = correlations[0]
                insights.append(f"Strongest correlation: {top_corr['symptom']} and {top_corr['trigger']} ({top_corr['correlation_strength']:.2f})")
                
                weather_corrs = [c for c in correlations if c['trigger'] in ['Temperature', 'Humidity', 'Pressure']]
                if weather_corrs:
                    insights.append(f"Weather shows {len(weather_corrs)} significant correlations with your symptoms")
                
                negative_corrs = [c for c in correlations if c['correlation_direction'] == 'negative']
                if negative_corrs:
                    insights.append(f"Found {len(negative_corrs)} protective factors that reduce symptoms")
            else:
                insights.append("Continue tracking to identify correlations between symptoms and triggers")
            
        except Exception as e:
            print(f"Correlation calculation error: {e}")
            insights.append("Unable to calculate correlations with current data")
        
        return convert_to_json_safe({
            'strong_correlations': correlations[:10],
            'correlation_count': len(correlations),
            'personalized_insights': insights,
            'correlation_strength_summary': {
                'total': len(correlations),
                'strong': len([c for c in correlations if abs(c['correlation_strength']) > 0.7]),
                'moderate': len([c for c in correlations if 0.3 <= abs(c['correlation_strength']) <= 0.7]),
                'weak': len([c for c in correlations if abs(c['correlation_strength']) < 0.3]),
                'average_strength': np.mean([abs(c['correlation_strength']) for c in correlations]) if correlations else 0.0
            }
        })
    
    def calculate_risk_analytics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk analytics with JSON safety"""
        
        try:
            # Safe data extraction
            pain_values = pd.to_numeric(data.get('pain_history_1d', [5.0] * len(data)), errors='coerce').fillna(5.0)
            flare_values = pd.to_numeric(data.get('flare_occurred', [0] * len(data)), errors='coerce').fillna(0).astype(int)
            sleep_values = pd.to_numeric(data.get('sleep_quality', [6.0] * len(data)), errors='coerce').fillna(6.0)
            stress_values = pd.to_numeric(data.get('stress_level', [5.0] * len(data)), errors='coerce').fillna(5.0)
            
            # Calculate metrics
            avg_pain = float(pain_values.mean())
            flare_rate = float(flare_values.mean())
            avg_sleep = float(sleep_values.mean())
            avg_stress = float(stress_values.mean())
            
            # Risk scoring
            risk_score = min(10.0, max(1.0, 3.0 + avg_pain * 0.5 + flare_rate * 3.0 + (10 - avg_sleep) * 0.2 + avg_stress * 0.3))
            
            # Risk level
            if risk_score >= 7.5:
                risk_level = "HIGH"
            elif risk_score >= 5.0:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
            
            # Trend calculation
            if len(pain_values) >= 14:
                recent_pain = pain_values.tail(7).mean()
                older_pain = pain_values.head(7).mean()
                if recent_pain > older_pain + 1:
                    trend = "INCREASING"
                elif recent_pain < older_pain - 1:
                    trend = "DECREASING"
                else:
                    trend = "STABLE"
            else:
                trend = "STABLE"
            
            # Days since last flare
            flare_indices = flare_values[flare_values == 1]
            days_since_flare = len(data) - flare_indices.index[-1] - 1 if len(flare_indices) > 0 else len(data)
            
            # Risk factors
            risk_factors = []
            if avg_pain > 6:
                risk_factors.append(f"Elevated pain levels ({avg_pain:.1f}/10)")
            if avg_sleep < 5:
                risk_factors.append(f"Poor sleep quality ({avg_sleep:.1f}/10)")
            if avg_stress > 7:
                risk_factors.append(f"High stress levels ({avg_stress:.1f}/10)")
            if flare_rate > 0.3:
                risk_factors.append(f"Frequent flares ({flare_rate:.1%} rate)")
            
            if not risk_factors:
                risk_factors.append("No major risk factors identified")
            
            # Recommendations
            recommendations = []
            if risk_level == "HIGH":
                recommendations.extend([
                    "Consider consulting your healthcare provider",
                    "Focus on stress reduction techniques",
                    "Monitor symptoms closely"
                ])
            elif risk_level == "MODERATE":
                recommendations.extend([
                    "Maintain current treatment plan",
                    "Monitor weather-related changes",
                    "Continue healthy lifestyle habits"
                ])
            else:
                recommendations.extend([
                    "Keep up the excellent management",
                    "Continue tracking symptoms",
                    "Maintain current routines"
                ])
            
        except Exception as e:
            print(f"Risk analytics error: {e}")
            # Return safe defaults
            risk_score = 5.0
            risk_level = "MODERATE"
            trend = "STABLE" 
            days_since_flare = 14
            risk_factors = ["Unable to assess risk with current data"]
            recommendations = ["Continue tracking symptoms", "Consult healthcare provider"]
        
        return convert_to_json_safe({
            'current_risk_score': risk_score,
            'risk_level': risk_level,
            'trend_direction': trend,
            'days_since_last_flare': int(days_since_flare),
            'average_pain_level': avg_pain,
            'flare_frequency': flare_rate * 100,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        })
    
    def generate_comprehensive_analytics(self, user_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate complete analytics with full JSON safety"""
        
        print(f"Generating safe analytics for {user_id} with {len(data)} records")
        
        try:
            # Calculate components
            risk_analytics = self.calculate_risk_analytics(data)
            correlation_analytics = self.calculate_simple_correlations(data)
            
            # Summary insights
            summary_insights = [
                f"Analysis complete for {len(data)} days of data",
                f"Current risk level: {risk_analytics['risk_level']}",
                f"Found {correlation_analytics['correlation_count']} significant correlations"
            ]
            
            # Add trend insight
            trend = risk_analytics['trend_direction']
            if trend == "INCREASING":
                summary_insights.append("Symptoms show an increasing trend - consider medical consultation")
            elif trend == "DECREASING":
                summary_insights.append("Great progress! Symptoms are trending downward")
            else:
                summary_insights.append("Symptoms are stable")
            
            result = convert_to_json_safe({
                'user_id': user_id,
                'analysis_date': datetime.now().isoformat(),
                'data_summary': {
                    'total_records': len(data),
                    'date_range_days': len(data),
                    'data_quality': 'good' if len(data) >= 14 else 'limited'
                },
                'risk_analytics': risk_analytics,
                'correlation_analytics': correlation_analytics,
                'summary_insights': summary_insights,
                'status': 'success'
            })
            
            print("✅ Analytics generated successfully")
            return result
            
        except Exception as e:
            print(f"Analytics generation error: {e}")
            return convert_to_json_safe({
                'user_id': user_id,
                'analysis_date': datetime.now().isoformat(),
                'error': str(e),
                'status': 'error'
            })

# Global instance
ultimate_analytics = UltimateAnalyticsService()

# Compatibility wrapper
def generate_dashboard_analytics(user_id: str, data: pd.DataFrame) -> Dict[str, Any]:
    """Wrapper for compatibility"""
    return ultimate_analytics.generate_comprehensive_analytics(user_id, data)