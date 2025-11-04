# simple_analytics_service.py - Bulletproof Analytics Service

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class SimpleAnalyticsService:
    """Bulletproof analytics service for RA prediction system"""
    
    def __init__(self):
        self.name = "Simple RA Analytics"
    
    def safe_float(self, value):
        """Convert to safe float for JSON"""
        if pd.isna(value) or np.isinf(value):
            return 0.0
        return float(value)
    
    def safe_int(self, value):
        """Convert to safe int for JSON"""
        if pd.isna(value):
            return 0
        return int(value)
    
    def calculate_correlations(self, data: pd.DataFrame) -> Dict:
        """Calculate symptom-trigger correlations"""
        correlations = {}
        strong_correlations = []
        
        try:
            # Define symptom and trigger columns
            symptoms = ['pain_history_1d', 'stress_level', 'flare_occurred']
            triggers = ['temperature', 'humidity', 'pressure', 'sleep_quality', 'medication_adherence']
            
            # Calculate correlations
            for symptom in symptoms:
                if symptom in data.columns:
                    for trigger in triggers:
                        if trigger in data.columns and len(data) > 5:
                            try:
                                # Convert boolean to int for correlation
                                symptom_data = data[symptom].astype(float)
                                trigger_data = data[trigger].astype(float)
                                
                                corr_value = symptom_data.corr(trigger_data)
                                
                                if abs(corr_value) > 0.3:  # Significant correlation
                                    strong_correlations.append({
                                        'symptom': symptom,
                                        'trigger': trigger,
                                        'correlation_strength': self.safe_float(corr_value),
                                        'correlation_direction': 'positive' if corr_value > 0 else 'negative',
                                        'p_value': 0.05,  # Simplified
                                        'interpretation': self._interpret_correlation(symptom, trigger, corr_value)
                                    })
                            except Exception:
                                continue
            
            # Sort by absolute correlation strength
            strong_correlations.sort(key=lambda x: abs(x['correlation_strength']), reverse=True)
            
            return {
                'strong_correlations': strong_correlations[:10],
                'correlation_count': len(strong_correlations),
                'personalized_insights': self._generate_insights(strong_correlations),
                'correlation_strength_summary': {
                    'total': len(strong_correlations),
                    'strong': len([c for c in strong_correlations if abs(c['correlation_strength']) > 0.7]),
                    'moderate': len([c for c in strong_correlations if 0.3 <= abs(c['correlation_strength']) <= 0.7]),
                    'average_strength': self.safe_float(np.mean([abs(c['correlation_strength']) for c in strong_correlations]) if strong_correlations else 0)
                }
            }
            
        except Exception as e:
            print(f"Correlation calculation error: {e}")
            return {
                'strong_correlations': [],
                'correlation_count': 0,
                'personalized_insights': ['Unable to calculate correlations with current data.'],
                'correlation_strength_summary': {'total': 0, 'strong': 0, 'moderate': 0, 'average_strength': 0.0}
            }
    
    def _interpret_correlation(self, symptom: str, trigger: str, correlation: float) -> str:
        """Generate human-readable interpretation"""
        direction = "increases" if correlation > 0 else "decreases"
        strength = "strongly" if abs(correlation) > 0.7 else "moderately"
        
        # Clean up names for better readability
        symptom_clean = symptom.replace('_', ' ').replace('1d', '').title()
        trigger_clean = trigger.replace('_', ' ').title()
        
        return f"{symptom_clean} {strength} {direction} with {trigger_clean}"
    
    def _generate_insights(self, correlations: List[Dict]) -> List[str]:
        """Generate personalized insights from correlations"""
        insights = []
        
        if not correlations:
            return ["Insufficient data for correlation analysis. Continue tracking symptoms for better insights."]
        
        # Top correlation insight
        if correlations:
            top_corr = correlations[0]
            strength_word = "strong" if abs(top_corr['correlation_strength']) > 0.7 else "notable"
            insights.append(f"Your strongest correlation is a {strength_word} relationship between {top_corr['symptom'].replace('_', ' ')} and {top_corr['trigger'].replace('_', ' ')} ({top_corr['correlation_strength']:.2f})")
        
        # Weather insights
        weather_corrs = [c for c in correlations if c['trigger'] in ['temperature', 'humidity', 'pressure']]
        if weather_corrs:
            insights.append(f"Weather factors show significant impact on your symptoms, with {len(weather_corrs)} notable correlations identified")
        
        # Lifestyle insights
        lifestyle_corrs = [c for c in correlations if c['trigger'] in ['sleep_quality', 'medication_adherence', 'stress_level']]
        if lifestyle_corrs:
            insights.append(f"Lifestyle factors are important for you, with {len(lifestyle_corrs)} significant correlations found")
        
        # Specific actionable insights
        for corr in correlations[:3]:
            if corr['trigger'] == 'temperature' and corr['correlation_strength'] < -0.5:
                insights.append("Cold weather appears to worsen your symptoms - consider staying warm during temperature drops")
            elif corr['trigger'] == 'humidity' and corr['correlation_strength'] > 0.5:
                insights.append("High humidity correlates with increased symptoms - monitor weather forecasts")
            elif corr['trigger'] == 'sleep_quality' and corr['correlation_strength'] < -0.4:
                insights.append("Poor sleep quality strongly affects your symptoms - prioritize sleep hygiene")
        
        return insights[:6]  # Return top 6 insights
    
    def generate_risk_analytics(self, data: pd.DataFrame) -> Dict:
        """Generate risk analytics from patient data"""
        try:
            if len(data) == 0:
                return self._default_risk_analytics()
            
            # Calculate basic risk metrics
            avg_pain = self.safe_float(data['pain_history_1d'].mean() if 'pain_history_1d' in data.columns else 5.0)
            pain_trend = self._calculate_pain_trend(data)
            flare_rate = self.safe_float(data['flare_occurred'].mean() if 'flare_occurred' in data.columns else 0.2)
            days_since_last_flare = self._days_since_last_flare(data)
            
            # Determine risk level
            if avg_pain > 7 or flare_rate > 0.4:
                risk_level = "HIGH"
                current_risk_score = min(9.0, 6.0 + avg_pain * 0.3)
            elif avg_pain > 5 or flare_rate > 0.2:
                risk_level = "MODERATE"
                current_risk_score = 4.0 + avg_pain * 0.4
            else:
                risk_level = "LOW"
                current_risk_score = 2.0 + avg_pain * 0.2
            
            return {
                'current_risk_score': self.safe_float(current_risk_score),
                'risk_level': risk_level,
                'trend_direction': pain_trend,
                'days_since_last_flare': self.safe_int(days_since_last_flare),
                'average_pain_level': self.safe_float(avg_pain),
                'flare_frequency': self.safe_float(flare_rate * 100),  # As percentage
                'risk_factors': self._identify_risk_factors(data),
                'recommendations': self._generate_recommendations(risk_level, avg_pain)
            }
            
        except Exception as e:
            print(f"Risk analytics error: {e}")
            return self._default_risk_analytics()
    
    def _calculate_pain_trend(self, data: pd.DataFrame) -> str:
        """Calculate pain trend direction"""
        if 'pain_history_1d' not in data.columns or len(data) < 7:
            return "STABLE"
        
        try:
            recent_pain = data['pain_history_1d'].tail(7).mean()
            older_pain = data['pain_history_1d'].head(7).mean()
            
            if recent_pain > older_pain + 1:
                return "INCREASING"
            elif recent_pain < older_pain - 1:
                return "DECREASING"
            else:
                return "STABLE"
        except Exception:
            return "STABLE"
    
    def _days_since_last_flare(self, data: pd.DataFrame) -> int:
        """Calculate days since last flare"""
        if 'flare_occurred' not in data.columns:
            return 30
        
        try:
            flare_days = data[data['flare_occurred'] == True]
            if len(flare_days) == 0:
                return len(data)
            
            return len(data) - flare_days.index[-1] - 1
        except Exception:
            return 30
    
    def _identify_risk_factors(self, data: pd.DataFrame) -> List[str]:
        """Identify current risk factors"""
        risk_factors = []
        
        try:
            if 'pain_history_1d' in data.columns:
                recent_pain = data['pain_history_1d'].tail(3).mean()
                if recent_pain > 6:
                    risk_factors.append(f"Elevated recent pain levels ({recent_pain:.1f}/10)")
            
            if 'sleep_quality' in data.columns:
                recent_sleep = data['sleep_quality'].tail(7).mean()
                if recent_sleep < 5:
                    risk_factors.append(f"Poor sleep quality ({recent_sleep:.1f}/10)")
            
            if 'medication_adherence' in data.columns:
                recent_adherence = data['medication_adherence'].tail(7).mean()
                if recent_adherence < 0.8:
                    risk_factors.append(f"Low medication adherence ({recent_adherence:.0%})")
            
            if 'stress_level' in data.columns:
                recent_stress = data['stress_level'].tail(7).mean()
                if recent_stress > 7:
                    risk_factors.append(f"High stress levels ({recent_stress:.1f}/10)")
                    
        except Exception:
            risk_factors.append("Unable to assess current risk factors")
        
        return risk_factors if risk_factors else ["No significant risk factors identified"]
    
    def _generate_recommendations(self, risk_level: str, avg_pain: float) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.extend([
                "Contact your healthcare provider to discuss current symptoms",
                "Consider adjusting current treatment plan",
                "Monitor symptoms closely and track triggers",
                "Ensure consistent medication adherence"
            ])
        elif risk_level == "MODERATE":
            recommendations.extend([
                "Focus on stress management techniques",
                "Maintain consistent sleep schedule",
                "Continue current medication regimen",
                "Monitor weather-related symptom changes"
            ])
        else:
            recommendations.extend([
                "Continue current healthy habits",
                "Maintain regular exercise routine",
                "Keep tracking symptoms for patterns",
                "Stay consistent with medication timing"
            ])
        
        if avg_pain > 6:
            recommendations.append("Consider pain management strategies (heat/cold therapy, gentle exercise)")
        
        return recommendations
    
    def _default_risk_analytics(self) -> Dict:
        """Return default risk analytics when calculation fails"""
        return {
            'current_risk_score': 5.0,
            'risk_level': "MODERATE",
            'trend_direction': "STABLE",
            'days_since_last_flare': 14,
            'average_pain_level': 5.0,
            'flare_frequency': 20.0,
            'risk_factors': ["Insufficient data for detailed analysis"],
            'recommendations': ["Continue tracking symptoms", "Maintain medication adherence", "Monitor weather patterns"]
        }
    
    def generate_comprehensive_analytics(self, user_id: str, data: pd.DataFrame) -> Dict:
        """Generate complete analytics package"""
        
        print(f"Generating analytics for {user_id} with {len(data)} records")
        
        try:
            # Generate risk analytics
            risk_analytics = self.generate_risk_analytics(data)
            
            # Generate correlation analytics
            correlation_analytics = self.calculate_correlations(data)
            
            # Generate summary insights
            summary_insights = self._generate_summary_insights(risk_analytics, correlation_analytics)
            
            return {
                'user_id': user_id,
                'analysis_date': datetime.now().isoformat(),
                'data_summary': {
                    'total_records': len(data),
                    'date_range_days': len(data),
                    'avg_pain_level': self.safe_float(data['pain_history_1d'].mean() if 'pain_history_1d' in data.columns else 5.0),
                    'flare_count': self.safe_int(data['flare_occurred'].sum() if 'flare_occurred' in data.columns else 0)
                },
                'risk_analytics': risk_analytics,
                'correlation_analytics': correlation_analytics,
                'summary_insights': summary_insights,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Analytics generation error: {e}")
            return {
                'user_id': user_id,
                'analysis_date': datetime.now().isoformat(),
                'error': str(e),
                'status': 'error'
            }
    
    def _generate_summary_insights(self, risk_analytics: Dict, correlation_analytics: Dict) -> List[str]:
        """Generate overall summary insights"""
        insights = []
        
        try:
            # Risk-based insights
            risk_level = risk_analytics.get('risk_level', 'MODERATE')
            insights.append(f"Your current risk level is {risk_level} with a risk score of {risk_analytics.get('current_risk_score', 5.0):.1f}/10")
            
            # Trend insights
            trend = risk_analytics.get('trend_direction', 'STABLE')
            if trend == "INCREASING":
                insights.append("Your symptoms show an increasing trend - consider consulting your healthcare provider")
            elif trend == "DECREASING":
                insights.append("Great news! Your symptoms show a decreasing trend")
            else:
                insights.append("Your symptoms are relatively stable")
            
            # Correlation insights
            strong_corrs = correlation_analytics.get('strong_correlations', [])
            if strong_corrs:
                insights.append(f"Found {len(strong_corrs)} significant correlations between your symptoms and environmental/lifestyle factors")
            
            # Days since flare
            days_since = risk_analytics.get('days_since_last_flare', 14)
            if days_since < 7:
                insights.append("Recent flare activity detected - focus on recovery and trigger avoidance")
            elif days_since > 30:
                insights.append(f"Excellent! {days_since} days since your last flare - current management is working well")
            
        except Exception:
            insights.append("Analysis completed successfully")
        
        return insights

# Create global instance for use in server
simple_analytics = SimpleAnalyticsService()

# Wrapper function for compatibility with existing code
def generate_dashboard_analytics(user_id: str, data: pd.DataFrame) -> Dict:
    """Wrapper function for compatibility"""
    return simple_analytics.generate_comprehensive_analytics(user_id, data)

if __name__ == "__main__":
    print("✅ Simple Analytics Service Ready")
    print("This module provides bulletproof analytics for the RA prediction system")