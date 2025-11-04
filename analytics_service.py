import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AnalyticsResult:
    """Structured analytics result"""
    metric_name: str
    value: float
    trend: str  # 'improving', 'stable', 'worsening'
    confidence: float
    time_period: str
    insights: List[str]

class PersonalRiskAnalytics:
    """Personal risk score and trending analytics"""
    
    def __init__(self):
        self.baseline_window = 30  # days for baseline calculation
        
    def calculate_risk_trends(self, user_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk trending analysis"""
        
        if len(user_data) < 7:
            return self._minimal_risk_analysis(user_data)
        
        # Sort by date to ensure chronological order
        user_data = user_data.sort_values('date') if 'date' in user_data.columns else user_data
        
        # Current risk metrics
        current_risk = self._calculate_current_risk_score(user_data)
        
        # Risk trends over different time periods
        trends = {
            '7_day': self._calculate_risk_trend(user_data, 7),
            '30_day': self._calculate_risk_trend(user_data, 30),
            '90_day': self._calculate_risk_trend(user_data, 90)
        }
        
        # Flare frequency analysis
        flare_analysis = self._analyze_flare_frequency(user_data)
        
        # Seasonal patterns
        seasonal_patterns = self._analyze_seasonal_patterns(user_data)
        
        return {
            'current_risk_score': current_risk['score'],
            'risk_level': current_risk['level'],
            'risk_change_7d': trends['7_day']['change'],
            'risk_change_30d': trends['30_day']['change'],
            'trend_direction': trends['30_day']['direction'],
            'confidence': trends['30_day']['confidence'],
            'flare_frequency': flare_analysis,
            'seasonal_patterns': seasonal_patterns,
            'days_since_last_flare': self._days_since_last_flare(user_data),
            'predicted_flare_window': self._predict_next_flare_window(user_data),
            'risk_factors_trending': self._trending_risk_factors(user_data)
        }
    
    def _calculate_current_risk_score(self, data: pd.DataFrame) -> Dict:
        """Calculate current personalized risk score"""
        if len(data) == 0:
            return {'score': 0.5, 'level': 'UNKNOWN'}
        
        recent_data = data.tail(7)  # Last 7 days
        
        # Weight different factors
        pain_score = recent_data['pain_history_1d'].mean() / 10 * 0.3 if 'pain_history_1d' in recent_data.columns else 0.15
        weather_risk = self._calculate_weather_risk(recent_data) * 0.25
        medication_risk = (1 - recent_data['medication_adherence'].mean()) * 0.2 if 'medication_adherence' in recent_data.columns else 0.1
        stress_risk = recent_data['stress_level'].mean() / 10 * 0.15 if 'stress_level' in recent_data.columns else 0.075
        sleep_risk = (10 - recent_data['sleep_quality'].mean()) / 10 * 0.1 if 'sleep_quality' in recent_data.columns else 0.05
        
        total_risk = pain_score + weather_risk + medication_risk + stress_risk + sleep_risk
        
        # Convert to risk level
        if total_risk >= 0.7:
            level = 'HIGH'
        elif total_risk >= 0.4:
            level = 'MODERATE'
        elif total_risk >= 0.2:
            level = 'LOW'
        else:
            level = 'MINIMAL'
        
        return {'score': float(total_risk), 'level': level}
    
    def _calculate_weather_risk(self, data: pd.DataFrame) -> float:
        """Calculate weather-based risk component"""
        if len(data) == 0:
            return 0.5
        
        # Weather risk factors
        temp_risk = (data['temperature'] < 10).mean() * 0.4 if 'temperature' in data.columns else 0.2
        humidity_risk = (data['humidity'] > 70).mean() * 0.3 if 'humidity' in data.columns else 0.15
        pressure_risk = (data['pressure'] < 1000).mean() * 0.3 if 'pressure' in data.columns else 0.15
        
        return temp_risk + humidity_risk + pressure_risk
    
    def _calculate_risk_trend(self, data: pd.DataFrame, days: int) -> Dict:
        """Calculate risk trend over specified period"""
        if len(data) < days:
            return {'change': 0.0, 'direction': 'stable', 'confidence': 0.0}
        
        recent_period = data.tail(days)
        previous_period = data.iloc[-2*days:-days] if len(data) >= 2*days else data.head(days)
        
        recent_risk = self._calculate_current_risk_score(recent_period)['score']
        previous_risk = self._calculate_current_risk_score(previous_period)['score']
        
        change = recent_risk - previous_risk
        
        # Determine trend direction
        if abs(change) < 0.05:
            direction = 'stable'
        elif change > 0:
            direction = 'worsening'
        else:
            direction = 'improving'
        
        # Calculate confidence based on data consistency
        confidence = min(1.0, len(recent_period) / days)
        
        return {
            'change': float(change),
            'direction': direction,
            'confidence': float(confidence)
        }
    
    def _analyze_flare_frequency(self, data: pd.DataFrame) -> Dict:
        """Analyze flare frequency patterns"""
        if 'flare_occurred' not in data.columns or len(data) < 30:
            return {
                'monthly_average': 0,
                'days_between_flares': [],
                'flare_severity_trend': 'stable',
                'frequency_trend': 'stable'
            }
        
        flare_data = data[data['flare_occurred'] == 1]
        
        if len(flare_data) == 0:
            return {
                'monthly_average': 0,
                'days_between_flares': [],
                'flare_severity_trend': 'stable',
                'frequency_trend': 'stable'
            }
        
        # Calculate days between flares
        if 'date' in flare_data.columns:
            flare_dates = pd.to_datetime(flare_data['date'])
            days_between = flare_dates.diff().dt.days.dropna().tolist()
        else:
            days_between = []
        
        # Monthly flare rate
        total_days = len(data)
        monthly_rate = len(flare_data) * 30 / total_days if total_days > 0 else 0
        
        return {
            'monthly_average': float(monthly_rate),
            'days_between_flares': days_between,
            'average_days_between': float(np.mean(days_between)) if days_between else 0,
            'flare_severity_trend': self._calculate_severity_trend(data),
            'frequency_trend': self._calculate_frequency_trend(data)
        }
    
    def _analyze_seasonal_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze seasonal flare patterns"""
        if 'month' not in data.columns or len(data) < 90:
            return {'worst_months': [], 'best_months': [], 'seasonal_risk': 'unknown'}
        
        # Group by month and calculate flare rates
        if 'flare_occurred' in data.columns:
            monthly_flares = data.groupby('month')['flare_occurred'].mean().to_dict()
            worst_months = sorted(monthly_flares.items(), key=lambda x: x[1], reverse=True)[:3]
            best_months = sorted(monthly_flares.items(), key=lambda x: x[1])[:3]
            
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            
            return {
                'worst_months': [month_names[month] for month, _ in worst_months],
                'best_months': [month_names[month] for month, _ in best_months],
                'monthly_risk_scores': {month_names[k]: float(v) for k, v in monthly_flares.items()},
                'seasonal_variation': float(max(monthly_flares.values()) - min(monthly_flares.values()))
            }
        
        return {'worst_months': [], 'best_months': [], 'seasonal_risk': 'insufficient_data'}
    
    def _minimal_risk_analysis(self, data: pd.DataFrame) -> Dict:
        """Minimal analysis for insufficient data"""
        if len(data) == 0:
            return {
                'current_risk_score': 0.5,
                'risk_level': 'UNKNOWN',
                'trend_direction': 'stable',
                'confidence': 0.0,
                'data_insufficient': True
            }
        
        current_risk = self._calculate_current_risk_score(data)
        return {
            'current_risk_score': current_risk['score'],
            'risk_level': current_risk['level'],
            'trend_direction': 'stable',
            'confidence': 0.3,
            'data_insufficient': True,
            'message': 'More data needed for comprehensive analysis'
        }
    
    # FIXED: Add missing methods
    def _calculate_severity_trend(self, data):
        """Calculate flare severity trend over time"""
        if 'flare_occurred' not in data.columns or len(data) < 14:
            return 'insufficient_data'
        
        # Get recent and older flare data
        recent_flares = data.tail(30)['flare_occurred'].sum()
        older_flares = data.head(30)['flare_occurred'].sum()
        
        if recent_flares > older_flares * 1.2:
            return 'worsening'
        elif recent_flares < older_flares * 0.8:
            return 'improving'
        else:
            return 'stable'

    def _calculate_frequency_trend(self, data):
        """Calculate flare frequency trend over time"""
        if 'flare_occurred' not in data.columns or len(data) < 30:
            return 'insufficient_data'
        
        # Calculate flare rate for first and second half
        mid_point = len(data) // 2
        first_half_rate = data.iloc[:mid_point]['flare_occurred'].mean()
        second_half_rate = data.iloc[mid_point:]['flare_occurred'].mean()
        
        if second_half_rate > first_half_rate * 1.1:
            return 'increasing'
        elif second_half_rate < first_half_rate * 0.9:
            return 'decreasing'
        else:
            return 'stable'

    def _days_since_last_flare(self, data):
        """Calculate days since last flare occurred"""
        if 'flare_occurred' not in data.columns:
            return 0
        
        # Find last flare occurrence
        flare_indices = data[data['flare_occurred'] == 1].index
        if len(flare_indices) == 0:
            return len(data)  # No flares in dataset
        
        last_flare_index = flare_indices[-1]
        days_since = len(data) - 1 - last_flare_index
        return int(days_since)

    def _predict_next_flare_window(self, data):
        """Predict time window for next potential flare"""
        if 'flare_occurred' not in data.columns:
            return 'unknown'
        
        flare_data = data[data['flare_occurred'] == 1]
        if len(flare_data) < 2:
            return 'insufficient_data'
        
        # Calculate average days between flares
        flare_indices = flare_data.index.tolist()
        intervals = [flare_indices[i] - flare_indices[i-1] for i in range(1, len(flare_indices))]
        
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            days_since_last = self._days_since_last_flare(data)
            
            if days_since_last >= avg_interval * 0.8:
                return 'within_7_days'
            elif days_since_last >= avg_interval * 0.6:
                return 'within_14_days'
            else:
                return 'low_risk_period'
        
        return 'unknown'

    def _trending_risk_factors(self, data):
        """Identify trending risk factors"""
        trending_factors = []
        
        if len(data) < 14:
            return trending_factors
        
        recent_data = data.tail(7)
        older_data = data.iloc[-14:-7]
        
        # Check weather trend
        if 'humidity' in data.columns:
            if recent_data['humidity'].mean() > older_data['humidity'].mean() + 10:
                trending_factors.append('increasing_humidity')
        
        if 'temperature' in data.columns:
            if recent_data['temperature'].mean() < older_data['temperature'].mean() - 5:
                trending_factors.append('dropping_temperature')
        
        # Check lifestyle trends
        if 'stress_level' in data.columns:
            if recent_data['stress_level'].mean() > older_data['stress_level'].mean() + 1:
                trending_factors.append('increasing_stress')
        
        if 'sleep_quality' in data.columns:
            if recent_data['sleep_quality'].mean() < older_data['sleep_quality'].mean() - 1:
                trending_factors.append('declining_sleep')
        
        if 'medication_adherence' in data.columns:
            if recent_data['medication_adherence'].mean() < older_data['medication_adherence'].mean() - 0.1:
                trending_factors.append('medication_adherence_decline')
        
        return trending_factors

class CorrelationAnalysisEngine:
    """Advanced correlation analysis between symptoms and triggers (PP-14)"""
    
    def __init__(self):
        self.correlation_methods = ['pearson', 'spearman', 'kendall']
        self.significance_threshold = 0.05
        
    def analyze_symptom_trigger_correlations(self, data: pd.DataFrame) -> Dict:
        """Comprehensive correlation analysis between symptoms and potential triggers"""
        
        if len(data) < 10:
            return {'error': 'Insufficient data for correlation analysis'}
        
        # Define symptom and trigger categories
        symptoms = self._identify_symptom_columns(data)
        triggers = self._identify_trigger_columns(data)
        
        if not symptoms or not triggers:
            return {'error': 'No valid symptom or trigger columns found'}
        
        # Calculate correlations using multiple methods
        correlations = {}
        for method in self.correlation_methods:
            correlations[method] = self._calculate_correlations(data, symptoms, triggers, method)
        
        # Statistical significance testing
        significance_results = self._test_correlation_significance(data, symptoms, triggers)
        
        # Identify strongest correlations
        strong_correlations = self._identify_strong_correlations(correlations, significance_results)
        
        # Time-lagged correlations (e.g., weather yesterday affecting pain today)
        lagged_correlations = self._calculate_lagged_correlations(data, symptoms, triggers)
        
        # Personalized correlation insights
        insights = self._generate_correlation_insights(strong_correlations, lagged_correlations)
        
        return {
            'correlation_matrix': correlations,
            'statistical_significance': significance_results,
            'strong_correlations': strong_correlations,
            'lagged_correlations': lagged_correlations,
            'personalized_insights': insights,
            'correlation_strength_summary': self._summarize_correlation_strength(strong_correlations)
        }
    
    def _identify_symptom_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify columns related to symptoms"""
        symptom_patterns = [
            'pain', 'flare', 'stiff', 'swell', 'fatigue', 'joint'
        ]
        
        symptom_cols = []
        for col in data.columns:
            if any(pattern in col.lower() for pattern in symptom_patterns):
                if data[col].dtype in ['int64', 'float64']:
                    symptom_cols.append(col)
        
        # Always include these if available
        standard_symptoms = ['pain_history_1d', 'pain_history_3d', 'pain_history_7d', 'flare_occurred']
        for col in standard_symptoms:
            if col in data.columns and col not in symptom_cols:
                symptom_cols.append(col)
        
        return symptom_cols
    
    def _identify_trigger_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify columns related to potential triggers"""
        trigger_patterns = [
            'temperature', 'humidity', 'pressure', 'weather', 'stress', 'sleep', 
            'medication', 'exercise', 'diet'
        ]
        
        trigger_cols = []
        for col in data.columns:
            if any(pattern in col.lower() for pattern in trigger_patterns):
                if data[col].dtype in ['int64', 'float64']:
                    trigger_cols.append(col)
        
        return trigger_cols
    
    def _calculate_correlations(self, data: pd.DataFrame, symptoms: List[str], 
                              triggers: List[str], method: str) -> Dict:
        """Calculate correlations between symptoms and triggers"""
        correlations = {}
        
        for symptom in symptoms:
            correlations[symptom] = {}
            for trigger in triggers:
                try:
                    if method == 'pearson':
                        corr, p_value = stats.pearsonr(data[symptom], data[trigger])
                    elif method == 'spearman':
                        corr, p_value = stats.spearmanr(data[symptom], data[trigger])
                    elif method == 'kendall':
                        corr, p_value = stats.kendalltau(data[symptom], data[trigger])
                    
                    correlations[symptom][trigger] = {
                        'correlation': float(corr),
                        'p_value': float(p_value),
                        'significant': p_value < self.significance_threshold
                    }
                    
                except Exception as e:
                    correlations[symptom][trigger] = {
                        'correlation': 0.0,
                        'p_value': 1.0,
                        'significant': False,
                        'error': str(e)
                    }
        
        return correlations
    
    def _test_correlation_significance(self, data: pd.DataFrame, symptoms: List[str],
                                     triggers: List[str]) -> Dict:
        """Test statistical significance of correlations"""
        significance_results = {}
        
        for symptom in symptoms:
            significance_results[symptom] = {}
            for trigger in triggers:
                try:
                    # Pearson correlation with confidence interval
                    corr, p_value = stats.pearsonr(data[symptom], data[trigger])
                    n = len(data)
                    
                    # Calculate confidence interval
                    if abs(corr) < 1:
                        stderr = np.sqrt((1 - corr**2) / (n - 2))
                        ci_lower = corr - 1.96 * stderr
                        ci_upper = corr + 1.96 * stderr
                    else:
                        ci_lower = ci_upper = corr
                    
                    significance_results[symptom][trigger] = {
                        'correlation': float(corr),
                        'p_value': float(p_value),
                        'significant': p_value < self.significance_threshold,
                        'confidence_interval': [float(ci_lower), float(ci_upper)],
                        'sample_size': int(n)
                    }
                    
                except Exception as e:
                    significance_results[symptom][trigger] = {
                        'error': str(e)
                    }
        
        return significance_results
    
    def _calculate_lagged_correlations(self, data: pd.DataFrame, symptoms: List[str],
                                     triggers: List[str], max_lag: int = 3) -> Dict:
        """Calculate time-lagged correlations (e.g., weather yesterday -> pain today)"""
        lagged_correlations = {}
        
        for symptom in symptoms:
            lagged_correlations[symptom] = {}
            
            for trigger in triggers:
                lagged_correlations[symptom][trigger] = {}
                
                for lag in range(1, max_lag + 1):
                    try:
                        # Shift trigger data backward by lag days
                        trigger_lagged = data[trigger].shift(lag)
                        
                        # Calculate correlation with current symptom
                        valid_data = pd.DataFrame({
                            'symptom': data[symptom],
                            'trigger_lagged': trigger_lagged
                        }).dropna()
                        
                        if len(valid_data) > 10:
                            corr, p_value = stats.pearsonr(valid_data['symptom'], valid_data['trigger_lagged'])
                            
                            lagged_correlations[symptom][trigger][f'lag_{lag}d'] = {
                                'correlation': float(corr),
                                'p_value': float(p_value),
                                'significant': p_value < self.significance_threshold,
                                'sample_size': len(valid_data)
                            }
                    
                    except Exception as e:
                        lagged_correlations[symptom][trigger][f'lag_{lag}d'] = {
                            'error': str(e)
                        }
        
        return lagged_correlations
    
    def _identify_strong_correlations(self, correlations: Dict, significance: Dict) -> List[Dict]:
        """Identify and rank strong, significant correlations"""
        strong_correlations = []
        
        for symptom in correlations.get('pearson', {}):
            for trigger in correlations['pearson'][symptom]:
                corr_data = correlations['pearson'][symptom][trigger]
                sig_data = significance.get(symptom, {}).get(trigger, {})
                
                if ('correlation' in corr_data and 'significant' in corr_data and
                    corr_data['significant'] and abs(corr_data['correlation']) > 0.3):
                    
                    strong_correlations.append({
                        'symptom': symptom,
                        'trigger': trigger,
                        'correlation_strength': abs(corr_data['correlation']),
                        'correlation_direction': 'positive' if corr_data['correlation'] > 0 else 'negative',
                        'p_value': corr_data['p_value'],
                        'confidence_interval': sig_data.get('confidence_interval', [0, 0])
                    })
        
        # Sort by correlation strength
        strong_correlations.sort(key=lambda x: x['correlation_strength'], reverse=True)
        
        return strong_correlations[:20]  # Top 20 strongest correlations
    
    def _generate_correlation_insights(self, strong_correlations: List[Dict], 
                                     lagged_correlations: Dict) -> List[str]:
        """Generate personalized insights from correlation analysis"""
        insights = []
        
        if not strong_correlations:
            insights.append("No strong correlations found. More data may be needed for analysis.")
            return insights
        
        # Top correlation insight
        top_corr = strong_correlations[0]
        direction = "increases" if top_corr['correlation_direction'] == 'positive' else "decreases"
        insights.append(
            f"Your strongest trigger is {top_corr['trigger']}: when it {direction}, "
            f"your {top_corr['symptom']} tends to change significantly "
            f"(correlation: {top_corr['correlation_strength']:.2f})"
        )
        
        # Weather-related insights
        weather_correlations = [c for c in strong_correlations 
                              if any(w in c['trigger'].lower() for w in ['temp', 'humid', 'pressure', 'weather'])]
        
        if weather_correlations:
            weather_triggers = [c['trigger'] for c in weather_correlations[:3]]
            insights.append(f"Weather sensitivity detected: {', '.join(weather_triggers)} affect your symptoms")
        
        # Lifestyle factor insights
        lifestyle_correlations = [c for c in strong_correlations 
                                if any(l in c['trigger'].lower() for l in ['stress', 'sleep', 'medication'])]
        
        if lifestyle_correlations:
            lifestyle_factors = [c['trigger'] for c in lifestyle_correlations[:2]]
            insights.append(f"Key lifestyle factors: {', '.join(lifestyle_factors)} show strong correlation with symptoms")
        
        # Lagged correlation insights
        strongest_lagged = self._find_strongest_lagged_correlation(lagged_correlations)
        if strongest_lagged:
            insights.append(
                f"Delayed effect detected: {strongest_lagged['trigger']} from "
                f"{strongest_lagged['lag_days']} days ago affects current {strongest_lagged['symptom']}"
            )
        
        return insights
    
    def _find_strongest_lagged_correlation(self, lagged_correlations: Dict) -> Optional[Dict]:
        """Find the strongest time-lagged correlation"""
        strongest = None
        max_strength = 0
        
        for symptom in lagged_correlations:
            for trigger in lagged_correlations[symptom]:
                for lag_key, lag_data in lagged_correlations[symptom][trigger].items():
                    if 'correlation' in lag_data and 'significant' in lag_data:
                        if lag_data['significant'] and abs(lag_data['correlation']) > max_strength:
                            max_strength = abs(lag_data['correlation'])
                            strongest = {
                                'symptom': symptom,
                                'trigger': trigger,
                                'correlation': lag_data['correlation'],
                                'lag_days': int(lag_key.split('_')[1][:-1]),  # Extract number from 'lag_1d'
                                'p_value': lag_data['p_value']
                            }
        
        return strongest
    
    def _summarize_correlation_strength(self, strong_correlations: List[Dict]) -> Dict:
        """Summarize correlation strength distribution"""
        if not strong_correlations:
            return {'total': 0, 'strong': 0, 'moderate': 0, 'weak': 0}
        
        total = len(strong_correlations)
        strong = len([c for c in strong_correlations if c['correlation_strength'] > 0.7])
        moderate = len([c for c in strong_correlations if 0.5 <= c['correlation_strength'] <= 0.7])
        weak = len([c for c in strong_correlations if c['correlation_strength'] < 0.5])
        
        return {
            'total': total,
            'strong': strong,
            'moderate': moderate,
            'weak': weak
        }

# Example usage functions
def generate_dashboard_analytics(user_id: str, data: pd.DataFrame) -> Dict:
    """Generate comprehensive analytics for dashboard display"""
    
    # Initialize analytics engines
    risk_analyzer = PersonalRiskAnalytics()
    correlation_analyzer = CorrelationAnalysisEngine()
    
    # Generate analytics
    risk_analytics = risk_analyzer.calculate_risk_trends(data)
    correlation_analytics = correlation_analyzer.analyze_symptom_trigger_correlations(data)
    
    # Combine results
    dashboard_data = {
        'user_id': user_id,
        'generated_at': datetime.now().isoformat(),
        'data_period': {
            'start_date': data.index.min() if hasattr(data.index, 'min') else 'unknown',
            'end_date': data.index.max() if hasattr(data.index, 'max') else 'unknown',
            'total_days': len(data)
        },
        'risk_analytics': risk_analytics,
        'correlation_analytics': correlation_analytics,
        'summary_insights': _generate_summary_insights(risk_analytics, correlation_analytics)
    }
    
    return dashboard_data

def _generate_summary_insights(risk_analytics: Dict, correlation_analytics: Dict) -> List[str]:
    """Generate high-level summary insights for the dashboard"""
    insights = []
    
    # Risk trend insight
    if risk_analytics.get('trend_direction') == 'improving':
        insights.append("✅ Your overall risk trend is improving!")
    elif risk_analytics.get('trend_direction') == 'worsening':
        insights.append("⚠️ Your risk trend shows some concerning patterns")
    else:
        insights.append("📊 Your risk levels are stable")
    
    # Correlation insights
    if correlation_analytics.get('personalized_insights'):
        insights.extend(correlation_analytics['personalized_insights'][:2])
    
    # Seasonal insight
    seasonal = risk_analytics.get('seasonal_patterns', {})
    if seasonal.get('worst_months'):
        worst_month = seasonal['worst_months'][0]
        insights.append(f"🗓️ Historical data shows {worst_month} tends to be your most challenging month")
    
    return insights


if __name__ == "__main__":
    # Example usage
    print("RA Analytics Engine - Example Usage")
    print("="*50)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=180, freq='D')
    
    sample_data = pd.DataFrame({
        'date': dates,
        'pain_history_1d': np.random.normal(4, 2, 180).clip(1, 10),
        'temperature': np.random.normal(15, 10, 180),
        'humidity': np.random.normal(60, 20, 180).clip(0, 100),
        'pressure': np.random.normal(1013, 20, 180),
        'stress_level': np.random.normal(5, 2, 180).clip(1, 10),
        'sleep_quality': np.random.normal(6, 2, 180).clip(1, 10),
        'medication_adherence': np.random.beta(8, 2, 180),
        'month': dates.month,
        'flare_occurred': np.random.binomial(1, 0.15, 180)
    })
    
    # Generate analytics
    analytics_result = generate_dashboard_analytics('user_123', sample_data)
    
    # Display results
    print(f"Risk Score: {analytics_result['risk_analytics']['current_risk_score']:.2f}")
    print(f"Risk Level: {analytics_result['risk_analytics']['risk_level']}")
    print(f"Trend: {analytics_result['risk_analytics']['trend_direction']}")
    
    print("\nTop Insights:")
    for insight in analytics_result['summary_insights']:
        print(f"  {insight}")
    
    print(f"\nAnalytics generated for {analytics_result['data_period']['total_days']} days of data")