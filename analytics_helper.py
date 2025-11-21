# analytics_helper.py
# Analytics utility functions for RA Flare Prediction API

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import joblib
from collections import defaultdict, Counter

def load_training_metadata() -> Dict:
    """Load latest training metadata from JSON file"""
    try:
        if os.path.exists('training_metadata.json'):
            with open('training_metadata.json', 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return {}

def get_feature_importance() -> Dict:
    """
    Get current model feature importances
    Returns structured data with rankings and categories
    """
    try:
        # Load model
        model = joblib.load('ra_model_gradient_boosting.pkl')
        features = joblib.load('ra_features_gradient_boosting.pkl')
        
        # Get importances
        importances = model.feature_importances_
        
        # Feature categories
        categories = {
            'age': 'demographic',
            'sex': 'demographic',
            'disease_duration': 'clinical',
            'bmi': 'demographic',
            'sleep_hours': 'lifestyle',
            'smoking_status': 'lifestyle',
            'air_quality_index': 'environmental',
            'min_temperature': 'environmental',
            'max_temperature': 'environmental',
            'humidity': 'environmental',
            'barometric_pressure': 'environmental',
            'precipitation': 'environmental',
            'wind_speed': 'environmental',
            'current_pain_score': 'clinical',
            'tender_joint_count': 'clinical',
            'swollen_joint_count': 'clinical'
        }
        
        # Feature descriptions
        descriptions = {
            'age': 'Patient age in years',
            'sex': '0=female, 1=male',
            'disease_duration': 'Years since RA diagnosis',
            'bmi': 'Body Mass Index',
            'sleep_hours': 'Hours of sleep per night',
            'smoking_status': '0=never, 1=former, 2=current',
            'air_quality_index': 'Air Quality Index (0-500)',
            'min_temperature': 'Daily minimum temperature (°C)',
            'max_temperature': 'Daily maximum temperature (°C)',
            'humidity': 'Relative humidity (%)',
            'barometric_pressure': 'Atmospheric pressure (hPa)',
            'precipitation': 'Daily rainfall (mm)',
            'wind_speed': 'Wind speed (km/h)',
            'current_pain_score': 'Current pain level (0-10)',
            'tender_joint_count': 'Number of tender joints',
            'swollen_joint_count': 'Number of swollen joints'
        }
        
        # Create structured response
        feature_data = []
        for i, (feature, importance) in enumerate(zip(features, importances)):
            feature_data.append({
                'name': feature,
                'importance': float(importance),
                'rank': i + 1,
                'category': categories.get(feature, 'other'),
                'description': descriptions.get(feature, 'No description')
            })
        
        # Sort by importance
        feature_data.sort(key=lambda x: x['importance'], reverse=True)
        
        # Update ranks
        for i, item in enumerate(feature_data):
            item['rank'] = i + 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(features),
            'features': feature_data,
            'top_5': feature_data[:5],
            'by_category': _group_by_category(feature_data)
        }
        
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def _group_by_category(feature_data: List[Dict]) -> Dict:
    """Group features by category"""
    by_category = defaultdict(list)
    for feature in feature_data:
        by_category[feature['category']].append({
            'name': feature['name'],
            'importance': feature['importance'],
            'rank': feature['rank']
        })
    
    # Calculate category totals
    category_totals = {}
    for category, features in by_category.items():
        total_importance = sum(f['importance'] for f in features)
        category_totals[category] = {
            'features': features,
            'total_importance': round(total_importance, 4),
            'count': len(features)
        }
    
    return category_totals

def calculate_trigger_statistics(predictions: List[Dict]) -> Dict:
    """
    Calculate trigger frequency and statistics from predictions
    """
    trigger_counts = defaultdict(int)
    trigger_severity = defaultdict(lambda: {'high': 0, 'moderate': 0, 'low': 0})
    trigger_impact_scores = defaultdict(list)
    
    for pred in predictions:
        triggers = pred.get('triggers', [])
        for trigger in triggers:
            trigger_name = trigger.get('trigger', '')
            severity = trigger.get('severity', 'low')
            impact = trigger.get('impact_score', 0.0)
            
            trigger_counts[trigger_name] += 1
            trigger_severity[trigger_name][severity] += 1
            trigger_impact_scores[trigger_name].append(impact)
    
    # Calculate statistics
    total_predictions = len(predictions)
    trigger_stats = {}
    
    for trigger_name, count in trigger_counts.items():
        trigger_stats[trigger_name] = {
            'count': count,
            'percentage': round((count / total_predictions) * 100, 1) if total_predictions > 0 else 0,
            'severity_distribution': dict(trigger_severity[trigger_name]),
            'average_impact_score': round(
                sum(trigger_impact_scores[trigger_name]) / len(trigger_impact_scores[trigger_name]),
                3
            ) if trigger_impact_scores[trigger_name] else 0
        }
    
    # Sort by frequency
    sorted_triggers = dict(sorted(trigger_stats.items(), key=lambda x: x[1]['count'], reverse=True))
    
    return {
        'total_predictions': total_predictions,
        'unique_triggers': len(trigger_counts),
        'trigger_counts': sorted_triggers,
        'timestamp': datetime.now().isoformat()
    }

def analyze_trigger_combinations(predictions: List[Dict]) -> Dict:
    """
    Analyze common trigger combinations
    """
    combinations = []
    combination_counter = Counter()
    
    for pred in predictions:
        triggers = pred.get('triggers', [])
        if len(triggers) >= 2:
            trigger_names = tuple(sorted([t['trigger'] for t in triggers]))
            combination_counter[trigger_names] += 1
            
            # Track flare outcome
            is_flare = pred.get('prediction', 0) == 1
            probability = pred.get('probability', 0.0)
            
            combinations.append({
                'triggers': list(trigger_names),
                'is_flare': is_flare,
                'probability': probability
            })
    
    # Calculate statistics for common combinations
    common_combinations = []
    for trigger_combo, count in combination_counter.most_common(10):
        combo_predictions = [
            c for c in combinations 
            if tuple(sorted(c['triggers'])) == trigger_combo
        ]
        
        flare_count = sum(1 for c in combo_predictions if c['is_flare'])
        avg_probability = sum(c['probability'] for c in combo_predictions) / len(combo_predictions)
        
        common_combinations.append({
            'triggers': list(trigger_combo),
            'frequency': count,
            'flare_rate': round(flare_count / count, 2) if count > 0 else 0,
            'average_probability': round(avg_probability, 2)
        })
    
    return {
        'total_combinations': len(combination_counter),
        'common_combinations': common_combinations,
        'timestamp': datetime.now().isoformat()
    }

def calculate_trigger_impact(predictions: List[Dict]) -> Dict:
    """
    Calculate individual trigger impact on flare predictions
    """
    trigger_present = defaultdict(lambda: {'flares': 0, 'total': 0})
    trigger_absent = defaultdict(lambda: {'flares': 0, 'total': 0})
    
    # Get all unique triggers
    all_triggers = set()
    for pred in predictions:
        for trigger in pred.get('triggers', []):
            all_triggers.add(trigger['trigger'])
    
    # Calculate presence/absence statistics
    for pred in predictions:
        is_flare = pred.get('prediction', 0) == 1
        present_triggers = set(t['trigger'] for t in pred.get('triggers', []))
        
        for trigger_name in all_triggers:
            if trigger_name in present_triggers:
                trigger_present[trigger_name]['total'] += 1
                if is_flare:
                    trigger_present[trigger_name]['flares'] += 1
            else:
                trigger_absent[trigger_name]['total'] += 1
                if is_flare:
                    trigger_absent[trigger_name]['flares'] += 1
    
    # Calculate impact statistics
    trigger_impacts = []
    for trigger_name in all_triggers:
        present_data = trigger_present[trigger_name]
        absent_data = trigger_absent[trigger_name]
        
        flare_rate_present = (
            present_data['flares'] / present_data['total']
            if present_data['total'] > 0 else 0
        )
        flare_rate_absent = (
            absent_data['flares'] / absent_data['total']
            if absent_data['total'] > 0 else 0
        )
        
        relative_risk = (
            flare_rate_present / flare_rate_absent
            if flare_rate_absent > 0 else 0
        )
        
        # Categorize impact
        if relative_risk >= 2.0:
            impact_category = 'Major Trigger'
        elif relative_risk >= 1.5:
            impact_category = 'Moderate Trigger'
        elif relative_risk >= 1.2:
            impact_category = 'Minor Trigger'
        else:
            impact_category = 'Low Impact'
        
        trigger_impacts.append({
            'name': trigger_name,
            'total_occurrences': present_data['total'],
            'flare_rate_when_present': round(flare_rate_present, 2),
            'flare_rate_when_absent': round(flare_rate_absent, 2),
            'relative_risk': round(relative_risk, 2),
            'impact_category': impact_category
        })
    
    # Sort by relative risk
    trigger_impacts.sort(key=lambda x: x['relative_risk'], reverse=True)
    
    return {
        'triggers': trigger_impacts,
        'timestamp': datetime.now().isoformat()
    }

def get_model_insights() -> Dict:
    """
    Combine model performance with feature importance for clinical insights
    """
    metadata = load_training_metadata()
    feature_data = get_feature_importance()
    
    if not metadata or 'error' in feature_data:
        return {'error': 'Unable to load model data'}
    
    # Top predictive features with clinical significance
    top_features = []
    for feature in feature_data.get('features', [])[:5]:
        clinical_significance = _get_clinical_significance(feature['name'])
        trigger_info = _get_trigger_info(feature['name'])
        
        top_features.append({
            'feature': feature['name'],
            'importance': feature['importance'],
            'category': feature['category'],
            'trigger_name': trigger_info['name'],
            'activation_threshold': trigger_info['threshold'],
            'clinical_significance': clinical_significance
        })
    
    return {
        'model_performance': {
            'accuracy': metadata.get('accuracy', 0),
            'auc': metadata.get('auc', 0),
            'f1_score': metadata.get('f1', 0),
            'training_date': metadata.get('timestamp', '')
        },
        'top_predictive_features': top_features,
        'model_type': metadata.get('model_type', 'unknown'),
        'total_features': metadata.get('n_features', 0)
    }

def _get_clinical_significance(feature_name: str) -> str:
    """Get clinical significance description for a feature"""
    significance = {
        'sleep_hours': 'Major modifiable risk factor - sleep deprivation increases inflammation',
        'current_pain_score': 'Direct inflammation indicator - early warning sign',
        'bmi': 'Obesity linked to increased inflammation and joint stress',
        'disease_duration': 'Longer disease duration associated with chronic inflammation',
        'swollen_joint_count': 'Direct measure of active inflammation',
        'tender_joint_count': 'Indicates joint inflammation and damage',
        'humidity': 'High humidity may affect joint fluid and inflammation',
        'barometric_pressure': 'Pressure changes may trigger joint pain and inflammation',
        'air_quality_index': 'Poor air quality increases systemic inflammation'
    }
    return significance.get(feature_name, 'Contributing factor to flare risk')

def _get_trigger_info(feature_name: str) -> Dict:
    """Get trigger information for a feature"""
    trigger_mapping = {
        'sleep_hours': {'name': 'Poor Sleep', 'threshold': '< 6 hours'},
        'current_pain_score': {'name': 'Elevated Pain Score', 'threshold': '> 6/10'},
        'bmi': {'name': 'Elevated BMI', 'threshold': '> 30'},
        'humidity': {'name': 'High Humidity', 'threshold': '> 75%'},
        'barometric_pressure': {'name': 'Low Barometric Pressure', 'threshold': '< 1000 hPa'},
        'air_quality_index': {'name': 'Poor Air Quality', 'threshold': '> 100'},
        'min_temperature': {'name': 'Cold Weather', 'threshold': '< 10°C'}
    }
    return trigger_mapping.get(feature_name, {'name': 'Unknown', 'threshold': 'N/A'})
