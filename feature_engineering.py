# feature_engineering.py - Advanced Feature Engineering for RA Flare Prediction

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Enhanced feature engineering for better RA flare prediction"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_history = {}
        
    def create_weather_features(self, df):
        """Enhanced weather-based features"""
        print("Creating advanced weather features...")
        
        # Weather volatility (captures instability)
        df['weather_volatility_3d'] = df['temperature'].rolling(3, min_periods=1).std().fillna(0)
        df['pressure_volatility_7d'] = df['pressure'].rolling(7, min_periods=1).std().fillna(0)
        df['humidity_volatility_5d'] = df['humidity'].rolling(5, min_periods=1).std().fillna(0)
        
        # Weather comfort index (optimal conditions score)
        df['comfort_index'] = (
            (df['temperature'].between(18, 24)).astype(int) * 0.4 +
            (df['humidity'].between(40, 60)).astype(int) * 0.3 +
            (df['pressure'] > 1010).astype(int) * 0.3
        )
        
        # Rapid weather changes (more sensitive thresholds)
        df['rapid_temp_drop'] = (df['temp_change_24h'] < -8).astype(int)
        df['rapid_pressure_drop'] = (df['pressure_change_24h'] < -15).astype(int)
        df['rapid_humidity_rise'] = (df['humidity_change_24h'] > 20).astype(int)
        
        # Weather pattern combinations (interaction features)
        df['cold_humid_combo'] = ((df['temperature'] < 10) & (df['humidity'] > 70)).astype(int)
        df['pressure_temp_interaction'] = df['pressure'] * df['temperature'] / 1000
        
        # Seasonal adjustments
        if 'month' not in df.columns:
            df['month'] = pd.to_datetime(df.index if df.index.dtype == 'datetime64[ns]' else range(len(df))).month
        
        df['seasonal_temp_anomaly'] = df.groupby('month')['temperature'].transform(
            lambda x: x - x.mean()
        )
        
        # Weather severity score
        df['weather_severity'] = (
            (df['temperature'] < 5).astype(int) * 2 +
            (df['humidity'] > 80).astype(int) * 2 +
            (df['pressure'] < 1000).astype(int) * 2 +
            (df['rapid_temp_drop']) * 1 +
            (df['rapid_pressure_drop']) * 1
        )
        
        return df
    
    def create_temporal_features(self, df):
        """Advanced time-based features"""
        print("Creating temporal features...")
        
        # Enhanced circadian features
        df['is_morning'] = (df['hour_of_day'].between(6, 12)).astype(int)
        df['is_afternoon'] = (df['hour_of_day'].between(12, 18)).astype(int)
        df['is_evening'] = (df['hour_of_day'].between(18, 22)).astype(int)
        df['is_night'] = (df['hour_of_day'].between(22, 24) | df['hour_of_day'].between(0, 6)).astype(int)
        
        # Weekend and workday effects
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)  # Monday effect
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)  # Friday effect
        
        # Seasonal patterns
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,    # Winter
            3: 1, 4: 1, 5: 1,     # Spring  
            6: 2, 7: 2, 8: 2,     # Summer
            9: 3, 10: 3, 11: 3    # Fall
        })
        
        # Monthly patterns (RA often has monthly cycles)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time since last flare (requires flare history)
        if 'flare_occurred' in df.columns:
            df['days_since_last_flare'] = 0
            for i in range(1, len(df)):
                if df.iloc[i-1]['flare_occurred'] == 1:
                    df.iloc[i:, df.columns.get_loc('days_since_last_flare')] = 0
                else:
                    df.iloc[i, df.columns.get_loc('days_since_last_flare')] = \
                        df.iloc[i-1]['days_since_last_flare'] + 1
        
        return df
    
    def create_personal_risk_features(self, df):
        """Personalized patient-specific features"""
        print("Creating personalized risk features...")
        
        # Personal pain baselines (rolling statistics)
        df['pain_baseline_7d'] = df['pain_history_1d'].rolling(7, min_periods=1).mean()
        df['pain_baseline_30d'] = df['pain_history_1d'].rolling(30, min_periods=1).mean()
        
        # Pain above personal baseline
        df['pain_above_baseline_7d'] = (df['pain_history_1d'] > df['pain_baseline_7d']).astype(int)
        df['pain_above_baseline_30d'] = (df['pain_history_1d'] > df['pain_baseline_30d']).astype(int)
        
        # Pain trend analysis
        df['pain_trend_3d'] = df['pain_history_1d'].rolling(3, min_periods=1).apply(
            lambda x: 1 if len(x) >= 2 and x.iloc[-1] > x.iloc[0] else 0
        )
        
        # Stress-pain interaction
        df['stress_pain_interaction'] = df['stress_level'] * df['pain_history_1d']
        df['stress_above_threshold'] = (df['stress_level'] > 7).astype(int)
        
        # Sleep debt calculation
        df['sleep_debt'] = np.maximum(0, 8 - df['sleep_quality'])
        df['chronic_sleep_issues'] = (df['sleep_quality'] < 4).astype(int)
        
        # Medication effectiveness indicators
        df['med_adherence_category'] = pd.cut(df['medication_adherence'], 
                                            bins=[0, 0.6, 0.8, 1.0], 
                                            labels=['poor', 'fair', 'good'])
        
        # Convert categorical to numeric
        if 'med_adherence_category' in df.columns:
            le = LabelEncoder()
            df['med_adherence_encoded'] = le.fit_transform(df['med_adherence_category'].astype(str))
            self.encoders['med_adherence'] = le
        
        # Patient risk profile score
        df['patient_risk_score'] = (
            (df['disease_duration'] > 10).astype(int) * 2 +
            (df['age'] > 60).astype(int) * 1 +
            (df['medication_adherence'] < 0.7).astype(int) * 2 +
            (df['stress_level'] > 7).astype(int) * 1 +
            (df['sleep_quality'] < 4).astype(int) * 1
        )
        
        return df
    
    def create_interaction_features(self, df):
        """Feature interactions and combinations"""
        print("Creating interaction features...")
        
        # Weather-patient interactions
        df['age_weather_sensitivity'] = df['age'] * df['weather_severity'] / 100
        df['disease_duration_humidity'] = df['disease_duration'] * df['humidity'] / 100
        
        # Pain-weather correlations
        df['pain_pressure_interaction'] = df['pain_history_1d'] * (1020 - df['pressure']) / 100
        df['pain_temp_interaction'] = df['pain_history_1d'] * np.maximum(0, 20 - df['temperature']) / 10
        
        # Medication-lifestyle interactions
        df['med_sleep_interaction'] = df['medication_adherence'] * df['sleep_quality']
        df['stress_sleep_deficit'] = df['stress_level'] * df['sleep_debt']
        
        # Compound risk factors
        df['triple_risk'] = (
            (df['weather_severity'] > 3).astype(int) +
            (df['pain_above_baseline_7d']).astype(int) +
            (df['patient_risk_score'] > 3).astype(int)
        )
        
        return df
    
    def create_rolling_statistics(self, df):
        """Rolling window statistics for trend detection"""
        print("Creating rolling statistics...")
        
        # Multi-window pain statistics
        for window in [3, 7, 14, 30]:
            if len(df) >= window:
                df[f'pain_rolling_mean_{window}d'] = df['pain_history_1d'].rolling(window, min_periods=1).mean()
                df[f'pain_rolling_std_{window}d'] = df['pain_history_1d'].rolling(window, min_periods=1).std().fillna(0)
                df[f'pain_rolling_max_{window}d'] = df['pain_history_1d'].rolling(window, min_periods=1).max()
                
                # Weather volatility across different windows
                df[f'weather_volatility_{window}d'] = df[['temperature', 'humidity', 'pressure']].rolling(window, min_periods=1).std().mean(axis=1).fillna(0)
        
        # Pain pattern indicators - safe for single row processing
        try:
            pain_7d = df.get('pain_rolling_mean_7d', df['pain_history_1d'])
            pain_14d = df.get('pain_rolling_mean_14d', df['pain_history_1d'])
            
            if isinstance(pain_7d, pd.Series) and isinstance(pain_14d, pd.Series):
                df['pain_increasing_trend'] = (pain_7d > pain_14d).fillna(0).astype(int)
            else:
                # Handle single value case
                df['pain_increasing_trend'] = int(pain_7d.iloc[0] > pain_14d.iloc[0]) if hasattr(pain_7d, 'iloc') else int(pain_7d > pain_14d)
            
            std_7d = df.get('pain_rolling_std_7d', 0)
            std_14d = df.get('pain_rolling_std_14d', 0)
            
            if isinstance(std_7d, pd.Series) and isinstance(std_14d, pd.Series):
                df['pain_variability_high'] = (std_7d > std_14d).fillna(0).astype(int)
            else:
                # Handle single value case
                df['pain_variability_high'] = int(std_7d.iloc[0] > std_14d.iloc[0]) if hasattr(std_7d, 'iloc') else int(std_7d > std_14d)

        except Exception as e:
            # Fallback for any edge cases
            df['pain_increasing_trend'] = 0
            df['pain_variability_high'] = 0
        return df
    
    def transform_dataset(self, df):
        """Apply all feature engineering transformations"""
        print("Starting comprehensive feature engineering...")
        print(f"Initial dataset shape: {df.shape}")
        
        # Ensure required base columns exist
        required_cols = ['temperature', 'humidity', 'pressure', 'pain_history_1d', 
                        'stress_level', 'sleep_quality', 'medication_adherence',
                        'age', 'disease_duration', 'hour_of_day', 'day_of_week', 'month']
        
        for col in required_cols:
            if col not in df.columns:
                if col in ['hour_of_day', 'day_of_week', 'month']:
                    # Generate time features if missing
                    now = datetime.now()
                    if col == 'hour_of_day':
                        df[col] = now.hour
                    elif col == 'day_of_week':
                        df[col] = now.weekday()
                    else:  # month
                        df[col] = now.month
                else:
                    # Fill with median values for missing features
                    df[col] = np.random.normal(0, 1, len(df))
                    print(f"Warning: Generated synthetic data for missing column: {col}")
        
        # Apply feature engineering steps
        df = self.create_weather_features(df)
        df = self.create_temporal_features(df)
        df = self.create_personal_risk_features(df)
        df = self.create_interaction_features(df)
        df = self.create_rolling_statistics(df)
        
        # FIXED: Handle NaN values properly for different column types
        for col in df.columns:
            if df[col].dtype.name == 'category':
                # For categorical columns, fill with mode or drop the category
                if col == 'med_adherence_category' and col in df.columns:
                    df = df.drop(columns=[col])  # Drop problematic categorical column
                    print(f"Dropped categorical column: {col}")
            else:
                # For numeric columns, fill with 0
                df[col] = df[col].fillna(0)
        
        # Remove any remaining categorical columns that might cause issues
        categorical_cols = df.select_dtypes(include=['category']).columns
        if len(categorical_cols) > 0:
            print(f"Removing categorical columns: {list(categorical_cols)}")
            df = df.drop(columns=categorical_cols)
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Added {df.shape[1] - len(required_cols)} new features")
        
        return df
    
    def get_feature_importance_groups(self):
        """Return feature groups for analysis"""
        return {
            'weather_basic': ['temperature', 'humidity', 'pressure', 'weather_condition_encoded'],
            'weather_advanced': ['weather_volatility_3d', 'pressure_volatility_7d', 'comfort_index', 
                                'weather_severity', 'rapid_temp_drop', 'rapid_pressure_drop'],
            'temporal': ['hour_of_day', 'day_of_week', 'month', 'season', 'is_weekend', 'is_morning'],
            'pain_patterns': ['pain_history_1d', 'pain_baseline_7d', 'pain_above_baseline_7d', 
                            'pain_trend_3d', 'pain_rolling_mean_7d'],
            'patient_factors': ['age', 'disease_duration', 'medication_adherence', 'patient_risk_score'],
            'lifestyle': ['stress_level', 'sleep_quality', 'sleep_debt', 'stress_pain_interaction'],
            'interactions': ['age_weather_sensitivity', 'pain_pressure_interaction', 'triple_risk']
        }


# Example usage
if __name__ == "__main__":
    # Load your existing training data
    df = pd.read_csv('training_data.csv')
    
    # Initialize feature engineer
    fe = AdvancedFeatureEngineer()
    
    # Transform dataset
    enhanced_df = fe.transform_dataset(df)
    
    # Save enhanced dataset
    enhanced_df.to_csv('enhanced_training_data.csv', index=False)
    
    # Display feature groups
    feature_groups = fe.get_feature_importance_groups()
    for group, features in feature_groups.items():
        print(f"\n{group.upper()}: {len(features)} features")
        print(f"  {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")