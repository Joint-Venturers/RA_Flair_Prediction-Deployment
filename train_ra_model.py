# train_ra_model.py
# RA Flare Prediction Model Training Script with 14 Features

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib
import json
from datetime import datetime
from supabase import create_client, Client
import os

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Initialize Supabase client
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Supabase client initialized")
else:
    print("⚠ Supabase credentials not found - will skip database logging")

# Define the 14 essential features (MUST MATCH TRAINING DATA COLUMNS)
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

# Feature categories for analytics
FEATURE_CATEGORIES = {
    'demographic': ['age', 'sex', 'disease_duration'],
    'lifestyle': ['sleep_hours', 'smoking_status', 'bmi'],
    'environmental': ['air_quality_index', 'min_temperature', 'max_temperature', 'humidity', 'change_in_barometric_pressure'],
    'clinical': ['current_pain_score', 'last_episode_duration', 'days_since_last_episode']
}

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'age': 'Patient age in years',
    'sex': 'Biological sex (0: Female, 1: Male)',
    'disease_duration': 'Years since RA diagnosis',
    'bmi': 'Body Mass Index (kg/m²)',
    'sleep_hours': 'Average sleep duration per night in hours',
    'smoking_status': 'Current smoking status (0: No, 1: Yes, 2: Quit)',
    'air_quality_index': 'Air Quality Index (0-500 scale)',
    'min_temperature': 'Minimum temperature in °C',
    'max_temperature': 'Maximum temperature in °C',
    'humidity': 'Relative humidity percentage',
    'change_in_barometric_pressure': '24-hour barometric pressure change in hPa',
    'current_pain_score': 'Self-reported pain level (0-10 scale)',
    'last_episode_duration': 'Duration of last flare episode in days',
    'days_since_last_episode': 'Days since last inflammation episode'
}

def load_data(filepath='training_data.csv'):
    """Load and prepare training data"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Verify all required columns exist
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if 'inflammation' not in df.columns:
        raise ValueError("Target column 'inflammation' not found in dataset")
    
    print(f"Loaded {len(df)} samples")
    print(f"Features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")
    print(f"\nTarget distribution:")
    print(df['inflammation'].value_counts())
    
    return df

def prepare_features(df):
    """Extract features and target"""
    X = df[FEATURE_COLUMNS].copy()
    y = df['inflammation'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    print(f"\nFeature statistics:")
    print(X.describe())
    
    return X, y

def train_model(X, y):
    """Train gradient boosting model"""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    print("\nSplitting data (80/20 train/test split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining Gradient Boosting model...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"AUC-ROC: {test_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=['No Flare', 'Flare']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': model.feature_importances_,
        'category': [next((cat for cat, features in FEATURE_CATEGORIES.items() if feat in features), 'other') 
                     for feat in FEATURE_COLUMNS]
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))
    
    return model, scaler, {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'f1_score': float(test_f1),
        'auc': float(test_auc),
        'feature_importance': feature_importance.to_dict('records'),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'total_samples': int(len(X)),
        'n_features': len(FEATURE_COLUMNS)
    }

def save_model(model, scaler, metrics):
    """Save model, scaler, and metadata"""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    # Save model and scaler
    joblib.dump(model, 'ra_flare_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("✓ Saved model to ra_flare_model.pkl")
    print("✓ Saved scaler to scaler.pkl")
    
    # Save metadata
    metadata = {
        'model_type': 'gradient_boosting',
        'n_features': len(FEATURE_COLUMNS),
        'feature_names': FEATURE_COLUMNS,
        'feature_categories': FEATURE_CATEGORIES,
        'feature_descriptions': FEATURE_DESCRIPTIONS,
        'training_date': datetime.now().isoformat(),
        'metrics': metrics,
        'sklearn_version': '1.3.0',
        'python_version': '3.11'
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✓ Saved metadata to model_metadata.json")
    
    return metadata

def log_to_supabase(metadata):
    """Log training run to Supabase"""
    if not supabase:
        print("\n⚠ Supabase not configured. Skipping database logging.")
        return
    
    try:
        print("\nLogging training run to Supabase...")
        
        training_record = {
            'timestamp': metadata['training_date'],
            'model_type': metadata['model_type'],
            'accuracy': metadata['metrics']['test_accuracy'],
            'f1_score': metadata['metrics']['f1_score'],
            'auc': metadata['metrics']['auc'],
            'feature_importances': json.dumps({
                item['feature']: item['importance'] 
                for item in metadata['metrics']['feature_importance']
            }),
            'training_samples': metadata['metrics']['train_samples'],
            'test_samples': metadata['metrics']['test_samples'],
            'total_samples': metadata['metrics']['total_samples'],
            'n_features': metadata['n_features'],
            'branch': os.getenv('GITHUB_REF_NAME', 'local')
        }
        
        result = supabase.table('model_training_history').insert(training_record).execute()
        print("✓ Successfully logged to Supabase")
        
    except Exception as e:
        print(f"⚠ Warning: Could not log to Supabase: {e}")

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("RA FLARE PREDICTION MODEL TRAINING")
    print("14 Features Version (with Episode History)")
    print("="*60)
    
    # Load data
    df = load_data('training_data.csv')
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Train model
    model, scaler, metrics = train_model(X, y)
    
    # Save model
    metadata = save_model(model, scaler, metrics)
    
    # Log to Supabase
    log_to_supabase(metadata)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['auc']:.4f}")
    print(f"Total Features: {len(FEATURE_COLUMNS)}")
    print("\nFiles saved:")
    print("  - ra_flare_model.pkl")
    print("  - scaler.pkl")
    print("  - model_metadata.json")
    print("\n🎯 Model ready for deployment!")

if __name__ == "__main__":
    main()