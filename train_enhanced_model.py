# train_enhanced_model.py
# Train enhanced RA flare prediction model with personalized trigger identification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import json
from datetime import datetime

def load_enhanced_data(file_path='ra_data_enhanced.csv'):
    """Load enhanced RA dataset"""
    print(f"📂 Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} samples")
    print(f"\n📊 Dataset Info:")
    print(f"  Features: {len(df.columns) - 1}")
    print(f"  Target: inflammation (0=No Flare, 1=Flare)")
    print(f"  Flare Rate: {df['inflammation'].mean()*100:.1f}%")
    return df

def prepare_features(df):
    """
    Prepare features and target for training
    
    Features (16 total):
    - Demographics: age, sex, disease_duration, bmi
    - Lifestyle: sleep_hours, smoking_status
    - Environmental: air_quality_index, min_temperature, max_temperature, 
                     humidity, barometric_pressure, precipitation, wind_speed
    - Current State: current_pain_score, tender_joint_count, swollen_joint_count
    """
    
    feature_columns = [
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
        'barometric_pressure',
        'precipitation',
        'wind_speed',
        'current_pain_score',
        'tender_joint_count',
        'swollen_joint_count'
    ]
    
    X = df[feature_columns]
    y = df['inflammation']
    
    print(f"\n🔧 Feature Engineering:")
    print(f"  Total Features: {len(feature_columns)}")
    print(f"  Feature List: {', '.join(feature_columns)}")
    
    return X, y, feature_columns

def train_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train multiple models and select the best one
    
    Models:
    1. Gradient Boosting (best for feature importance)
    2. Random Forest (robust, good for interactions)
    3. Logistic Regression (interpretable baseline)
    """
    
    print(f"\n🤖 Training Models...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    models = {
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n📊 Performance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        print(f"\n📋 Confusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        
        # Feature Importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            
            print(f"\n🎯 Top 10 Feature Importances:")
            for i, (feature, importance) in enumerate(feature_importance[:10], 1):
                print(f"  {i:2d}. {feature:25s} {importance:.4f}")
        
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist()
        }
    
    return results

def save_best_model(results, scaler, feature_names):
    """Save the best performing model and metadata"""
    
    # Select best model based on F1 score (balanced metric)
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_model_data = results[best_model_name]
    
    print(f"\n{'='*60}")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"{'='*60}")
    print(f"  F1 Score: {best_model_data['f1']:.4f}")
    print(f"  Accuracy: {best_model_data['accuracy']:.4f}")
    print(f"  AUC:      {best_model_data['auc']:.4f}")
    
    # Save model
    model_filename = 'ra_model_gradient_boosting.pkl'
    joblib.dump(best_model_data['model'], model_filename)
    print(f"\n💾 Saved model: {model_filename}")
    
    # Save scaler
    scaler_filename = 'ra_scaler_gradient_boosting.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"💾 Saved scaler: {scaler_filename}")
    
    # Save feature names
    features_filename = 'ra_features_gradient_boosting.pkl'
    joblib.dump(feature_names, features_filename)
    print(f"💾 Saved features: {features_filename}")
    
    # Save model type
    model_type_filename = 'ra_model_type.pkl'
    joblib.dump(best_model_name.lower().replace(' ', '_'), model_type_filename)
    print(f"💾 Saved model type: {model_type_filename}")
    
    # Save training metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': best_model_name.lower().replace(' ', '_'),
        'accuracy': float(best_model_data['accuracy']),
        'precision': float(best_model_data['precision']),
        'recall': float(best_model_data['recall']),
        'f1': float(best_model_data['f1']),
        'auc': float(best_model_data['auc']),
        'features': feature_names,
        'n_features': len(feature_names),
        'confusion_matrix': best_model_data['confusion_matrix']
    }
    
    metadata_filename = 'training_metadata.json'
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata: {metadata_filename}")
    
    return best_model_name, best_model_data

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("🏥 RA FLARE PREDICTION - ENHANCED MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = load_enhanced_data('ra_data_enhanced.csv')
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Split data
    print(f"\n📊 Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print(f"🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
    
    # Save best model
    best_model_name, best_model_data = save_best_model(results, scaler, feature_names)
    
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n🎯 Model Performance Summary:")
    print(f"  Best Model: {best_model_name}")
    print(f"  Accuracy:  {best_model_data['accuracy']:.4f}")
    print(f"  F1 Score:  {best_model_data['f1']:.4f}")
    print(f"  AUC:       {best_model_data['auc']:.4f}")
    print(f"\n📦 Files Created:")
    print(f"  - ra_model_gradient_boosting.pkl")
    print(f"  - ra_scaler_gradient_boosting.pkl")
    print(f"  - ra_features_gradient_boosting.pkl")
    print(f"  - ra_model_type.pkl")
    print(f"  - training_metadata.json")
    print(f"\n🚀 Model ready for deployment!")

if __name__ == "__main__":
    main()
