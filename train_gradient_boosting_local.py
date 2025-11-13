# Local Training Script - Gradient Boosting
# Train RA Inflammation model using Gradient Boosting locally (FREE)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, 
                             confusion_matrix)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("LOCAL TRAINING - GRADIENT BOOSTING MODEL")
print("=" * 70)
print()

# Load data
print("📂 Loading training data...")
df = pd.read_csv('ra_data_simple.csv')
print(f"✅ Loaded {len(df)} samples")
print()

# Define features (12 raw features)
FEATURES = [
    'age',
    'gender',
    'disease_duration',
    'bmi',
    'min_temperature',
    'max_temperature',
    'humidity',
    'barometric_pressure',
    'precipitation',
    'wind_speed',
    'tender_joint_count',
    'swollen_joint_count'
]

TARGET = 'inflammation'

print("📊 Features (12 raw features):")
for i, feat in enumerate(FEATURES, 1):
    print(f"   {i:2d}. {feat}")
print(f"   Target: {TARGET} (0=Remission, 1=Inflammation)")
print()

# Prepare data
X = df[FEATURES]
y = df[TARGET]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data split:")
print(f"   Training: {len(X_train)} samples ({(y_train==1).mean()*100:.1f}% inflammation)")
print(f"   Testing: {len(X_test)} samples ({(y_test==1).mean()*100:.1f}% inflammation)")
print()

# Scale features
print("⚙️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled")
print()

# Train Gradient Boosting model
print("=" * 70)
print("🚀 TRAINING GRADIENT BOOSTING MODEL")
print("=" * 70)
print()

print("⚙️ Hyperparameters:")
hyperparameters = {
    'n_estimators': 100,        # Number of boosting stages
    'learning_rate': 0.1,       # Learning rate
    'max_depth': 5,             # Maximum tree depth
    'min_samples_split': 5,     # Min samples to split
    'min_samples_leaf': 2,      # Min samples per leaf
    'subsample': 0.8,           # Row sampling
    'max_features': 'sqrt',     # Column sampling
    'random_state': 42
}

for key, value in hyperparameters.items():
    print(f"   {key}: {value}")
print()

print("🔨 Training model...")
model = GradientBoostingClassifier(**hyperparameters)
model.fit(X_train_scaled, y_train)
print("✅ Training complete!")
print()

# Evaluate model
print("=" * 70)
print("📊 MODEL EVALUATION")
print("=" * 70)
print()

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("📈 Performance Metrics:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print(f"   ROC AUC:   {auc:.4f}")
print()

# Classification report
print("📋 Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Remission', 'Inflammation']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                Predicted")
print(f"              Remission  Inflammation")
print(f"Actual Remission      {cm[0,0]:4d}      {cm[0,1]:4d}")
print(f"       Inflammation   {cm[1,0]:4d}      {cm[1,1]:4d}")
print()

# Feature importance
print("🔍 Top 5 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(5).iterrows():
    print(f"   {row['feature']:25s}: {row['importance']:.4f}")
print()

# Save model
print("=" * 70)
print("💾 SAVING MODEL")
print("=" * 70)
print()

print("💾 Saving model artifacts...")
joblib.dump(model, 'ra_model_gradient_boosting.pkl')
print("✅ Saved: ra_model_gradient_boosting.pkl")

joblib.dump(scaler, 'ra_scaler_gradient_boosting.pkl')
print("✅ Saved: ra_scaler_gradient_boosting.pkl")

joblib.dump(FEATURES, 'ra_features_gradient_boosting.pkl')
print("✅ Saved: ra_features_gradient_boosting.pkl")

joblib.dump('gradient_boosting', 'ra_model_type.pkl')
print("✅ Saved: ra_model_type.pkl")
print()

# Test sample prediction
print("=" * 70)
print("🧪 TESTING SAMPLE PREDICTION")
print("=" * 70)
print()

test_sample = X_test.iloc[0].values.reshape(1, -1)
test_sample_scaled = scaler.transform(test_sample)

probability = model.predict_proba(test_sample_scaled)[0][1]
prediction = int(probability >= 0.5)
risk_level = 'HIGH' if probability > 0.6 else 'MODERATE' if probability > 0.3 else 'LOW'

print("📝 Sample Input:")
for feat, val in zip(FEATURES, test_sample[0]):
    print(f"   {feat:25s}: {val}")
print()

print("📊 Prediction:")
print(f"   Inflammation probability: {probability:.4f} ({probability*100:.2f}%)")
print(f"   Prediction: {'Inflammation' if prediction == 1 else 'Remission'}")
print(f"   Risk level: {risk_level}")
print(f"   Confidence: {max(probability, 1-probability):.4f}")
print()

print("=" * 70)
print("✅ TRAINING COMPLETE")
print("=" * 70)
print()
print("📋 Summary:")
print(f"   Algorithm: Gradient Boosting")
print(f"   Accuracy: {accuracy*100:.2f}%")
print(f"   AUC: {auc:.4f}")
print(f"   Features: {len(FEATURES)}")
print()
print("📦 Files created:")
print("   - ra_model_gradient_boosting.pkl")
print("   - ra_scaler_gradient_boosting.pkl")
print("   - ra_features_gradient_boosting.pkl")
print("   - ra_model_type.pkl")
print()
print("🚀 Next steps:")
print("   1. Test locally: python test_local_api.py")
print("   2. Deploy to Render: Follow RENDER_DEPLOYMENT.md")
print()
print("💰 Cost: $0 (completely FREE!)")