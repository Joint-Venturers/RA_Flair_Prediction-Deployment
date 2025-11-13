# Step 3 FINAL: Train Models - Combined vs Separate (RAW FEATURES ONLY)
# NO engineered features - just the 12 raw features you specified

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, confusion_matrix)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🤖 RA INFLAMMATION - COMBINED vs SEPARATE (RAW FEATURES ONLY)")
print("=" * 70)
print("")

# Load data
df = pd.read_csv('ra_data_simple.csv')
print(f"✅ Loaded {df.shape[0]} samples")
print("")

# Define feature sets - ONLY the raw features you specified
BASE_FEATURES = [
    'age',
    'gender',
    'disease_duration',
    'bmi',
    'min_temperature',
    'max_temperature',
    'humidity',
    'barometric_pressure',
    'precipitation',
    'wind_speed'
]

# MODEL A: Using COMBINED joint count
COMBINED_FEATURES = BASE_FEATURES + ['combined_joint_count']

# MODEL B: Using SEPARATE joint counts
SEPARATE_FEATURES = BASE_FEATURES + ['tender_joint_count', 'swollen_joint_count']

TARGET = 'inflammation'

print("📊 Feature Sets (RAW FEATURES ONLY):")
print(f"   Model A (COMBINED): {len(COMBINED_FEATURES)} features")
print(f"      {COMBINED_FEATURES}")
print("")
print(f"   Model B (SEPARATE): {len(SEPARATE_FEATURES)} features")
print(f"      {SEPARATE_FEATURES}")
print("")

# Prepare data
X_combined = df[COMBINED_FEATURES]
X_separate = df[SEPARATE_FEATURES]
y = df[TARGET]

# Split data
X_comb_train, X_comb_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

X_sep_train, X_sep_test = train_test_split(
    X_separate, test_size=0.2, random_state=42, stratify=y
)[0:2]

print("✅ Data split:")
print(f"   Training: {len(y_train)} samples ({(y_train==1).mean()*100:.1f}% inflammation)")
print(f"   Testing: {len(y_test)} samples ({(y_test==1).mean()*100:.1f}% inflammation)")
print("")

# Scale features
scaler_comb = StandardScaler()
X_comb_train_scaled = scaler_comb.fit_transform(X_comb_train)
X_comb_test_scaled = scaler_comb.transform(X_comb_test)

scaler_sep = StandardScaler()
X_sep_train_scaled = scaler_sep.fit_transform(X_sep_train)
X_sep_test_scaled = scaler_sep.transform(X_sep_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Results storage
results = []

print("🔬 Training Models")
print("=" * 70)

for model_name, model in models.items():
    print(f"\n{'='*70}")
    print(f"🎯 {model_name}")
    print(f"{'='*70}")
    
    # MODEL A: COMBINED
    print("\n📌 MODEL A: COMBINED_JOINT_COUNT (11 features)")
    model_comb = model
    model_comb.fit(X_comb_train_scaled, y_train)
    y_pred_comb = model_comb.predict(X_comb_test_scaled)
    y_prob_comb = model_comb.predict_proba(X_comb_test_scaled)[:, 1]
    
    comb_acc = accuracy_score(y_test, y_pred_comb)
    comb_prec = precision_score(y_test, y_pred_comb)
    comb_rec = recall_score(y_test, y_pred_comb)
    comb_f1 = f1_score(y_test, y_pred_comb)
    comb_auc = roc_auc_score(y_test, y_prob_comb)
    
    print(f"   Accuracy:  {comb_acc:.4f}")
    print(f"   Precision: {comb_prec:.4f}")
    print(f"   Recall:    {comb_rec:.4f}")
    print(f"   F1 Score:  {comb_f1:.4f}")
    print(f"   ROC AUC:   {comb_auc:.4f}")
    
    # MODEL B: SEPARATE
    print(f"\n📌 MODEL B: SEPARATE TJC + SJC (12 features)")
    model_sep = model.__class__(**model.get_params())
    model_sep.fit(X_sep_train_scaled, y_train)
    y_pred_sep = model_sep.predict(X_sep_test_scaled)
    y_prob_sep = model_sep.predict_proba(X_sep_test_scaled)[:, 1]
    
    sep_acc = accuracy_score(y_test, y_pred_sep)
    sep_prec = precision_score(y_test, y_pred_sep)
    sep_rec = recall_score(y_test, y_pred_sep)
    sep_f1 = f1_score(y_test, y_pred_sep)
    sep_auc = roc_auc_score(y_test, y_prob_sep)
    
    print(f"   Accuracy:  {sep_acc:.4f}")
    print(f"   Precision: {sep_prec:.4f}")
    print(f"   Recall:    {sep_rec:.4f}")
    print(f"   F1 Score:  {sep_f1:.4f}")
    print(f"   ROC AUC:   {sep_auc:.4f}")
    
    # Comparison
    print(f"\n🔍 DIFFERENCE (Combined - Separate):")
    print(f"   Accuracy:  {(comb_acc-sep_acc)*100:+.2f}%")
    print(f"   Precision: {(comb_prec-sep_prec)*100:+.2f}%")
    print(f"   Recall:    {(comb_rec-sep_rec)*100:+.2f}%")
    print(f"   F1 Score:  {(comb_f1-sep_f1)*100:+.2f}%")
    print(f"   ROC AUC:   {(comb_auc-sep_auc)*100:+.2f}%")
    
    if sep_acc > comb_acc:
        print(f"   ✅ SEPARATE is better by {(sep_acc-comb_acc)*100:.2f}%")
    elif comb_acc > sep_acc:
        print(f"   ✅ COMBINED is better by {(comb_acc-sep_acc)*100:.2f}%")
    else:
        print(f"   🟰 TIED")
    
    # Store results
    results.append({
        'Model': model_name,
        'Type': 'Combined',
        'Features': len(COMBINED_FEATURES),
        'Accuracy': comb_acc,
        'Precision': comb_prec,
        'Recall': comb_rec,
        'F1': comb_f1,
        'AUC': comb_auc
    })
    
    results.append({
        'Model': model_name,
        'Type': 'Separate',
        'Features': len(SEPARATE_FEATURES),
        'Accuracy': sep_acc,
        'Precision': sep_prec,
        'Recall': sep_rec,
        'F1': sep_f1,
        'AUC': sep_auc
    })

# Summary
print(f"\n\n{'='*70}")
print("📊 FINAL COMPARISON")
print(f"{'='*70}\n")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("")

# Best model
best_model_row = results_df.loc[results_df['Accuracy'].idxmax()]
print(f"🏆 BEST MODEL: {best_model_row['Model']} ({best_model_row['Type']})")
print(f"   Accuracy: {best_model_row['Accuracy']:.4f} ({best_model_row['Accuracy']*100:.2f}%)")
print(f"   F1 Score: {best_model_row['F1']:.4f}")
print(f"   AUC: {best_model_row['AUC']:.4f}")
print("")

# Average comparison
combined_avg = results_df[results_df['Type']=='Combined']['Accuracy'].mean()
separate_avg = results_df[results_df['Type']=='Separate']['Accuracy'].mean()

print(f"📈 AVERAGE ACCURACY:")
print(f"   Combined: {combined_avg:.4f} ({combined_avg*100:.2f}%)")
print(f"   Separate: {separate_avg:.4f} ({separate_avg*100:.2f}%)")
print(f"   Difference: {(separate_avg-combined_avg)*100:+.2f}%")
print("")

if separate_avg > combined_avg:
    winner = "SEPARATE"
    print(f"🎯 CONCLUSION: SEPARATE joint counts perform better on average")
elif combined_avg > separate_avg:
    winner = "COMBINED"
    print(f"🎯 CONCLUSION: COMBINED joint count performs better on average")
else:
    winner = "TIE"
    print(f"🎯 CONCLUSION: Both approaches perform equally")

print("")

# Save best model
best_model_name = best_model_row['Model']
best_is_combined = best_model_row['Type'] == 'Combined'

if best_is_combined:
    X_final_train = X_comb_train_scaled
    X_final_test = X_comb_test_scaled
    scaler_final = scaler_comb
    features_final = COMBINED_FEATURES
    model_type = "combined"
else:
    X_final_train = X_sep_train_scaled
    X_final_test = X_sep_test_scaled
    scaler_final = scaler_sep
    features_final = SEPARATE_FEATURES
    model_type = "separate"

final_model = models[best_model_name]
final_model.fit(X_final_train, y_train)

# Save
joblib.dump(final_model, f'ra_model_{model_type}.pkl')
joblib.dump(scaler_final, f'ra_scaler_{model_type}.pkl')
joblib.dump(features_final, f'ra_features_{model_type}.pkl')
joblib.dump(model_type, 'ra_model_type.pkl')

print("💾 Saved:")
print(f"   - ra_model_{model_type}.pkl")
print(f"   - ra_scaler_{model_type}.pkl")
print(f"   - ra_features_{model_type}.pkl")
print(f"   - ra_model_type.pkl")
print("")

# Classification report
print(f"{'='*70}")
print(f"📋 CLASSIFICATION REPORT - {best_model_name} ({model_type.upper()})")
print(f"{'='*70}\n")

y_pred_final = final_model.predict(X_final_test)
print(classification_report(y_test, y_pred_final, target_names=['Remission', 'Inflammation']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
print("\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Remission  Inflammation")
print(f"Actual Remission      {cm[0,0]:4d}      {cm[0,1]:4d}")
print(f"       Inflammation   {cm[1,0]:4d}      {cm[1,1]:4d}")
print("")

# Save comparison
results_df.to_csv('comparison_combined_vs_separate.csv', index=False)
print("💾 Saved: comparison_combined_vs_separate.csv")
print("")
print("✅ Training complete!")
print(f"🎯 Winner: {model_type.upper()} joint count approach")
print(f"📊 Final accuracy: {best_model_row['Accuracy']*100:.2f}%")