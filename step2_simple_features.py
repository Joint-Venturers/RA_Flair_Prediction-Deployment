# Step 2 FINAL: Simple Feature Engineering - ONLY SPECIFIED FEATURES
# NO extra engineered features - just raw features + binary target

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def simple_feature_engineering(df_raw):
    """
    Minimal feature engineering - ONLY keep specified features
    """
    print("⚙️ SIMPLE FEATURE ENGINEERING - RAW FEATURES ONLY")
    print("=" * 60)
    
    df = df_raw.copy()
    
    # Convert DAS28ESR to binary inflammation indicator
    print("✅ Converting DAS28ESR to binary inflammation target")
    df['inflammation'] = (df['das28esr'] >= 2.6).astype(int)
    print(f"   - Remission (DAS28 < 2.6): {(df['inflammation']==0).sum()} samples ({(df['inflammation']==0).mean()*100:.1f}%)")
    print(f"   - Active inflammation (DAS28 >= 2.6): {(df['inflammation']==1).sum()} samples ({(df['inflammation']==1).mean()*100:.1f}%)")
    print("")
    
    # Create combined joint count for comparison
    print("✅ Creating combined_joint_count for model comparison")
    df['combined_joint_count'] = df['tender_joint_count'] + df['swollen_joint_count']
    print(f"   - TJC range: {df['tender_joint_count'].min():.2f} to {df['tender_joint_count'].max():.2f}")
    print(f"   - SJC range: {df['swollen_joint_count'].min():.2f} to {df['swollen_joint_count'].max():.2f}")
    print(f"   - Combined range: {df['combined_joint_count'].min():.2f} to {df['combined_joint_count'].max():.2f}")
    print("")
    
    print("📊 Features Kept:")
    print("   ✅ age")
    print("   ✅ gender")
    print("   ✅ disease_duration")
    print("   ✅ bmi")
    print("   ✅ min_temperature")
    print("   ✅ max_temperature")
    print("   ✅ humidity")
    print("   ✅ barometric_pressure")
    print("   ✅ precipitation")
    print("   ✅ wind_speed")
    print("   ✅ tender_joint_count (for separate model)")
    print("   ✅ swollen_joint_count (for separate model)")
    print("   ✅ combined_joint_count (for combined model)")
    print("")
    print("❌ NO engineered features")
    print("❌ NO season")
    print("❌ NO temperature ranges")
    print("❌ NO weather flags")
    print("")
    
    return df

# Load raw data
print("📂 Loading raw dataset...")
df_raw = pd.read_csv('ra_research_data_raw.csv')
print(f"✅ Loaded {df_raw.shape[0]} samples with {df_raw.shape[1]} features")
print("")

# Apply simple feature engineering
df_simple = simple_feature_engineering(df_raw)

# Save simple dataset
df_simple.to_csv('ra_data_simple.csv', index=False)
print("💾 Saved: ra_data_simple.csv")
print("")

# Display sample
print("📊 Sample of dataset (first 5 rows):")
print(df_simple[['age', 'gender', 'disease_duration', 'bmi', 'min_temperature', 
                'max_temperature', 'humidity', 'tender_joint_count', 
                'swollen_joint_count', 'combined_joint_count', 'inflammation']].head())
print("")

# Target variable distribution
print("🎯 Target Variable (Inflammation) Distribution:")
print(df_simple['inflammation'].value_counts())
print("")
print(f"Remission: {(df_simple['inflammation']==0).sum()} ({(df_simple['inflammation']==0).mean()*100:.1f}%)")
print(f"Inflammation: {(df_simple['inflammation']==1).sum()} ({(df_simple['inflammation']==1).mean()*100:.1f}%)")
print("")

# Feature statistics
print("📈 Key Feature Statistics:")
print(df_simple[['age', 'disease_duration', 'bmi', 'tender_joint_count', 
                'swollen_joint_count', 'combined_joint_count', 
                'min_temperature', 'humidity', 'inflammation']].describe())