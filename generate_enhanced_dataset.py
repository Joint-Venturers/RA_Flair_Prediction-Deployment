# generate_enhanced_dataset.py
# Generate enhanced RA flare prediction dataset with lifestyle and environmental factors
# BMI based on CDC age-group averages

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def get_age_based_bmi(age, sex):
    """
    Get realistic BMI based on age group and sex
    Based on CDC National Health Statistics (2023)
    
    Age Groups and Average BMI:
    - 20-39: Male 29.1, Female 29.6
    - 40-59: Male 29.9, Female 30.9
    - 60-79: Male 29.5, Female 29.3
    
    Source: CDC NHANES Data (2017-2020)
    """
    
    # BMI averages by age group and sex
    bmi_data = {
        'male': {
            (20, 39): (29.1, 4.5),   # (mean, std)
            (40, 59): (29.9, 4.8),
            (60, 79): (29.5, 4.2),
            (80, 100): (27.8, 4.0)   # Older adults tend to have lower BMI
        },
        'female': {
            (20, 39): (29.6, 6.2),
            (40, 59): (30.9, 6.5),
            (60, 79): (29.3, 5.8),
            (80, 100): (27.5, 5.5)
        }
    }
    
    sex_key = 'male' if sex == 1 else 'female'
    
    # Find appropriate age group
    for (min_age, max_age), (mean_bmi, std_bmi) in bmi_data[sex_key].items():
        if min_age <= age <= max_age:
            # Generate BMI with realistic variance
            bmi = np.random.normal(mean_bmi, std_bmi)
            # Clip to realistic range (18.5 underweight to 50 severe obesity)
            return np.clip(bmi, 18.5, 50)
    
    # Fallback (shouldn't reach here)
    return 27.0

def generate_enhanced_ra_dataset(n_samples=3000):
    """
    Generate enhanced RA flare prediction dataset with personalized triggers
    
    Features:
    - Realistic age-based BMI distributions
    - Lifestyle factors (sleep, smoking)
    - Demographics (age, sex, disease_duration)
    - Environmental (weather, air quality)
    - Current state (pain score, joint counts)
    
    Target: Predict inflammation (flare)
    """
    
    print(f"Generating {n_samples} samples with enhanced features...")
    print(f"\nUsing CDC-based BMI distributions:")
    print(f"  Age 20-39: Male 29.1±4.5, Female 29.6±6.2")
    print(f"  Age 40-59: Male 29.9±4.8, Female 30.9±6.5")
    print(f"  Age 60-79: Male 29.5±4.2, Female 29.3±5.8")
    
    data = []
    
    for i in range(n_samples):
        # Demographics
        age = np.random.randint(25, 80)
        sex = np.random.choice([0, 1], p=[0.65, 0.35])  # RA more common in females (65%)
        disease_duration = np.random.randint(1, 30)  # years since diagnosis
        
        # Body metrics - Age and sex-specific BMI
        bmi = get_age_based_bmi(age, sex)
        
        # Lifestyle factors
        # Sleep: Younger adults sleep less on average
        sleep_base = 7.5 - (age < 40) * 0.5  # Younger adults average 7 hours
        sleep_hours = np.clip(np.random.normal(sleep_base, 1.5), 3, 12)
        
        # Smoking: Higher rates in 40-59 age group, lower in 60+
        if age < 40:
            smoking_probs = [0.75, 0.15, 0.10]  # no, yes, quit
        elif age < 60:
            smoking_probs = [0.65, 0.15, 0.20]
        else:
            smoking_probs = [0.80, 0.05, 0.15]  # older adults less likely to smoke
        smoking_status = np.random.choice([0, 1, 2], p=smoking_probs)
        
        # Weather factors (seasonal variation)
        season = np.random.choice(['winter', 'spring', 'summer', 'fall'])
        if season == 'winter':
            temp_mean = np.random.uniform(0, 10)
            humidity_mean = 65
        elif season == 'summer':
            temp_mean = np.random.uniform(20, 35)
            humidity_mean = 70
        else:
            temp_mean = np.random.uniform(10, 20)
            humidity_mean = 60
        
        min_temperature = temp_mean - np.random.uniform(2, 5)
        max_temperature = temp_mean + np.random.uniform(2, 5)
        
        humidity = np.clip(np.random.normal(humidity_mean, 15), 20, 95)
        barometric_pressure = np.random.normal(1013, 10)  # hPa
        precipitation = np.random.exponential(2) if np.random.rand() < 0.3 else 0
        wind_speed = np.random.exponential(5)
        
        # Air quality (worse in urban areas, worse with smoking)
        aqi_base = 50 + (smoking_status == 1) * 30  # smokers often in higher AQI areas
        air_quality_index = np.clip(np.random.normal(aqi_base, 30), 0, 300)
        
        # Current pain score (0-10 scale)
        # Baseline increases with disease duration and age
        pain_base = 2 + (disease_duration / 15) + (age > 60) * 0.5
        current_pain_score = np.clip(np.random.normal(pain_base, 2), 0, 10)
        
        # Joint assessments
        tender_joint_count = np.random.poisson(2) + (current_pain_score > 5) * 2
        swollen_joint_count = np.random.poisson(1.5) + (current_pain_score > 5) * 1.5
        
        # --- INFLAMMATION LOGIC (Target Variable) ---
        # Complex multi-factor model for personalized triggers
        
        inflammation_score = 0
        
        # Lifestyle triggers
        if sleep_hours < 6:
            inflammation_score += 0.25  # Poor sleep is major trigger
        if sleep_hours > 9:
            inflammation_score += 0.10  # Excessive sleep can indicate inflammation
        if smoking_status == 1:
            inflammation_score += 0.20  # Active smoking major trigger
        
        # Demographic risk factors
        if age > 60:
            inflammation_score += 0.10  # Older adults more susceptible
        if disease_duration > 15:
            inflammation_score += 0.15  # Longer disease duration = more damage
        if bmi > 30:
            inflammation_score += 0.15  # Obesity is inflammation trigger (adipokines)
        elif bmi < 20:
            inflammation_score += 0.10  # Underweight also risk factor
        
        # Environmental triggers (well-documented in RA literature)
        if humidity > 75:
            inflammation_score += 0.15  # High humidity trigger
        if barometric_pressure < 1000:
            inflammation_score += 0.15  # Low pressure (weather changes)
        if air_quality_index > 100:
            inflammation_score += 0.15  # Air pollution trigger
        if min_temperature < 10:
            inflammation_score += 0.12  # Cold weather trigger
        if precipitation > 3:
            inflammation_score += 0.08  # Rain/storms
        
        # Current inflammatory state
        if current_pain_score > 6:
            inflammation_score += 0.20  # High pain indicates inflammation
        if tender_joint_count > 3:
            inflammation_score += 0.15  # Multiple tender joints
        if swollen_joint_count > 2:
            inflammation_score += 0.15  # Swelling indicates active inflammation
        
        # Interaction effects
        if bmi > 30 and sleep_hours < 6:
            inflammation_score += 0.10  # Combined obesity + poor sleep
        if smoking_status == 1 and air_quality_index > 100:
            inflammation_score += 0.10  # Smoking + pollution
        
        # Add some randomness (individual variation)
        inflammation_score += np.random.uniform(-0.2, 0.2)
        
        # Convert to binary (0=no flare, 1=flare)
        # Threshold tuned for ~40-45% flare rate (realistic for RA)
        inflammation = 1 if inflammation_score > 0.55 else 0
        
        # Create record
        record = {
            'age': round(age),
            'sex': sex,
            'disease_duration': round(disease_duration),
            'bmi': round(bmi, 1),
            'sleep_hours': round(sleep_hours, 1),
            'smoking_status': smoking_status,
            'air_quality_index': round(air_quality_index),
            'min_temperature': round(min_temperature, 1),
            'max_temperature': round(max_temperature, 1),
            'humidity': round(humidity, 1),
            'barometric_pressure': round(barometric_pressure, 1),
            'precipitation': round(precipitation, 1),
            'wind_speed': round(wind_speed, 1),
            'current_pain_score': round(current_pain_score, 1),
            'tender_joint_count': round(tender_joint_count, 1),
            'swollen_joint_count': round(swollen_joint_count, 1),
            'inflammation': inflammation
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Print dataset statistics
    print(f"\n✅ Dataset generated: {len(df)} samples")
    print(f"\n📊 Target Distribution:")
    print(f"  No Flare (0): {(df['inflammation'] == 0).sum()} ({(df['inflammation'] == 0).mean()*100:.1f}%)")
    print(f"  Flare (1):    {(df['inflammation'] == 1).sum()} ({(df['inflammation'] == 1).mean()*100:.1f}%)")
    
    print(f"\n📈 BMI Statistics by Age Group:")
    for age_group in [(25, 39), (40, 59), (60, 79)]:
        age_mask = (df['age'] >= age_group[0]) & (df['age'] <= age_group[1])
        male_bmi = df[age_mask & (df['sex'] == 1)]['bmi'].mean()
        female_bmi = df[age_mask & (df['sex'] == 0)]['bmi'].mean()
        print(f"  Age {age_group[0]}-{age_group[1]}: Male {male_bmi:.1f}, Female {female_bmi:.1f}")
    
    print(f"\n📋 Feature Ranges:")
    for col in df.columns:
        if col != 'inflammation':
            print(f"  {col}: {df[col].min():.1f} - {df[col].max():.1f} (mean: {df[col].mean():.1f})")
    
    return df

if __name__ == "__main__":
    # Generate dataset
    df = generate_enhanced_ra_dataset(n_samples=3000)
    
    # Save to CSV
    output_file = 'ra_data_enhanced.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")
    
    # Show sample
    print(f"\n📝 Sample records:")
    print(df.head(10).to_string())
    
    print(f"\n🎯 Dataset ready for training!")