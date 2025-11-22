# generate_enhanced_dataset.py
# Generate enhanced RA flare prediction dataset with 14 features
# Includes episode history features

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def get_age_based_bmi(age, sex):
    """Get realistic BMI based on age group and sex"""
    bmi_data = {
        'male': {
            (20, 39): (29.1, 4.5),
            (40, 59): (29.9, 4.8),
            (60, 79): (29.5, 4.2),
            (80, 100): (27.8, 4.0)
        },
        'female': {
            (20, 39): (29.6, 6.2),
            (40, 59): (30.9, 6.5),
            (60, 79): (29.3, 5.8),
            (80, 100): (27.5, 5.5)
        }
    }
    
    sex_key = 'male' if sex == 1 else 'female'
    
    for (min_age, max_age), (mean_bmi, std_bmi) in bmi_data[sex_key].items():
        if min_age <= age <= max_age:
            bmi = np.random.normal(mean_bmi, std_bmi)
            return np.clip(bmi, 18.5, 50)
    
    return 27.0

def generate_enhanced_ra_dataset(n_samples=3000):
    """
    Generate enhanced RA flare prediction dataset with 14 features
    
    New Episode Features:
    - last_episode_duration: Duration of last flare episode (0-30 days)
    - days_since_last_episode: Days since last inflammation (0-180 days)
    """
    
    print(f"Generating {n_samples} samples with 14 features...")
    print(f"Including episode history features:")
    print(f"  - last_episode_duration")
    print(f"  - days_since_last_episode")
    
    data = []
    
    for i in range(n_samples):
        # Demographics
        age = np.random.randint(25, 80)
        sex = np.random.choice([0, 1], p=[0.65, 0.35])  # RA more common in females
        disease_duration = np.random.randint(1, 30)
        
        # Body metrics
        bmi = get_age_based_bmi(age, sex)
        
        # Lifestyle factors
        sleep_base = 7.5 - (age < 40) * 0.5
        sleep_hours = np.clip(np.random.normal(sleep_base, 1.5), 3, 12)
        
        # Smoking rates by age
        if age < 40:
            smoking_probs = [0.75, 0.15, 0.10]
        elif age < 60:
            smoking_probs = [0.65, 0.15, 0.20]
        else:
            smoking_probs = [0.80, 0.05, 0.15]
        smoking_status = np.random.choice([0, 1, 2], p=smoking_probs)
        
        # Weather factors
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
        
        # Barometric pressure change
        change_in_barometric_pressure = np.random.normal(0, 8)
        change_in_barometric_pressure = np.clip(change_in_barometric_pressure, -25, 25)
        
        # Air quality
        aqi_base = 50 + (smoking_status == 1) * 30
        air_quality_index = np.clip(np.random.normal(aqi_base, 30), 0, 300)
        
        # Current pain score
        pain_base = 2 + (disease_duration / 15) + (age > 60) * 0.5
        current_pain_score = np.clip(np.random.normal(pain_base, 2), 0, 10)
        
        # NEW: Episode history features
        # Days since last episode (0-180 days, with higher frequency for recent episodes)
        days_since_dist = np.random.exponential(30)
        days_since_last_episode = int(np.clip(days_since_dist, 0, 180))
        
        # Last episode duration (1-30 days, avg ~5-7 days for RA flares)
        # Only meaningful if they've had a recent episode
        if days_since_last_episode < 90:
            # Had recent episode
            last_episode_duration = int(np.clip(np.random.gamma(3, 2), 1, 30))
        else:
            # No recent episode or first episode
            last_episode_duration = 0
        
        # --- INFLAMMATION LOGIC (Target Variable) ---
        inflammation_score = 0
        
        # Lifestyle triggers
        if sleep_hours < 6:
            inflammation_score += 0.22
        if sleep_hours > 9:
            inflammation_score += 0.08
        if smoking_status == 1:
            inflammation_score += 0.18
        
        # Demographic risk factors
        if age > 60:
            inflammation_score += 0.08
        if disease_duration > 15:
            inflammation_score += 0.12
        if bmi > 30:
            inflammation_score += 0.13
        elif bmi < 20:
            inflammation_score += 0.08
        
        # Environmental triggers
        if humidity > 75:
            inflammation_score += 0.12
        if air_quality_index > 100:
            inflammation_score += 0.13
        if min_temperature < 10:
            inflammation_score += 0.10
        
        # Barometric pressure change trigger
        if change_in_barometric_pressure < -10:
            inflammation_score += 0.18
        elif change_in_barometric_pressure < -5:
            inflammation_score += 0.10
        
        # Current pain
        if current_pain_score > 6:
            inflammation_score += 0.18
        
        # NEW: Episode history impact (STRONG PREDICTORS)
        # Recent history of flares is a strong predictor
        if days_since_last_episode < 14:
            inflammation_score += 0.25  # Very recent episode
        elif days_since_last_episode < 30:
            inflammation_score += 0.15  # Recent episode
        elif days_since_last_episode < 60:
            inflammation_score += 0.08  # Somewhat recent
        
        # Longer previous episodes indicate more severe disease
        if last_episode_duration > 14:
            inflammation_score += 0.15  # Long previous episode
        elif last_episode_duration > 7:
            inflammation_score += 0.10  # Moderate previous episode
        elif last_episode_duration > 0:
            inflammation_score += 0.05  # Had previous episode
        
        # Interaction effects
        if bmi > 30 and sleep_hours < 6:
            inflammation_score += 0.08
        if smoking_status == 1 and air_quality_index > 100:
            inflammation_score += 0.08
        if days_since_last_episode < 30 and current_pain_score > 6:
            inflammation_score += 0.10  # Recent episode + high pain
        
        # Add randomness
        inflammation_score += np.random.uniform(-0.2, 0.2)
        
        # Convert to binary
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
            'change_in_barometric_pressure': round(change_in_barometric_pressure, 1),
            'current_pain_score': round(current_pain_score, 1),
            'last_episode_duration': last_episode_duration,
            'days_since_last_episode': days_since_last_episode,
            'inflammation': inflammation
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Print statistics
    print(f"\n✅ Dataset generated: {len(df)} samples")
    print(f"\n📊 Target Distribution:")
    print(f"  No Flare (0): {(df['inflammation'] == 0).sum()} ({(df['inflammation'] == 0).mean()*100:.1f}%)")
    print(f"  Flare (1):    {(df['inflammation'] == 1).sum()} ({(df['inflammation'] == 1).mean()*100:.1f}%)")
    
    print(f"\n📈 Feature Statistics:")
    for col in df.columns:
        if col != 'inflammation':
            print(f"  {col}: {df[col].min():.1f} - {df[col].max():.1f} (mean: {df[col].mean():.1f})")
    
    return df

if __name__ == "__main__":
    # Generate dataset
    df = generate_enhanced_ra_dataset(n_samples=3000)
    
    # Save to CSV
    output_file = 'training_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")
    
    # Show sample
    print(f"\n📝 Sample records:")
    print(df.head(10).to_string())
    
    print(f"\n🎯 Dataset ready for training!")