# Step 1: Generate Synthetic Dataset Based on Research Paper
# Based on PMC7492902 - Seasonal and Weather Effects on Rheumatoid Arthritis

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_research_based_synthetic_data(n_samples=2000, random_seed=42):
    """
    Generate synthetic RA dataset based on actual research paper parameters
    Source: PMC7492902 Tables 1 and 2
    
    Parameters from Table 2:
    - Age: 50.45 ± 12.17 years
    - Disease duration: 10.57 ± 8.49 years
    - BMI: 26.63 ± 8.04
    - TJC (Tender Joint Count): Winter 0.82±1.61, Spring 1.04±1.99, Summer 0.89±2.35
    - SJC (Swollen Joint Count): Winter 0.48±1.14, Spring 0.72±1.38, Summer 0.57±1.41
    - DAS28ESR: Winter 2.44±0.95, Spring 2.60±0.98, Summer 2.57±0.95
    
    Parameters from Table 1 (Weather):
    - Temperature: Winter 10-15°C, Spring 15.9-19.5°C, Summer 24.5-32.7°C
    - Humidity: Winter 58.1%, Spring 71%, Summer 49.7%
    - Pressure: Winter 1023.9 Mb, Spring 1016.4 Mb, Summer 1016.9 Mb
    - Precipitation: Winter 1.2 mm, Spring 3.14 mm, Summer 0.08 mm
    - Wind speed: Winter 4.1 m/s, Spring 6.1 m/s, Summer 5.9 m/s
    """
    
    np.random.seed(random_seed)
    
    print("🔬 Generating Research-Based Synthetic RA Dataset")
    print("=" * 60)
    print(f"📊 Target samples: {n_samples}")
    print(f"📄 Source: PMC7492902 - Azzouzi et al. (2020)")
    print("")
    
    # Patient demographics (from Table 2)
    age = np.clip(np.random.normal(50.45, 12.17, n_samples), 25, 85)
    disease_duration = np.clip(np.random.normal(10.57, 8.49, n_samples), 0, 40)
    bmi = np.clip(np.random.normal(26.63, 8.04, n_samples), 15, 50)
    
    # Gender distribution (86.3% women from research)
    gender = np.random.choice([0, 1], n_samples, p=[0.137, 0.863])  # 0=Male, 1=Female
    
    # Sjogren's syndrome (46.8% prevalence - BUT WE'LL DROP THIS)
    # sjogren = np.random.choice([0, 1], n_samples, p=[0.532, 0.468])
    
    # Season assignment (equal distribution)
    seasons = np.random.choice(['winter', 'spring', 'summer'], n_samples, p=[0.33, 0.34, 0.33])
    
    # Initialize arrays
    min_temp = np.zeros(n_samples)
    max_temp = np.zeros(n_samples)
    humidity = np.zeros(n_samples)
    pressure = np.zeros(n_samples)
    precipitation = np.zeros(n_samples)
    wind_speed = np.zeros(n_samples)
    tjc = np.zeros(n_samples)
    sjc = np.zeros(n_samples)
    das28esr = np.zeros(n_samples)
    
    # Generate season-specific parameters
    for i in range(n_samples):
        season = seasons[i]
        
        if season == 'winter':
            # Table 1: Winter parameters
            min_temp[i] = np.random.normal(10, 2.5)
            max_temp[i] = np.random.normal(15, 2.5)
            humidity[i] = np.clip(np.random.normal(58.1, 10), 30, 90)
            pressure[i] = np.random.normal(1023.9, 5)
            precipitation[i] = np.clip(np.random.exponential(1.2), 0, 20)
            wind_speed[i] = np.clip(np.random.normal(4.1, 1.5), 0, 15)
            
            # Table 2: Winter clinical parameters
            tjc[i] = np.clip(np.random.normal(0.82, 1.61), 0, 28)
            sjc[i] = np.clip(np.random.normal(0.48, 1.14), 0, 28)
            das28esr[i] = np.clip(np.random.normal(2.44, 0.95), 0.5, 9.5)
            
        elif season == 'spring':
            # Table 1: Spring parameters
            min_temp[i] = np.random.normal(15.9, 2.5)
            max_temp[i] = np.random.normal(19.5, 2.5)
            humidity[i] = np.clip(np.random.normal(71, 10), 40, 95)
            pressure[i] = np.random.normal(1016.4, 5)
            precipitation[i] = np.clip(np.random.exponential(3.14), 0, 30)
            wind_speed[i] = np.clip(np.random.normal(6.1, 1.5), 0, 15)
            
            # Table 2: Spring clinical parameters
            tjc[i] = np.clip(np.random.normal(1.04, 1.99), 0, 28)
            sjc[i] = np.clip(np.random.normal(0.72, 1.38), 0, 28)
            das28esr[i] = np.clip(np.random.normal(2.60, 0.98), 0.5, 9.5)
            
        else:  # summer
            # Table 1: Summer parameters
            min_temp[i] = np.random.normal(24.5, 3)
            max_temp[i] = np.random.normal(32.7, 3)
            humidity[i] = np.clip(np.random.normal(49.7, 10), 25, 80)
            pressure[i] = np.random.normal(1016.9, 5)
            precipitation[i] = np.clip(np.random.exponential(0.08), 0, 5)
            wind_speed[i] = np.clip(np.random.normal(5.9, 1.5), 0, 15)
            
            # Table 2: Summer clinical parameters
            tjc[i] = np.clip(np.random.normal(0.89, 2.35), 0, 28)
            sjc[i] = np.clip(np.random.normal(0.57, 1.41), 0, 28)
            das28esr[i] = np.clip(np.random.normal(2.57, 0.95), 0.5, 9.5)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'disease_duration': disease_duration,
        'bmi': bmi,
        'season': seasons,
        'min_temperature': min_temp,
        'max_temperature': max_temp,
        'humidity': humidity,
        'barometric_pressure': pressure,
        'precipitation': precipitation,
        'wind_speed': wind_speed,
        'tender_joint_count': tjc,
        'swollen_joint_count': sjc,
        'das28esr': das28esr
    })
    
    print("✅ Dataset generation complete!")
    print(f"📊 Shape: {df.shape}")
    print(f"👥 Gender distribution: {(gender.mean()*100):.1f}% female")
    print(f"📅 Season distribution:")
    print(f"   - Winter: {(seasons=='winter').sum()} samples")
    print(f"   - Spring: {(seasons=='spring').sum()} samples")
    print(f"   - Summer: {(seasons=='summer').sum()} samples")
    print("")
    
    return df

# Generate the dataset
df_raw = generate_research_based_synthetic_data(n_samples=2000)

# Save raw dataset
df_raw.to_csv('ra_research_data_raw.csv', index=False)
print("💾 Saved: ra_research_data_raw.csv")
print("")

# Display sample
print("📊 First 10 rows of raw dataset:")
print(df_raw.head(10))
print("")
print("📈 Statistical summary:")
print(df_raw.describe())