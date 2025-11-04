# analytics_test.py - Test Analytics Endpoints with Perfect Data

import requests
import json
from datetime import datetime, timedelta
import numpy as np

def test_correlation_analytics_with_safe_data():
    """Test correlation analytics with perfectly safe data types"""
    
    base_url = "http://localhost:8000"
    print("🔗 Testing Correlation Analytics with Safe Data...")
    
    # Generate data with EXPLICIT type safety
    historical_data = []
    for i in range(45):  # 45 days for better correlations
        date = (datetime.now() - timedelta(days=45-i)).strftime("%Y-%m-%d")
        
        # Create strong correlations with explicit float conversion
        temp = float(10 + 15 * np.sin(i / 12))
        humidity = float(50 + 30 * np.cos(i / 10))
        
        # Pain strongly correlated with weather (explicit float)
        weather_pain = 3 + (15 - temp) * 0.4 + (humidity - 50) * 0.03
        weather_pain = float(max(1.0, min(10.0, weather_pain + np.random.normal(0, 0.8))))
        
        # Other metrics (explicit float/int conversion)
        stress = float(max(1.0, min(10.0, 3 + weather_pain * 0.5 + np.random.normal(0, 1))))
        sleep = float(max(1.0, min(10.0, 9 - weather_pain * 0.4 + np.random.normal(0, 0.8))))
        med_adh = float(max(0.1, min(1.0, 0.9 - (weather_pain - 3) * 0.05)))
        
        # CRITICAL: Convert flare_occurred to EXPLICIT int (not boolean)
        flare_occurred_int = 1 if weather_pain > 7 else 0  # Explicit int conversion
        
        historical_data.append({
            "date": str(date),  # Explicit string
            "weather_data": {
                "temperature": temp,      # float
                "humidity": humidity,     # float
                "pressure": float(1013 + np.random.normal(0, 8)),  # explicit float
                "weather_condition": "cloudy",  # string
                "temp_change_24h": float(0),     # explicit float
                "pressure_change_24h": float(0), # explicit float  
                "humidity_change_24h": float(0)  # explicit float
            },
            "pain_history": {
                "day_1_avg": weather_pain,  # already float
                "day_3_avg": weather_pain,  # already float
                "day_7_avg": weather_pain   # already float
            },
            "patient_data": {
                "age": int(58),                    # explicit int
                "disease_duration": int(12),       # explicit int
                "medication_adherence": med_adh,   # already float
                "sleep_quality": sleep,            # already float
                "stress_level": stress             # already float
            },
            "flare_occurred": flare_occurred_int   # CRITICAL: explicit int (not bool)
        })
    
    correlation_request = {
        "user_id": "safe_correlation_test_patient",  # string
        "historical_data": historical_data,
        "analytics_type": "correlation"  # string
    }
    
    # Print sample data for debugging
    print(f"Sample data types:")
    sample = historical_data[0]
    print(f"  flare_occurred: {sample['flare_occurred']} (type: {type(sample['flare_occurred'])})")
    print(f"  temperature: {sample['weather_data']['temperature']} (type: {type(sample['weather_data']['temperature'])})")
    
    try:
        response = requests.post(
            f"{base_url}/analytics/correlations",
            json=correlation_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("🎉 CORRELATION ANALYTICS SUCCESS!")
            
            correlation_analysis = result.get('correlation_analysis', {})
            
            # Display strong correlations
            strong_correlations = correlation_analysis.get('strong_correlations', [])
            if strong_correlations:
                print(f"🔗 Found {len(strong_correlations)} strong correlations:")
                for i, corr in enumerate(strong_correlations[:5], 1):
                    direction = "📈" if corr.get('correlation_direction') == 'positive' else "📉"
                    print(f"   {i}. {direction} {corr['symptom']} ↔ {corr['trigger']}")
                    print(f"      Strength: {corr['correlation_strength']:.3f}")
            
            # Display insights
            insights = correlation_analysis.get('personalized_insights', [])
            if insights:
                print(f"\n💡 PERSONALIZED INSIGHTS:")
                for i, insight in enumerate(insights, 1):
                    print(f"   {i}. {insight}")
            
            # Display summary
            summary = correlation_analysis.get('correlation_strength_summary', {})
            if summary:
                print(f"\n📊 CORRELATION SUMMARY:")
                print(f"   Total Correlations: {summary.get('total', 0)}")
                print(f"   Strong (>0.7): {summary.get('strong', 0)}")
                print(f"   Moderate (0.3-0.7): {summary.get('moderate', 0)}")
                print(f"   Average Strength: {summary.get('average_strength', 0):.3f}")
            
            return True
            
        else:
            print(f"❌ Correlation analytics failed: {response.status_code}")
            print(f"Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Correlation request failed: {e}")
        return False

def test_main_analytics_with_safe_data():
    """Test main analytics with safe data types"""
    
    base_url = "http://localhost:8000"
    print("📊 Testing Main Analytics with Safe Data...")
    
    # Generate 30 days of perfectly safe data
    historical_data = []
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(30):
        date = start_date + timedelta(days=i)
        
        # Explicit type conversions throughout
        temp = float(15 + 10 * np.sin(i / 10) + np.random.normal(0, 3))
        humidity = float(max(30, min(90, 60 + 15 * np.sin(i / 8) + np.random.normal(0, 10))))
        pressure = float(1013 + np.random.normal(0, 15))
        
        # Pain calculation (explicit float)
        weather_effect = (temp < 10) * 2 + (humidity > 75) * 1.5 + (pressure < 1000) * 1
        base_pain = 4 + weather_effect + np.random.normal(0, 1.5)
        pain = float(max(1, min(10, base_pain)))
        
        # Patient factors (explicit float)
        stress = float(max(1, min(10, 5 + weather_effect * 0.5 + np.random.normal(0, 1.5))))
        sleep = float(max(1, min(10, 7 - pain * 0.3 + np.random.normal(0, 1))))
        med_adherence = float(max(0.1, min(1.0, 0.85 - (stress - 5) * 0.05 + np.random.normal(0, 0.1))))
        
        # Flare calculation (explicit int)
        flare_prob = 0.1 + (pain > 6) * 0.3 + (stress > 7) * 0.2 + (sleep < 4) * 0.2
        flare = 1 if np.random.random() < flare_prob else 0  # EXPLICIT int
        
        historical_data.append({
            "date": date.strftime("%Y-%m-%d"),  # string
            "weather_data": {
                "temperature": temp,                                    # float
                "humidity": humidity,                                   # float
                "pressure": pressure,                                   # float
                "weather_condition": ["sunny", "cloudy", "rainy"][i % 3],  # string
                "temp_change_24h": float(np.random.normal(0, 5)),       # explicit float
                "pressure_change_24h": float(np.random.normal(0, 8)),   # explicit float
                "humidity_change_24h": float(np.random.normal(0, 10))   # explicit float
            },
            "pain_history": {
                "day_1_avg": pain,                                      # float
                "day_3_avg": float(max(1, min(10, pain + np.random.normal(0, 0.5)))),  # explicit float
                "day_7_avg": float(max(1, min(10, pain + np.random.normal(0, 1))))     # explicit float
            },
            "patient_data": {
                "age": int(55),              # explicit int
                "disease_duration": int(8),  # explicit int
                "medication_adherence": med_adherence,  # float
                "sleep_quality": sleep,      # float
                "stress_level": stress       # float
            },
            "flare_occurred": flare  # EXPLICIT int
        })
    
    analytics_request = {
        "user_id": "safe_analytics_test_patient",
        "historical_data": historical_data,
        "analytics_type": "comprehensive"
    }
    
    try:
        response = requests.post(
            f"{base_url}/analytics",
            json=analytics_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("🎉 MAIN ANALYTICS SUCCESS!")
            print(f"✅ User ID: {result['user_id']}")
            print(f"✅ Data Period: {result['data_period']['total_records']} records")
            
            # Display analytics results (same as before)
            analytics = result.get('analytics', {})
            
            if 'risk_analytics' in analytics:
                risk_data = analytics['risk_analytics']
                print(f"\n🎯 RISK ANALYTICS:")
                print(f"   Current Risk Score: {risk_data.get('current_risk_score', 'N/A'):.2f}")
                print(f"   Risk Level: {risk_data.get('risk_level', 'N/A')}")
                print(f"   Trend Direction: {risk_data.get('trend_direction', 'N/A')}")
                print(f"   Days Since Last Flare: {risk_data.get('days_since_last_flare', 'N/A')}")
            
            return True
            
        else:
            print(f"❌ Analytics failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analytics request failed: {e}")
        return False

def run_perfect_test_suite():
    """Run the perfect test suite with safe data types"""
    print("🚀 PERFECT ANALYTICS TEST SUITE")
    print("=" * 60)
    
    # Test both endpoints with perfectly safe data
    main_analytics_result = test_main_analytics_with_safe_data()
    correlation_analytics_result = test_correlation_analytics_with_safe_data()
    
    print("\n" + "=" * 60)
    print("📊 PERFECT TEST RESULTS")
    print("=" * 60)
    print(f"Main Analytics: {'✅ WORKING' if main_analytics_result else '❌ NEEDS FIX'}")
    print(f"Correlation Analytics: {'✅ WORKING' if correlation_analytics_result else '❌ NEEDS FIX'}")
    
    if main_analytics_result and correlation_analytics_result:
       
        print("🏥 Your RA prediction system has complete analytics capabilities!")
       
    else:
        print(f"\n🔧 Some analytics endpoints still need attention.")

if __name__ == "__main__":
    run_perfect_test_suite()