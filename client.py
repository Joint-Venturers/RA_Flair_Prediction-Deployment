import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Sample data for testing
test_data = {
    "weather_data": {
        "temperature": 5.0,
        "humidity": 85.0,
        "pressure": 995.0,
        "weather_condition": "rainy",
        "temp_change_24h": -12.0,
        "pressure_change_24h": -20.0,
        "humidity_change_24h": 25.0
    },
    "pain_history": {
        "1_day_avg": 7.5,
        "3_day_avg": 6.8,
        "7_day_avg": 5.2
    },
    "patient_data": {
        "age": 62,
        "disease_duration": 15,
        "medication_adherence": 0.7,
        "sleep_quality": 3,
        "stress_level": 8
    }
}

# Send request
try:
    response = requests.post(url, json=test_data)
    
    if response.status_code == 200:
        result = response.json()
        print("🎯 Prediction Results:")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Flare Probability: {result['flare_probability']:.1%}")
        print(f"Confidence: {result['confidence_score']:.1%}")
        print(f"Risk Factors: {result['risk_factors']}")
        print(f"Recommendations: {result['recommendations'][:3]}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to API. Make sure the server is running.")
except Exception as e:
    print(f"❌ Error: {e}")

# Test multiple predictions
test_scenarios = [
    {
        "name": "Low Risk",
        "data": {
            "weather_data": {"temperature": 22, "humidity": 40, "pressure": 1020, "weather_condition": "sunny"},
            "pain_history": {"1_day_avg": 2.5, "3_day_avg": 2.8, "7_day_avg": 3.1},
            "patient_data": {"age": 45, "medication_adherence": 0.95, "sleep_quality": 8, "stress_level": 2}
        }
    },
    {
        "name": "High Risk",
        "data": {
            "weather_data": {"temperature": 2, "humidity": 90, "pressure": 985, "weather_condition": "stormy"},
            "pain_history": {"1_day_avg": 8.5, "3_day_avg": 7.2, "7_day_avg": 6.1},
            "patient_data": {"age": 65, "medication_adherence": 0.6, "sleep_quality": 2, "stress_level": 9}
        }
    }
]

print("\n" + "="*50)
print("TESTING MULTIPLE SCENARIOS")
print("="*50)

for scenario in test_scenarios:
    try:
        response = requests.post(url, json=scenario["data"])
        if response.status_code == 200:
            result = response.json()
            print(f"\n🔬 {scenario['name']} Scenario:")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Probability: {result['flare_probability']:.1%}")
        else:
            print(f"❌ {scenario['name']} failed: {response.status_code}")
    except Exception as e:
        print(f"❌ {scenario['name']} error: {e}")
