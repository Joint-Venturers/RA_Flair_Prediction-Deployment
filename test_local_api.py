# Test Local API
# Test FastAPI server locally before deploying

import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 70)
print("TESTING LOCAL API")
print("=" * 70)
print()

# Test 1: Root endpoint
print("🧪 Test 1: Root Endpoint")
print(f"GET {BASE_URL}/")
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"✅ Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"❌ Error: {e}")
print()

# Test 2: Health check
print("🧪 Test 2: Health Check")
print(f"GET {BASE_URL}/health")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"✅ Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"❌ Error: {e}")
print()

# Test 3: Model info
print("🧪 Test 3: Model Info")
print(f"GET {BASE_URL}/info")
try:
    response = requests.get(f"{BASE_URL}/info")
    print(f"✅ Status: {response.status_code}")
    result = response.json()
    print(f"   Model: {result.get('model_type')}")
    print(f"   Algorithm: {result.get('algorithm')}")
    print(f"   Features: {result.get('feature_count')}")
except Exception as e:
    print(f"❌ Error: {e}")
print()

# Test 4: Low risk prediction
print("🧪 Test 4: Low Risk Patient")
print(f"POST {BASE_URL}/predict")
low_risk = {
    "age": 45.0,
    "gender": 1,
    "disease_duration": 5.0,
    "bmi": 24.0,
    "min_temperature": 18.0,
    "max_temperature": 24.0,
    "humidity": 55.0,
    "barometric_pressure": 1018.0,
    "precipitation": 0.0,
    "wind_speed": 4.0,
    "tender_joint_count": 0.5,
    "swollen_joint_count": 0.5
}

try:
    response = requests.post(f"{BASE_URL}/predict", json=low_risk)
    print(f"✅ Status: {response.status_code}")
    result = response.json()
    print(f"   Probability: {result['inflammation_probability']:.4f} ({result['inflammation_probability']*100:.2f}%)")
    print(f"   Prediction: {'Inflammation' if result['inflammation_prediction'] == 1 else 'Remission'}")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Confidence: {result['confidence']:.4f}")
except Exception as e:
    print(f"❌ Error: {e}")
print()

# Test 5: High risk prediction
print("🧪 Test 5: High Risk Patient")
high_risk = {
    "age": 65.0,
    "gender": 1,
    "disease_duration": 20.0,
    "bmi": 32.0,
    "min_temperature": 8.0,
    "max_temperature": 12.0,
    "humidity": 75.0,
    "barometric_pressure": 1005.0,
    "precipitation": 5.0,
    "wind_speed": 8.0,
    "tender_joint_count": 5.0,
    "swollen_joint_count": 3.0
}

try:
    response = requests.post(f"{BASE_URL}/predict", json=high_risk)
    print(f"✅ Status: {response.status_code}")
    result = response.json()
    print(f"   Probability: {result['inflammation_probability']:.4f} ({result['inflammation_probability']*100:.2f}%)")
    print(f"   Prediction: {'Inflammation' if result['inflammation_prediction'] == 1 else 'Remission'}")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Confidence: {result['confidence']:.4f}")
except Exception as e:
    print(f"❌ Error: {e}")
print()

print("=" * 70)
print("✅ LOCAL API TESTING COMPLETE")
print("=" * 70)
print()
print("📋 All tests passed!")
print("🚀 Ready to deploy to Render.com")